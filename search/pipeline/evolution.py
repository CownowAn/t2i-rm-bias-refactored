"""EvolutionEngine: orchestrates all evo steps (0..N) across topics."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING

from loguru import logger

from search.data.results import SearchResults, FoundAttribute
from search.data.state import TopicState
from search.pipeline.baselines import load_topic_states, load_baselines_from_manifest, score_baselines
from search.pipeline.evaluator import CounterfactualEvaluator
from search.pipeline.scoring import AmplificationScorer
from search.pipeline.selector import TopKSelector
from search.planner.cluster import AttributeClusterer
from search.planner.initial import InitialPlanner
from search.planner.mutator import AttributeMutator

if TYPE_CHECKING:
    from search.config import SearchConfig
    from search.logging.tracker import ExperimentTracker
    from search.models.base import RewardModel, JudgeModel, DetectorModel


class EvolutionEngine:
    """Runs the full evolutionary search for undesirable T2I attributes."""

    def __init__(
        self,
        config: "SearchConfig",
        topic_states: list[TopicState],
        reward_model: "RewardModel",
        judge_model: "JudgeModel",
        detector_model: "DetectorModel",
        evaluator: CounterfactualEvaluator,
        amp_scorer: AmplificationScorer,
        initial_planner: InitialPlanner,
        mutator: AttributeMutator,
        clusterer: AttributeClusterer,
        tracker: "ExperimentTracker",
    ):
        self.config = config
        self.topic_states = topic_states
        self.reward_model = reward_model
        self.judge_model = judge_model
        self.detector_model = detector_model
        self.evaluator = evaluator
        self.amp_scorer = amp_scorer
        self.initial_planner = initial_planner
        self.mutator = mutator
        self.clusterer = clusterer
        self.tracker = tracker

        self._rng = Random(config.run.random_seed)
        self._total_cost_usd = 0.0
        self._all_found: dict[tuple[str, int], FoundAttribute] = {}

    # ─── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: "SearchConfig",
        tracker: "ExperimentTracker",
    ) -> "EvolutionEngine":
        from search.models.reward.imagereward import ImageRewardModel
        from search.models.judge.vlm_judge import VisionLLMJudge
        from search.models.detector import build_detector
        from search.models.editor.instruction_gen import EditInstructionGenerator
        from search.models.editor.flux_kontext import FluxKontextApplier
        from search.utils.async_utils import GpuApplierPool

        reward_model = ImageRewardModel(
            device=config.models.reward_model.device,
            hf_cache_dir=config.models.reward_model.hf_cache_dir,
        )

        judge_model = VisionLLMJudge(
            model_name=config.models.judge.model,
            max_tokens=config.models.judge.max_tokens,
            max_parallel=config.models.judge.max_parallel,
        )

        detector_model = build_detector(config.models.detector)

        instruction_gen = EditInstructionGenerator(
            model_name=config.models.editor.instruction_model,
        )

        appliers = [
            FluxKontextApplier(
                model_name=config.models.editor.flux_model,
                device=device,
                guidance_scale=config.models.editor.guidance_scale,
            )
            for device in config.models.editor.flux_devices
        ]
        gpu_pool = GpuApplierPool(appliers)

        output_dir = config.run_output_dir() / "edited_images"

        evaluator = CounterfactualEvaluator(
            instruction_gen=instruction_gen,
            gpu_pool=gpu_pool,
            reward_model=reward_model,
            output_dir=output_dir,
        )
        amp_scorer = AmplificationScorer(detector=detector_model)

        initial_planner = InitialPlanner(
            model_name=config.models.planner.model,
            reasoning=config.models.planner.reasoning,
            max_tokens=config.models.planner.max_tokens,
            max_parallel=config.models.planner.max_parallel,
            n_attrs_per_prompt=config.evolution.n_attrs_per_prompt,
            n_per_user_prompt=config.evolution.n_per_user_prompt,
            n_context_imgs=config.evolution.n_context_imgs,
            n_initial_plan_prompts=config.evolution.n_initial_plan_prompts,
            initial_context_sampling=config.evolution.initial_context_sampling,
            use_cluster_summary=config.evolution.use_cluster_summary,
            direction=config.evolution.direction,
            order=config.evolution.image_order,
            random_seed=config.run.random_seed,
            score_normalization=config.evolution.initial_score_normalization,
            output_dir=config.run_output_dir() / "planner",
        )

        mutator = AttributeMutator(
            model_name=config.models.planner.model,
            reasoning=config.models.planner.reasoning,
            max_tokens=config.models.planner.max_tokens,
            max_parallel=config.models.planner.max_parallel,
            n_mutations=config.evolution.n_mutations,
            context=config.evolution.context,
            direction=config.evolution.direction,
            random_seed=config.run.random_seed,
            reg_min_pairs=config.evolution.reg_min_pairs,
            mutation_context_source=config.evolution.mutation_context_source,
            use_cluster_summary=config.evolution.use_cluster_summary,
            use_outlier_removal=config.evaluation.use_outlier_removal,
        )

        clusterer = AttributeClusterer(
            model_name=config.models.cluster.model,
            reasoning=config.models.cluster.reasoning,
            max_tokens=config.models.cluster.max_tokens,
            max_parallel=config.models.cluster.max_parallel,
        )

        topic_states = load_topic_states(
            prompts_dir=config.data.prompts_dir,
            topic_ids=config.data.topic_ids,
            val_split_size=config.data.val_split_size,
            random_seed=config.run.random_seed,
            summary_field=config.data.cluster_summary_field,
        )

        return cls(
            config=config,
            topic_states=topic_states,
            reward_model=reward_model,
            judge_model=judge_model,
            detector_model=detector_model,
            evaluator=evaluator,
            amp_scorer=amp_scorer,
            initial_planner=initial_planner,
            mutator=mutator,
            clusterer=clusterer,
            tracker=tracker,
        )

    # ─── Main run ─────────────────────────────────────────────────────────────

    async def run(self) -> SearchResults:
        t_start = time.time()
        cfg = self.config
        reward_name = cfg.models.reward_model.name

        # Load and score baselines
        logger.info("Loading and scoring baselines...")
        for ts in self.topic_states:
            load_baselines_from_manifest(ts, cfg.data.baseline_manifest, cfg.data.baseline_root)
        await asyncio.gather(*(score_baselines(ts, self.reward_model) for ts in self.topic_states))

        # Step 0: initial planning
        logger.info("=== Step 0: Initial Planning ===")
        await self.initial_planner.plan(self.topic_states, reward_model_name=reward_name)

        # Log candidates from initial planning
        for ts in self.topic_states:
            if ts.history:
                attrs = list(ts.history[0].attributes.keys())
                logger.info(f"Topic {ts.topic_id} step 0: {len(attrs)} candidates after planning")
                for a in attrs:
                    logger.info(f"  {a}")

        # Cluster/deduplicate step 0
        for ts in self.topic_states:
            if ts.history:
                await self._cluster_step(ts, step_idx=0, n_pop=cfg.evolution.initial_pop_size * 4)

        await self._evaluate_and_select(step_idx=0)
        n_steps_completed = 1

        # Steps 1..N-1: mutate + evaluate
        for step_idx in range(1, cfg.evolution.n_steps):
            logger.info(f"=== Step {step_idx}: Mutation ===")
            await self.mutator.mutate(self.topic_states)

            # Log new mutations vs carry-overs
            for ts in self.topic_states:
                if len(ts.history) > step_idx:
                    step = ts.history[step_idx]
                    new_attrs = [(a, s) for a, s in step.attributes.items()
                                 if s.meta.operation in ("mutate", "replan")]
                    carry_attrs = [(a, s) for a, s in step.attributes.items()
                                   if s.meta.operation == "carry_over"]
                    logger.info(
                        f"Topic {ts.topic_id} step {step_idx}: "
                        f"{len(new_attrs)} new mutation(s), {len(carry_attrs)} carry-over(s)"
                    )
                    for a, _ in new_attrs:
                        logger.info(f"  [new]   {a}")
                    for a, _ in carry_attrs:
                        logger.info(f"  [carry] {a}")

            # Replan fallback: if no truly undesirable survivors, add fresh candidates
            if cfg.evolution.replan_if_no_undesirable:
                topics_to_replan = [
                    ts for ts in self.topic_states
                    if not any(
                        self._all_found.get((attr, ts.topic_id),
                                           FoundAttribute(attribute="", delta_rm=0.0, delta_j=0.0,
                                                          amplification_score=0.0, step_found=0,
                                                          step_last_survived=0, topic_id=0,
                                                          is_undesirable=False)).is_undesirable
                        for attr in ts.surviving
                    )
                ]
                if topics_to_replan:
                    logger.info(
                        f"Step {step_idx}: no undesirable survivors in topic(s) "
                        f"{[ts.topic_id for ts in topics_to_replan]} — re-planning fresh candidates"
                    )
                    await self.initial_planner.plan_into_step(
                        topics_to_replan, step_idx, reward_name
                    )

            # Cluster after mutation (+ replan if triggered)
            for ts in self.topic_states:
                if len(ts.history) > step_idx:
                    n_pop = cfg.evolution.target_pop_sizes[step_idx] * 4
                    await self._cluster_step(ts, step_idx=step_idx, n_pop=n_pop)

            await self._evaluate_and_select(step_idx=step_idx)
            n_steps_completed += 1

        from search.utils.cost import estimate_cost
        estimated_cost = estimate_cost(cfg)["total"]

        results = SearchResults(
            run_id=cfg.run.name,
            config_snapshot=cfg.to_dict(),
            top_attributes=list(self._all_found.values()),
            n_steps_completed=n_steps_completed,
            cost_usd=estimated_cost,
            wall_time_seconds=time.time() - t_start,
        )
        return results

    async def shutdown(self) -> None:
        """Close all caller cache connections before the event loop exits."""
        for caller_owner in (self.initial_planner, self.mutator, self.clusterer,
                             self.judge_model, self.detector_model):
            if hasattr(caller_owner, "caller"):
                await caller_owner.caller.shutdown()

    # ─── Clustering helper ────────────────────────────────────────────────────

    async def _cluster_step(self, ts: TopicState, step_idx: int, n_pop: int) -> None:
        """Remove semantically duplicate attributes from a step (in-place)."""
        step = ts.history[step_idx]
        attrs = list(step.attributes.keys())
        if len(attrs) <= n_pop:
            return
        kept = await self.clusterer.cluster(
            attrs,
            cluster_summary=ts.cluster_summary,
            n_pop=n_pop,
        )
        kept_set = set(kept)
        removed = [a for a in attrs if a not in kept_set]
        for a in removed:
            del step.attributes[a]
        if removed:
            logger.info(
                f"Topic {ts.topic_id} step {step_idx}: "
                f"clustered {len(attrs)} → {len(step.attributes)} attributes"
            )

    # ─── Step helper ─────────────────────────────────────────────────────────

    async def _evaluate_and_select(self, step_idx: int) -> None:
        cfg = self.config
        eval_cfg = cfg.evaluation
        reward_name = cfg.models.reward_model.name
        pop_size = cfg.evolution.target_pop_sizes[step_idx]
        batch_size = eval_cfg.train_batch_size[step_idx]

        for ts in self.topic_states:
            if len(ts.history) <= step_idx:
                logger.warning(f"Topic {ts.topic_id}: no history at step {step_idx}")
                continue

            step = ts.history[step_idx]
            train_prompts = ts.train_prompts()

            # Sample a batch of prompts for this step
            rng = Random(cfg.run.random_seed + step_idx + ts.topic_id)
            batch_prompts = rng.sample(train_prompts, min(batch_size, len(train_prompts)))

            # A — counterfactual editing + ΔRM scoring
            await self.evaluator.evaluate_step(
                topic_state=ts,
                step=step,
                batch_prompts=batch_prompts,
                n_rollouts=eval_cfg.n_rollouts_per_prompt,
                reward_model_name=reward_name,
                rng=rng,
            )

            # B — judge scoring (ΔJ) on all batch prompts
            judge_prompts = batch_prompts
            await self._score_judge(ts, step, judge_prompts, eval_cfg.judge_first_n_rollouts)

            # C — filter to undesirable candidates before expensive A(g) computation
            ror = eval_cfg.use_outlier_removal
            scored_attrs = [
                (attr, s) for attr, s in step.attributes.items()
                if s.delta_rm(ror) is not None and s.delta_j() is not None
            ]
            undesirable = [(attr, s) for attr, s in scored_attrs if s.is_undesirable(ror)]
            if len(undesirable) >= pop_size * 2:
                amp_candidates = [attr for attr, _ in undesirable]
            else:
                # Pad to pop_size*2 buffer by ΔRM descending
                undesirable_set = {attr for attr, _ in undesirable}
                rest = sorted(
                    [(attr, s) for attr, s in scored_attrs if attr not in undesirable_set],
                    key=lambda x: x[1].delta_rm(ror) or float("-inf"),
                    reverse=True,
                )
                amp_candidates = [attr for attr, _ in undesirable] + [
                    attr for attr, _ in rest[: pop_size * 2 - len(undesirable)]
                ]

            logger.info(
                f"  Pre-filter: {len(scored_attrs)} scored → {len(amp_candidates)} "
                f"candidates for A(g) (undesirable={len(undesirable)}, pop_size={pop_size})"
            )

            # D — amplification scores A(g) on filtered candidates only
            # Sample prompts + images from the manifest baseline pool
            # amp_prompts = rng.sample(
            #     [p for p in train_prompts if p in ts.baselines],
            #     min(eval_cfg.amp_n_prompts, sum(1 for p in train_prompts if p in ts.baselines)),
            # )
            amp_prompts = batch_prompts
            amp_baselines: dict[str, list] = {
                p: rng.sample(ts.baselines[p], min(eval_cfg.amp_n_images_per_prompt, len(ts.baselines[p])))
                for p in amp_prompts
            }
            amp_scores = await self.amp_scorer.compute_batch(
                attributes=amp_candidates,
                baselines_by_prompt=amp_baselines,
                reward_model_name=reward_name,
                attrs_stats_map=step.attributes,
            )
            for attr, score in amp_scores.items():
                if attr in step.attributes:
                    step.attributes[attr].meta.amplification_score = score

            # E — TopK selection from amp-scored candidates
            amp_stats = [step.attributes[a] for a in amp_candidates if a in step.attributes]
            selector = TopKSelector(
                direction=cfg.evolution.direction,
                target_pop_size=pop_size,
                use_outlier_removal=eval_cfg.use_outlier_removal,
                strict_undesirable_selection=cfg.evolution.strict_undesirable_selection,
            )
            result = selector.select(amp_stats, ts.surviving, step_idx, ts.topic_id)
            ts.surviving = result.surviving
            for fa in result.selected:
                key = (fa.attribute, fa.topic_id)
                if key not in self._all_found:
                    self._all_found[key] = fa
                else:
                    # Re-selected carry-over: update scores with latest evaluation
                    # step_found is already correct from selector (surviving.get(attr, step_idx))
                    prev = self._all_found[key]
                    self._all_found[key] = FoundAttribute(
                        attribute=fa.attribute,
                        delta_rm=fa.delta_rm,
                        delta_j=fa.delta_j,
                        amplification_score=fa.amplification_score,
                        step_found=fa.step_found,
                        step_last_survived=fa.step_last_survived,
                        topic_id=fa.topic_id,
                        is_undesirable=fa.is_undesirable or prev.is_undesirable,
                    )

            # Log
            should_log_images = (step_idx % cfg.logging.log_images_every_n_steps == 0)
            self.tracker.log_step(
                step_idx=step_idx,
                topic_id=ts.topic_id,
                stats_map=step.attributes,
                selected=result.selected,
                cost_usd=0.0,
                use_outlier_removal=eval_cfg.use_outlier_removal,
            )
            if should_log_images:
                self.tracker.log_image_pairs(
                    step_idx=step_idx,
                    topic_id=ts.topic_id,
                    stats_map=step.attributes,
                )

    async def _score_judge(
        self,
        topic_state: TopicState,
        step,
        judge_prompts: list[str],
        n_rollouts: int,
    ) -> None:
        """Populate delta_j for pairs in step using batched judge comparisons."""
        edited_paths: list[str] = []
        baseline_paths: list[str] = []
        prompt_texts: list[str] = []
        pair_refs: list[tuple[str, str, int]] = []  # (attr, prompt_text, pair_idx)

        for attr, stats in step.attributes.items():
            for prompt_text in judge_prompts:
                pairs = stats.pairs.get(prompt_text, [])
                for pidx, pair in enumerate(pairs[:n_rollouts]):
                    if (
                        pair.delta_j is not None
                        or not pair.edited_image_path.exists()
                        or not pair.baseline.image_path.exists()
                    ):
                        continue
                    edited_paths.append(str(pair.edited_image_path))
                    baseline_paths.append(str(pair.baseline.image_path))
                    prompt_texts.append(prompt_text)
                    pair_refs.append((attr, prompt_text, pidx))

        if not edited_paths:
            return

        logger.info(f"  Judge scoring {len(edited_paths)} pairs...")
        results = await self.judge_model.compare(
            image_A_paths=edited_paths,
            image_B_paths=baseline_paths,
            prompts=prompt_texts,
        )

        for (attr, prompt_text, pidx), result in zip(pair_refs, results):
            if result is None or result.score_diff is None:
                continue
            pairs = step.attributes[attr].pairs.get(prompt_text, [])
            if pidx < len(pairs):
                # score_diff: +1.0 = edited (A) wins, -1.0 = baseline (B) wins
                # delta_j < 0 means judge prefers baseline (undesirable attribute criterion)
                pairs[pidx].delta_j = result.score_diff
