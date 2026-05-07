"""BaselinePairEvolutionEngine: finds undesirable T2I attributes without image editing.

Step lifecycle
──────────────
EVALUATE  humanness filter → detect new attrs → μ1>μ0 → A(g) → top-K select
          → merges new detection into engine-level detection_cache
          → appends selected attrs to engine-level acc_pool
          → writes BaselinePairStep.{attribute_pool, acc_pool_snapshot, detection, amp_scores}

EXPAND    construct pairs using detection_cache (full acc_pool attr vectors)
          → [judge filter] → D matrix (acc_pool full cols) → linear regression (W_rm, residuals)
          → high-residual pairs → LLM proposes raw new attrs → EvoStep[step_idx+1]
          → writes BaselinePairStep.{pairs, D, delta_rm_vec, W_rm, residuals}

Key invariants
──────────────
• Fixed baselines (_fixed_baselines): sampled once per topic, reused every step.
• Cumulative detection cache (_detection_cache): only new attrs detected each step,
  merged in. Full cache used for regression → residuals = "what entire pool can't explain".
• acc_pool (_acc_pool): union of all selected attrs across steps. Regression always uses
  full acc_pool columns so residuals improve as pool grows.
• New proposed attrs enter EvoStep unfiltered; EVALUATE handles filtering next iteration.
• Bug guard: _expand_step checks bp_history[-1].step_idx == step_idx to avoid stale data.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from search.data.baseline_pair_types import BaselinePair, BaselinePairStep
from search.data.results import SearchResults, FoundAttribute
from search.data.state import TopicState, EvoStep, AttributeStats, AttributeMeta
from search.pipeline.baselines import load_topic_states, load_baselines_from_manifest, score_baselines
from search.pipeline.attribute_filter import AttributeUndesirabilityFilter
from search.pipeline.baseline_pair_constructor import (
    BaselinePairConstructor,
    AttributeStratifiedPairConstructor,
)
from search.planner.residual_proposer import ResidualAttributeProposer
from search.utils.linear_probing import compute_regression_residuals_from_matrix

if TYPE_CHECKING:
    from search.config import SearchConfig
    from search.logging.tracker import ExperimentTracker
    from search.data.types import BaselineImage


class BaselinePairEvolutionEngine:

    def __init__(
        self,
        config: "SearchConfig",
        topic_states: list[TopicState],
        reward_model,
        detector_model,
        judge_model,
        attr_filter: AttributeUndesirabilityFilter,
        pair_constructor: BaselinePairConstructor,
        residual_proposer: ResidualAttributeProposer,
        initial_planner,
        clusterer,
        tracker: "ExperimentTracker",
    ):
        self.config = config
        self.topic_states = topic_states
        self.reward_model = reward_model
        self.detector_model = detector_model
        self.judge_model = judge_model
        self.attr_filter = attr_filter
        self.pair_constructor = pair_constructor
        self.residual_proposer = residual_proposer
        self.initial_planner = initial_planner
        self.clusterer = clusterer
        self.tracker = tracker

        self._rng = Random(config.run.random_seed)
        self._all_found: dict[tuple[str, int], FoundAttribute] = {}

        # Per-topic state — populated during SETUP
        # Fixed baselines: sampled once, reused across all steps
        self._fixed_baselines: dict[int, dict[str, list]] = {}
        # Cumulative detection cache: {topic_id: {image_id: {attr: 0/1}}}
        self._detection_cache: dict[int, dict[str, dict[str, int]]] = {}
        # Accumulated attribute pool: union of all selected attrs across steps
        self._acc_pool: dict[int, list[str]] = {}
        # Permanently rejected attrs (humanness or μ1>μ0 failures) — step-independent
        self._rejected_pool: dict[int, list[str]] = {}

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: "SearchConfig",
        tracker: "ExperimentTracker",
    ) -> "BaselinePairEvolutionEngine":
        from search.models.reward.imagereward import ImageRewardModel
        from search.models.judge.vlm_judge import VisionLLMDetector, VisionLLMJudge
        from search.planner.initial import InitialPlanner
        from search.planner.cluster import AttributeClusterer

        reward_model = ImageRewardModel(
            device=config.models.reward_model.device,
            hf_cache_dir=config.models.reward_model.hf_cache_dir,
        )
        detector_model = VisionLLMDetector(
            model_name=config.models.detector.model,
            max_tokens=config.models.detector.max_tokens,
            max_parallel=config.models.detector.max_parallel,
            image_detail=config.models.detector.image_detail,
            use_batch_api=config.models.detector.use_batch_api,
        )
        judge_model = (
            VisionLLMJudge(
                model_name=config.models.judge.model,
                max_tokens=config.models.judge.max_tokens,
                max_parallel=config.models.judge.max_parallel,
                image_detail=config.models.judge.image_detail,
                use_batch_api=config.models.judge.use_batch_api,
            )
            if config.baseline_pairs.use_judge
            else None
        )
        attr_filter = AttributeUndesirabilityFilter(
            model_name=config.models.planner.model,
            max_tokens=config.models.planner.max_tokens,
            max_parallel=config.models.planner.max_parallel,
        )
        bp_cfg = config.baseline_pairs
        if bp_cfg.pair_constructor == "stratified":
            pair_constructor = AttributeStratifiedPairConstructor(
                n_pairs_per_stratum=bp_cfg.n_pairs_per_stratum,
            )
        else:
            pair_constructor = BaselinePairConstructor(
                n_pairs_per_prompt=bp_cfg.n_pairs_per_prompt,
            )
        residual_proposer = ResidualAttributeProposer(
            model_name=config.models.planner.model,
            reasoning=config.models.planner.reasoning,
            max_tokens=config.models.planner.max_tokens,
            max_parallel=config.models.planner.max_parallel,
            use_cluster_summary=config.evolution.use_cluster_summary,
        )
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
            require_editable=False,  # baseline-pairs: VLM detection, not FLUX editing
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
        )
        return cls(
            config=config, topic_states=topic_states,
            reward_model=reward_model, detector_model=detector_model, judge_model=judge_model,
            attr_filter=attr_filter, pair_constructor=pair_constructor,
            residual_proposer=residual_proposer,
            initial_planner=initial_planner, clusterer=clusterer, tracker=tracker,
        )

    # ── Main run ──────────────────────────────────────────────────────────────

    async def run(self) -> SearchResults:
        t_start = time.time()
        cfg = self.config
        bp_cfg = cfg.baseline_pairs
        eval_cfg = cfg.evaluation
        reward_name = cfg.models.reward_model.name

        # ── SETUP ────────────────────────────────────────────────────────────

        logger.info("=== Baseline-Pairs: Loading and scoring baselines ===")
        for ts in self.topic_states:
            load_baselines_from_manifest(ts, cfg.data.baseline_manifest, cfg.data.baseline_root)
        await asyncio.gather(*(score_baselines(ts, self.reward_model) for ts in self.topic_states))

        # Sample fixed baselines once per topic (reused across all steps)
        for ts in self.topic_states:
            rng = Random(cfg.run.random_seed + ts.topic_id + 77777)
            train_prompts = [
                p for p in ts.train_prompts() if p in ts.baselines and ts.baselines[p]
            ]
            sample_prompts = rng.sample(
                train_prompts, min(eval_cfg.amp_n_prompts, len(train_prompts))
            )
            self._fixed_baselines[ts.topic_id] = {
                p: rng.sample(
                    ts.baselines[p],
                    min(eval_cfg.amp_n_images_per_prompt, len(ts.baselines[p]))
                )
                for p in sample_prompts
            }
            self._detection_cache[ts.topic_id] = {}
            self._acc_pool[ts.topic_id] = []
            self._rejected_pool[ts.topic_id] = []
            if cfg.baseline_pairs.detection_cache_path:
                _model_key = f"{cfg.models.detector.model}::{cfg.models.detector.image_detail}"
                self._load_detection_cache(ts.topic_id, Path(cfg.baseline_pairs.detection_cache_path), _model_key)
            n_imgs = sum(len(v) for v in self._fixed_baselines[ts.topic_id].values())
            logger.info(
                f"Topic {ts.topic_id}: fixed baselines = "
                f"{len(sample_prompts)} prompts × up to {eval_cfg.amp_n_images_per_prompt} "
                f"imgs = {n_imgs} total"
            )

        logger.info("=== Step 0: Initial Planning ===")
        await self.initial_planner.plan(self.topic_states, reward_model_name=reward_name)

        for ts in self.topic_states:
            if ts.history:
                await self._cluster_step(ts, step_idx=0, n_pop=cfg.evolution.initial_pop_size * 2)

        # ── MAIN LOOP ────────────────────────────────────────────────────────

        n_steps_completed = 0
        for step_idx in range(cfg.evolution.n_steps):
            logger.info(f"=== Step {step_idx}: EVALUATE ===")
            any_evaluated = False
            for ts in self.topic_states:
                if len(ts.history) <= step_idx:
                    logger.warning(f"Topic {ts.topic_id}: no EvoStep at {step_idx}, skipping")
                    continue
                evaluated = await self._evaluate_step(ts, step_idx, reward_name)
                any_evaluated = any_evaluated or evaluated
            if any_evaluated:
                n_steps_completed += 1

            if step_idx < cfg.evolution.n_steps - 1:
                logger.info(f"=== Step {step_idx}: EXPAND ===")
                for ts in self.topic_states:
                    await self._expand_step(ts, step_idx, reward_name, bp_cfg)

        from search.utils.cost import estimate_cost_bp
        estimated_cost = estimate_cost_bp(cfg)["total"]

        return SearchResults(
            run_id=cfg.run.name,
            config_snapshot=cfg.to_dict(),
            top_attributes=list(self._all_found.values()),
            n_steps_completed=n_steps_completed,
            cost_usd=estimated_cost,
            wall_time_seconds=time.time() - t_start,
        )

    async def shutdown(self) -> None:
        for obj in (self.initial_planner, self.clusterer, self.attr_filter,
                    self.residual_proposer, self.detector_model):
            if hasattr(obj, "caller"):
                await obj.caller.shutdown()
            elif hasattr(obj, "shutdown"):
                await obj.shutdown()

    # ── EVALUATE ──────────────────────────────────────────────────────────────
    # Returns True if at least one attr was processed, False otherwise.

    # ── Detection cache persistence ───────────────────────────────────────────

    def _load_detection_cache(self, topic_id: int, path: Path, model_key: str) -> None:
        """Merge on-disk detection results for model_key into the in-memory cache.

        File format: {"{model}::{detail}": {image_id: {attr: 0|1}}}
        Only entries matching model_key and belonging to the current fixed baselines are loaded.
        """
        if not path.exists():
            logger.info(f"Detection cache file not found at {path}, starting fresh")
            return
        try:
            with open(path) as f:
                all_saved: dict = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load detection cache from {path}: {e}")
            return
        saved = all_saved.get(model_key, {})
        if not saved:
            logger.info(f"Topic {topic_id}: no cached entries for {model_key!r} in {path}")
            return
        fixed_image_ids = {
            img.image_id
            for imgs in self._fixed_baselines[topic_id].values()
            for img in imgs
        }
        n_loaded = 0
        cache = self._detection_cache[topic_id]
        for image_id, attr_vals in saved.items():
            if image_id not in fixed_image_ids:
                continue
            cache.setdefault(image_id, {}).update(attr_vals)
            n_loaded += len(attr_vals)
        logger.info(
            f"Topic {topic_id}: loaded detection cache [{model_key}] from {path} "
            f"({len(cache)} images, {n_loaded} attr entries)"
        )

    def _save_detection_cache(self, topic_id: int, path: Path, model_key: str) -> None:
        """Persist the in-memory detection cache for model_key to disk (merge with existing)."""
        cache = self._detection_cache[topic_id]
        if not cache:
            return
        all_existing: dict = {}
        if path.exists():
            try:
                with open(path) as f:
                    all_existing = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read existing cache at {path}, overwriting: {e}")
        model_cache = all_existing.setdefault(model_key, {})
        for image_id, attr_vals in cache.items():
            model_cache.setdefault(image_id, {}).update(attr_vals)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(all_existing, f)
        n_entries = sum(len(v) for v in model_cache.values())
        logger.debug(
            f"Topic {topic_id}: saved detection cache [{model_key}] → {path} "
            f"({len(model_cache)} images, {n_entries} attr entries)"
        )

    # ── EVALUATE ──────────────────────────────────────────────────────────────
    # Returns True if at least one attr was processed, False otherwise.

    async def _evaluate_step(
        self,
        ts: TopicState,
        step_idx: int,
        reward_name: str,
    ) -> bool:
        cfg = self.config
        step = ts.history[step_idx]
        fixed_baselines = self._fixed_baselines[ts.topic_id]
        detection_cache = self._detection_cache[ts.topic_id]
        acc_pool = self._acc_pool[ts.topic_id]

        rejected_pool = self._rejected_pool[ts.topic_id]

        attr_pool = list(step.attributes.keys())
        if not attr_pool:
            logger.warning(f"Topic {ts.topic_id} step {step_idx}: empty pool, skipping")
            return False
        if not fixed_baselines:
            logger.warning(f"Topic {ts.topic_id} step {step_idx}: no fixed baselines, skipping")
            return False

        # [A] Humanness filter (LLM text-only, cheap)
        before_humanness = set(attr_pool)
        attr_pool = await self.attr_filter.filter_by_humanness(attr_pool)
        humanness_failed = before_humanness - set(attr_pool)
        _add_to_rejected(rejected_pool, humanness_failed)
        _trim_step(step, attr_pool)
        if not attr_pool:
            logger.warning(f"Topic {ts.topic_id} step {step_idx}: all attrs failed humanness filter")
            return False

        # [B] Detect ONLY new attrs on fixed baselines (cache miss only)
        new_attrs_to_detect = [a for a in attr_pool if not _all_cached(a, detection_cache)]
        if new_attrs_to_detect:
            n_images = sum(len(v) for v in fixed_baselines.values())
            logger.info(
                f"  [B] Detecting {len(new_attrs_to_detect)} new attrs in {n_images} images"
            )
            new_det = await _detect_all_attributes(
                self.detector_model, new_attrs_to_detect, fixed_baselines
            )
            # Merge into cumulative cache
            for image_id, attr_vals in new_det.items():
                detection_cache.setdefault(image_id, {}).update(attr_vals)
        else:
            logger.debug(f"  [B] All {len(attr_pool)} attrs already in detection cache")

        # [C] μ1 > μ0 filter using full detection_cache
        # Always compute μ stats (used for logging); filter pool only when use_mu_filter=True.
        bp_cfg = cfg.baseline_pairs
        attr_pool_for_mu = attr_pool if bp_cfg.use_mu_filter else attr_pool[:]
        mu_passed, all_mu_stats = self.attr_filter.filter_by_mu(
            attr_pool_for_mu, detection_cache, fixed_baselines, reward_name
        )
        if bp_cfg.use_mu_filter:
            mu_failed = set(all_mu_stats.keys()) - set(mu_passed)
            mu_failed_stats = {a: all_mu_stats[a] for a in mu_failed}
            _add_to_rejected(rejected_pool, mu_failed)
            attr_pool = mu_passed
            _trim_step(step, attr_pool)
            if not attr_pool:
                logger.warning(f"Topic {ts.topic_id} step {step_idx}: all attrs failed μ1>μ0 filter")
                return False
        else:
            mu_failed_stats = {}
            logger.debug(f"  [C] μ filter skipped (use_mu_filter=False); {len(attr_pool)} attrs remain")

        # [D] A(g) for current attrs using detection_cache (no extra VLM calls)
        amp_scores = _compute_amp_from_detection(
            detection_cache, fixed_baselines, attr_pool, reward_name
        )

        # [E] A(g) > 0 filter (optional) then top-K selection
        if bp_cfg.use_amp_filter:
            amp_failed = {a for a in attr_pool if amp_scores.get(a, 0.0) <= 0}
            if amp_failed:
                logger.info(
                    f"  [E] A(g)>0 filter: removing {len(amp_failed)} attrs with A(g)≤0: "
                    + ", ".join(sorted(amp_failed))
                )
            _add_to_rejected(rejected_pool, amp_failed)
            attr_pool = [a for a in attr_pool if a not in amp_failed]
            _trim_step(step, attr_pool)
            if not attr_pool:
                logger.warning(f"Topic {ts.topic_id} step {step_idx}: all attrs failed A(g)>0 filter")
                return False

        pop_size = (
            cfg.evolution.target_pop_sizes[step_idx]
            if step_idx < len(cfg.evolution.target_pop_sizes)
            else cfg.evolution.target_pop_sizes[-1]
        )
        selected = sorted(attr_pool, key=lambda a: amp_scores.get(a, 0.0), reverse=True)[:pop_size]
        _trim_step(step, selected)

        # [F] Update engine-level acc_pool (dedup, preserve order)
        for attr in selected:
            if attr not in acc_pool:
                acc_pool.append(attr)
        # acc_pool is modified in-place (it's a list stored in self._acc_pool[topic_id])

        # [G] Update AttributeStats and _all_found
        for attr in selected:
            amp_score = amp_scores.get(attr, 0.0)
            step.attributes[attr].meta.amplification_score = amp_score
            key = (attr, ts.topic_id)
            if key not in self._all_found:
                self._all_found[key] = FoundAttribute(
                    attribute=attr, delta_rm=None, delta_j=None,
                    amplification_score=amp_score,
                    step_found=step_idx, step_last_survived=step_idx,
                    topic_id=ts.topic_id, is_undesirable=True,
                )
                ts.surviving[attr] = step_idx
            else:
                prev = self._all_found[key]
                self._all_found[key] = FoundAttribute(
                    attribute=attr, delta_rm=prev.delta_rm, delta_j=prev.delta_j,
                    amplification_score=amp_score,
                    step_found=prev.step_found, step_last_survived=step_idx,
                    topic_id=ts.topic_id, is_undesirable=True,
                )

        # [H] Store bp_step (EXPAND will fill pairs/regression fields)
        # amp_scores covers ALL acc_pool attrs (new + historical) for complete EXPAND logging
        ts.bp_history.append(BaselinePairStep(
            step_idx=step_idx,
            attribute_pool=selected,
            acc_pool_snapshot=list(acc_pool),
            detection=dict(new_det) if new_attrs_to_detect else {},
            amp_scores=amp_scores,
        ))

        # [I] Log EVALUATE phase
        self.tracker.log_bp_evaluate(
            step_idx=step_idx,
            topic_id=ts.topic_id,
            new_attrs=selected,
            amp_scores=amp_scores,
            acc_pool_size=len(acc_pool),
            humanness_failed=sorted(humanness_failed),
            mu_failed_stats=mu_failed_stats,
        )

        # [J] Persist detection cache to disk (if configured)
        if self.config.baseline_pairs.detection_cache_path:
            _model_key = f"{self.config.models.detector.model}::{self.config.models.detector.image_detail}"
            self._save_detection_cache(ts.topic_id, Path(self.config.baseline_pairs.detection_cache_path), _model_key)

        return True

    # ── EXPAND ────────────────────────────────────────────────────────────────

    async def _expand_step(
        self,
        ts: TopicState,
        step_idx: int,
        reward_name: str,
        bp_cfg,
    ) -> None:
        cfg = self.config

        # Guard: only proceed if EVALUATE produced a bp_step for this exact step
        if not ts.bp_history or ts.bp_history[-1].step_idx != step_idx:
            logger.info(
                f"Topic {ts.topic_id} step {step_idx}: "
                "EVALUATE produced no output, skipping EXPAND"
            )
            self._create_empty_next_step(ts, step_idx)
            return

        bp_step = ts.bp_history[-1]
        fixed_baselines = self._fixed_baselines[ts.topic_id]
        detection_cache = self._detection_cache[ts.topic_id]
        acc_pool = self._acc_pool[ts.topic_id]  # full accumulated pool

        if not acc_pool or not fixed_baselines:
            self._create_empty_next_step(ts, step_idx)
            return

        # [A] Construct pairs using FULL acc_pool attr vectors
        # Hamming distance is computed on all known attrs → pairs are meaningful across pool
        pairs = self.pair_constructor.construct(
            fixed_baselines, detection_cache, acc_pool, reward_name
        )

        # [B] Judge scoring + filter — before_regression position (optional)
        if (bp_cfg.use_judge and self.judge_model is not None and pairs
                and bp_cfg.judge_filter_position == "before_regression"):
            pairs = await self._judge_filter_pairs(pairs)

        bp_step.pairs = pairs

        if not pairs:
            logger.warning(f"  Topic {ts.topic_id}: no pairs after construction/judge filter")
            self._create_empty_next_step(ts, step_idx)
            return

        # [C] D matrix with FULL acc_pool columns → linear regression (W_rm, residuals)
        # residuals = "what entire accumulated pool can't explain"
        D, delta_rm_vec, pair_keys = _build_D_matrix(pairs, detection_cache, acc_pool)
        reg_result = compute_regression_residuals_from_matrix(
            D.astype(np.float32), delta_rm_vec.astype(np.float32),
            acc_pool, pair_keys,
            min_pairs=cfg.evolution.reg_min_pairs,
            fit_intercept=cfg.baseline_pairs.reg_fit_intercept,
            regression_model=cfg.baseline_pairs.regression_model,
            l1_ratio=cfg.baseline_pairs.elasticnet_l1_ratio,
            n_alphas=cfg.baseline_pairs.elasticnet_n_alphas,
            cv=cfg.baseline_pairs.elasticnet_cv,
        )
        W_rm = np.array([reg_result.attribute_weights.get(a, 0.0) for a in acc_pool])
        residuals_vec = np.array([reg_result.residuals.get(pk, 0.0) for pk in pair_keys])

        bp_step.D = D
        bp_step.delta_rm_vec = delta_rm_vec
        bp_step.W_rm = W_rm
        bp_step.reg_intercept = reg_result.reg_intercept
        bp_step.reg_alpha = reg_result.reg_alpha
        bp_step.reg_l1_ratio = reg_result.l1_ratio
        bp_step.residuals = residuals_vec

        # Update delta_rm in _all_found from regression weights
        for k, attr in enumerate(acc_pool):
            key = (attr, ts.topic_id)
            if key in self._all_found:
                prev = self._all_found[key]
                self._all_found[key] = FoundAttribute(
                    attribute=attr, delta_rm=float(W_rm[k]), delta_j=prev.delta_j,
                    amplification_score=prev.amplification_score,
                    step_found=prev.step_found, step_last_survived=step_idx,
                    topic_id=ts.topic_id, is_undesirable=float(W_rm[k]) > 0,
                )

        # [B] Judge scoring + filter — before_residual_select position (optional)
        # bp_step.residuals[i] corresponds to bp_step.pairs[i], so remap both together
        if (bp_cfg.use_judge and self.judge_model is not None and bp_step.pairs
                and bp_cfg.judge_filter_position == "before_residual_select"):
            judge_passed = await self._judge_filter_pairs(bp_step.pairs)
            passed_ids = {id(p) for p in judge_passed}
            kept_idx = [i for i, p in enumerate(bp_step.pairs) if id(p) in passed_ids]
            bp_step.pairs = [bp_step.pairs[i] for i in kept_idx]
            if bp_step.residuals is not None:
                bp_step.residuals = bp_step.residuals[kept_idx]
            _log_attr_stats_after_judge(bp_step.pairs, acc_pool, detection_cache, ts.topic_id)

        # [D] Select high-residual pairs
        high_res_pairs = _select_high_residual_pairs(bp_step, bp_cfg.n_high_residual_pairs)

        # [E] LLM proposes new attrs — full acc_pool + rejected pool as context
        proposed, diverse_pairs = await self.residual_proposer.propose(
            topic_state=ts,
            high_residual_pairs=high_res_pairs,
            current_pool=acc_pool,
            detection=detection_cache,
            n_proposed=bp_cfg.n_proposed_per_step,
            avoid_attrs=self._rejected_pool[ts.topic_id],
        )

        # [F] EvoStep[step_idx+1] with raw proposals (EVALUATE filters next iteration)
        next_step = EvoStep(step_idx=step_idx + 1)
        for attr in proposed:
            next_step.attributes[attr] = AttributeStats(
                attribute=attr,
                meta=AttributeMeta(
                    time_step=step_idx + 1, parent=None, parent_time_step=None,
                    operation="residual_proposed",
                    planner_model=cfg.models.planner.model,
                    reasoning_effort=cfg.models.planner.reasoning,
                ),
            )
        ts.history.append(next_step)

        # [G] Log EXPAND phase
        # Build acc_pool A(g) from _all_found (already stored at EVALUATE time)
        all_amp_scores = {
            attr: self._all_found[(attr, ts.topic_id)].amplification_score
            for attr in acc_pool
            if (attr, ts.topic_id) in self._all_found
        }
        self.tracker.log_bp_expand(
            step_idx=step_idx,
            topic_id=ts.topic_id,
            bp_step=bp_step,
            acc_pool=acc_pool,
            proposed_attrs=proposed,
            high_residual_pairs=high_res_pairs,
            diverse_pairs=diverse_pairs,
            all_amp_scores=all_amp_scores,
            output_dir=self.config.run_output_dir(),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _cluster_step(self, ts: TopicState, step_idx: int, n_pop: int) -> None:
        step = ts.history[step_idx]
        attrs = list(step.attributes.keys())
        if len(attrs) <= n_pop:
            return
        kept = await self.clusterer.cluster(attrs, cluster_summary=ts.cluster_summary, n_pop=n_pop)
        kept_set = set(kept)
        for a in list(step.attributes.keys()):
            if a not in kept_set:
                del step.attributes[a]
        logger.info(
            f"Topic {ts.topic_id} step {step_idx}: clustered {len(attrs)} → {len(step.attributes)}"
        )

    async def _judge_filter_pairs(self, pairs: list[BaselinePair]) -> list[BaselinePair]:
        """Keep pairs where judge prefers the low-reward image (or ties): RM bias evidence."""
        results = await self.judge_model.compare(
            image_A_paths=[str(p.high_reward.image_path) for p in pairs],
            image_B_paths=[str(p.low_reward.image_path) for p in pairs],
            prompts=[p.high_reward.prompt.text for p in pairs],
        )
        filtered = []
        for pair, result in zip(pairs, results):
            if result is None or result.score_diff is None:
                continue
            pair.delta_j = result.score_diff
            pair.judge_reasoning = getattr(result, "reasoning", None)
            # score_diff ≤ 0: judge doesn't prefer the high-reward image → RM bias evidence
            if result.score_diff <= 0:
                filtered.append(pair)
        logger.info(f"  Judge filter: {len(pairs)} → {len(filtered)} pairs (δj ≤ 0)")
        return filtered

    def _create_empty_next_step(self, ts: TopicState, step_idx: int) -> None:
        ts.history.append(EvoStep(step_idx=step_idx + 1))
        logger.debug(f"Topic {ts.topic_id}: empty EvoStep added for step {step_idx + 1}")


# ── Module-level helpers ──────────────────────────────────────────────────────


def _log_attr_stats_after_judge(
    pairs: list,
    acc_pool: list[str],
    detection_cache: dict[str, dict[str, int]],
    topic_id: str,
) -> None:
    N = len(pairs)
    col_counts = []
    for attr_k in acc_pool:
        n_pos = sum(
            1 for p in pairs
            if (detection_cache.get(p.high_reward.image_id, {}).get(attr_k, 0)
                - detection_cache.get(p.low_reward.image_id, {}).get(attr_k, 0)) > 0
        )
        col_counts.append(n_pos)
        status = "✓" if n_pos > 0 else "✗"
        logger.info(
            f"  {status} [{attr_k[:45]:45s}] pairs={n_pos:3d}/{N}  D_col={n_pos/N:.0%}" if N > 0
            else f"  ✗ [{attr_k[:45]:45s}] pairs=0/0"
        )
    if col_counts and N > 0:
        densities = [c / N for c in col_counts]
        logger.info(
            f"  Topic {topic_id}: after judge filter — {N} pairs  "
            f"D_col min={min(densities):.0%}  mean={float(np.mean(densities)):.0%}  max={max(densities):.0%}"
        )


def _add_to_rejected(rejected_pool: list[str], new_rejects: set[str]) -> None:
    """Append newly rejected attrs to the pool, avoiding duplicates."""
    for attr in new_rejects:
        if attr not in rejected_pool:
            rejected_pool.append(attr)


def _all_cached(attr: str, detection_cache: dict[str, dict[str, int]]) -> bool:
    """True if every image in cache already has a detection entry for this attr."""
    if not detection_cache:
        return False
    return all(attr in v for v in detection_cache.values())


def _trim_step(step: EvoStep, keep: list[str]) -> None:
    keep_set = set(keep)
    for a in list(step.attributes.keys()):
        if a not in keep_set:
            del step.attributes[a]


async def _detect_all_attributes(
    detector_model,
    attrs_to_detect: list[str],
    amp_baselines: dict[str, list["BaselineImage"]],
) -> dict[str, dict[str, int]]:
    """Returns {image_id: {attr: 0/1}}.  One batched VLM call per attribute."""
    all_images: list["BaselineImage"] = []
    all_prompts: list[str] = []
    for prompt, images in amp_baselines.items():
        for img in images:
            all_images.append(img)
            all_prompts.append(prompt)

    detection: dict[str, dict[str, int]] = {}
    for attr in attrs_to_detect:
        det_results = await detector_model.detect(
            [str(img.image_path) for img in all_images], all_prompts, attr
        )
        for img, d in zip(all_images, det_results):
            detection.setdefault(img.image_id, {})[attr] = int(d)
    return detection


def _build_D_matrix(
    pairs: list[BaselinePair],
    detection: dict[str, dict[str, int]],
    attr_pool: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """D[i,k] = detection(high_i, k) - detection(low_i, k).  Rows=pairs, cols=acc_pool."""
    K, N = len(attr_pool), len(pairs)
    D = np.zeros((N, K), dtype=np.float32)
    delta_rm_vec = np.zeros(N, dtype=np.float32)
    pair_keys: list[str] = []
    for i, pair in enumerate(pairs):
        hi = detection.get(pair.high_reward.image_id, {})
        lo = detection.get(pair.low_reward.image_id, {})
        for k, attr in enumerate(attr_pool):
            D[i, k] = float(hi.get(attr, 0)) - float(lo.get(attr, 0))
        delta_rm_vec[i] = pair.delta_rm
        pair_keys.append(f"{pair.high_reward.image_id}|{pair.low_reward.image_id}")
    return D, delta_rm_vec, pair_keys


def _compute_amp_from_detection(
    detection: dict[str, dict[str, int]],
    amp_baselines: dict[str, list["BaselineImage"]],
    attr_pool: list[str],
    reward_model_name: str,
) -> dict[str, float]:
    """A(g) = E_x[p1*p0*(μ1-μ0)].  Uses pre-computed detection cache, no extra VLM calls."""
    amp_scores: dict[str, float] = {}
    for attr in attr_pool:
        per_prompt: list[float] = []
        per_p1: list[float] = []
        per_p0: list[float] = []
        per_mu1: list[float] = []
        per_mu0: list[float] = []

        for prompt_text, images in amp_baselines.items():
            scored = [
                img for img in images
                if reward_model_name in img.reward_scores
                and img.image_id in detection
                and attr in detection[img.image_id]
            ]
            if len(scored) < 2:
                continue
            rewards = [img.reward_scores[reward_model_name] for img in scored]
            dets = [detection[img.image_id][attr] for img in scored]
            g1 = [r for r, d in zip(rewards, dets) if d == 1]
            g0 = [r for r, d in zip(rewards, dets) if d == 0]
            if not g1 and not g0:
                logger.debug(
                    f"  A(g) '{attr}' | '{prompt_text}': skipped — attr undetected in all images"
                )
                continue
            if not g1:
                logger.debug(
                    f"  A(g) '{attr}' | '{prompt_text}': skipped — "
                    f"attr never present (g1=0, len(g0)={len(g0)}, p1=0)"
                )
                per_prompt.append(0.0)
                per_p1.append(0.0)
                per_p0.append(1.0)
                per_mu1.append(0.0)
                per_mu0.append(float(np.mean(g0)) if g0 else 0.0)
                continue
            if not g0:
                logger.debug(
                    f"  A(g) '{attr}' | '{prompt_text}': skipped — "
                    f"attr always present (len(g1)={len(g1)}, g0=0, p0=0)"
                )
                per_prompt.append(0.0)
                per_p1.append(1.0)
                per_p0.append(0.0)
                per_mu1.append(float(np.mean(g1)) if g1 else 0.0)
                per_mu0.append(0.0)
                continue
            n = len(scored)
            p1 = len(g1) / n
            p0 = len(g0) / n
            mu1 = float(np.mean(g1))
            mu0 = float(np.mean(g0))
            prompt_score = p1 * p0 * (mu1 - mu0)
            per_prompt.append(prompt_score)
            per_p1.append(p1)
            per_p0.append(p0)
            per_mu1.append(mu1)
            per_mu0.append(mu0)
            logger.info(
                f"  A(g) '{attr}' | '{prompt_text}': "
                f"p1={p1:.3f} p0={p0:.3f} μ1={mu1:.3f} μ0={mu0:.3f} "
                f"→ {prompt_score:.4f}"
            )

        score = float(np.mean(per_prompt)) if per_prompt else 0.0
        amp_scores[attr] = score

        if per_p1:
            logger.info(
                f"  A(g) '{attr}': {score:.4f}  "
                f"(p1={np.mean(per_p1):.3f} p0={np.mean(per_p0):.3f} "
                f"μ1={np.mean(per_mu1):.3f} μ0={np.mean(per_mu0):.3f} "
                f"over {len(per_p1)} prompts)"
            )
        else:
            logger.info(f"  A(g) '{attr}': 0.0 (no valid prompts)")

    return amp_scores


def _select_high_residual_pairs(bp_step: BaselinePairStep, n: int) -> list[BaselinePair]:
    if bp_step.residuals is None or not bp_step.pairs:
        return []
    pairs = bp_step.pairs
    abs_res = np.abs(bp_step.residuals)
    # Primary: delta_j < 0 first (strong judge disagreement > tie)
    # Secondary: |residual| descending
    # Falls back to pure residual order when delta_j is unavailable (judge disabled)
    indices = sorted(
        range(len(pairs)),
        key=lambda i: (pairs[i].delta_j is not None and pairs[i].delta_j < 0, float(abs_res[i])),
        reverse=True,
    )
    return [pairs[i] for i in indices[:n]]
