"""BonAmplifiedEvolutionEngine: BoN-amplified bias discovery on individual images.

Implements NEW_SEARCH_ALGO.md:
  - Unit of analysis: individual images (not pairs)
  - A(g) = N · E_x[Cov_x(g, U^{N-1})]  via existing 'bon' amp_mode
  - Residuals: within-prompt centered OLS of U^{N-1} on g matrix
  - Mining: P+/P- images per prompt → one VLM call → m proposals
  - Validation: detect proposed candidates, keep only A_hat > tau

Step lifecycle
──────────────
SETUP     load_baselines → score_baselines → InitialPlanner.plan
          → [★] humanness filter on initial pool (before any detection)
          → (optional) AttributeClusterer

EVALUATE  detect new attrs (cache miss only)
          → A(g) with bon formula
          → prune: A_hat > tau, then top-K or select_all
          → acc_pool update → BonAmplifiedStep append → log

EXPAND    _compute_bon_residuals: within-prompt OLS of U^{N-1} on g matrix
          → _extract_pplus_pminus: top/bottom n_top images per prompt
          → BonResidualProposer.propose → raw candidates
          → [★] humanness filter on candidates (before detection)
          → detect + A_hat > tau validate
          → EvoStep[step_idx+1] with validated attrs

Key invariants
──────────────
• Fixed baselines (_fixed_baselines): sampled once, reused every step.
• Cumulative detection cache (_detection_cache): only new attrs detected each step.
• acc_pool (_acc_pool): union of all selected attrs across steps.
• Humanness filter applied twice: after InitialPlanner.plan, and after VLM propose.
• Validated candidates only: A_hat > tau is enforced before adding to next step.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import random
import time
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from search.data.bon_amplified_types import BonAmplifiedStep
from search.data.results import SearchResults, FoundAttribute
from search.data.state import TopicState, EvoStep, AttributeStats, AttributeMeta
from search.pipeline.baselines import load_topic_states, load_baselines_from_manifest, score_baselines
from search.pipeline.attribute_filter import AttributeUndesirabilityFilter
from search.utils.io import save_json
from search.pipeline.baseline_evo import (
    _compute_amp_from_detection,
    _detect_all_attributes,
    _all_cached,
    _trim_step,
    _add_to_rejected,
)
from search.planner.bon_residual_proposer import BonResidualProposer

if TYPE_CHECKING:
    from search.config import SearchConfig
    from search.logging.tracker import ExperimentTracker
    from search.data.types import BaselineImage


class BonAmplifiedEvolutionEngine:

    def __init__(
        self,
        config: "SearchConfig",
        topic_states: list[TopicState],
        reward_model,
        detector_model,
        attr_filter: AttributeUndesirabilityFilter,
        residual_proposer: BonResidualProposer,
        initial_planner,
        clusterer,
        tracker: "ExperimentTracker",
    ):
        self.config = config
        self.topic_states = topic_states
        self.reward_model = reward_model
        self.detector_model = detector_model
        self.attr_filter = attr_filter
        self.residual_proposer = residual_proposer
        self.initial_planner = initial_planner
        self.clusterer = clusterer
        self.tracker = tracker

        self._rng = Random(config.run.random_seed)
        self._all_found: dict[tuple[str, int], FoundAttribute] = {}

        self._fixed_baselines: dict[int, dict[str, list]] = {}
        self._detection_cache: dict[int, dict[str, dict[str, int]]] = {}
        self._acc_pool: dict[int, list[str]] = {}
        self._rejected_pool: dict[int, list[str]] = {}
        # Rejected attrs loaded from the detection cache (previous runs). Kept separate
        # from _rejected_pool (current run) so the proposer avoid list can exclude them.
        self._cached_rejected: dict[int, list[str]] = {}
        # First step at which each attr was admitted to the pool (for age tracking)
        self._attr_first_seen: dict[int, dict[str, int]] = {}
        # Cumulative per-prompt R² history across steps (topic_id → list of dicts)
        self._per_prompt_r2_history: dict[int, list[dict]] = {}

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: "SearchConfig",
        tracker: "ExperimentTracker",
    ) -> "BonAmplifiedEvolutionEngine":
        from search.models.detector import build_detector
        from search.planner.initial import InitialPlanner
        from search.planner.cluster import AttributeClusterer

        rm_name = config.models.reward_model.name

        # Skip loading the heavy reward model if every baseline in the manifest
        # already has reward_scores[rm_name] — score_baselines() would early-return
        # anyway, so the GPU load is pure waste (esp. for HPSv3 ~22 GB / ~15 min NFS).
        from search.pipeline.baselines import all_baselines_have_scores
        if all_baselines_have_scores(config.data.baseline_manifest, rm_name):
            from search.models.reward.noop import NoOpRewardModel
            logger.info(
                f"All baselines in {config.data.baseline_manifest} already have "
                f"reward_scores['{rm_name}'] — skipping reward model GPU load."
            )
            reward_model = NoOpRewardModel(rm_name)
        elif rm_name == "pickscore":
            from search.models.reward.pickscore import PickScoreModel
            reward_model = PickScoreModel(
                device=config.models.reward_model.device,
                hf_cache_dir=config.models.reward_model.hf_cache_dir,
            )
        elif rm_name == "hpsv3":
            from search.models.reward.hpsv3 import HPSv3Model
            reward_model = HPSv3Model(
                device=config.models.reward_model.device,
                hf_cache_dir=config.models.reward_model.hf_cache_dir,
            )
        else:
            from search.models.reward.imagereward import ImageRewardModel
            reward_model = ImageRewardModel(
                device=config.models.reward_model.device,
                hf_cache_dir=config.models.reward_model.hf_cache_dir,
            )

        cache_config = config.caller_cache.build()
        detector_model = build_detector(config.models.detector, cache_config=cache_config)
        attr_filter = AttributeUndesirabilityFilter(
            model_name=config.models.attr_filter.model,
            max_tokens=config.models.attr_filter.max_tokens,
            max_parallel=config.models.attr_filter.max_parallel,
            cache_config=cache_config,
        )
        residual_proposer = BonResidualProposer(
            model_name=config.models.proposer.model,
            reasoning=config.models.proposer.reasoning,
            max_tokens=config.models.proposer.max_tokens,
            image_detail=config.models.proposer.image_detail,
            output_dir=config.run_output_dir() / "proposer",
            cache_config=cache_config,
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
            require_editable=False,
            n_prompts_per_plan_call=config.evolution.n_prompts_per_plan_call,
            score_normalization=config.evolution.initial_score_normalization,
            output_dir=config.run_output_dir() / "planner",
            cache_config=cache_config,
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
            detector_model=detector_model,
            attr_filter=attr_filter, 
            residual_proposer=residual_proposer,
            initial_planner=initial_planner, 
            clusterer=clusterer,
            tracker=tracker,
        )

    # ── Main run ──────────────────────────────────────────────────────────────

    async def run(self) -> SearchResults:
        t_start = time.time()
        cfg = self.config
        ba_cfg = cfg.bon_amplified
        eval_cfg = cfg.evaluation
        reward_name = cfg.models.reward_model.name

        # ── SETUP ────────────────────────────────────────────────────────────

        logger.info("=== BoN-Amplified: Loading and scoring baselines ===")
        for ts in self.topic_states:
            load_baselines_from_manifest(ts, cfg.data.baseline_manifest, cfg.data.baseline_root)
        await asyncio.gather(*(score_baselines(ts, self.reward_model) for ts in self.topic_states))

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
            self._cached_rejected[ts.topic_id] = []
            self._attr_first_seen[ts.topic_id] = {}
            self._per_prompt_r2_history[ts.topic_id] = []

            if ba_cfg.detection_cache_path:
                _model_key = f"{cfg.models.detector.model}::{cfg.models.detector.image_detail}"
                self._load_detection_cache(
                    ts.topic_id, Path(ba_cfg.detection_cache_path), _model_key
                )
            n_imgs = sum(len(v) for v in self._fixed_baselines[ts.topic_id].values())
            logger.info(
                f"Topic {ts.topic_id}: fixed baselines = "
                f"{len(sample_prompts)} prompts × up to {eval_cfg.amp_n_images_per_prompt} "
                f"imgs = {n_imgs} total"
            )

        logger.info("=== Step 0: Initial pool ===")
        if ba_cfg.initial_pool_path:
            # Load a pre-filtered pool from a file or a previous run's planner/ dir;
            # planner + humanness + clustering are skipped.
            self._load_initial_pool_into_step0(Path(ba_cfg.initial_pool_path))
        else:
            fixed_prompts = {
                ts.topic_id: list(self._fixed_baselines[ts.topic_id].keys())
                for ts in self.topic_states
            }
            await self.initial_planner.plan(
                self.topic_states, reward_model_name=reward_name, fixed_prompts=fixed_prompts,
            )

            # [★] Humanness filter on initial pool — before any detection
            for ts in self.topic_states:
                if not ts.history:
                    continue
                step = ts.history[0]
                attr_pool = list(step.attributes.keys())
                if attr_pool:
                    passed = await self.attr_filter.filter_by_humanness(attr_pool)
                    failed = set(attr_pool) - set(passed)
                    _add_to_rejected(self._rejected_pool[ts.topic_id], failed)
                    _trim_step(step, passed)
                    self._save_planner_stage("humanness_initial", ts.topic_id, 0, {
                        "input_attrs": attr_pool,
                        "passed": passed,
                        "failed": sorted(failed),
                        "n_in": len(attr_pool),
                        "n_out": len(passed),
                    })
                    logger.info(
                        f"Topic {ts.topic_id}: initial humanness filter: "
                        f"{len(attr_pool)} → {len(passed)} attrs"
                    )

            for ts in self.topic_states:
                if ts.history:
                    await self._cluster_step(ts, step_idx=0, n_pop=cfg.evolution.initial_pop_size)

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

            logger.info(f"=== Step {step_idx}: EXPAND ===")
            for ts in self.topic_states:
                await self._expand_step(ts, step_idx, reward_name, ba_cfg)

        # Dump per-prompt R² history per topic (#8)
        for ts in self.topic_states:
            self.tracker.log_per_prompt_r2_history(
                topic_id=ts.topic_id,
                history=self._per_prompt_r2_history.get(ts.topic_id, []),
                output_dir=cfg.run_output_dir(),
            )

        from search.utils.cost import estimate_cost_ba
        estimated_cost = estimate_cost_ba(cfg)["total"]

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
    # Returns True if at least one attr was processed.

    async def _evaluate_step(
        self,
        ts: TopicState,
        step_idx: int,
        reward_name: str,
    ) -> bool:
        cfg = self.config
        ba_cfg = cfg.bon_amplified
        step = ts.history[step_idx]
        fixed_baselines = self._fixed_baselines[ts.topic_id]
        detection_cache = self._detection_cache[ts.topic_id]
        acc_pool = self._acc_pool[ts.topic_id]

        attr_pool = list(step.attributes.keys())
        if not attr_pool:
            logger.warning(f"Topic {ts.topic_id} step {step_idx}: empty pool, skipping")
            return False
        if not fixed_baselines:
            logger.warning(f"Topic {ts.topic_id} step {step_idx}: no fixed baselines, skipping")
            return False

        # [A] Detect only new attrs (cache miss)
        new_attrs_to_detect = [a for a in attr_pool if not _all_cached(a, detection_cache)]
        new_det: dict[str, dict[str, int]] = {}
        if new_attrs_to_detect:
            n_images = sum(len(v) for v in fixed_baselines.values())
            logger.info(
                f"  [A] Detecting {len(new_attrs_to_detect)} new attrs in {n_images} images"
            )
            new_det = await _detect_all_attributes(
                self.detector_model, 
                new_attrs_to_detect, 
                fixed_baselines,
                existing_cache=detection_cache,
            )
            for image_id, attr_vals in new_det.items():
                detection_cache.setdefault(image_id, {}).update(attr_vals)
        else:
            logger.debug(f"  [A] All {len(attr_pool)} attrs already in detection cache")

        # [B] A(g) with BoN formula: N · E_x[Cov_x(g, U^{N-1})]
        amp_scores = _compute_amp_from_detection(
            detection_cache, fixed_baselines, attr_pool, reward_name,
            amp_mode="bon", bon_n=ba_cfg.N,
        )

        # [C] Prune: A_hat > tau, then top-K or select_all
        amp_failed = {a for a in attr_pool if amp_scores.get(a, 0.0) <= ba_cfg.tau}
        if amp_failed:
            logger.info(
                f"  [C] A(g)≤{ba_cfg.tau} filter: removing {len(amp_failed)} attrs: "
                + ", ".join(sorted(amp_failed))
            )
            _add_to_rejected(self._rejected_pool[ts.topic_id], amp_failed)
            attr_pool = [a for a in attr_pool if a not in amp_failed]
            _trim_step(step, attr_pool)
        if not attr_pool:
            logger.warning(f"Topic {ts.topic_id} step {step_idx}: all attrs failed A(g) filter")
            return False

        # [C-p1] Prevalence filter: reject attrs whose global p1 is outside [p1_min, p1_max]
        if ba_cfg.use_p1_filter:
            total_cached = sum(
                1 for imgs in fixed_baselines.values()
                for img in imgs if img.image_id in detection_cache
            )
            p1_failed: set[str] = set()
            for attr in attr_pool:
                present = sum(
                    1 for imgs in fixed_baselines.values()
                    for img in imgs
                    if img.image_id in detection_cache
                    and detection_cache[img.image_id].get(attr, 0) == 1
                )
                global_p1 = present / total_cached if total_cached > 0 else 0.0
                if not (ba_cfg.p1_min <= global_p1 <= ba_cfg.p1_max):
                    p1_failed.add(attr)
                    logger.info(
                        f"  [C-p1] p1={global_p1:.3f} out of [{ba_cfg.p1_min},{ba_cfg.p1_max}]"
                        f" → rejected: {attr[:60]}"
                    )
            if p1_failed:
                _add_to_rejected(self._rejected_pool[ts.topic_id], p1_failed)
                attr_pool = [a for a in attr_pool if a not in p1_failed]
                _trim_step(step, attr_pool)
                logger.info(f"  [C-p1] prevalence filter removed {len(p1_failed)} attrs")
            if not attr_pool:
                logger.warning(f"Topic {ts.topic_id} step {step_idx}: all attrs failed p1 filter")
                return False

        if ba_cfg.use_monotonic_pool:
            # Monotonic mode: tau / p1 filters already applied above; skip Top-K
            # (pool grows without artificial K cap; quality enforced by tau and p1)
            selected = sorted(attr_pool, key=lambda a: amp_scores.get(a, 0.0), reverse=True)
        elif ba_cfg.select_all_passing:
            selected = sorted(attr_pool, key=lambda a: amp_scores.get(a, 0.0), reverse=True)
        else:
            pop_size = (
                cfg.evolution.target_pop_sizes[step_idx]
                if step_idx < len(cfg.evolution.target_pop_sizes)
                else cfg.evolution.target_pop_sizes[-1]
            )
            selected = sorted(attr_pool, key=lambda a: amp_scores.get(a, 0.0), reverse=True)[:pop_size]
        _trim_step(step, selected)

        # [D] Update acc_pool = S_t (replace, not accumulate)
        # Pseudocode: S_t = Top K elements of B_t. acc_pool holds S_t for this step's regression.
        acc_pool[:] = selected

        # Track first-seen step for each attr (for age display in pool snapshots)
        first_seen = self._attr_first_seen[ts.topic_id]
        for attr in selected:
            if attr not in first_seen:
                first_seen[attr] = step_idx

        # [E] Update _all_found
        for attr in selected:
            amp_score = amp_scores.get(attr, 0.0)
            step.attributes[attr].meta.amplification_score = amp_score
            key = (attr, ts.topic_id)
            if key not in self._all_found:
                self._all_found[key] = FoundAttribute(
                    attribute=attr, 
                    delta_rm=None, 
                    delta_j=None,
                    amplification_score=amp_score,
                    step_found=step_idx, 
                    step_last_survived=step_idx,
                    topic_id=ts.topic_id, 
                    is_undesirable=True,
                )
                ts.surviving[attr] = step_idx
            else:
                prev = self._all_found[key]
                self._all_found[key] = FoundAttribute(
                    attribute=attr, 
                    delta_rm=prev.delta_rm, 
                    delta_j=prev.delta_j,
                    amplification_score=amp_score,
                    step_found=prev.step_found, 
                    step_last_survived=step_idx,
                    topic_id=ts.topic_id, 
                    is_undesirable=True,
                )

        # [F] Append BonAmplifiedStep + log
        ts.ba_history.append(BonAmplifiedStep(
            step_idx=step_idx, 
            N=ba_cfg.N,
            attribute_pool=selected,
            acc_pool_snapshot=list(acc_pool),
            detection=new_det,
            amp_scores=amp_scores,
            n_images=sum(len(v) for v in fixed_baselines.values()),
        ))

        self.tracker.log_bon_evaluate(
            step_idx=step_idx,
            topic_id=ts.topic_id,
            N=ba_cfg.N,
            selected=selected,
            amp_scores=amp_scores,
            tau_rejected=list(amp_failed),
            tau=ba_cfg.tau,
            first_seen=self._attr_first_seen[ts.topic_id],
        )

        if ba_cfg.detection_cache_path:
            _model_key = f"{cfg.models.detector.model}::{cfg.models.detector.image_detail}"
            self._save_detection_cache(
                ts.topic_id, Path(ba_cfg.detection_cache_path), _model_key
            )

        return True

    # ── EXPAND ────────────────────────────────────────────────────────────────

    async def _expand_step(
        self,
        ts: TopicState,
        step_idx: int,
        reward_name: str,
        ba_cfg,
    ) -> None:
        cfg = self.config

        if not ts.ba_history or ts.ba_history[-1].step_idx != step_idx:
            logger.info(
                f"Topic {ts.topic_id} step {step_idx}: "
                "EVALUATE produced no output, skipping EXPAND"
            )
            self._create_empty_next_step(ts, step_idx)
            return

        ba_step = ts.ba_history[-1]
        fixed_baselines = self._fixed_baselines[ts.topic_id]
        detection_cache = self._detection_cache[ts.topic_id]
        acc_pool = self._acc_pool[ts.topic_id]

        if not acc_pool or not fixed_baselines:
            self._create_empty_next_step(ts, step_idx)
            return

        # [A] Within-prompt centered OLS residuals of U^{N-1}
        ols_mode = "per_prompt" if ba_cfg.use_per_prompt_ols else "global"
        residuals, W, var_exp, mean_abs, max_abs, per_prompt_r2, per_prompt_W = (
            _compute_bon_residuals(
                detection_cache, fixed_baselines, acc_pool,
                reward_name, ba_cfg.N, mode=ols_mode,
            )
        )
        ba_step.residuals = {f"{k[0]}||{k[1]}": v for k, v in residuals.items()}
        ba_step.n_images = len(residuals)
        ba_step.W = W
        ba_step.W_mode = "mean_per_prompt" if ba_cfg.use_per_prompt_ols else "global"
        ba_step.reg_var_explained = var_exp
        ba_step.per_prompt_r2 = per_prompt_r2
        ba_step.per_prompt_W = per_prompt_W or {}

        # Append per-prompt R² to history (#8)
        self._per_prompt_r2_history[ts.topic_id].append({
            "step": step_idx, **per_prompt_r2,
        })
        logger.info(
            f"  [A] OLS [{ols_mode}]: N={len(residuals)} imgs, K={len(acc_pool)} attrs  "
            f"var_explained={var_exp:.4f}  mean|r|={mean_abs:.4f}  max|r|={max_abs:.4f}"
        )

        # [B] P+/P- extraction
        P_plus, P_minus = _extract_pplus_pminus(
            residuals, fixed_baselines, ba_cfg.n_top_residual,
            reward_name=reward_name,
            selection=ba_cfg.pplus_pminus_selection,
            reward_tol=ba_cfg.pplus_pminus_reward_tol,
            rng=self._rng,
        )
        logger.info(f"  [B] P+/P-: {len(P_plus)} prompts with both P+ and P- sets")

        # P+/P- per-prompt residual range (top-5 by spread) — #3 diagnostic
        if P_plus:
            ranges = []
            for prompt_text in P_plus:
                ress = [residuals[(prompt_text, img.image_id)]
                        for img in P_plus[prompt_text] + P_minus.get(prompt_text, [])]
                if ress:
                    ranges.append((prompt_text, min(ress), max(ress)))
            ranges.sort(key=lambda t: t[2] - t[1], reverse=True)
            logger.info(f"  [B] top-5 prompts by residual spread:")
            for prompt_text, lo, hi in ranges[:5]:
                logger.info(
                    f"      spread={hi - lo:+.3f}  range=[{lo:+.3f}, {hi:+.3f}]  '{prompt_text}'"
                )

        is_last_step = (step_idx >= cfg.evolution.n_steps - 1)
        if is_last_step or not P_plus:
            if is_last_step:
                logger.info("  Last step — skipping proposal")
            self._create_empty_next_step(ts, step_idx)
            # Still log OLS results even when skipping proposal
            self.tracker.log_bon_expand(
                step_idx=step_idx,
                topic_id=ts.topic_id,
                ba_step=ba_step,
                acc_pool=acc_pool,
                p_plus_n_prompts=len(P_plus),
                raw_candidates=[],
                humanness_rejected=[],
                validated=[],
                tau_rejected_cands=[],
                candidate_ahat={},
                tau=ba_cfg.tau,
                output_dir=self.config.run_output_dir(),
            )
            return

        # [C] VLM propose raw candidates (n_proposer_calls times, deduped across calls)
        raw_candidates: list[str] = []
        accumulated_proposals: list[str] = []
        for call_idx in range(ba_cfg.n_proposer_calls):
            batch = await self.residual_proposer.propose(
                p_plus=P_plus,
                p_minus=P_minus,
                # this-step proposals go into current_pool ("already identified"),
                # not avoid_attrs ("unsuitable") — dedup without the negative signal
                current_pool=acc_pool + accumulated_proposals,
                n_proposals=ba_cfg.n_proposals,
                n_prompts_vlm=ba_cfg.n_prompts_vlm,
                avoid_attrs=(
                    self._rejected_pool[ts.topic_id]
                    + ([] if ba_cfg.proposer_avoid_current_run_only
                       else self._cached_rejected.get(ts.topic_id, []))
                ),
                cluster_summary=ts.cluster_summary if ba_cfg.proposer_use_cluster_summary else None,
                per_prompt_r2=per_prompt_r2,
                selection_strategy=ba_cfg.prompt_select_strategy,
                exclude_pct=ba_cfg.prompt_select_exclude_pct,
                call_idx=call_idx,
                step_idx=step_idx,
                topic_id=ts.topic_id,
            )
            accumulated_proposals.extend(batch)
            raw_candidates.extend(batch)
            if ba_cfg.n_proposer_calls > 1:
                logger.info(f"  [C] call {call_idx + 1}/{ba_cfg.n_proposer_calls}: {len(batch)} proposals")
        raw_candidates_before_humanness = list(raw_candidates)
        logger.info(
            f"  [C] VLM proposed {len(raw_candidates)} raw candidates "
            f"({ba_cfg.n_proposer_calls} call(s))"
        )

        # [★] Humanness filter on candidates before detection
        humanness_failed: set[str] = set()
        if raw_candidates:
            passed_humanness = await self.attr_filter.filter_by_humanness(raw_candidates)
            humanness_failed = set(raw_candidates) - set(passed_humanness)
            _add_to_rejected(self._rejected_pool[ts.topic_id], humanness_failed)
            raw_candidates = passed_humanness
            logger.info(f"  [★] Humanness filter: {len(raw_candidates)} candidates pass")

        # [D] Detect + validate A_hat > tau
        validated: list[str] = []
        tau_rejected: list[str] = []
        cand_amp: dict[str, float] = {}
        partial_ahats: dict[str, float] = {}
        if raw_candidates:
            new_det = await _detect_all_attributes(
                self.detector_model, raw_candidates, fixed_baselines,
                existing_cache=detection_cache,
            )
            for image_id, attr_vals in new_det.items():
                detection_cache.setdefault(image_id, {}).update(attr_vals)

            # A_hat is always computed (for logging/diagnostics)
            cand_amp = _compute_amp_from_detection(
                detection_cache, fixed_baselines, raw_candidates, reward_name,
                amp_mode="bon", bon_n=ba_cfg.N,
            )

            if ba_cfg.use_monotonic_pool:
                # ── Monotonic mode: admit via partial_A_hat + Top-K' ─────────
                partial_ahats = _compute_partial_a_hat(
                    candidates=raw_candidates,
                    detection_cache=detection_cache,
                    residuals=residuals,
                    fixed_baselines=fixed_baselines,
                    N=ba_cfg.N,
                )
                passing = [
                    c for c in raw_candidates
                    if partial_ahats.get(c, 0.0) > ba_cfg.tau_partial
                ]
                passing.sort(key=lambda c: partial_ahats[c], reverse=True)
                validated = passing[:ba_cfg.n_admit_per_step]
                tau_rejected = [c for c in raw_candidates if c not in validated]

                for cand in validated:
                    logger.info(
                        f"  [D] Admitted '{cand}'  "
                        f"partial_A_hat={partial_ahats[cand]:+.4f}  "
                        f"A_hat={cand_amp.get(cand, 0.0):+.4f}"
                    )
                for cand in tau_rejected:
                    pa = partial_ahats.get(cand, 0.0)
                    reason = (
                        "≤ tau_partial" if pa <= ba_cfg.tau_partial
                        else f"not in Top-{ba_cfg.n_admit_per_step}"
                    )
                    _add_to_rejected(self._rejected_pool[ts.topic_id], {cand})
                    logger.info(
                        f"  [D] Rejected  '{cand}'  partial_A_hat={pa:+.4f}  ({reason})"
                    )
            else:
                # ── Legacy A_hat / tau validation ─────────────────────────────
                partial_ahats = {}
                for cand in raw_candidates:
                    a_hat = cand_amp.get(cand, 0.0)
                    if a_hat > ba_cfg.tau:
                        validated.append(cand)
                        logger.info(f"  [D] Validated '{cand}'  A_hat={a_hat:.4f}")
                    else:
                        tau_rejected.append(cand)
                        _add_to_rejected(self._rejected_pool[ts.topic_id], {cand})
                        logger.info(
                            f"  [D] Rejected  '{cand}'  A_hat={a_hat:.4f} (≤ tau={ba_cfg.tau})"
                        )

            if ba_cfg.detection_cache_path:
                _model_key = f"{cfg.models.detector.model}::{cfg.models.detector.image_detail}"
                self._save_detection_cache(
                    ts.topic_id, Path(ba_cfg.detection_cache_path), _model_key
                )

        logger.info(
            f"  [EXPAND] step {step_idx} topic {ts.topic_id}: "
            f"{len(validated)}/{len(raw_candidates)} validated → EvoStep[{step_idx + 1}]"
        )

        # [E] Create next EvoStep: B_{t+1} = S_t ∪ {admitted candidates}
        # S_t = acc_pool (selected from EVALUATE); admitted = validated
        next_step = EvoStep(step_idx=step_idx + 1)
        prev_attrs = ts.history[step_idx].attributes
        for attr in acc_pool:  # S_t carried forward (already detected, cache will hit)
            prev = prev_attrs.get(attr)
            if prev is not None:
                # Preserve original provenance (planner_model / reasoning / prompt);
                # only bump the step index and mark as survived.
                meta = dataclasses.replace(
                    prev.meta, 
                    time_step=step_idx + 1, 
                    operation="survived"
                )
            else:
                meta = AttributeMeta(
                    time_step=step_idx + 1, 
                    parent=None, 
                    parent_time_step=None,
                    operation="survived",
                    planner_model=cfg.models.planner.model,
                    reasoning_effort=cfg.models.planner.reasoning,
                )
            next_step.attributes[attr] = AttributeStats(attribute=attr, meta=meta)
        for attr in validated:  # newly admitted candidates — produced by the residual proposer
            if attr not in next_step.attributes:
                next_step.attributes[attr] = AttributeStats(
                    attribute=attr,
                    meta=AttributeMeta(
                        time_step=step_idx + 1,
                        parent=None,
                        parent_time_step=None,
                        operation="bon_residual_proposed",
                        planner_model=cfg.models.proposer.model,
                        reasoning_effort=cfg.models.proposer.reasoning,
                    ),
                )
        ts.history.append(next_step)

        # [F] Log EXPAND phase
        self.tracker.log_bon_expand(
            step_idx=step_idx,
            topic_id=ts.topic_id,
            ba_step=ba_step,
            acc_pool=acc_pool,
            p_plus_n_prompts=len(P_plus),
            raw_candidates=raw_candidates_before_humanness,
            humanness_rejected=list(humanness_failed),
            validated=validated,
            tau_rejected_cands=tau_rejected,
            candidate_ahat=cand_amp,
            candidate_partial_ahat=partial_ahats if ba_cfg.use_monotonic_pool else None,
            tau=ba_cfg.tau,
            output_dir=self.config.run_output_dir(),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_initial_pool_into_step0(self, path: Path) -> None:
        """Populate each topic's history[0] from a pool file or a previous run's
        planner/ dir (no planner / humanness / clustering)."""
        for ts in self.topic_states:
            attrs = (_pool_from_planner_dir(path, ts.topic_id) if path.is_dir()
                     else _pool_from_file(path, ts.topic_id))
            step = EvoStep(step_idx=0)
            ts.history.append(step)
            for a in attrs:
                a = str(a).strip()
                if a and a not in step.attributes:
                    step.attributes[a] = AttributeStats(
                        attribute=a,
                        meta=AttributeMeta(
                            time_step=0, parent=None, parent_time_step=None,
                            operation="initial", planner_model="loaded_pool",
                            reasoning_effort=None,
                        ),
                    )
            logger.info(
                f"Topic {ts.topic_id}: loaded {len(step.attributes)} attrs "
                f"from initial pool {path}"
            )

    def _save_planner_stage(self, stage: str, topic_id: int, step_idx: int, payload: dict) -> None:
        """Persist a step-0 planner-stage artifact (humanness / clustering) into the
        run's planner/ subdir, alongside the InitialPlanner per-call JSONs."""
        out_dir = self.config.run_output_dir() / "planner"
        record = {"stage": stage, "topic_id": topic_id, "step_idx": step_idx, **payload}
        save_json(record, out_dir / f"{stage}_step{step_idx}_topic{topic_id}.json")

    async def _cluster_step(self, ts: TopicState, step_idx: int, n_pop: int) -> None:
        step = ts.history[step_idx]
        attrs = list(step.attributes.keys())
        if len(attrs) <= n_pop:
            return
        kept, clusters, reasoning = await self.clusterer.cluster(
            attrs, cluster_summary=ts.cluster_summary, n_pop=n_pop, return_clusters=True
        )
        kept_set = set(kept)
        dropped = [a for a in attrs if a not in kept_set]
        for a in list(step.attributes.keys()):
            if a not in kept_set:
                del step.attributes[a]
        self._save_planner_stage("clustering", ts.topic_id, step_idx, {
            "input_attrs": attrs,
            "kept": kept,
            "dropped": dropped,
            "n_in": len(attrs),
            "n_out": len(kept),
            "clusters": clusters,
            "reasoning": reasoning,
        })
        logger.info(
            f"Topic {ts.topic_id} step {step_idx}: clustered {len(attrs)} → {len(step.attributes)}"
        )

    def _create_empty_next_step(self, ts: TopicState, step_idx: int) -> None:
        ts.history.append(EvoStep(step_idx=step_idx + 1))
        logger.debug(f"Topic {ts.topic_id}: empty EvoStep added for step {step_idx + 1}")

    def _load_detection_cache(self, topic_id: int, path: Path, model_key: str) -> None:
        if not path.exists():
            logger.info(f"Detection cache not found at {path}, starting fresh")
            return
        try:
            with open(path) as f:
                all_saved: dict = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load detection cache from {path}: {e}")
            return
        saved = all_saved.get(model_key, {})
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
        rejected_key = f"_rejected::{model_key}::{topic_id}"
        saved_rejected: list[str] = all_saved.get(rejected_key, [])
        cached = self._cached_rejected.setdefault(topic_id, [])
        existing_rejected = set(cached)
        added = 0
        for attr in saved_rejected:
            if attr not in existing_rejected:
                cached.append(attr)
                existing_rejected.add(attr)
                added += 1
        if added:
            logger.info(f"Topic {topic_id}: loaded {added} rejected attrs from cache")

    def _save_detection_cache(self, topic_id: int, path: Path, model_key: str) -> None:
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
        rejected_key = f"_rejected::{model_key}::{topic_id}"
        existing_rejected = set(all_existing.get(rejected_key, []))
        existing_rejected.update(self._rejected_pool[topic_id])
        existing_rejected.update(self._cached_rejected.get(topic_id, []))
        all_existing[rejected_key] = sorted(existing_rejected)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(all_existing, f)
        n_entries = sum(len(v) for v in model_cache.values())
        logger.debug(
            f"Topic {topic_id}: saved detection cache [{model_key}] → {path} "
            f"({len(model_cache)} images, {n_entries} attr entries)"
        )


# ── Module-level helpers ──────────────────────────────────────────────────────


def _pool_from_planner_dir(d: Path, topic_id: int) -> list[str]:
    """Reconstruct the final step-0 pool from a run's planner/ dir:
    clustering 'kept' (if present) -> humanness 'passed' -> planner parsed_attributes union."""
    from search.utils.io import load_json
    cl = d / f"clustering_step0_topic{topic_id}.json"
    if cl.exists():
        return list(load_json(cl).get("kept", []))
    hu = d / f"humanness_initial_step0_topic{topic_id}.json"
    if hu.exists():
        return list(load_json(hu).get("passed", []))
    out: list[str] = []
    seen: set[str] = set()
    for f in sorted(d.glob(f"initial_step0_topic{topic_id}_call*.json")):
        for a in (load_json(f).get("parsed_attributes") or []):
            a = str(a).strip()
            if a and a not in seen:
                seen.add(a)
                out.append(a)
    return out


def _pool_from_file(path: Path, topic_id: int) -> list[str]:
    """Load an attribute pool from a JSON file: list (shared across topics) /
    per-topic dict {"0": [...]} / common dict with a 'kept'|'passed'|'attributes'|'acc_pool' list."""
    from search.utils.io import load_json
    data = load_json(path)
    if isinstance(data, list):
        return list(data)
    if isinstance(data, dict):
        if str(topic_id) in data:
            return list(data[str(topic_id)])
        if topic_id in data:
            return list(data[topic_id])
        for k in ("kept", "passed", "attributes", "acc_pool"):
            if isinstance(data.get(k), list):
                return list(data[k])
    return []


def _compute_bon_residuals(
    detection_cache: "dict[str, dict[str, int]]",
    fixed_baselines: "dict[str, list[BaselineImage]]",
    acc_pool: list[str],
    reward_model_name: str,
    N: int,
    mode: str = "global",
) -> "tuple[dict[tuple[str, str], float], list[float], float, float, float, dict[str, float], dict[str, list[float]] | None]":
    """Within-prompt centered OLS of U^{N-1} on g matrix.

    For each prompt x:
      1. Compute U_{x,i} = #{j: r_j ≤ r_i} / M_x  (empirical BoN quantile)
      2. Center both G_x (M_x × K) and U_pow_x = U^{N-1}  within the prompt

    mode="global":
      Stack all prompts and solve a single global lstsq → residuals.
    mode="per_prompt":
      Fit a separate W_x per prompt → per-prompt residuals.

    Returns (7-tuple):
      residuals   : (prompt_text, image_id) → residual e_{x,i}
      W           : list[float] regression weights (K,) — global W or mean(W_x)
      var_explained: 1 - Var(resid) / Var(u_all)   (global mode)
                     or mean(per-prompt R²)        (per_prompt mode)
      mean_abs_res: mean |residual|
      max_abs_res : max  |residual|
      per_prompt_r2: dict[prompt → R²_x]  (always populated, for diagnostics)
      per_prompt_W : dict[prompt → W_x as list[float]] or None
                     (only populated when mode="per_prompt")

    Prompts with <2 fully-detected scored images are skipped.
    """
    K = len(acc_pool)
    _empty = ({}, [0.0] * K, 0.0, 0.0, 0.0, {}, None)
    if not acc_pool:
        return _empty

    # ── Collect per-prompt centered (G_x_c, U_pow_c) ──────────────────────────
    per_prompt_data: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    all_keys: list[tuple[str, str]] = []

    for prompt_text, images in fixed_baselines.items():
        scored = [
            img for img in images
            if reward_model_name in img.reward_scores
            and img.image_id in detection_cache
            and all(attr in detection_cache[img.image_id] for attr in acc_pool)
        ]
        if len(scored) < 2:
            continue

        n = len(scored)
        rewards = np.array(
            [img.reward_scores[reward_model_name] for img in scored], dtype=float
        )

        sorted_r = np.sort(rewards)
        U = np.searchsorted(sorted_r, rewards, side="right") / n
        U_pow = U ** (N - 1)

        G_x = np.array(
            [[float(detection_cache[img.image_id].get(attr, 0)) for attr in acc_pool]
             for img in scored]
        )

        G_x_c = G_x - G_x.mean(axis=0, keepdims=True)
        U_pow_c = U_pow - U_pow.mean()

        ids = [img.image_id for img in scored]
        per_prompt_data[prompt_text] = (G_x_c, U_pow_c, ids)
        all_keys.extend((prompt_text, iid) for iid in ids)

    if not per_prompt_data:
        return _empty

    # ── Per-prompt OLS (always — used for diagnostics + per_prompt mode) ──────
    per_prompt_r2: dict[str, float] = {}
    per_prompt_W: dict[str, np.ndarray] = {}
    per_prompt_resid: dict[str, np.ndarray] = {}

    for prompt_text, (G_x_c, U_x_c, _ids) in per_prompt_data.items():
        var_u_x = float(np.var(U_x_c))
        if var_u_x < 1e-10 or G_x_c.shape[0] < G_x_c.shape[1]:
            W_x = np.zeros(K)
            res_x = U_x_c.copy()
            r2_x = 0.0
        else:
            W_x, *_ = np.linalg.lstsq(G_x_c, U_x_c, rcond=None)
            res_x = U_x_c - G_x_c @ W_x
            r2_x = max(0.0, 1.0 - float(np.var(res_x)) / var_u_x)
        per_prompt_W[prompt_text] = W_x
        per_prompt_resid[prompt_text] = res_x
        per_prompt_r2[prompt_text] = float(r2_x)

    # ── Branch on mode ────────────────────────────────────────────────────────
    if mode == "per_prompt":
        # Residuals are per-prompt residuals
        residuals_dict: dict[tuple[str, str], float] = {}
        residual_chunks: list[np.ndarray] = []
        for prompt_text, (_, _, ids) in per_prompt_data.items():
            res_x = per_prompt_resid[prompt_text]
            residual_chunks.append(res_x)
            for iid, r in zip(ids, res_x):
                residuals_dict[(prompt_text, iid)] = float(r)
        residuals_vec = np.concatenate(residual_chunks)

        # W output: mean across prompts (for backward-compat logging)
        W_stack = np.stack(list(per_prompt_W.values()))   # (P, K)
        W_out = W_stack.mean(axis=0).tolist()

        # var_explained: mean of per-prompt R²
        var_exp = float(np.mean(list(per_prompt_r2.values())))

        # per-prompt W output
        per_prompt_W_out: dict[str, list[float]] | None = {
            p: w.tolist() for p, w in per_prompt_W.items()
        }
    else:
        # ── Global OLS (legacy behaviour) ─────────────────────────────────────
        G_all = np.vstack([d[0] for d in per_prompt_data.values()])
        u_all = np.concatenate([d[1] for d in per_prompt_data.values()])

        if G_all.shape[0] < G_all.shape[1]:
            residuals_vec = u_all
            W_vec = np.zeros(K)
        else:
            W_vec, *_ = np.linalg.lstsq(G_all, u_all, rcond=None)
            residuals_vec = u_all - G_all @ W_vec

        residuals_dict = {key: float(r) for key, r in zip(all_keys, residuals_vec)}
        var_u = float(np.var(u_all))
        var_exp = (max(0.0, 1.0 - float(np.var(residuals_vec)) / var_u)
                   if var_u > 1e-10 else 0.0)
        W_out = W_vec.tolist()
        per_prompt_W_out = None

    mean_abs = float(np.mean(np.abs(residuals_vec)))
    max_abs  = float(np.max(np.abs(residuals_vec)))

    return residuals_dict, W_out, var_exp, mean_abs, max_abs, per_prompt_r2, per_prompt_W_out


def _compute_partial_a_hat(
    candidates: list[str],
    detection_cache: "dict[str, dict[str, int]]",
    residuals: "dict[tuple[str, str], float]",
    fixed_baselines: "dict[str, list[BaselineImage]]",
    N: int,
) -> dict[str, float]:
    """For each candidate g*, compute partial_A_hat = N × E_x[Cov_x(g*, residuals)].

    Centering within each prompt before computing covariance. Prompts where the
    candidate has <2 valid images (with detection AND residual) are skipped.
    """
    result: dict[str, float] = {}
    for g_star in candidates:
        cov_per_prompt: list[float] = []
        for prompt_text, images in fixed_baselines.items():
            valid = [
                img for img in images
                if img.image_id in detection_cache
                and g_star in detection_cache[img.image_id]
                and (prompt_text, img.image_id) in residuals
            ]
            if len(valid) < 2:
                continue
            g_x = np.array(
                [float(detection_cache[img.image_id][g_star]) for img in valid]
            )
            e_x = np.array(
                [residuals[(prompt_text, img.image_id)] for img in valid]
            )
            cov_x = float(np.mean((g_x - g_x.mean()) * (e_x - e_x.mean())))
            cov_per_prompt.append(cov_x)
        result[g_star] = (
            float(N) * float(np.mean(cov_per_prompt)) if cov_per_prompt else 0.0
        )
    return result


def _match_by_reward(pos_pool, neg_pool, n_top, reward_tol=None):
    """Greedy: largest-residual P+ first, match each to the reward-closest unused P-.

    pos_pool/neg_pool elements = (img, residual, reward). Returns (P+ imgs, P- imgs)
    of equal length (≤ n_top); P+ and P- at the same index form a reward-matched pair.
    """
    pos_sorted = sorted(pos_pool, key=lambda t: -t[1])   # largest residual first (informative)
    selected_pos, selected_neg = [], []
    used_neg: set[int] = set()
    for p_img, _p_e, p_r in pos_sorted:
        best, best_diff = None, float("inf")
        for cand in neg_pool:
            n_img, _n_e, n_r = cand
            if id(n_img) in used_neg:
                continue
            diff = abs(p_r - n_r)
            if reward_tol is not None and diff > reward_tol:
                continue
            if diff < best_diff:
                best_diff, best = diff, cand
        if best is None:
            continue
        selected_pos.append(p_img)
        selected_neg.append(best[0])
        used_neg.add(id(best[0]))
        if len(selected_pos) >= n_top:
            break
    return selected_pos, selected_neg


def _extract_pplus_pminus(
    residuals: "dict[tuple[str, str], float]",
    fixed_baselines: "dict[str, list[BaselineImage]]",
    n_top: int = 2,
    *,
    reward_name: str = "",
    selection: str = "extreme",
    reward_tol: float | None = None,
    rng=None,
) -> "tuple[dict[str, list[BaselineImage]], dict[str, list[BaselineImage]]]":
    """Return P+ (positive-residual) / P- (negative-residual) images per prompt.

    P_plus[prompt]  = images with POSITIVE residual (more BoN-friendly than predicted)
    P_minus[prompt] = images with NEGATIVE residual (less BoN-friendly than predicted)

    Prompts lacking either positive- or negative-residual images are skipped, so a
    proposed attribute is contrasted between genuinely over- and under-predicted images.

    selection (how n_top are picked within each sign-split pool):
      "extreme"        — most-positive / most-negative residual (default)
      "reward_matched" — greedy match so P+/P- have similar reward (reward_tol caps |Δreward|)
      "random"         — random n_top from each pool (uses rng if provided)
    """
    P_plus: dict[str, list] = {}
    P_minus: dict[str, list] = {}
    _rng = rng or random

    for prompt_text, images in fixed_baselines.items():
        scored = [
            (img, residuals[(prompt_text, img.image_id)], img.reward_scores.get(reward_name, 0.0))
            for img in images
            if (prompt_text, img.image_id) in residuals
        ]
        if len(scored) < 2:
            continue
        scored.sort(key=lambda t: t[1])
        neg = [t for t in scored if t[1] < 0]   # negative residual
        pos = [t for t in scored if t[1] > 0]   # positive residual
        if not neg or not pos:
            continue

        if selection == "reward_matched":
            sel_pos, sel_neg = _match_by_reward(pos, neg, n_top, reward_tol)
            if not sel_pos:
                continue
            P_plus[prompt_text] = sel_pos
            P_minus[prompt_text] = sel_neg
        elif selection == "random":
            P_plus[prompt_text] = [t[0] for t in _rng.sample(pos, min(n_top, len(pos)))]
            P_minus[prompt_text] = [t[0] for t in _rng.sample(neg, min(n_top, len(neg)))]
        else:  # "extreme"
            P_minus[prompt_text] = [t[0] for t in neg[:n_top]]     # most-negative n_top
            P_plus[prompt_text] = [t[0] for t in pos[-n_top:]]     # most-positive n_top

    return P_plus, P_minus
