"""BonRunner: Best-of-N prevalence analysis for T2I reward model bias."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from search.data.results import SearchResults
from search.data.types import BaselineImage
from search.models.base import DetectorModel, RewardModel
from search.pipeline.baselines import (
    load_baselines_from_manifest,
    load_val_topic_state,
    score_baselines,
)
from bon.results import BonResults

if TYPE_CHECKING:
    from bon.config import BonConfig
    from bon.logging.tracker import BonTracker


class BonRunner:
    def __init__(
        self,
        config: "BonConfig",
        reward_model: RewardModel,
        detector: DetectorModel,
        tracker: "BonTracker",
    ):
        self.config = config
        self.reward_model = reward_model
        self.detector = detector
        self.tracker = tracker
        self.rng = Random(config.run.random_seed)

    @classmethod
    def from_config(cls, config: "BonConfig", tracker: "BonTracker") -> "BonRunner":
        from search.models.reward.imagereward import ImageRewardModel
        from search.models.judge.vlm_judge import VisionLLMDetector

        reward_model = ImageRewardModel(
            device=config.models.reward_model.device,
            hf_cache_dir=config.models.reward_model.hf_cache_dir,
        )
        detector = VisionLLMDetector(
            model_name=config.models.detector.model,
            max_tokens=config.models.detector.max_tokens,
            max_parallel=config.models.detector.max_parallel,
        )
        return cls(config=config, reward_model=reward_model, detector=detector, tracker=tracker)

    async def run(self) -> list[BonResults]:
        t0 = time.monotonic()
        config = self.config

        attributes = _resolve_attributes(config)
        if not attributes:
            logger.warning("No attributes to track — check attributes.attributes or attributes.search_results_path")
        logger.info(f"Tracking {len(attributes)} attributes: {attributes}")

        all_results: list[BonResults] = []
        for topic_id in config.data.topic_ids:
            results = await self._run_topic(topic_id, attributes, t0)
            all_results.append(results)

        return all_results

    async def _run_topic(
        self,
        topic_id: int,
        attributes: list[str],
        t0: float,
    ) -> BonResults:
        config = self.config

        val_state = load_val_topic_state(
            prompts_dir=config.data.prompts_dir,
            topic_id=topic_id,
            val_split_size=config.data.val_split_size,
            random_seed=config.run.random_seed,
        )
        load_baselines_from_manifest(
            topic_state=val_state,
            manifest_path=config.data.baseline_manifest,
            baseline_root=config.data.baseline_root,
        )

        n_val_prompts = len(val_state.baselines)
        logger.info(f"Topic {topic_id}: {n_val_prompts} val prompts with baselines loaded")

        if n_val_prompts == 0:
            logger.warning(f"Topic {topic_id}: no val baselines found, skipping")
            return self._empty_results(topic_id, attributes, t0)

        await score_baselines(val_state, self.reward_model)

        output_dir = config.run_output_dir()
        global_cache_path = Path(config.run.output_dir) / "cache" / config.data.name / f"topic{topic_id}.json"
        run_cache_path = output_dir / f"detection_cache_topic{topic_id}.json"

        detection_cache = await self._build_detection_cache(
            baselines_by_prompt=val_state.baselines,
            attributes=attributes,
            global_cache_path=global_cache_path,
            run_cache_path=run_cache_path,
        )

        curves = self._compute_bon_curves(
            baselines_by_prompt=val_state.baselines,
            detection_cache=detection_cache,
            attributes=attributes,
        )

        self.tracker.log_curves(topic_id, attributes, curves, config.sampling.n_values)

        return BonResults(
            run_id=config.run.name,
            search_results_path=config.attributes.search_results_path,
            topic_id=topic_id,
            attributes=attributes,
            n_values=config.sampling.n_values,
            n_trials=config.sampling.n_trials,
            n_val_prompts=n_val_prompts,
            prevalence=curves,
            cost_usd=0.0,
            wall_time_seconds=time.monotonic() - t0,
        )

    async def _build_detection_cache(
        self,
        baselines_by_prompt: dict[str, list[BaselineImage]],
        attributes: list[str],
        global_cache_path: Path,
        run_cache_path: Path,
    ) -> dict[str, dict[str, bool]]:
        """
        Two-level detection cache:
        - global: shared across all runs at outputs/bon/cache/topic{N}.json
        - run snapshot: what this specific run used at {run_output_dir}/detection_cache_topic{N}.json
        """
        cache: dict[str, dict[str, bool]] = {}
        if global_cache_path.exists():
            with open(global_cache_path) as f:
                raw = json.load(f)
            cache = {img_id: {k: bool(v) for k, v in attrs.items()} for img_id, attrs in raw.items()}
            logger.info(f"Loaded global cache: {len(cache)} images ({global_cache_path})")

        all_images = [img for imgs in baselines_by_prompt.values() for img in imgs]
        global_cache_dirty = False

        for attr in attributes:
            missing = [img for img in all_images if attr not in cache.get(img.image_id, {})]
            if not missing:
                logger.info(f"  '{attr}': all {len(all_images)} images cached")
                continue

            logger.info(
                f"  '{attr}': detecting on {len(missing)} images "
                f"({len(all_images) - len(missing)} cached)"
            )
            detections = await self.detector.detect(
                image_paths=[str(img.image_path) for img in missing],
                prompts=[img.prompt.text for img in missing],
                attribute=attr,
            )
            for img, det in zip(missing, detections):
                cache.setdefault(img.image_id, {})[attr] = bool(det)
            global_cache_dirty = True

        if global_cache_dirty:
            _atomic_json_write(global_cache_path, cache)
            logger.info(f"Updated global cache: {len(cache)} images → {global_cache_path}")

        used_ids = {img.image_id for img in all_images}
        run_snapshot = {img_id: attrs for img_id, attrs in cache.items() if img_id in used_ids}
        _atomic_json_write(run_cache_path, run_snapshot)
        logger.info(f"Saved run cache snapshot: {len(run_snapshot)} images → {run_cache_path}")

        return cache

    def _compute_bon_curves(
        self,
        baselines_by_prompt: dict[str, list[BaselineImage]],
        detection_cache: dict[str, dict[str, bool]],
        attributes: list[str],
    ) -> dict[str, list[float]]:
        """Monte Carlo BoN prevalence curves. Pure numpy, no I/O."""
        reward_model_name = self.reward_model.model_name
        n_values = self.config.sampling.n_values
        n_trials = self.config.sampling.n_trials

        # Pre-filter prompts to those with at least one scored image
        prompts_data: list[tuple[np.ndarray, list[dict[str, bool]]]] = []
        for imgs in baselines_by_prompt.values():
            scored = [img for img in imgs if reward_model_name in img.reward_scores]
            if not scored:
                continue
            rewards = np.array([img.reward_scores[reward_model_name] for img in scored])
            dets = [detection_cache.get(img.image_id, {}) for img in scored]
            prompts_data.append((rewards, dets))

        if not prompts_data:
            logger.warning("No scored val images found; returning zero curves")
            return {attr: [0.0] * len(n_values) for attr in attributes}

        curves: dict[str, list[float]] = {attr: [] for attr in attributes}

        for n in n_values:
            usable = [(r, d) for r, d in prompts_data if len(r) >= n]
            if not usable:
                logger.warning(f"N={n}: no prompts have enough images, using min(n, m)")
                usable = prompts_data

            attr_presences: dict[str, list[float]] = {attr: [] for attr in attributes}

            for rewards, dets in usable:
                m = len(rewards)
                k = min(n, m)
                for _ in range(n_trials):
                    indices = self.rng.sample(range(m), k)
                    best = indices[int(np.argmax(rewards[indices]))]
                    best_det = dets[best]
                    for attr in attributes:
                        attr_presences[attr].append(float(best_det.get(attr, False)))

            for attr in attributes:
                vals = attr_presences[attr]
                curves[attr].append(float(np.mean(vals)) if vals else 0.0)

            summary = "  ".join(f"{a}={curves[a][-1]:.3f}" for a in attributes)
            logger.info(f"  N={n}: {len(usable)} prompts × {n_trials} trials  {summary}")

        return curves

    def _empty_results(self, topic_id: int, attributes: list[str], t0: float) -> BonResults:
        return BonResults(
            run_id=self.config.run.name,
            search_results_path=self.config.attributes.search_results_path,
            topic_id=topic_id,
            attributes=attributes,
            n_values=self.config.sampling.n_values,
            n_trials=self.config.sampling.n_trials,
            n_val_prompts=0,
            prevalence={attr: [0.0] * len(self.config.sampling.n_values) for attr in attributes},
            cost_usd=0.0,
            wall_time_seconds=time.monotonic() - t0,
        )

    async def shutdown(self) -> None:
        pass


def _resolve_attributes(config: "BonConfig") -> list[str]:
    """
    Explicit list takes precedence; otherwise load from search results JSON.
    """
    if config.attributes.attributes:
        return list(config.attributes.attributes)

    search_results = SearchResults.load(config.attributes.search_results_path)
    if config.attributes.only_undesirable:
        return [fa.attribute for fa in search_results.top_attributes if fa.is_undesirable]
    return [fa.attribute for fa in search_results.top_attributes]


def _atomic_json_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
