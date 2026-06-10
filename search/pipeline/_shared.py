"""Helpers shared across pipeline phases.

These used to live in `search.pipeline.baseline_evo`, which has been retired
along with the baseline-pairs engine. They are now imported from here by the
live BoN-amplified engine and by the few analysis scripts that need to redo
detection or amplification scoring offline.

Public surface:
    - _add_to_rejected, _all_cached, _trim_step   (pool/cache micro-helpers)
    - _detect_all_attributes                       (one batched VLM call per attr)
    - _compute_amp_from_detection                  (A(g) from a detection cache)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from search.config import DetectorConfig
    from search.data.state import BaselineImage, EvoStep


def detector_model_key(detector_cfg: "DetectorConfig") -> str:
    """Canonical detection-cache key for a detector config.

    The same model id with different ``image_detail`` settings yields different
    detector outputs, so both are baked into the key. All cache I/O and lookups
    go through this single helper so the format never drifts between sites.
    """
    return f"{detector_cfg.model}::{detector_cfg.image_detail}"


# ── Pool / cache micro-helpers ────────────────────────────────────────────────

def _add_to_rejected(rejected_pool: list[str], new_rejects: set[str]) -> None:
    """Append newly rejected attrs to the pool, avoiding duplicates."""
    for attr in new_rejects:
        if attr not in rejected_pool:
            rejected_pool.append(attr)


def _all_cached(attr: str, detection_cache: dict[str, dict[str, int]]) -> bool:
    """True if every image in the cache already has a detection entry for this attr."""
    if not detection_cache:
        return False
    return all(attr in v for v in detection_cache.values())


def _trim_step(step: "EvoStep", keep: list[str]) -> None:
    """Drop attrs from the EvoStep that are not in `keep` (in-place)."""
    keep_set = set(keep)
    for a in list(step.attributes.keys()):
        if a not in keep_set:
            del step.attributes[a]


# ── Detection — one batched VLM call per attribute ────────────────────────────

async def _detect_all_attributes(
    detector_model,
    attrs_to_detect: list[str],
    amp_baselines: dict[str, list["BaselineImage"]],
    existing_cache: "dict[str, dict[str, int]] | None" = None,
    _retry: bool = True,
) -> dict[str, dict[str, int]]:
    """Return ``{image_id: {attr: 0/1/-1}}``. One batched VLM call per attribute.

    Args:
        existing_cache: the engine-level detection cache; used to find images
            that are missing after the first pass so they can be retried once.
        _retry: internal flag — set to False on the retry pass to avoid
            infinite recursion.
    """
    import time as _time

    all_images: list["BaselineImage"] = []
    all_prompts: list[str] = []
    for prompt, images in amp_baselines.items():
        for img in images:
            all_images.append(img)
            all_prompts.append(prompt)

    n_attrs = len(attrs_to_detect)
    n_images = len(all_images)
    t_total_start = _time.monotonic()

    async def _detect_one(attr: str) -> tuple[str, float, list]:
        t0 = _time.monotonic()
        results = await detector_model.detect(
            [str(img.image_path) for img in all_images], all_prompts, attr
        )
        return attr, _time.monotonic() - t0, results

    from tqdm.asyncio import tqdm as atqdm
    logger.info(f"  detecting {n_attrs} attrs × {n_images} imgs in parallel")

    tasks = [_detect_one(attr) for attr in attrs_to_detect]
    attr_results = await atqdm.gather(
        *tasks,
        desc=f"detecting ({n_images} imgs)",
        unit="attr",
        dynamic_ncols=True,
        leave=True,
    )

    detection: dict[str, dict[str, int]] = {}
    for attr, elapsed, det_results in attr_results:
        for img, d in zip(all_images, det_results):
            detection.setdefault(img.image_id, {})[attr] = int(d)
        logger.info(f"  detection done in {elapsed:.1f}s  attr: {attr[:60]}")

    total_elapsed = _time.monotonic() - t_total_start
    logger.info(f"  detection total: {total_elapsed:.1f}s for {n_attrs} attrs")

    # ── Coverage retry ────────────────────────────────────────────────────────
    # Find images that are still missing after this pass (not in new detection
    # results AND not already in the engine-level cache).
    if _retry and existing_cache is not None:
        all_image_ids = {img.image_id for img in all_images}
        covered = set(detection.keys()) | set(existing_cache.keys())
        missing_ids = all_image_ids - covered
        if missing_ids:
            logger.warning(
                f"  {len(missing_ids)} images missing from detection — retrying once"
            )
            missing_baselines: dict = {}
            for prompt, images in amp_baselines.items():
                sub = [img for img in images if img.image_id in missing_ids]
                if sub:
                    missing_baselines[prompt] = sub
            retry_det = await _detect_all_attributes(
                detector_model, attrs_to_detect, missing_baselines,
                existing_cache=None, _retry=False,
            )
            for image_id, attr_vals in retry_det.items():
                detection.setdefault(image_id, {}).update(attr_vals)
            logger.info(f"  retry recovered {len(retry_det)} images")

    return detection


# ── Amplification score from a detection cache ────────────────────────────────

def _compute_amp_from_detection(
    detection: dict[str, dict[str, int]],
    amp_baselines: dict[str, list["BaselineImage"]],
    attr_pool: list[str],
    reward_model_name: str,
    amp_mode: str = "kl_rlhf",
    bon_n: int = 16,
    not_applicable_as_absent: bool = False,
) -> dict[str, float]:
    """Per-attribute A(g) score using a pre-computed detection cache.

    amp_mode="kl_rlhf": A(g) = E_x[p1·p0·(μ1−μ0)]  (Cov(g,r) proxy, small-β KL-RLHF limit)
    amp_mode="bon":      A(g) = E_x[N·p1·p0·(E[U^{N-1}|g=1]−E[U^{N-1}|g=0])]
                         where U_x(y_i) = #{j: r_j ≤ r_i}/n  (empirical reward quantile)
    """
    amp_scores: dict[str, float] = {}
    for attr in attr_pool:
        per_prompt: list[float] = []
        per_p1: list[float] = []
        per_p0: list[float] = []
        per_s1: list[float] = []  # μ1 (kl_rlhf) or E[U^{N-1}|g=1] (bon)
        per_s0: list[float] = []  # μ0 (kl_rlhf) or E[U^{N-1}|g=0] (bon)

        for prompt_text, images in amp_baselines.items():
            scored = [
                img for img in images
                if reward_model_name in img.reward_scores
                and img.image_id in detection
                and attr in detection[img.image_id]
            ]
            if len(scored) < 2:
                continue
            n = len(scored)
            rewards = np.array([img.reward_scores[reward_model_name] for img in scored])
            dets = np.array([detection[img.image_id][attr] for img in scored])
            if not_applicable_as_absent:
                # Cache stores -1 for "not applicable"; treat as absent (=0).
                dets = np.where(dets == -1, 0, dets)
            g1_mask = dets == 1
            g0_mask = dets == 0

            if not g1_mask.any() and not g0_mask.any():
                logger.debug(
                    f"  A(g) '{attr}' | '{prompt_text}': skipped — attr undetected in all images"
                )
                continue

            p1 = float(g1_mask.sum()) / n
            p0 = float(g0_mask.sum()) / n

            # Per-prompt statistic vector — units depend on amp_mode:
            #   bon     → stat = U^{N-1} (empirical BoN quantile power, [0, 1])
            #   kl_rlhf → stat = raw reward (reward-model native scale)
            if amp_mode == "bon":
                sorted_r = np.sort(rewards)
                U = np.searchsorted(sorted_r, rewards, side="right") / n
                stat_vec = U ** (bon_n - 1)
            else:
                stat_vec = rewards

            if not g1_mask.any():
                logger.debug(
                    f"  A(g) '{attr}' | '{prompt_text}': skipped — attr never present (g1=0, n={n})"
                )
                per_prompt.append(0.0); per_p1.append(0.0); per_p0.append(1.0)
                per_s1.append(0.0); per_s0.append(float(np.mean(stat_vec[g0_mask])))
                continue
            if not g0_mask.any():
                logger.debug(
                    f"  A(g) '{attr}' | '{prompt_text}': skipped — attr always present (n={n})"
                )
                per_prompt.append(0.0); per_p1.append(1.0); per_p0.append(0.0)
                per_s1.append(float(np.mean(stat_vec[g1_mask]))); per_s0.append(0.0)
                continue

            if amp_mode == "bon":
                eu1 = float(np.mean(stat_vec[g1_mask]))
                eu0 = float(np.mean(stat_vec[g0_mask]))
                prompt_score = bon_n * p1 * p0 * (eu1 - eu0)
                per_s1.append(eu1); per_s0.append(eu0)
            else:  # kl_rlhf
                mu1 = float(np.mean(stat_vec[g1_mask]))
                mu0 = float(np.mean(stat_vec[g0_mask]))
                prompt_score = p1 * p0 * (mu1 - mu0)
                per_s1.append(mu1); per_s0.append(mu0)
            per_prompt.append(prompt_score)
            per_p1.append(p1)
            per_p0.append(p0)

        score = float(np.mean(per_prompt)) if per_prompt else 0.0
        amp_scores[attr] = score

        s1_label, s0_label = ("eu1", "eu0") if amp_mode == "bon" else ("μ1", "μ0")
        if per_p1:
            logger.info(
                f"  A(g) '{attr}': {score:.4f}  "
                f"(p1={np.mean(per_p1):.3f} p0={np.mean(per_p0):.3f} "
                f"{s1_label}={np.mean(per_s1):.4f} {s0_label}={np.mean(per_s0):.4f} "
                f"over {len(per_p1)} prompts)"
            )
        else:
            logger.info(f"  A(g) '{attr}': 0.0 (no valid prompts)")

    return amp_scores
