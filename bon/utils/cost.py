"""Pre-run API cost estimator for BoN analysis."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from search.utils.cost import (
    _call_cost,
    _img_tok,
    _JUDGE_DETECT_TEXT_TOK,
    _JUDGE_DETECT_OUT_TOK,
    _BASELINE_IMG_W,
    _BASELINE_IMG_H,
)

if TYPE_CHECKING:
    from bon.config import BonConfig


def estimate_bon_cost(config: "BonConfig") -> tuple[dict[str, float], dict]:
    """
    Estimate API cost for a BoN run.

    Returns (breakdown, meta) where breakdown has 'detection'/'total' keys (USD).
    Detection cost is worst-case (cold cache). Hot-cache cost is $0.

    The only API expense is VLM detection — one call per (image, attribute).
    Reward scoring (ImageReward) runs locally on GPU at zero API cost.
    """
    detector_model = config.models.detector.model
    detector_is_local = bool(config.models.detector.vllm_base_url)
    if detector_is_local:
        detect_per_call = 0.0
    else:
        img_tok = _img_tok(
            detector_model, _BASELINE_IMG_W, _BASELINE_IMG_H,
            detail=config.models.detector.image_detail,
        )
        batch_discount = 0.5 if config.models.detector.use_batch_api else 1.0
        detect_per_call = _call_cost(
            detector_model,
            _JUDGE_DETECT_TEXT_TOK + img_tok,
            _JUDGE_DETECT_OUT_TOK,
        ) * batch_discount

    n_attrs, attrs_source = _count_attributes(config)
    n_images_per_prompt, manifest_source = _sample_images_per_prompt(config.data.baseline_manifest)
    n_topics = len(config.data.topic_ids)
    n_val_prompts = config.data.val_split_size
    n_total_calls = n_topics * n_val_prompts * n_images_per_prompt * n_attrs

    cost_detection = n_total_calls * detect_per_call

    meta = {
        "detector_model": detector_model,
        "detector_is_local": detector_is_local,
        "vllm_base_url": config.models.detector.vllm_base_url,
        "image_detail": config.models.detector.image_detail,
        "use_batch_api": config.models.detector.use_batch_api,
        "n_topics": n_topics,
        "n_val_prompts": n_val_prompts,
        "n_images_per_prompt": n_images_per_prompt,
        "manifest_source": manifest_source,
        "n_attrs": n_attrs,
        "attrs_source": attrs_source,
        "n_total_detection_calls": n_total_calls,
        "detect_per_call_usd": detect_per_call,
    }
    breakdown = {
        "detection": cost_detection,
        "reward_scoring": 0.0,
        "total": cost_detection,
    }
    return breakdown, meta


def log_bon_cost_estimate(config: "BonConfig") -> None:
    from loguru import logger

    breakdown, meta = estimate_bon_cost(config)
    n_calls = meta["n_total_detection_calls"]
    if meta["detector_is_local"]:
        serving_tag = f"  [local vLLM: {meta['vllm_base_url']}]"
    elif meta["use_batch_api"]:
        serving_tag = "  [batch API 50% discount]"
    else:
        serving_tag = ""
    lines = [
        "── Estimated API Cost (BoN) ─────────────────────────────────",
        f"  Detector model:          {meta['detector_model']}{serving_tag}",
        f"  Image detail:            {meta['image_detail']}",
        f"  Topics:                  {meta['n_topics']}",
        f"  Val prompts per topic:   {meta['n_val_prompts']}",
        f"  Images per prompt:       ~{meta['n_images_per_prompt']}  ({meta['manifest_source']})",
        f"  Attributes tracked:      {meta['n_attrs']}  ({meta['attrs_source']})",
        f"  Total detection calls:   {n_calls:,}  "
        + (f"(local vLLM — $0)" if meta["detector_is_local"] else f"(@ ${meta['detect_per_call_usd'] * 1000:.4f} per 1k calls)"),
        "  ─────────────────────────────────────────────────────────────",
        f"  Detection  (cold cache):    ${breakdown['detection']:>8.2f}",
        f"  Detection  (hot cache):     ${0.0:>8.2f}",
        f"  Reward scoring (local GPU): ${0.0:>8.2f}",
        "  ─────────────────────────────────────────────────────────────",
        f"  TOTAL  worst / best:   ${breakdown['total']:>8.2f} / $0.00",
        "──────────────────────────────────────────────────────────────",
    ]
    for line in lines:
        logger.info(line)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _count_attributes(config: "BonConfig") -> tuple[int, str]:
    """Return (n_attrs, source_description). Explicit list takes precedence."""
    if config.attributes.attributes:
        return len(config.attributes.attributes), "explicit attributes list"
    try:
        from search.data.results import SearchResults
        sr = SearchResults.load(config.attributes.search_results_path)
        if config.attributes.only_undesirable:
            n = sum(1 for fa in sr.top_attributes if fa.is_undesirable)
            return n, "search results (only_undesirable=True)"
        return len(sr.top_attributes), "search results"
    except Exception:
        return 5, "fallback estimate (search results not loaded)"


def _sample_images_per_prompt(manifest_path: str, n_sample: int = 5) -> tuple[int, str]:
    """Sample a few entries from the manifest to estimate images per prompt."""
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        baselines = manifest.get("baselines", {})
        sample = list(baselines.values())[:n_sample]
        if not sample:
            return 64, "fallback (empty manifest)"
        avg = round(sum(len(v) for v in sample) / len(sample))
        return avg, f"sampled {len(sample)} prompts from manifest"
    except Exception:
        return 64, "fallback (manifest not readable)"
