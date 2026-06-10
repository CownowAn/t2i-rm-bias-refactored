"""Pre-run API cost estimator for the BoN-amplified search.

Edit-mode and baseline-pairs-mode estimators have been removed along with the
modes themselves; only the bon_amplified estimator remains.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from search.config import SearchConfig

# ─── Model pricing (OpenAI, May 2026) — (input $/1M tok, output $/1M tok) ────
# Source: platform.openai.com/docs/pricing
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "openai/gpt-4o-mini": (0.15,   0.60),
    "openai/gpt-4o":      (2.50,  10.00),
    "openai/gpt-5":       (1.25,  10.00),
    "openai/gpt-5-nano":  (0.20,   1.25),
    "openai/gpt-5-mini":  (0.25,   2.00),
    "openai/gpt-5.2":     (2.50,  15.00),
}
_DEFAULT_PRICING = (2.50, 15.00)  # conservative fallback for unlisted models

# ─── Fixed token estimates ────────────────────────────────────────────────────
_PLANNER_OUTPUT_TOK    = 15_000  # reasoning trace + attribute list (gpt-5 high)
_PLANNER_TEXT_STEP0    = 1_000   # system + user-prompt text at step 0

_JUDGE_DETECT_TEXT_TOK  = 200   # system prompt (~25) + template (~100) + attr (~20) + img prompt (~30) + overhead
_JUDGE_DETECT_OUT_TOK   = 40    # JSON: {"present":bool, "confidence":float, "reasoning":"..."}

_CLUSTER_ATTR_TOK   = 60        # ~60 tokens per attribute entry in JSON list
_CLUSTER_OUTPUT_TOK = 1_000

_HUMANNESS_TEXT_TOK   = 80   # system/user prompt + attr name
_HUMANNESS_OUTPUT_TOK = 2    # "YES" or "NO"

# Proposer: P+ and P- images per prompt, plus pool/avoid text. 600 tok mid-run.
_PROPOSER_TEXT_TOK   = 600

# Baseline images are FLUX.1-dev generated at 512×512 (confirmed from manifest metadata).
_BASELINE_IMG_W = 512
_BASELINE_IMG_H = 512


def _call_cost(model: str, input_tok: int, output_tok: int) -> float:
    in_cpm, out_cpm = _MODEL_PRICING.get(model, _DEFAULT_PRICING)
    return (input_tok * in_cpm + output_tok * out_cpm) / 1_000_000


def _img_tok(model: str, width: int, height: int, detail: str = "auto") -> int:
    """OpenAI image token count (source: developers.openai.com/api/docs/guides/images-vision).

    detail="low"  → base tokens only (no tile calculation).
    detail="high" | "auto" → tile-based high-detail formula:
      1. If max(w,h) > 2048: scale down to fit within 2048×2048.
      2. If min(w,h) > 768:  scale down so shortest side = 768.
      3. n_tiles = ceil(w/512) × ceil(h/512).
      4. tokens  = base + per_tile × n_tiles.

    Per-model constants:
      gpt-4o / gpt-4.1 / gpt-5.x  : base=85,   per_tile=170
      gpt-4o-mini / gpt-4.1-mini  : base=2833, per_tile=5667
    """
    is_mini = "mini" in model or "nano" in model
    base, per_tile = (2833, 5667) if is_mini else (85, 170)
    if detail == "low":
        return base
    w, h = float(width), float(height)
    if max(w, h) > 2048:
        scale = 2048 / max(w, h)
        w, h = w * scale, h * scale
    if min(w, h) > 768:
        scale = 768 / min(w, h)
        w, h = w * scale, h * scale
    n_tiles = math.ceil(w / 512) * math.ceil(h / 512)
    return base + per_tile * n_tiles


def _load_n_train_prompts(config: "SearchConfig") -> dict[int, int]:
    """Try to read actual train prompt counts per topic from the cluster JSON files."""
    counts: dict[int, int] = {}
    prompts_dir = Path(config.data.prompts_dir)
    val = config.data.val_split_size
    for tid in config.data.topic_ids:
        path = prompts_dir / f"cluster_{tid}.json"
        try:
            with open(path) as f:
                data = json.load(f)
            total = len(data.get("prompts", []))
            counts[tid] = max(1, total - val)
        except Exception:
            counts[tid] = 20  # fallback estimate
    return counts


def estimate_cost(config: "SearchConfig") -> dict[str, float]:
    """Estimate total API cost (USD) for a bon_amplified search run.

    Components per topic per step:
      1. Planner    — InitialPlanner at step 0
      2. Cluster    — AttributeClusterer at step 0
      3. Humanness  — filter_by_humanness (text-only). Applied at step 0 (initial pool)
                      and once per step thereafter on the proposer's candidates.
      4. Detection  — VisionLLMDetector: 1 image per (new_attr × fixed_baseline_image).
                      $0 when vllm_base_url is set (local inference).
      5. Proposer   — BonResidualProposer: 1 call with n_prompts_vlm × 2 images (P+/P-).
      6. Validation — detect proposed candidates + compute A_hat (same rate as detection).
                      $0 when vllm_base_url is set.
    """
    cfg = config
    ecfg = cfg.evaluation
    evcfg = cfg.evolution
    bacfg = cfg.bon_amplified
    n_steps = evcfg.n_steps

    planner_model     = cfg.models.planner.model
    attr_filter_model = cfg.models.attr_filter.model
    proposer_model    = cfg.models.proposer.model
    detector_model    = cfg.models.detector.model
    cluster_model     = cfg.models.cluster.model

    img_tok_detector = _img_tok(detector_model, _BASELINE_IMG_W, _BASELINE_IMG_H,
                                detail=cfg.models.detector.image_detail)
    img_tok_planner  = _img_tok(planner_model,  _BASELINE_IMG_W, _BASELINE_IMG_H)
    img_tok_proposer = _img_tok(proposer_model, _BASELINE_IMG_W, _BASELINE_IMG_H,
                                detail=cfg.models.proposer.image_detail)

    n_fixed_images = ecfg.amp_n_prompts * ecfg.amp_n_images_per_prompt

    detector_is_local = bool(cfg.models.detector.vllm_base_url)
    _BATCH_DISCOUNT = 0.5
    detector_discount = _BATCH_DISCOUNT if cfg.models.detector.use_batch_api else 1.0

    detect_call = (
        0.0 if detector_is_local
        else _call_cost(detector_model,
                        _JUDGE_DETECT_TEXT_TOK + img_tok_detector,
                        _JUDGE_DETECT_OUT_TOK) * detector_discount
    )
    humanness_call = _call_cost(attr_filter_model, _HUMANNESS_TEXT_TOK, _HUMANNESS_OUTPUT_TOK)

    # BonResidualProposer: 1 call, showing n_prompts_vlm prompts each with P+ + P- images.
    n_proposer_imgs = bacfg.n_prompts_vlm * bacfg.n_top_residual * 2
    proposer_call = _call_cost(
        proposer_model,
        _PROPOSER_TEXT_TOK + n_proposer_imgs * img_tok_proposer,
        bacfg.n_proposals * 15,  # ~15 tokens per proposed attr string
    )

    n_train_prompts_by_topic = _load_n_train_prompts(cfg)

    costs = dict(planner=0.0, cluster=0.0, humanness=0.0,
                 detection=0.0, proposer=0.0, validation=0.0)

    for tid in cfg.data.topic_ids:
        n_train = n_train_prompts_by_topic[tid]
        n_effective = (
            min(n_train, evcfg.n_initial_plan_prompts)
            if evcfg.n_initial_plan_prompts is not None
            else n_train
        )

        # ── Step 0: Initial Planning + Cluster ─────────────────────────────
        n_ppc = evcfg.n_prompts_per_plan_call
        if n_ppc > 1:
            n_planner_calls = math.ceil(n_effective / n_ppc)
            planner_in_tok  = n_ppc * evcfg.n_context_imgs * img_tok_planner + _PLANNER_TEXT_STEP0
        else:
            n_planner_calls = n_effective * evcfg.n_per_user_prompt
            planner_in_tok  = evcfg.n_context_imgs * img_tok_planner + _PLANNER_TEXT_STEP0
        costs["planner"] += n_planner_calls * _call_cost(
            planner_model, planner_in_tok, _PLANNER_OUTPUT_TOK
        )
        n_raw_attrs = n_planner_calls * evcfg.n_attrs_per_prompt
        costs["cluster"] += _call_cost(
            cluster_model, n_raw_attrs * _CLUSTER_ATTR_TOK, _CLUSTER_OUTPUT_TOK
        )

        # ── Main loop ──────────────────────────────────────────────────────
        for step_idx in range(n_steps):
            # Attrs entering EVALUATE:
            #   step 0: initial pool (post-cluster + humanness from SETUP)
            #   step k>0: S_{t-1} (survived) + validated candidates from step k-1
            if step_idx == 0:
                n_attrs_in = min(n_raw_attrs, evcfg.initial_pop_size * 2)
                costs["humanness"] += n_attrs_in * humanness_call
            else:
                # EXPAND humanness from previous step (n_proposals candidates)
                costs["humanness"] += bacfg.n_proposals * humanness_call

            # Detection — only NEW attrs (S_{t-1} already cached).
            n_new_attrs = n_attrs_in if step_idx == 0 else bacfg.n_proposals
            costs["detection"] += n_new_attrs * n_fixed_images * detect_call

            # Proposer + Validation (skipped on the last step).
            if step_idx < n_steps - 1:
                costs["proposer"] += proposer_call * bacfg.n_proposer_calls
                costs["validation"] += (
                    bacfg.n_proposals * bacfg.n_proposer_calls * n_fixed_images * detect_call
                )

    costs["total"] = sum(v for k, v in costs.items() if k != "detector_is_local")
    costs["detector_is_local"] = detector_is_local
    return costs


def log_cost_estimate(config: "SearchConfig") -> None:
    """Print a formatted cost estimate breakdown to the logger."""
    breakdown = estimate_cost(config)
    det_note = " [local vLLM, $0]" if breakdown["detector_is_local"] else ""
    lines = [
        "── Estimated API Cost (bon-amplified) ─────────────",
        f"  Planner    (step 0 initial plan): ${breakdown['planner']:>8.2f}",
        f"  Cluster    (deduplication):       ${breakdown['cluster']:>8.2f}",
        f"  Humanness  (undesir. filter ×2):  ${breakdown['humanness']:>8.2f}",
        f"  Detection  (VLM attr detection):  ${breakdown['detection']:>8.2f}{det_note}",
        f"  Proposer   (P+/P- VLM mining):    ${breakdown['proposer']:>8.2f}",
        f"  Validation (A_hat check):         ${breakdown['validation']:>8.2f}{det_note}",
        "  ──────────────────────────────────────────────────",
        f"  TOTAL                             ${breakdown['total']:>8.2f}",
        "────────────────────────────────────────────────────",
    ]
    for line in lines:
        logger.info(line)
