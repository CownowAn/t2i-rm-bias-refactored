"""Pre-run API cost estimator for the evolutionary T2I bias search."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from search.config import SearchConfig

# ─── Model pricing (OpenAI, May 2026) — (input $/1M tok, output $/1M tok) ────
# Source: platform.openai.com/docs/pricing
# FluxKontextApplier runs locally on GPU → zero API cost.
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
_PLANNER_TEXT_MUTATION = 1_200   # MUTATE_PRE + POST text (all context modes)

_EDITOR_TEXT_TOK   = 500   # instruction-generation prompt text
_EDITOR_OUTPUT_TOK = 50    # short edit instruction string

_JUDGE_COMPARE_TEXT_TOK = 200   # image comparison question text
_JUDGE_COMPARE_OUT_TOK  = 100   # preference + score_diff + brief reasoning

_JUDGE_DETECT_TEXT_TOK  = 200   # system prompt (~25) + template (~100) + attr (~20) + img prompt (~30) + overhead
_JUDGE_DETECT_OUT_TOK   = 40    # JSON: {"present":bool, "confidence":float, "reasoning":"..."}

_CLUSTER_ATTR_TOK   = 60    # ~60 tokens per attribute entry in JSON list
_CLUSTER_OUTPUT_TOK = 1_000

_MUTATOR_ROLLOUTS_IN_CONTEXT = 4
_MUTATOR_ANCESTRY_IMGS_PER_NODE = 4

# ── Baseline-pairs specific ───────────────────────────────────────────────────
_HUMANNESS_TEXT_TOK   = 80   # system/user prompt + attr name
_HUMANNESS_OUTPUT_TOK = 2    # "YES" or "NO"

# Proposer: 2 images + context (pool list + rejected list + attr-vector block)
# Text scales with acc_pool size; 600 tok is a mid-run average estimate.
_PROPOSER_TEXT_TOK   = 600
_PROPOSER_OUTPUT_TOK = 50    # single JSON attr string

# Baseline images are FLUX.1-dev generated at 512×512 (confirmed from manifest metadata).
_BASELINE_IMG_W = 512
_BASELINE_IMG_H = 512


def _call_cost(model: str, input_tok: int, output_tok: int) -> float:
    in_cpm, out_cpm = _MODEL_PRICING.get(model, _DEFAULT_PRICING)
    return (input_tok * in_cpm + output_tok * out_cpm) / 1_000_000


# ─── Image token formula ─────────────────────────────────────────────────────

def _img_tok(model: str, width: int, height: int, detail: str = "auto") -> int:
    """OpenAI image token count (source: developers.openai.com/api/docs/guides/images-vision).

    detail="low"  → base tokens only (no tile calculation).
    detail="high" | "auto" → tile-based high-detail formula:
      1. If max(w,h) > 2048: scale down to fit within 2048×2048.
      2. If min(w,h) > 768:  scale down so shortest side = 768.
         (images ≤ 768 on shortest side are NOT scaled up)
      3. n_tiles = ceil(w/512) × ceil(h/512).
      4. tokens  = base + per_tile × n_tiles.

    Per-model constants:
      gpt-4o / gpt-4.1 / gpt-5.x : base=85,   per_tile=170
      gpt-4o-mini / gpt-4.1-mini  : base=2833,  per_tile=5667

    Examples:
      gpt-4o,      512×512  high →  85 + 170×1 =    255 tokens
      gpt-4o,     1024×1024 high →  85 + 170×4 =    765 tokens
      gpt-4o-mini, 512×512  high → 2833 + 5667×1 =  8500 tokens
      gpt-4o-mini, 512×512  low  → 2833 tokens
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
    """
    Estimate total API cost (USD) for a full evolutionary search run.

    Components accounted for per topic per step:
      1. Planner   — initial planning (step 0) or mutation (steps 1..N-1)
      2. Editor    — EditInstructionGenerator: 1 image in per (attr × prompt × rollout)
      3. Judge/compare — VisionLLMJudge.compare: 2 images in per (attr × prompt × rollout)
      4. Amp/detect    — VisionLLMJudge.detect: 1 image per (attr × amp_prompt × baseline_img)
      5. Cluster   — AttributeClusterer LLM call after each step
    """
    cfg = config
    ecfg = cfg.evaluation
    evcfg = cfg.evolution
    n_steps = evcfg.n_steps

    planner_model  = cfg.models.planner.model
    editor_model   = cfg.models.editor.instruction_model
    judge_model    = cfg.models.judge.model
    detector_model = cfg.models.detector.model
    cluster_model  = cfg.models.cluster.model

    img_tok_editor   = _img_tok(editor_model,   _BASELINE_IMG_W, _BASELINE_IMG_H)
    img_tok_judge    = _img_tok(judge_model,    _BASELINE_IMG_W, _BASELINE_IMG_H,
                                detail=cfg.models.judge.image_detail)
    img_tok_detector = _img_tok(detector_model, _BASELINE_IMG_W, _BASELINE_IMG_H,
                                detail=cfg.models.detector.image_detail)
    img_tok_planner  = _img_tok(planner_model,  _BASELINE_IMG_W, _BASELINE_IMG_H)

    editor_call = _call_cost(editor_model,
                             _EDITOR_TEXT_TOK + img_tok_editor, _EDITOR_OUTPUT_TOK)
    judge_compare_call = _call_cost(judge_model,
                                    _JUDGE_COMPARE_TEXT_TOK + 2 * img_tok_judge,
                                    _JUDGE_COMPARE_OUT_TOK)
    judge_detect_call = _call_cost(detector_model,
                                   _JUDGE_DETECT_TEXT_TOK + img_tok_detector,
                                   _JUDGE_DETECT_OUT_TOK)

    n_train_prompts_by_topic = _load_n_train_prompts(cfg)

    cost_planner = 0.0
    cost_editor  = 0.0
    cost_judge   = 0.0
    cost_amp     = 0.0
    cost_cluster = 0.0

    for tid in cfg.data.topic_ids:
        n_train = n_train_prompts_by_topic[tid]

        for step_idx in range(n_steps):
            batch_size = ecfg.train_batch_size[step_idx]
            pop_size   = evcfg.target_pop_sizes[step_idx]

            n_effective_prompts = (
                min(n_train, evcfg.n_initial_plan_prompts)
                if evcfg.n_initial_plan_prompts is not None
                else n_train
            )

            if step_idx == 0:
                n_initial_raw = n_effective_prompts * evcfg.n_per_user_prompt * evcfg.n_attrs_per_prompt
                n_attrs = min(n_initial_raw, evcfg.initial_pop_size * 2)
            else:
                prev_surviving = evcfg.target_pop_sizes[step_idx - 1]
                n_attrs_raw = prev_surviving * (evcfg.n_mutations + 1)
                n_attrs = min(n_attrs_raw, pop_size * 2)

            # ── 1. Planner cost ───────────────────────────────────────────
            if step_idx == 0:
                n_planner_calls = n_effective_prompts * evcfg.n_per_user_prompt
                planner_in_tok = evcfg.n_context_imgs * img_tok_planner + _PLANNER_TEXT_STEP0
            else:
                prev_surviving = evcfg.target_pop_sizes[step_idx - 1]
                n_planner_calls = prev_surviving
                n_imgs = 2 * _MUTATOR_ROLLOUTS_IN_CONTEXT
                if evcfg.context not in ("vanilla",):
                    n_imgs += _MUTATOR_ANCESTRY_IMGS_PER_NODE * step_idx
                planner_in_tok = n_imgs * img_tok_planner + _PLANNER_TEXT_MUTATION

            cost_planner += n_planner_calls * _call_cost(
                planner_model, planner_in_tok, _PLANNER_OUTPUT_TOK
            )

            # ── 2. Clustering cost ────────────────────────────────────────
            if step_idx == 0:
                n_cluster_attrs = n_initial_raw
            else:
                n_cluster_attrs = n_attrs_raw  # type: ignore[possibly-undefined]

            cluster_in_tok = n_cluster_attrs * _CLUSTER_ATTR_TOK
            cost_cluster += _call_cost(cluster_model, cluster_in_tok, _CLUSTER_OUTPUT_TOK)

            # ── 3. Editor cost ────────────────────────────────────────────
            n_editor = n_attrs * batch_size * ecfg.n_rollouts_per_prompt
            cost_editor += n_editor * editor_call

            # ── 4. Judge/compare cost ─────────────────────────────────────
            n_judge = n_attrs * batch_size * ecfg.judge_first_n_rollouts
            cost_judge += n_judge * judge_compare_call

            # ── 5. Amplification detect cost ──────────────────────────────
            n_amp_candidates = pop_size
            n_detect = n_amp_candidates * batch_size * ecfg.amp_n_images_per_prompt
            cost_amp += n_detect * judge_detect_call

    total = cost_planner + cost_editor + cost_judge + cost_amp + cost_cluster

    return {
        "planner": cost_planner,
        "editor":  cost_editor,
        "judge":   cost_judge,
        "amp":     cost_amp,
        "cluster": cost_cluster,
        "total":   total,
    }


def estimate_cost_bp(config: "SearchConfig") -> dict[str, float]:
    """
    Estimate total API cost (USD) for a baseline-pairs search run.

    Components per topic per step:
      1. Planner    — InitialPlanner at step 0 (n_initial_plan_prompts × n_per_user_prompt calls)
      2. Cluster    — AttributeClusterer at step 0 (1 call)
      3. Humanness  — filter_by_humanness: 1 text-only call per candidate attr (uses planner_model)
      4. Detection  — VisionLLMDetector: 1 image call per (new_attr × fixed_baseline_image)
                      Only NEW attrs detected each step (cache hit for attrs from prior steps)
      5. Judge      — VisionLLMJudge.compare: 1 call per pair with 2 images (if use_judge=True)
      6. Proposer   — ResidualAttributeProposer: n_proposed_per_step calls × 2 images each

    Fixed baselines: amp_n_prompts × amp_n_images_per_prompt images, sampled once and reused.
    Baseline images are assumed to be 1024×1024 (FLUX.1-dev default).
    """
    cfg = config
    ecfg = cfg.evaluation
    evcfg = cfg.evolution
    bpcfg = cfg.baseline_pairs
    n_steps = evcfg.n_steps

    planner_model  = cfg.models.planner.model
    judge_model    = cfg.models.judge.model
    detector_model = cfg.models.detector.model
    cluster_model  = cfg.models.cluster.model

    # Per-model image token counts (detail level from config)
    img_tok_detector = _img_tok(detector_model, _BASELINE_IMG_W, _BASELINE_IMG_H,
                                detail=cfg.models.detector.image_detail)
    img_tok_judge    = _img_tok(judge_model,    _BASELINE_IMG_W, _BASELINE_IMG_H,
                                detail=cfg.models.judge.image_detail)
    img_tok_planner  = _img_tok(planner_model,  _BASELINE_IMG_W, _BASELINE_IMG_H)

    # Total fixed baseline images (sampled once, reused every step)
    n_fixed_images = ecfg.amp_n_prompts * ecfg.amp_n_images_per_prompt

    # Batch API discount (50%) applies to detector and judge when use_batch_api=True
    _BATCH_DISCOUNT = 0.5
    detector_discount = _BATCH_DISCOUNT if cfg.models.detector.use_batch_api else 1.0
    judge_discount    = _BATCH_DISCOUNT if cfg.models.judge.use_batch_api    else 1.0

    # Local vLLM detector → zero API cost
    detector_is_local = bool(cfg.models.detector.vllm_base_url)

    # Per-call costs
    humanness_call = _call_cost(planner_model,
                                _HUMANNESS_TEXT_TOK, _HUMANNESS_OUTPUT_TOK)
    detect_call = (
        0.0 if detector_is_local
        else _call_cost(detector_model,
                        _JUDGE_DETECT_TEXT_TOK + img_tok_detector,
                        _JUDGE_DETECT_OUT_TOK) * detector_discount
    )
    judge_call     = _call_cost(judge_model,
                                _JUDGE_COMPARE_TEXT_TOK + 2 * img_tok_judge,
                                _JUDGE_COMPARE_OUT_TOK) * judge_discount
    proposer_call  = _call_cost(planner_model,
                                _PROPOSER_TEXT_TOK + 2 * img_tok_planner,
                                _PROPOSER_OUTPUT_TOK)

    n_train_prompts_by_topic = _load_n_train_prompts(cfg)

    cost_planner   = 0.0
    cost_cluster   = 0.0
    cost_humanness = 0.0
    cost_detection = 0.0
    cost_judge     = 0.0
    cost_proposer  = 0.0

    for tid in cfg.data.topic_ids:
        n_train = n_train_prompts_by_topic[tid]
        n_effective = (
            min(n_train, evcfg.n_initial_plan_prompts)
            if evcfg.n_initial_plan_prompts is not None
            else n_train
        )

        # ── Step 0: Initial Planning ──────────────────────────────────────────
        n_planner_calls = n_effective * evcfg.n_per_user_prompt
        planner_in_tok  = evcfg.n_context_imgs * img_tok_planner + _PLANNER_TEXT_STEP0
        cost_planner += n_planner_calls * _call_cost(
            planner_model, planner_in_tok, _PLANNER_OUTPUT_TOK
        )

        # Clustering: reduce initial candidates to initial_pop_size*2
        n_raw_attrs = n_effective * evcfg.n_per_user_prompt * evcfg.n_attrs_per_prompt
        cost_cluster += _call_cost(
            cluster_model, n_raw_attrs * _CLUSTER_ATTR_TOK, _CLUSTER_OUTPUT_TOK
        )

        # ── Main loop ─────────────────────────────────────────────────────────
        for step_idx in range(n_steps):

            # Attrs entering EVALUATE this step:
            #   step 0: post-cluster, ≤ initial_pop_size × 2
            #   step k>0: new proposals from previous EXPAND (= n_proposed_per_step)
            n_attrs_in = (
                min(n_raw_attrs, evcfg.initial_pop_size * 2)
                if step_idx == 0
                else bpcfg.n_proposed_per_step
            )

            # [1] Humanness filter — one text-only call per candidate attr
            cost_humanness += n_attrs_in * humanness_call

            # [2] Detection — only NEW attrs need VLM detection (cache hit for prior attrs)
            #     Each (attr, image) pair → one separate API call (confirmed: no batching)
            cost_detection += n_attrs_in * n_fixed_images * detect_call

            # [3] Judge — one call per pair, 2 images per call
            #     Number of pairs judged depends on pair_constructor AND judge_filter_position.
            if bpcfg.use_judge:
                if bpcfg.judge_filter_position == "lazy":
                    # lazy: judge top-residual pairs until n_high_residual_pairs × overshoot confirmed.
                    # Actual judged count depends on judge pass rate; assume ~20% pass rate as
                    # a conservative upper bound (5× more pairs judged than confirmed target).
                    n_confirmed_target = int(bpcfg.n_high_residual_pairs * bpcfg.judge_lazy_overshoot)
                    n_judged = n_confirmed_target * 5
                elif bpcfg.pair_constructor == "all":
                    # all valid (high, low) pairs where D[i,:]≠0.
                    # Upper bound: C(amp_n_images_per_prompt, 2) × amp_n_prompts / 2
                    # (÷2 for RM(high)>RM(low) direction; assume most pairs have D≠0)
                    n_img = ecfg.amp_n_images_per_prompt
                    n_judged = n_img * (n_img - 1) // 2 * ecfg.amp_n_prompts // 2
                elif bpcfg.pair_constructor == "hamming":
                    n_judged = bpcfg.n_pairs_per_prompt * ecfg.amp_n_prompts
                else:  # stratified
                    # Global quota ≈ n_pairs_per_stratum × amp_n_prompts (constant across K)
                    n_judged = bpcfg.n_pairs_per_stratum * ecfg.amp_n_prompts
                cost_judge += n_judged * judge_call

            # [4] Residual proposer — n_proposed_per_step calls, each with 2 images
            #     Skipped on the final step (EXPAND runs but proposal is omitted)
            if step_idx < n_steps - 1:
                cost_proposer += bpcfg.n_proposed_per_step * proposer_call

    total = cost_planner + cost_cluster + cost_humanness + cost_detection + cost_judge + cost_proposer
    return {
        "planner":   cost_planner,
        "cluster":   cost_cluster,
        "humanness": cost_humanness,
        "detection": cost_detection,
        "judge":     cost_judge,
        "proposer":  cost_proposer,
        "total":     total,
    }


def estimate_cost_ba(config: "SearchConfig") -> dict[str, float]:
    """
    Estimate total API cost (USD) for a bon_amplified search run.

    Components per topic per step:
      1. Planner    — InitialPlanner at step 0
      2. Cluster    — AttributeClusterer at step 0
      3. Humanness  — filter_by_humanness: text-only, planner model
                      Applied twice: after InitialPlanner AND after VLM propose
      4. Detection  — VisionLLMDetector: 1 image per (new_attr × fixed_baseline_image)
                      $0 if vllm_base_url is set (local inference)
      5. Proposer   — BonResidualProposer: 1 call with n_prompts_vlm × 2 images (P+/P-)
      6. Validation — detect proposed candidates + compute A_hat (same rate as detection)
                      $0 if vllm_base_url is set
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

    # BonResidualProposer: 1 call, showing n_prompts_vlm prompts each with P+ + P- images
    n_proposer_imgs = bacfg.n_prompts_vlm * bacfg.n_top_residual * 2
    proposer_call = _call_cost(
        proposer_model,
        _PROPOSER_TEXT_TOK + n_proposer_imgs * img_tok_proposer,
        bacfg.n_proposals * 15,  # ~15 tokens per proposed attr string
    )

    n_train_prompts_by_topic = _load_n_train_prompts(cfg)

    cost_planner   = 0.0
    cost_cluster   = 0.0
    cost_humanness = 0.0
    cost_detection = 0.0
    cost_proposer  = 0.0
    cost_validation = 0.0

    for tid in cfg.data.topic_ids:
        n_train = n_train_prompts_by_topic[tid]
        n_effective = (
            min(n_train, evcfg.n_initial_plan_prompts)
            if evcfg.n_initial_plan_prompts is not None
            else n_train
        )

        # ── Step 0: Initial Planning + Cluster ──────────────────────────────────
        n_ppc = evcfg.n_prompts_per_plan_call  # prompts per VLM call
        if n_ppc > 1:
            # Multi-prompt mode: ceil(n_effective / n_ppc) calls, each showing
            # n_ppc prompts × n_context_imgs images
            n_planner_calls = math.ceil(n_effective / n_ppc)
            planner_in_tok  = n_ppc * evcfg.n_context_imgs * img_tok_planner + _PLANNER_TEXT_STEP0
        else:
            n_planner_calls = n_effective * evcfg.n_per_user_prompt
            planner_in_tok  = evcfg.n_context_imgs * img_tok_planner + _PLANNER_TEXT_STEP0
        cost_planner += n_planner_calls * _call_cost(
            planner_model, planner_in_tok, _PLANNER_OUTPUT_TOK
        )
        n_raw_attrs = n_planner_calls * evcfg.n_attrs_per_prompt
        cost_cluster += _call_cost(
            cluster_model, n_raw_attrs * _CLUSTER_ATTR_TOK, _CLUSTER_OUTPUT_TOK
        )

        # ── Main loop ─────────────────────────────────────────────────────────
        for step_idx in range(n_steps):
            # Attrs entering EVALUATE:
            #   step 0: initial pool (post-cluster + humanness from SETUP)
            #   step k>0: S_{t-1} (survived) + validated candidates from step k-1
            if step_idx == 0:
                n_attrs_in = min(n_raw_attrs, evcfg.initial_pop_size * 2)
            else:
                top_k = evcfg.target_pop_sizes[step_idx - 1]
                n_attrs_in = top_k + bacfg.n_proposals  # S_{t-1} + new validated

            # [1] Humanness filter on SETUP (step 0 only) — already applied before main loop;
            #     included once in the step 0 humanness budget
            # [1] Humanness on EVALUATE candidates (but these were already filtered in EXPAND)
            #     → Only counts the EXPAND humanness call (n_proposals text calls)
            if step_idx == 0:
                # SETUP humanness on initial pool
                cost_humanness += n_attrs_in * humanness_call
            else:
                # EXPAND humanness from previous step (n_proposals candidates)
                cost_humanness += bacfg.n_proposals * humanness_call

            # [2] Detection — only NEW attrs (S_{t-1} already cached; new = validated from EXPAND)
            n_new_attrs = n_attrs_in if step_idx == 0 else bacfg.n_proposals
            cost_detection += n_new_attrs * n_fixed_images * detect_call

            # [3] Proposer + Validation (skipped on last step)
            if step_idx < n_steps - 1:
                cost_proposer += proposer_call * bacfg.n_proposer_calls
                # Validation: detect all proposed candidates on all fixed images
                cost_validation += bacfg.n_proposals * bacfg.n_proposer_calls * n_fixed_images * detect_call

    total = (cost_planner + cost_cluster + cost_humanness
             + cost_detection + cost_proposer + cost_validation)
    return {
        "planner":    cost_planner,
        "cluster":    cost_cluster,
        "humanness":  cost_humanness,
        "detection":  cost_detection,
        "proposer":   cost_proposer,
        "validation": cost_validation,
        "total":      total,
        "detector_is_local": detector_is_local,
    }


def log_cost_estimate(config: "SearchConfig") -> None:
    """Print a formatted cost estimate breakdown to the logger."""
    from loguru import logger

    if config.pipeline.mode == "baseline_pairs":
        breakdown = estimate_cost_bp(config)
        det_local = bool(config.models.detector.vllm_base_url)
        det_note = " [local vLLM, $0]" if det_local else ""
        lines = [
            "── Estimated API Cost (baseline-pairs) ────────────",
            f"  Planner   (step 0 initial plan):  ${breakdown['planner']:>8.2f}",
            f"  Cluster   (deduplication):        ${breakdown['cluster']:>8.2f}",
            f"  Humanness (undesirability filter):${breakdown['humanness']:>8.2f}",
            f"  Detection (VLM attr detection):   ${breakdown['detection']:>8.2f}{det_note}",
            f"  Judge     (pair comparison):      ${breakdown['judge']:>8.2f}",
            f"  Proposer  (residual attr proposal):${breakdown['proposer']:>8.2f}",
            "  ──────────────────────────────────────────────────",
            f"  TOTAL                             ${breakdown['total']:>8.2f}",
            "────────────────────────────────────────────────────",
        ]
    elif config.pipeline.mode == "bon_amplified":
        breakdown = estimate_cost_ba(config)
        det_local = breakdown["detector_is_local"]
        det_note = " [local vLLM, $0]" if det_local else ""
        val_note = " [local vLLM, $0]" if det_local else ""
        lines = [
            "── Estimated API Cost (bon-amplified) ─────────────",
            f"  Planner    (step 0 initial plan): ${breakdown['planner']:>8.2f}",
            f"  Cluster    (deduplication):       ${breakdown['cluster']:>8.2f}",
            f"  Humanness  (undesir. filter ×2):  ${breakdown['humanness']:>8.2f}",
            f"  Detection  (VLM attr detection):  ${breakdown['detection']:>8.2f}{det_note}",
            f"  Proposer   (P+/P- VLM mining):    ${breakdown['proposer']:>8.2f}",
            f"  Validation (A_hat check):         ${breakdown['validation']:>8.2f}{val_note}",
            "  ──────────────────────────────────────────────────",
            f"  TOTAL                             ${breakdown['total']:>8.2f}",
            "────────────────────────────────────────────────────",
        ]
    else:
        breakdown = estimate_cost(config)
        lines = [
            "── Estimated API Cost ─────────────────────────────",
            f"  Planner  (step 0 + mutations):  ${breakdown['planner']:>8.2f}",
            f"  Editor   (instruction gen):     ${breakdown['editor']:>8.2f}",
            f"  Judge    (pairwise compare):    ${breakdown['judge']:>8.2f}",
            f"  Amp      (attribute detection): ${breakdown['amp']:>8.2f}",
            f"  Cluster  (deduplication):       ${breakdown['cluster']:>8.2f}",
            "  ──────────────────────────────────────────────────",
            f"  TOTAL                           ${breakdown['total']:>8.2f}",
            "────────────────────────────────────────────────────",
        ]
    for line in lines:
        logger.info(line)