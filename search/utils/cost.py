"""Pre-run API cost estimator for the evolutionary T2I bias search."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from search.config import SearchConfig

# ─── Model pricing (OpenAI, 2025) — (input $/1M tok, output $/1M tok) ────────
# FluxKontextApplier runs locally on GPU → zero API cost.
# Editor cost = vision LLM call for instruction generation only (1 image in).
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "openai/gpt-4o-mini": (0.15,   0.60),
    "openai/gpt-4o":      (2.50,  10.00),
    "openai/gpt-5":       (2.50,  15.00),  # reasoning tokens counted in output
    "openai/gpt-5-nano":  (0.20,   1.25),
    "openai/gpt-5-mini":  (0.40,   1.60),
    "openai/gpt-5.2":     (2.50,  15.00),
}
_DEFAULT_PRICING = (1.00, 4.00)  # conservative fallback for unlisted models

# ─── Fixed token estimates ────────────────────────────────────────────────────
_PLANNER_OUTPUT_TOK    = 15_000  # reasoning trace + attribute list (gpt-5 high)
_PLANNER_TEXT_STEP0    = 1_000   # system + user-prompt text at step 0
_PLANNER_TEXT_MUTATION = 1_200   # MUTATE_PRE + POST text (all context modes)

_EDITOR_TEXT_TOK   = 500   # instruction-generation prompt text
_EDITOR_OUTPUT_TOK = 50    # short edit instruction string

_JUDGE_COMPARE_TEXT_TOK = 200   # image comparison question text
_JUDGE_COMPARE_OUT_TOK  = 50    # short JSON decision
_JUDGE_DETECT_TEXT_TOK  = 150   # attribute detection question text
_JUDGE_DETECT_OUT_TOK   = 30    # short JSON present/absent

_CLUSTER_ATTR_TOK   = 60    # ~60 tokens per attribute entry in JSON list
_CLUSTER_OUTPUT_TOK = 1_000

_MUTATOR_ROLLOUTS_IN_CONTEXT = 4   # n_rollouts_in_context in AttributeMutator (half top + half bottom)
_MUTATOR_ANCESTRY_IMGS_PER_NODE = 4  # 2 pairs × 2 images shown per ancestor node


def _call_cost(model: str, input_tok: int, output_tok: int) -> float:
    in_cpm, out_cpm = _MODEL_PRICING.get(model, _DEFAULT_PRICING)
    return (input_tok * in_cpm + output_tok * out_cpm) / 1_000_000


def _image_tokens(width: int = 1024, height: int = 1024) -> int:
    """
    OpenAI vision 'auto' detail token count.
    Uses 'low' (85 tok) when both dims ≤ 512, else 'high' (170/tile + 85 base).
    FLUX.1-Kontext default output is 1024×1024.
    """
    if width <= 512 and height <= 512:
        return 85
    tiles_w = math.ceil(width / 512)
    tiles_h = math.ceil(height / 512)
    return 170 * tiles_w * tiles_h + 85


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

    Returns a breakdown dict with keys:
      planner, editor, judge, amp, cluster, total
    """
    cfg = config
    ecfg = cfg.evaluation
    evcfg = cfg.evolution
    n_steps = evcfg.n_steps
    n_topics = len(cfg.data.topic_ids)

    img_tok = _image_tokens(width=512, height=512)  # FLUX default 1024×1024

    planner_model  = cfg.models.planner.model
    editor_model   = cfg.models.editor.instruction_model
    judge_model    = cfg.models.judge.model
    detector_model = cfg.models.detector.model
    cluster_model  = cfg.models.cluster.model

    # Per-call costs
    editor_call = _call_cost(editor_model,
                             _EDITOR_TEXT_TOK + img_tok, _EDITOR_OUTPUT_TOK)
    judge_compare_call = _call_cost(judge_model,
                                    _JUDGE_COMPARE_TEXT_TOK + 2 * img_tok,
                                    _JUDGE_COMPARE_OUT_TOK)
    # Amp detection uses detector_model (separate from judge_model after the split)
    judge_detect_call = _call_cost(detector_model,
                                   _JUDGE_DETECT_TEXT_TOK + img_tok,
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

            # n_initial_plan_prompts caps how many prompts InitialPlanner actually processes
            n_effective_prompts = (
                min(n_train, evcfg.n_initial_plan_prompts)
                if evcfg.n_initial_plan_prompts is not None
                else n_train
            )

            # ── Number of attributes entering evaluate_and_select ──────────
            if step_idx == 0:
                # After initial planning + clustering: ≤ initial_pop_size × 2
                n_initial_raw = n_effective_prompts * evcfg.n_per_user_prompt * evcfg.n_attrs_per_prompt
                n_attrs = min(n_initial_raw, evcfg.initial_pop_size * 2)
            else:
                # After mutation + clustering: ≤ target_pop_sizes[step_idx] × 2
                # n_mutations new children + 1 carry-over per survivor
                prev_surviving = evcfg.target_pop_sizes[step_idx - 1]
                n_attrs_raw = prev_surviving * (evcfg.n_mutations + 1)
                n_attrs = min(n_attrs_raw, pop_size * 2)

            # ── 1. Planner cost ───────────────────────────────────────────
            if step_idx == 0:
                # One LLM call per (effective_prompt × n_per_user_prompt)
                n_planner_calls = n_effective_prompts * evcfg.n_per_user_prompt
                planner_in_tok = evcfg.n_context_imgs * img_tok + _PLANNER_TEXT_STEP0
            else:
                prev_surviving = evcfg.target_pop_sizes[step_idx - 1]
                # One mutation call per surviving attribute (carry-overs need no LLM call)
                n_planner_calls = prev_surviving

                # Current attribute: n_rollouts_in_context pairs = 2 images each
                n_imgs = 2 * _MUTATOR_ROLLOUTS_IN_CONTEXT
                # Ancestry: step_idx ancestor nodes, each shows 2 pairs = 4 images
                if evcfg.context not in ("vanilla",):
                    n_imgs += _MUTATOR_ANCESTRY_IMGS_PER_NODE * step_idx
                planner_in_tok = n_imgs * img_tok + _PLANNER_TEXT_MUTATION

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
            # amp uses the same batch_prompts as evaluation (not a separate amp_n_prompts sample)
            n_amp_candidates = pop_size
            n_detect = n_amp_candidates * batch_size * ecfg.amp_n_images_per_prompt
            cost_amp += n_detect * judge_detect_call

    # Scale by number of topics
    cost_planner *= 1  # already looped over topics
    total = cost_planner + cost_editor + cost_judge + cost_amp + cost_cluster

    return {
        "planner": cost_planner,
        "editor":  cost_editor,
        "judge":   cost_judge,
        "amp":     cost_amp,
        "cluster": cost_cluster,
        "total":   total,
    }


def log_cost_estimate(config: "SearchConfig") -> None:
    """Print a formatted cost estimate breakdown to the logger."""
    from loguru import logger

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
