"""Find within-prompt natural-counterfactual image pairs from the detection cache.

For each target attribute, builds pairs (g=1 image, g=0 image) within the same
prompt, ranked by ascending Hamming distance on the OTHER attributes — i.e.
pairs whose images are as similar as possible except for the target attr.

Outputs one mega-grid PNG per target attribute (each row = one prompt's best
pair), plus an optional `pairs.json` with the full machine-readable bundle.

Usage:
    python -m analysis.hamming_pair_finder \\
        --per_prompt_W_path outputs/search/<run>/per_prompt_W_step<N>_topic0.json \\
        --baseline_manifest /nfs/.../topic_0/.../manifest.json \\
        --topic_id 0 \\
        --limit_n_attrs 4 \\
        --max_prompts_per_attr 5 \\
        --save_json
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from debias.counterfactual.schemas import PerPromptW  # noqa: E402
from debias.counterfactual.selection.attr_selector import (  # noqa: E402
    group_by_attr,
    select_per_prompt,
)
from debias.counterfactual.selection.detection_lookup import load_detection_cache  # noqa: E402
from debias.counterfactual.selection.per_prompt_w_loader import (  # noqa: E402
    limit_attrs,
    load_per_prompt_w,
)
from search.pipeline.baseline_pair_constructor import _hamming  # noqa: E402
from search.pipeline.baselines import (  # noqa: E402
    load_baselines_from_manifest,
    load_topic_states,
)

if TYPE_CHECKING:
    from search.data.types import BaselineImage


# ── Dataclass ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PairInfo:
    """One within-prompt natural counterfactual pair on a target attribute."""
    prompt_text: str
    attr: str
    pos_image_id: str                     # g_target = 1
    neg_image_id: str                     # g_target = 0
    pos_image_path: Path
    neg_image_path: Path
    hamming_other: int                    # Hamming over (attrs \ {target})
    n_others: int
    pos_reward: float | None = None       # baseline reward of the g=1 image
    neg_reward: float | None = None       # baseline reward of the g=0 image
    reward_model_name: str | None = None  # which key from BaselineImage.reward_scores


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--per_prompt_W_path", required=True, type=Path,
                   help="Source of the attribute list (PPW.attrs) and topic_id.")
    p.add_argument("--detection_cache_path", type=Path,
                   default=Path("outputs/detection_cache/mjhq/black-forest-labs-FLUX.1-dev.json"))
    p.add_argument("--detector_key", default="Qwen/Qwen3.5-9B::auto")
    p.add_argument("--baseline_manifest", required=True, type=Path)
    p.add_argument("--baseline_root", default="")
    p.add_argument("--prompts_dir", default="clustering/output/mjhq", type=Path)
    p.add_argument("--topic_id", required=True, type=int)
    p.add_argument("--target_attrs", default=None,
                   help="Comma-separated case-insensitive substrings; each is matched "
                        "against PPW.attrs. Multiple matches allowed.")
    p.add_argument("--target_attr_indices", default=None,
                   help="Comma-separated indices into PPW.attrs (after --limit_n_attrs).")
    p.add_argument("--limit_n_attrs", type=int, default=0,
                   help="If >0, restrict PPW.attrs to the first N entries.")
    p.add_argument("--posthoc_cleanup_path", type=Path, default=None,
                   help="Optional: path to a posthoc_cleanup_topic{T}_step{N}.json. "
                        "When provided, attrs are taken from cleanup.kept_pool (NOT from "
                        "PPW.attrs), per-prompt OLS is re-fit on that subset using "
                        "cleanup.N + cleanup.reward_model, and the resulting weights are "
                        "used to build B_x. --limit_n_attrs is ignored in this mode.")
    p.add_argument("--tau", type=float, default=0.0,
                   help="W threshold for B_x = {a : W_{x,a} > tau ∧ undesirable}.")
    p.add_argument("--top_n_per_prompt", type=int, default=3,
                   help="Per-prompt B_x cap by descending W_{x,a}. For each attr we then "
                        "consider only prompts where that attr ∈ B_x. "
                        "Set 0 to disable the filter (use all PPW prompts everywhere).")
    p.add_argument("--reward_model_name", default="imagereward",
                   choices=["imagereward", "pickscore", "hpsv3"],
                   help="Which reward score to read from the manifest's reward_scores dict. "
                        "Always logged in pairs.json / grid annotation when present.")
    # Strict g=1 consistency check (re-query the detector and keep only images
    # that the detector calls g=1 in EVERY query). Filters out cache entries
    # driven by detector noise.
    p.add_argument("--source_consistency_n", type=int, default=0,
                   help="If >0, re-detect each candidate g=1 image this many times. Keep "
                        "only images where ALL queries return 1. 0 = skip the check.")
    p.add_argument("--detector_model", default="Qwen/Qwen3.5-9B",
                   help="Used only when --source_consistency_n > 0.")
    p.add_argument("--detector_vllm_base_url", default=None,
                   help="Local vLLM endpoint (e.g. http://localhost:8000/v1). "
                        "Required when --source_consistency_n > 0 and detector is Qwen.")
    p.add_argument("--detector_image_detail", default="auto")
    p.add_argument("--detector_max_parallel", type=int, default=32)
    p.add_argument("--detector_max_tokens", type=int, default=1024)
    p.add_argument("--detector_use_applicability", action="store_true",
                   help="Ask the detector to also report whether the attribute even "
                        "applies to the image (e.g. attribute talks about rocks but the "
                        "image has none). Such images are returned as -1 and excluded "
                        "from the g=1 consistency set.")
    p.add_argument("--also_verify_neg", action="store_true",
                   help="Also re-verify g=0 (neg) candidates with --source_consistency_n "
                        "rounds; keep only images that the detector calls present=false "
                        "(and applicable, when --detector_use_applicability is on) every "
                        "single round. Default is to verify g=1 (pos) only.")
    p.add_argument("--require_pos_higher_reward", action="store_true",
                   help="Keep only pairs where the g=1 (pos) image has strictly higher "
                        "reward than g=0 (neg). Pairs missing either score are dropped.")
    p.add_argument("--h0_min_reward_delta", action="store_true",
                   help="Restrict to pairs with hamming_other == 0 (identical in every OTHER "
                        "attr) and rank them by |pos_reward − neg_reward| ascending. The "
                        "top-k and prompt selection then surface the tightest counterfactual: "
                        "same on every other attribute, smallest reward gap. Pairs missing "
                        "either reward score are dropped.")
    p.add_argument("--top_k_pairs", type=int, default=1,
                   help="Top-K best pairs per (prompt, attr), ranked by ascending Hamming. "
                        "The mega-grid only uses the top-1 row.")
    p.add_argument("--max_prompts_per_attr", type=int, default=8,
                   help="Rows in the per-attr mega-grid.")
    p.add_argument("--out_root", default="outputs/hamming_pairs", type=Path)
    p.add_argument("--thumb_px", type=int, default=256)
    p.add_argument("--save_json", action="store_true")
    p.add_argument("--run_id", default=None,
                   help="Output subdir name. Default: current timestamp.")
    return p.parse_args()


# ── Loaders ──────────────────────────────────────────────────────────────────


def _load_ppw(ppw_path: Path, limit_n_attrs: int) -> PerPromptW:
    """Load the per_prompt_W artifact and optionally slice the attribute list."""
    ppw = load_per_prompt_w(ppw_path)
    K_before = len(ppw.attrs)
    if limit_n_attrs > 0 and limit_n_attrs < K_before:
        ppw = limit_attrs(ppw, limit_n_attrs)
        logger.info(f"  limit_n_attrs: {K_before} → {len(ppw.attrs)}")
    logger.info(
        f"per_prompt_W: topic={ppw.topic_id} K={len(ppw.attrs)} "
        f"P={len(ppw.per_prompt_W)} (from {ppw_path.name})"
    )
    return ppw


def _resolve_target_attrs(
    all_attrs: list[str],
    target_attrs_csv: str | None,
    target_attr_indices: str | None,
) -> list[str]:
    """Resolve user-specified target attrs. None+None ⇒ all attrs."""
    selected: list[str] = []
    if target_attrs_csv:
        needles = [s.strip().lower() for s in target_attrs_csv.split(",") if s.strip()]
        for needle in needles:
            matched = [a for a in all_attrs if needle in a.lower()]
            if not matched:
                logger.warning(f"  --target_attrs: no match for {needle!r}")
            for a in matched:
                if a not in selected:
                    selected.append(a)
    if target_attr_indices:
        for tok in target_attr_indices.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                idx = int(tok)
            except ValueError:
                logger.warning(f"  --target_attr_indices: not an integer: {tok!r}")
                continue
            if 0 <= idx < len(all_attrs):
                if all_attrs[idx] not in selected:
                    selected.append(all_attrs[idx])
            else:
                logger.warning(f"  --target_attr_indices: index {idx} out of range")
    if not selected:
        if target_attrs_csv or target_attr_indices:
            logger.warning("no target attrs matched — exiting without producing grids")
            return []
        selected = list(all_attrs)
    logger.info(f"target attrs: {len(selected)}/{len(all_attrs)}")
    return selected


def _refit_w_from_kept_pool(
    posthoc_path: Path,
    ppw: PerPromptW,
    baselines_by_prompt: "dict[str, list[BaselineImage]]",
    detection: dict[str, dict[str, int]],
) -> PerPromptW:
    """Load posthoc cleanup, run per-prompt OLS on its kept_pool, return new PPW.

    `baselines_by_prompt` should already be filtered to the prompts that were
    used by the original search OLS (i.e. ppw.per_prompt_W.keys()). Each
    BaselineImage must carry reward_scores[reward_model_name].
    """
    from search.pipeline.bon_amplified_evo import _compute_bon_residuals

    with open(posthoc_path) as f:
        cleanup = json.load(f)

    kept_pool: list[str] = list(cleanup["kept_pool"])
    N: int = int(cleanup.get("N", 2))
    reward_model_name: str = cleanup.get("reward_model", "imagereward")
    ols_mode: str = cleanup.get("ols_mode", "per_prompt")
    topic_id: int = int(cleanup["topic_id"])
    step_idx: int = int(cleanup.get("step", -1))

    if topic_id != ppw.topic_id:
        raise ValueError(
            f"posthoc cleanup topic_id={topic_id} ≠ ppw.topic_id={ppw.topic_id}"
        )

    logger.info(
        f"posthoc cleanup: step={step_idx}  "
        f"initial_pool={cleanup.get('n_initial')}  "
        f"kept_pool={len(kept_pool)}  removed={cleanup.get('n_removed')}  "
        f"reward_model={reward_model_name}  N={N}  ols_mode={ols_mode}"
    )
    if ols_mode != "per_prompt":
        logger.warning(
            f"  posthoc was {ols_mode!r} but we are running 'per_prompt' OLS for B_x"
        )

    # Refit OLS on the kept_pool. Mode is forced to 'per_prompt' because we need
    # the per-prompt W to construct B_x downstream.
    (
        _residuals, _W_out, var_exp, mean_abs, _max_abs,
        per_prompt_r2, per_prompt_W_out,
    ) = _compute_bon_residuals(
        detection, baselines_by_prompt, kept_pool,
        reward_model_name, N, mode="per_prompt",
    )
    if per_prompt_W_out is None:
        raise RuntimeError(
            "per-prompt OLS produced no per_prompt_W (likely all prompts had <2 "
            "scored+fully-detected images)"
        )

    logger.info(
        f"refit OLS [per_prompt] on kept_pool: K={len(kept_pool)}  "
        f"P={len(per_prompt_W_out)}  var_explained={var_exp:.4f}  "
        f"mean|r|={mean_abs:.4f}"
    )

    # Drop prompts the OLS couldn't fit (e.g. too few scored images)
    prompts_dropped = sorted(set(ppw.per_prompt_W.keys()) - set(per_prompt_W_out.keys()))
    if prompts_dropped:
        logger.info(
            f"  refit dropped {len(prompts_dropped)} prompts (insufficient OLS support)"
        )

    return PerPromptW(
        step_idx=step_idx,
        topic_id=topic_id,
        attrs=kept_pool,
        per_prompt_W=per_prompt_W_out,
        per_prompt_r2=per_prompt_r2,
    )


def _build_detector(args: argparse.Namespace):
    """Build a VisionLLMDetector from CLI args (only when consistency check is on)."""
    from search.config import DetectorConfig
    from search.models.detector import build_detector as _build
    cfg = DetectorConfig(
        model=args.detector_model,
        max_tokens=args.detector_max_tokens,
        max_parallel=args.detector_max_parallel,
        image_detail=args.detector_image_detail,
        use_batch_api=False,
        vllm_base_url=args.detector_vllm_base_url,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        } if args.detector_vllm_base_url else None,
        use_prompt=False,
        use_reasoning=False,
        use_applicability=args.detector_use_applicability,
    )
    return _build(cfg, cache_config=None)


async def _refine_consistency(
    candidates_by_attr: "dict[str, list[tuple[str, BaselineImage]]]",
    detector,
    n_repeats: int,
    target_value: int,                          # 1 = require all-present; 0 = require all-not-present
) -> dict[str, set[str]]:
    """Re-query the detector `n_repeats` times on each candidate image (batched
    per attr). Return `{attr: set of image_ids that returned `target_value` in
    EVERY round AND never returned -1 (not-applicable)}`.

    Rounds where the detector itself crashes are NOT counted toward
    `target_value` — noisy attributes are dropped rather than waved through.
    Any single -1 round drops the image (an image whose attribute does not
    apply is not a clean g=1 OR g=0).
    """
    label = f"g={target_value}"
    survived: dict[str, set[str]] = {}
    n_in = sum(len(v) for v in candidates_by_attr.values())
    logger.info(
        f"{label} consistency check: re-querying detector {n_repeats}× on "
        f"{n_in} (image, attr) pairs across {len(candidates_by_attr)} attrs"
    )
    n_out = 0
    n_na_total = 0                              # any -1 round  → "not applicable"
    for attr, pairs in candidates_by_attr.items():
        if not pairs:
            survived[attr] = set()
            continue
        prompts = [p for p, _ in pairs]
        paths = [str(img.image_path) for _, img in pairs]
        matched = [0] * len(pairs)
        na_seen = [False] * len(pairs)
        rounds_done = 0
        for _ in range(n_repeats):
            try:
                results = await detector.detect(paths, prompts, attr)
            except Exception as e:
                logger.exception(
                    f"  {label} consistency: detector failed on attr={attr[:50]!r}: {e}"
                )
                continue
            rounds_done += 1
            for i, r in enumerate(results):
                if r is None:
                    continue
                rv = int(r)
                if rv == -1:
                    na_seen[i] = True            # N/A in any round → drop
                elif rv == target_value:
                    matched[i] += 1
        ids = {
            pairs[i][1].image_id
            for i in range(len(pairs))
            if matched[i] == n_repeats and not na_seen[i]
        }
        n_na_attr = sum(1 for x in na_seen if x)
        survived[attr] = ids
        n_out += len(ids)
        n_na_total += n_na_attr
        logger.info(
            f"  {label} consistency  attr={attr[:60]!r}  "
            f"{rounds_done}/{n_repeats} rounds  kept={len(ids)}/{len(pairs)}  "
            f"n/a={n_na_attr}"
        )
    logger.info(
        f"{label} consistency: kept {n_out}/{n_in} "
        f"({100 * n_out / max(n_in, 1):.1f}% retention; "
        f"{n_na_total} dropped as not-applicable)"
    )
    return survived


def _load_baselines_for_topic(
    prompts_dir: Path,
    baseline_manifest: Path,
    baseline_root: str,
    topic_id: int,
) -> "dict[str, list[BaselineImage]]":
    """Load all prompts in the topic (train + val together via val_split_size=0)."""
    states = load_topic_states(
        prompts_dir=str(prompts_dir),
        topic_ids=[topic_id],
        val_split_size=0,
        random_seed=42,
    )
    if not states:
        raise ValueError(f"no topic state for topic_id={topic_id}")
    ts = states[0]
    load_baselines_from_manifest(ts, str(baseline_manifest), str(baseline_root))
    logger.info(
        f"baselines: {len(ts.baselines)} prompts loaded "
        f"({sum(len(v) for v in ts.baselines.values())} images total)"
    )
    return ts.baselines


# ── Core pair finding ────────────────────────────────────────────────────────


def _find_pairs_for_prompt_attr(
    images: "list[BaselineImage]",
    detection: dict[str, dict[str, int]],
    attr: str,
    others: list[str],
    top_k: int,
    reward_model_name: str = "imagereward",
    require_pos_higher_reward: bool = False,
    consistent_g1_ids: set[str] | None = None,
    consistent_g0_ids: set[str] | None = None,
    h0_min_reward_delta: bool = False,
) -> list[PairInfo]:
    """Return top-K (g=1, g=0) pairs by ascending Hamming on `others`.

    If `require_pos_higher_reward` is set, only keep pairs where
    `pos.reward > neg.reward` under `reward_model_name`. Pairs whose images are
    missing the score under that key are dropped.

    If `consistent_g1_ids` is provided (not None), g=1 images are additionally
    restricted to that allow-set — i.e. the upstream consistency check has
    confirmed the detector calls them g=1 across multiple re-queries.
    """
    scored = [
        img for img in images
        if img.image_id in detection
        and attr in detection[img.image_id]
    ]
    if not scored:
        return []
    G1 = [img for img in scored if detection[img.image_id][attr] == 1]
    if consistent_g1_ids is not None:
        G1 = [img for img in G1 if img.image_id in consistent_g1_ids]
    G0 = [img for img in scored if detection[img.image_id][attr] == 0]
    if consistent_g0_ids is not None:
        G0 = [img for img in G0 if img.image_id in consistent_g0_ids]
    if not G1 or not G0:
        return []
    prompt_text = scored[0].prompt.text         # all share the same prompt
    pairs: list[PairInfo] = []
    for g1, g0 in itertools.product(G1, G0):
        pos_r = g1.reward_scores.get(reward_model_name)
        neg_r = g0.reward_scores.get(reward_model_name)
        if require_pos_higher_reward:
            if pos_r is None or neg_r is None:
                continue
            if pos_r <= neg_r:
                continue
        h = _hamming(detection[g1.image_id], detection[g0.image_id], others)
        pairs.append(PairInfo(
            prompt_text=prompt_text,
            attr=attr,
            pos_image_id=g1.image_id,
            neg_image_id=g0.image_id,
            pos_image_path=Path(g1.image_path),
            neg_image_path=Path(g0.image_path),
            hamming_other=h,
            n_others=len(others),
            pos_reward=float(pos_r) if pos_r is not None else None,
            neg_reward=float(neg_r) if neg_r is not None else None,
            reward_model_name=reward_model_name,
        ))
    if h0_min_reward_delta:
        pairs = [
            p for p in pairs
            if p.hamming_other == 0
            and p.pos_reward is not None
            and p.neg_reward is not None
        ]
        pairs.sort(key=lambda p: (
            abs(p.pos_reward - p.neg_reward),
            p.pos_image_id, p.neg_image_id,
        ))
    else:
        pairs.sort(key=lambda p: (p.hamming_other, p.pos_image_id, p.neg_image_id))
    return pairs[: max(top_k, 1)]


def _select_prompts_per_attr(
    pairs_by_prompt: dict[str, list[PairInfo]],
    max_prompts_per_attr: int,
    h0_min_reward_delta: bool = False,
) -> dict[str, list[PairInfo]]:
    """Keep up to N prompts, ranked by each prompt's best pair.

    Default ranking: ascending hamming_other (smallest h first).
    When `h0_min_reward_delta` is on, all kept pairs have h=0 already, so we
    rank prompts by |Δr| of the best pair instead.
    """
    if h0_min_reward_delta:
        def _key(kv):
            pairs = kv[1]
            if not pairs:
                return (float("inf"),)
            p0 = pairs[0]
            if p0.pos_reward is None or p0.neg_reward is None:
                return (float("inf"),)
            return (abs(p0.pos_reward - p0.neg_reward),)
    else:
        def _key(kv):
            return (kv[1][0].hamming_other if kv[1] else 10**9,)
    ranked = sorted(pairs_by_prompt.items(), key=_key)
    return dict(ranked[: max(max_prompts_per_attr, 1)])


# ── Output ───────────────────────────────────────────────────────────────────


def _attr_slug(attr: str) -> str:
    return hashlib.sha1(attr.encode("utf-8")).hexdigest()[:8]


def _load_font(size: int):
    """Best-effort truetype load with default fallback."""
    from PIL import ImageFont
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _wrap_text(text: str, max_chars: int) -> list[str]:
    """Greedy word-wrap. Long single tokens are hard-split."""
    if max_chars <= 0:
        return [text]
    lines: list[str] = []
    current = ""
    for word in text.split():
        # Hard-split a single oversized token
        while len(word) > max_chars:
            if current:
                lines.append(current)
                current = ""
            lines.append(word[:max_chars])
            word = word[max_chars:]
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _render_per_attr_grid(
    attr: str,
    by_prompt: dict[str, list[PairInfo]],
    out_path: Path,
    thumb_px: int,
) -> None:
    """One mega-grid per attribute. Each row = top-1 pair from one prompt.

    Layout per row:  prompt text (wrapped) | (pos image | neg image) | h annotation
    Top of canvas:   full attribute text (wrapped, untruncated) + stats line.
    """
    from PIL import Image, ImageDraw

    rows: list[tuple[str, PairInfo]] = []
    for prompt_text, pairs in by_prompt.items():
        if pairs:
            rows.append((prompt_text, pairs[0]))           # top-1 per prompt
    if not rows:
        logger.warning(f"  no pairs to render for attr={attr[:50]!r}")
        return

    hammings = [p.hamming_other for _, p in rows]
    stats_line = (
        f"rows={len(rows)}  min_h={min(hammings)}  "
        f"mean_h={sum(hammings) / len(hammings):.1f}  "
        f"max_h={max(hammings)}  n_others={rows[0][1].n_others}"
    )

    # Fonts + per-char width
    title_font  = _load_font(18)
    body_font   = _load_font(13)
    anno_font   = _load_font(12)
    title_line_h = 22
    body_line_h  = 16

    # Layout constants
    pad         = 12
    images_w    = thumb_px * 2 + pad                # two images side by side
    prompt_w    = 320                               # left text column for the prompt
    anno_w      = 60                                # h=<n> annotation column on the right
    cell_w      = prompt_w + pad + images_w + pad + anno_w
    grid_w      = cell_w + 2 * pad

    # Title block: wrap attr text to the full width
    title_max_chars = max(40, grid_w // 9)          # ~9px/char at 18pt sans
    title_lines = _wrap_text(attr, title_max_chars)
    title_h = title_line_h * len(title_lines) + title_line_h + pad   # +1 line for stats

    # Per-row prompt wrap width
    prompt_max_chars = max(20, prompt_w // 7)       # ~7px/char at 13pt sans

    # Pre-compute row heights (variable, depending on prompt length)
    row_blocks: list[tuple[str, PairInfo, list[str], int]] = []
    for prompt_text, pi in rows:
        plines = _wrap_text(prompt_text, prompt_max_chars)
        text_h = body_line_h * len(plines)
        row_h = max(thumb_px, text_h) + pad
        row_blocks.append((prompt_text, pi, plines, row_h))

    total_h = pad + title_h + pad + sum(r[3] for r in row_blocks) + pad
    canvas = Image.new("RGB", (grid_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Title (full attr text)
    y = pad
    for line in title_lines:
        draw.text((pad, y), line, fill=(0, 0, 0), font=title_font)
        y += title_line_h
    draw.text((pad, y), stats_line, fill=(100, 100, 100), font=title_font)
    y += title_line_h + pad

    # Rows
    for prompt_text, pi, plines, row_h in row_blocks:
        # Separator line above each row (except the first)
        draw.line([(pad, y - 2), (grid_w - pad, y - 2)], fill=(220, 220, 220), width=1)

        # Prompt column (left)
        ty = y + max(0, (row_h - pad - body_line_h * len(plines)) // 2)
        for line in plines:
            draw.text((pad, ty), line, fill=(0, 0, 0), font=body_font)
            ty += body_line_h

        # Image pair (centre)
        img_x = pad + prompt_w + pad
        img_y = y + max(0, (row_h - pad - thumb_px) // 2)
        for j, p in enumerate([pi.pos_image_path, pi.neg_image_path]):
            try:
                im = Image.open(p).convert("RGB")
                im.thumbnail((thumb_px, thumb_px))
            except Exception as e:
                logger.warning(f"  cannot read {p}: {e}")
                im = Image.new("RGB", (thumb_px, thumb_px), (200, 200, 200))
            canvas.paste(im, (img_x + j * (thumb_px + pad // 2), img_y))

        # Annotation (right): h value + reward delta + pos/neg ids
        ax = pad + prompt_w + pad + images_w + pad
        n_anno_lines = 5 if (pi.pos_reward is not None and pi.neg_reward is not None) else 3
        ay = y + max(0, (row_h - pad - body_line_h * n_anno_lines) // 2)
        draw.text((ax, ay), f"h={pi.hamming_other}",
                  fill=(20, 100, 20), font=anno_font)
        next_y = ay + body_line_h
        if pi.pos_reward is not None and pi.neg_reward is not None:
            delta = pi.pos_reward - pi.neg_reward
            delta_color = (20, 100, 20) if delta > 0 else (140, 30, 30)
            draw.text((ax, next_y),
                      f"Δr={delta:+.3f}",
                      fill=delta_color, font=anno_font)
            next_y += body_line_h
            draw.text((ax, next_y),
                      f"+r={pi.pos_reward:+.3f}",
                      fill=(140, 30, 30), font=anno_font)
            next_y += body_line_h
            draw.text((ax, next_y),
                      f"-r={pi.neg_reward:+.3f}",
                      fill=(30, 30, 140), font=anno_font)
            next_y += body_line_h
        draw.text((ax, next_y),
                  f"+{pi.pos_image_id[:10]}",
                  fill=(140, 30, 30), font=anno_font)
        next_y += body_line_h
        draw.text((ax, next_y),
                  f"-{pi.neg_image_id[:10]}",
                  fill=(30, 30, 140), font=anno_font)

        y += row_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    logger.info(f"  mega-grid → {out_path}")


def _write_pairs_json(
    out_path: Path,
    by_attr: dict[str, dict[str, list[PairInfo]]],
    meta: dict,
) -> None:
    payload: dict = {"meta": meta, "by_attr": {}}
    for attr, by_prompt in by_attr.items():
        payload["by_attr"][attr] = {
            prompt_text: [
                {
                    "pos_image_id": p.pos_image_id,
                    "neg_image_id": p.neg_image_id,
                    "pos_image_path": str(p.pos_image_path),
                    "neg_image_path": str(p.neg_image_path),
                    "hamming_other": p.hamming_other,
                    "n_others": p.n_others,
                    "pos_reward": p.pos_reward,
                    "neg_reward": p.neg_reward,
                    "reward_delta": (
                        (p.pos_reward - p.neg_reward)
                        if (p.pos_reward is not None and p.neg_reward is not None)
                        else None
                    ),
                    "reward_model_name": p.reward_model_name,
                    "rank": rank,
                }
                for rank, p in enumerate(pairs)
            ]
            for prompt_text, pairs in by_prompt.items()
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"pairs.json → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


async def amain() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_root / run_id
    grids_dir = out_dir / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"output dir → {out_dir}")

    # 1. attrs + topic_id (full PerPromptW so we can also use its prompt list)
    if args.posthoc_cleanup_path and args.limit_n_attrs > 0:
        logger.warning(
            "--limit_n_attrs is ignored when --posthoc_cleanup_path is provided "
            "(kept_pool is already curated)."
        )
    ppw = _load_ppw(
        args.per_prompt_W_path,
        0 if args.posthoc_cleanup_path else args.limit_n_attrs,
    )
    assert ppw.topic_id == args.topic_id, (
        f"topic_id mismatch: ppw.topic_id={ppw.topic_id} vs --topic_id={args.topic_id}"
    )

    # 2. detection cache
    detection = load_detection_cache(args.detection_cache_path, args.detector_key)

    # 3. baselines, restricted to the prompts that actually appear in PPW —
    #    i.e. the same training prompts the search step used for OLS.
    baselines_by_prompt = _load_baselines_for_topic(
        args.prompts_dir, args.baseline_manifest, args.baseline_root, args.topic_id,
    )
    used_prompts = set(ppw.per_prompt_W.keys())
    before = len(baselines_by_prompt)
    baselines_by_prompt = {p: imgs for p, imgs in baselines_by_prompt.items() if p in used_prompts}
    logger.info(
        f"baselines filtered to PPW prompts: {before} → {len(baselines_by_prompt)} "
        f"(= search train set)"
    )

    # 3b. (optional) Replace ppw with a fresh per-prompt OLS fit on the
    #     posthoc-cleanup `kept_pool`. Downstream B_x / target_attrs / etc.
    #     all use this new ppw transparently.
    if args.posthoc_cleanup_path:
        ppw = _refit_w_from_kept_pool(
            args.posthoc_cleanup_path, ppw, baselines_by_prompt, detection,
        )

    all_attrs = list(ppw.attrs)

    # 4. target attrs to iterate (resolved against the final attribute list)
    target_attrs = _resolve_target_attrs(
        all_attrs, args.target_attrs, args.target_attr_indices,
    )
    if not target_attrs:
        return

    # 4b. (optional) Build P_a per attribute via B_x = top-N W_{x,a} > tau.
    #     `undesirable` defaults to all PPW attrs (already humanness-filtered upstream).
    attr_to_pa: "dict[str, set[str]] | None" = None
    if args.top_n_per_prompt > 0:
        undesirable = set(ppw.attrs)
        selections = select_per_prompt(
            ppw, undesirable, tau=args.tau, top_n=args.top_n_per_prompt,
        )
        by_attr_sel = group_by_attr(selections)
        attr_to_pa = {a: {s.prompt_text for s in sels} for a, sels in by_attr_sel.items()}
        logger.info(
            f"B_x filter on  (tau={args.tau}, top_n={args.top_n_per_prompt}) → "
            f"{len(attr_to_pa)} attrs have non-empty P_a"
        )
    else:
        logger.info("B_x filter off (top_n_per_prompt=0) — every PPW prompt considered for every attr")

    # 4c. (optional) Source consistency check via repeated detector queries.
    #     For each target attr, collect candidate images the iterate-loop will
    #     consider (post B_x filter) and re-run the detector `n_repeats` times
    #     per (image, attr). g=1 candidates must come back as 1 every round;
    #     g=0 candidates must come back as 0 every round (only enabled when
    #     --also_verify_neg is set). Images that return -1 (applicable=false)
    #     in any round are dropped from the surviving set.
    consistent_g1_by_attr: "dict[str, set[str]] | None" = None
    consistent_g0_by_attr: "dict[str, set[str]] | None" = None
    if args.source_consistency_n > 0:
        candidates_g1_by_attr: dict[str, list[tuple[str, "BaselineImage"]]] = {}
        candidates_g0_by_attr: dict[str, list[tuple[str, "BaselineImage"]]] = {}
        for attr in target_attrs:
            if attr_to_pa is not None:
                pa = attr_to_pa.get(attr, set())
                if not pa:
                    candidates_g1_by_attr[attr] = []
                    candidates_g0_by_attr[attr] = []
                    continue
                eligible_for_attr = {
                    p: imgs for p, imgs in baselines_by_prompt.items() if p in pa
                }
            else:
                eligible_for_attr = baselines_by_prompt
            items_g1: list[tuple[str, BaselineImage]] = []
            items_g0: list[tuple[str, BaselineImage]] = []
            for prompt_text, images in eligible_for_attr.items():
                for img in images:
                    d = detection.get(img.image_id)
                    if d is None:
                        continue
                    v = d.get(attr)
                    if v == 1:
                        items_g1.append((prompt_text, img))
                    elif v == 0:
                        items_g0.append((prompt_text, img))
            candidates_g1_by_attr[attr] = items_g1
            candidates_g0_by_attr[attr] = items_g0
        detector = _build_detector(args)
        try:
            consistent_g1_by_attr = await _refine_consistency(
                candidates_g1_by_attr, detector, args.source_consistency_n,
                target_value=1,
            )
            if args.also_verify_neg:
                consistent_g0_by_attr = await _refine_consistency(
                    candidates_g0_by_attr, detector, args.source_consistency_n,
                    target_value=0,
                )
        finally:
            if hasattr(detector, "shutdown"):
                try:
                    await detector.shutdown()
                except Exception as e:
                    logger.warning(f"detector shutdown failed: {e}")

    # 5. iterate target attrs
    by_attr_pairs: dict[str, dict[str, list[PairInfo]]] = {}
    attr_index_lines: list[str] = []
    for attr in target_attrs:
        # B_x filter: only iterate prompts that have this attr in their B_x
        if attr_to_pa is not None:
            pa = attr_to_pa.get(attr, set())
            if not pa:
                logger.info(
                    f"attr={attr[:60]!r}  not in any prompt's B_x — skip "
                    f"(tau={args.tau}, top_n={args.top_n_per_prompt})"
                )
                continue
            eligible_baselines = {
                p: imgs for p, imgs in baselines_by_prompt.items() if p in pa
            }
        else:
            eligible_baselines = baselines_by_prompt

        allow_g1_ids = (
            consistent_g1_by_attr.get(attr, set())
            if consistent_g1_by_attr is not None else None
        )
        if consistent_g1_by_attr is not None and not allow_g1_ids:
            logger.info(
                f"attr={attr[:60]!r}  no g=1 image survived consistency check — skip"
            )
            continue
        allow_g0_ids = (
            consistent_g0_by_attr.get(attr, set())
            if consistent_g0_by_attr is not None else None
        )
        if consistent_g0_by_attr is not None and not allow_g0_ids:
            logger.info(
                f"attr={attr[:60]!r}  no g=0 image survived consistency check — skip"
            )
            continue

        others = [x for x in all_attrs if x != attr]
        pairs_by_prompt: dict[str, list[PairInfo]] = {}
        for prompt_text, images in eligible_baselines.items():
            pairs = _find_pairs_for_prompt_attr(
                images, detection, attr, others, args.top_k_pairs,
                reward_model_name=args.reward_model_name,
                require_pos_higher_reward=args.require_pos_higher_reward,
                consistent_g1_ids=allow_g1_ids,
                consistent_g0_ids=allow_g0_ids,
                h0_min_reward_delta=args.h0_min_reward_delta,
            )
            if pairs:
                pairs_by_prompt[prompt_text] = pairs

        if not pairs_by_prompt:
            logger.info(
                f"attr={attr[:60]!r}  no eligible prompts after pair finding "
                f"(B_x had {len(eligible_baselines)} candidates)"
            )
            continue

        kept = _select_prompts_per_attr(
            pairs_by_prompt, args.max_prompts_per_attr,
            h0_min_reward_delta=args.h0_min_reward_delta,
        )
        hs = [pairs[0].hamming_other for pairs in kept.values()]
        logger.info(
            f"attr={attr[:60]!r}  prompts={len(kept)}/{len(pairs_by_prompt)} "
            f"(of {len(eligible_baselines)} in P_a)  "
            f"min_h={min(hs)}  mean_h={sum(hs) / len(hs):.1f}"
        )

        slug = _attr_slug(attr)
        out_png = grids_dir / f"{slug}.png"
        _render_per_attr_grid(attr, kept, out_png, args.thumb_px)
        attr_index_lines.append(f"{slug}\t{attr}")
        by_attr_pairs[attr] = kept

    # 6. attr index for human navigation
    if attr_index_lines:
        idx_path = grids_dir / "_attr_index.txt"
        idx_path.write_text("\n".join(attr_index_lines) + "\n", encoding="utf-8")
        logger.info(f"attr index → {idx_path}")

    # 7. optional pairs.json
    if args.save_json:
        meta = {
            "ppw_path": str(args.per_prompt_W_path),
            "posthoc_cleanup_path": (
                str(args.posthoc_cleanup_path)
                if args.posthoc_cleanup_path else None
            ),
            "attrs_source": (
                "posthoc_kept_pool" if args.posthoc_cleanup_path else "ppw"
            ),
            "detection_cache_path": str(args.detection_cache_path),
            "detector_key": args.detector_key,
            "baseline_manifest": str(args.baseline_manifest),
            "topic_id": args.topic_id,
            "attrs_used": all_attrs,
            "target_attrs": target_attrs,
            "n_other_attrs_per_target": len(all_attrs) - 1,
            "tau": args.tau,
            "top_n_per_prompt": args.top_n_per_prompt,
            "top_k_pairs": args.top_k_pairs,
            "max_prompts_per_attr": args.max_prompts_per_attr,
            "reward_model_name": args.reward_model_name,
            "require_pos_higher_reward": args.require_pos_higher_reward,
            "h0_min_reward_delta": args.h0_min_reward_delta,
            "source_consistency_n": args.source_consistency_n,
            "also_verify_neg": (
                bool(args.also_verify_neg) if args.source_consistency_n > 0 else None
            ),
            "detector_model": (
                args.detector_model if args.source_consistency_n > 0 else None
            ),
            "detector_use_applicability": (
                bool(args.detector_use_applicability)
                if args.source_consistency_n > 0 else None
            ),
            "run_id": run_id,
            "command_line": " ".join(sys.argv),
        }
        _write_pairs_json(out_dir / "pairs.json", by_attr_pairs, meta)

    logger.info(f"done → {out_dir}  ({len(by_attr_pairs)} attrs rendered)")


def main() -> None:
    import asyncio
    asyncio.run(amain())


if __name__ == "__main__":
    main()
