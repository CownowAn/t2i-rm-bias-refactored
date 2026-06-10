#!/usr/bin/env python3
"""Counterfactual Pair Construction for Reward-Model Debiasing.

Finds *naturally-occurring* counterfactual pairs among already-generated images,
following a statistically-grounded prompt-selection pipeline:

  1. Per-attribute, per-prompt: **bootstrap** W_xk to identify *reliable* prompts
     (95% CI lower bound > 0  →  statistically significant positive W).
     W_xk is the per-prompt OLS coefficient of  U^{N-1}  on the attribute
     indicator matrix — exactly the regression that `search` solves in
     `bon_amplified_evo._compute_bon_residuals` (per_prompt mode).
  2. For each reliable prompt: find counterfactual pairs (y+, y-)
       - g_k(y+) = 1, g_k(y-) = 0
       - other-attribute matching: Hamming distance over the *other* attrs
         <= --max_hamming
       - reward filter: r(y+) > r(y-)            (scenario B)
       - reward-gap cap: r(y+) - r(y-) <= cap    (other-quality control;
         cap = --max_reward_gap_quantile quantile of within-prompt positive gaps)
  3. Render each chosen pair to a side-by-side PNG + write pairs.json / pairs.csv.

The regression inputs (G_x, U^{N-1}) are reconstructed *exactly* from the search
artifacts — verified to reproduce the saved per_prompt_W to ~1e-15. N, the reward
model, the manifest, and the detection cache are read from the run's
`config_effective.yaml` so the analysis stays consistent with the search.

Example
-------
    python -m debias.counterfactual.find_natural_pairs \
        --search_dir outputs/search/20260521-232513 \
        --step 9 \
        --out_dir outputs/cf_pairs/topic4_step9
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

_PPW_PAT = re.compile(r"per_prompt_W_step(\d+)_topic(\d+)\.json$")


# =====================================================================
# Data structures
# =====================================================================

@dataclass
class CounterfactualPair:
    attribute_idx: int
    attribute: str
    prompt: str
    positive_id: str          # image with g_k = 1 (debias target)
    negative_id: str          # image with g_k = 0 (should be preferred)
    reward_pos: float
    reward_neg: float
    reward_gap: float         # r_pos - r_neg (> 0 by construction)
    hamming_distance: int     # other-attribute mismatch count
    w_mean: float             # bootstrap mean of W_xk for this prompt
    w_ci_low: float           # 5th percentile of bootstrap W_xk
    w_ci_high: float          # 95th percentile of bootstrap W_xk
    pair_weight: float        # composite quality weight
    png_path: str = ""


@dataclass
class PromptReliability:
    prompt: str
    attribute_idx: int
    w_mean: float
    w_ci_low: float
    w_ci_high: float
    is_reliable: bool         # CI lower bound > 0


# =====================================================================
# Artifact / config loading
# =====================================================================

def _find_ppw(search_dir: Path, topic_id: int | None, step: int | None
              ) -> tuple[Path, int, int]:
    """Locate the per_prompt_W JSON to use. Defaults to the latest step."""
    cands: list[tuple[int, int, Path]] = []
    for p in search_dir.glob("per_prompt_W_step*_topic*.json"):
        m = _PPW_PAT.search(p.name)
        if not m:
            continue
        s, t = int(m.group(1)), int(m.group(2))
        if topic_id is not None and t != topic_id:
            continue
        if step is not None and s != step:
            continue
        cands.append((t, s, p))
    if not cands:
        raise FileNotFoundError(
            f"no per_prompt_W_step*_topic*.json in {search_dir} "
            f"(topic_id={topic_id}, step={step})"
        )
    topics = sorted({t for t, _, _ in cands})
    if topic_id is None and len(topics) > 1:
        raise ValueError(f"multiple topics in {search_dir}: {topics}. Pass --topic_id.")
    chosen_topic = topic_id if topic_id is not None else topics[0]
    sub = [(s, p) for t, s, p in cands if t == chosen_topic]
    s, p = max(sub, key=lambda x: x[0])
    return p, chosen_topic, s


def _load_config(search_dir: Path) -> dict:
    for name in ("config_effective.yaml", "config_source.yaml"):
        path = search_dir / name
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)
    return {}


def _cfg_get(cfg: dict, *keys, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# =====================================================================
# Step 0: reconstruct per-prompt regression inputs from search artifacts
# =====================================================================

@dataclass
class RegressionInputs:
    attrs: list[str]
    G_per_prompt: dict[str, np.ndarray]          # prompt -> (M_x, K) centered indicators
    u_per_prompt: dict[str, np.ndarray]          # prompt -> (M_x,)  centered U^{N-1}
    image_ids_per_prompt: dict[str, list[str]]   # prompt -> ordered image ids
    labels_per_image: dict[str, np.ndarray]      # image_id -> (K,) binary attr vector
    rewards_per_image: dict[str, float]          # image_id -> reward
    path_per_image: dict[str, Path]              # image_id -> image path


def build_regression_inputs(
    *,
    ppw: dict,
    detection: dict[str, dict[str, int]],
    baselines: dict[str, list[dict]],
    reward_name: str,
    N: int,
    baseline_root: Path | None,
) -> RegressionInputs:
    """Reconstruct (G_x centered, U^{N-1} centered) per prompt — matching
    `bon_amplified_evo._compute_bon_residuals` exactly.

    The prompt set is taken from per_prompt_W's keys (the prompts the search
    actually regressed on); all scored + fully-detected baseline images for each
    are used (the search used `amp_n_images_per_prompt`, which equals the full
    manifest count, so this reproduces the original sample).
    """
    attrs: list[str] = list(ppw["attrs"])
    K = len(attrs)
    prompts = list(ppw["per_prompt_W"].keys())

    ri = RegressionInputs(attrs, {}, {}, {}, {}, {}, {})
    for prompt in prompts:
        entries = baselines.get(prompt, [])
        scored = [
            e for e in entries
            if reward_name in e.get("reward_scores", {})
            and e["image_id"] in detection
            and all(a in detection[e["image_id"]] for a in attrs)
        ]
        if len(scored) < 2:
            continue

        n = len(scored)
        rewards = np.array([e["reward_scores"][reward_name] for e in scored], dtype=float)
        sorted_r = np.sort(rewards)
        U = np.searchsorted(sorted_r, rewards, side="right") / n
        U_pow = U ** (N - 1)

        G_x = np.array(
            [[float(detection[e["image_id"]].get(a, 0)) for a in attrs] for e in scored]
        )
        G_x_c = G_x - G_x.mean(axis=0, keepdims=True)
        U_pow_c = U_pow - U_pow.mean()

        ids = [e["image_id"] for e in scored]
        ri.G_per_prompt[prompt] = G_x_c
        ri.u_per_prompt[prompt] = U_pow_c
        ri.image_ids_per_prompt[prompt] = ids
        for e, lbl, r in zip(scored, G_x.astype(int), rewards):
            iid = e["image_id"]
            ri.labels_per_image[iid] = lbl
            ri.rewards_per_image[iid] = float(r)
            ri.path_per_image[iid] = _resolve(e["image_path"], baseline_root)
    return ri


def validate_reconstruction(ri: RegressionInputs, ppw: dict) -> float:
    """Recompute per-prompt OLS W_x and compare to the saved per_prompt_W.

    Returns the max abs difference. A large value means the config (reward model
    / N) does not match the run that produced per_prompt_W.
    """
    saved = ppw["per_prompt_W"]
    max_diff = 0.0
    for prompt, G_c in ri.G_per_prompt.items():
        u_c = ri.u_per_prompt[prompt]
        if np.var(u_c) < 1e-10 or G_c.shape[0] < G_c.shape[1]:
            W_x = np.zeros(G_c.shape[1])
        else:
            W_x, *_ = np.linalg.lstsq(G_c, u_c, rcond=None)
        ref = np.asarray(saved.get(prompt, np.zeros(G_c.shape[1])), dtype=float)
        if ref.shape == W_x.shape:
            max_diff = max(max_diff, float(np.abs(ref - W_x).max()))
    return max_diff


# =====================================================================
# Step 1: bootstrap-based reliability  (one bootstrap per prompt, reused for all attrs)
# =====================================================================

def bootstrap_W_all_attrs(
    G_x_c: np.ndarray,                 # (M_x, K) centered indicators
    u_x_c: np.ndarray,                 # (M_x,)  centered U^{N-1}
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap the full per-prompt OLS coefficient vector.

    Returns (mean[K], ci_low[K], ci_high[K]) with (5, 95) percentiles so that a
    one-sided 5% test of "W_xk > 0" corresponds to ci_low > 0. Resamples the
    (already-centered) rows with replacement, matching the search's regression.

    Degenerate prompts (near-constant u, or M < K) get a wide CI (ci_low=-inf)
    so no attribute is ever marked reliable for them.
    """
    M, K = G_x_c.shape
    if np.var(u_x_c) < 1e-10 or M < K:
        return np.zeros(K), np.full(K, -np.inf), np.full(K, np.inf)

    samples = np.empty((n_boot, K), dtype=float)
    ok = 0
    for _ in range(n_boot):
        idx = rng.integers(0, M, size=M)
        try:
            W_b, *_ = np.linalg.lstsq(G_x_c[idx], u_x_c[idx], rcond=None)
        except np.linalg.LinAlgError:
            continue
        samples[ok] = W_b
        ok += 1

    if ok < n_boot * 0.8:
        return np.zeros(K), np.full(K, -np.inf), np.full(K, np.inf)

    arr = samples[:ok]
    return arr.mean(0), np.percentile(arr, 5, axis=0), np.percentile(arr, 95, axis=0)


def identify_reliable_prompts(
    ri: RegressionInputs,
    boot_stats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    attribute_idx: int,
) -> list[PromptReliability]:
    """For one attribute, mark prompts with statistically reliable positive W."""
    out: list[PromptReliability] = []
    for prompt, (mean, lo, hi) in boot_stats.items():
        out.append(PromptReliability(
            prompt=prompt,
            attribute_idx=attribute_idx,
            w_mean=float(mean[attribute_idx]),
            w_ci_low=float(lo[attribute_idx]),
            w_ci_high=float(hi[attribute_idx]),
            is_reliable=bool(lo[attribute_idx] > 0),
        ))
    return out


# =====================================================================
# Step 2: pair construction within reliable prompts
# =====================================================================

def hamming_distance_other_attrs(a: np.ndarray, b: np.ndarray, exclude_idx: int) -> int:
    diff = a != b
    diff[exclude_idx] = False
    return int(diff.sum())


def find_best_counterfactual(
    y_pos_labels: np.ndarray,
    y_pos_reward: float,
    negative_candidates: list[tuple[str, np.ndarray, float]],
    attribute_idx: int,
    max_hamming: int,
    max_reward_gap: float,
    hamming_weight: float = 0.5,
) -> tuple[str, int, float] | None:
    """Best y_neg (g_k=0) for a given y_pos (g_k=1): Hamming<=cap, r_pos>r_neg,
    gap<=cap. Best = minimize weighted (hamming, normalized gap). Returns
    (neg_id, hamming, gap) or None."""
    candidates: list[tuple[str, int, float]] = []
    for neg_id, neg_labels, neg_reward in negative_candidates:
        h = hamming_distance_other_attrs(y_pos_labels, neg_labels, attribute_idx)
        if h > max_hamming:
            continue
        gap = y_pos_reward - neg_reward
        if gap <= 0 or gap > max_reward_gap:
            continue
        candidates.append((neg_id, h, gap))
    if not candidates:
        return None

    def score(item: tuple[str, int, float]) -> float:
        _, h, gap = item
        h_norm = h / max(max_hamming, 1)
        gap_norm = gap / max(max_reward_gap, 1e-9)
        return hamming_weight * h_norm + (1 - hamming_weight) * gap_norm

    candidates.sort(key=score)
    return candidates[0]


def compute_pair_weight(
    hamming: int, reward_gap: float, w_mean: float, w_ci_low: float,
    w_ci_high: float, max_hamming: int, max_reward_gap: float,
) -> float:
    """Higher for better-matched (low hamming), small-gap, high-confidence pairs."""
    hamming_score = 1.0 - hamming / max(max_hamming, 1)
    gap_score = 1.0 - reward_gap / max(max_reward_gap, 1e-9)
    ci_width = max(w_ci_high - w_ci_low, 1e-9)
    confidence_score = float(np.clip(w_mean / ci_width, 0.0, 5.0)) / 5.0
    return (hamming_score + gap_score + confidence_score) / 3.0


def build_counterfactual_pairs(
    *,
    ri: RegressionInputs,
    boot_stats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    target_attributes: list[int],
    max_hamming: int,
    max_reward_gap_quantile: float,
    max_pairs_per_prompt: int,
) -> dict[int, list[CounterfactualPair]]:
    """Build counterfactual pairs for each target attribute."""
    pairs_per_attribute: dict[int, list[CounterfactualPair]] = {}

    for attr_k in target_attributes:
        attr_name = ri.attrs[attr_k]
        reliability = identify_reliable_prompts(ri, boot_stats, attr_k)
        reliable = [r for r in reliability if r.is_reliable]
        if not reliable:
            pairs_per_attribute[attr_k] = []
            continue

        # adaptive reward-gap cap from within-prompt positive gaps across reliable prompts
        all_gaps: list[float] = []
        for r in reliable:
            ids = ri.image_ids_per_prompt[r.prompt]
            pos = [i for i in ids if ri.labels_per_image[i][attr_k] == 1]
            neg = [i for i in ids if ri.labels_per_image[i][attr_k] == 0]
            for p in pos:
                for ng in neg:
                    g = ri.rewards_per_image[p] - ri.rewards_per_image[ng]
                    if g > 0:
                        all_gaps.append(g)
        if not all_gaps:
            pairs_per_attribute[attr_k] = []
            continue
        max_reward_gap = float(np.quantile(all_gaps, max_reward_gap_quantile))

        pairs: list[CounterfactualPair] = []
        for r in reliable:
            ids = ri.image_ids_per_prompt[r.prompt]
            pos_imgs = [(i, ri.labels_per_image[i], ri.rewards_per_image[i])
                        for i in ids if ri.labels_per_image[i][attr_k] == 1]
            neg_imgs = [(i, ri.labels_per_image[i], ri.rewards_per_image[i])
                        for i in ids if ri.labels_per_image[i][attr_k] == 0]
            if not pos_imgs or not neg_imgs:
                continue
            pos_imgs.sort(key=lambda t: -t[2])  # strongest-biased positives first

            used_neg: set[str] = set()
            for y_pos_id, y_pos_labels, y_pos_reward in pos_imgs:
                if len([p for p in pairs if p.prompt == r.prompt]) >= max_pairs_per_prompt:
                    break
                cand = [(i, l, rw) for i, l, rw in neg_imgs if i not in used_neg]
                if not cand:
                    break
                match = find_best_counterfactual(
                    y_pos_labels, y_pos_reward, cand, attr_k,
                    max_hamming, max_reward_gap,
                )
                if match is None:
                    continue
                neg_id, hamming, gap = match
                weight = compute_pair_weight(
                    hamming, gap, r.w_mean, r.w_ci_low, r.w_ci_high,
                    max_hamming, max_reward_gap,
                )
                pairs.append(CounterfactualPair(
                    attribute_idx=attr_k, attribute=attr_name, prompt=r.prompt,
                    positive_id=y_pos_id, negative_id=neg_id,
                    reward_pos=y_pos_reward, reward_neg=y_pos_reward - gap,
                    reward_gap=gap, hamming_distance=hamming,
                    w_mean=r.w_mean, w_ci_low=r.w_ci_low, w_ci_high=r.w_ci_high,
                    pair_weight=weight,
                ))
                used_neg.add(neg_id)

        pairs_per_attribute[attr_k] = pairs
    return pairs_per_attribute


# =====================================================================
# Rendering + summary
# =====================================================================

def _resolve(p: str, root: Path | None) -> Path:
    path = Path(p)
    if root is not None and not path.is_absolute():
        path = root / path
    return path


def _short(text: str, n: int = 10) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(draw, text: str, font, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render_pair(out_path: Path, pair: CounterfactualPair, ri: RegressionInputs,
                reward_name: str, tile: int = 512) -> bool:
    try:
        pos = Image.open(ri.path_per_image[pair.positive_id]).convert("RGB")
        neg = Image.open(ri.path_per_image[pair.negative_id]).convert("RGB")
    except (FileNotFoundError, OSError, KeyError) as e:
        print(f"    [skip render] {e}")
        return False

    pos.thumbnail((tile, tile))
    neg.thumbnail((tile, tile))
    th = max(pos.height, neg.height)
    tw, pad = tile, 8
    canvas_w = tw * 2 + pad * 3

    f_h, f_b = _font(16), _font(15)
    scratch = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    header = (
        _wrap(scratch, f"ATTR[{pair.attribute_idx}]: {pair.attribute}", f_h, canvas_w - 2 * pad)
        + _wrap(scratch, f"PROMPT: {pair.prompt}", f_b, canvas_w - 2 * pad)
        + [f"W_x: mean={pair.w_mean:.4f} CI95=[{pair.w_ci_low:.4f}, {pair.w_ci_high:.4f}]  "
           f"(reliable: CI_low>0)"]
        + [f"{reward_name}: pos={pair.reward_pos:.3f}  neg={pair.reward_neg:.3f}  "
           f"gap={pair.reward_gap:.3f}  |  hamming={pair.hamming_distance}  "
           f"weight={pair.pair_weight:.3f}"]
    )
    line_h, label_h = 20, 22
    header_h = pad + line_h * len(header) + pad
    H = header_h + label_h + th + pad

    canvas = Image.new("RGB", (canvas_w, H), "white")
    draw = ImageDraw.Draw(canvas)
    y = pad
    for i, ln in enumerate(header):
        draw.text((pad, y), ln, fill="black", font=f_h if i == 0 else f_b)
        y += line_h
    draw.text((pad, header_h), f"POS (g_k=1)  R={pair.reward_pos:.3f}",
              fill=(150, 0, 0), font=f_b)
    draw.text((tw + 2 * pad, header_h), f"NEG (g_k=0)  R={pair.reward_neg:.3f}",
              fill=(0, 100, 0), font=f_b)
    iy = header_h + label_h
    canvas.paste(pos, (pad, iy))
    canvas.paste(neg, (tw + 2 * pad, iy))
    draw.rectangle([pad, iy, pad + pos.width, iy + pos.height], outline=(200, 0, 0), width=3)
    draw.rectangle([tw + 2 * pad, iy, tw + 2 * pad + neg.width, iy + neg.height],
                   outline=(0, 150, 0), width=3)
    canvas.save(out_path)
    return True


def summarize_pairs(pairs_per_attribute: dict[int, list[CounterfactualPair]],
                    attrs: list[str]) -> dict:
    summary = {}
    for attr_k, pairs in pairs_per_attribute.items():
        if not pairs:
            summary[attr_k] = {"attribute": attrs[attr_k], "n_pairs": 0}
            continue
        summary[attr_k] = {
            "attribute": attrs[attr_k],
            "n_pairs": len(pairs),
            "n_unique_prompts": len({p.prompt for p in pairs}),
            "mean_hamming": float(np.mean([p.hamming_distance for p in pairs])),
            "mean_reward_gap": float(np.mean([p.reward_gap for p in pairs])),
            "mean_weight": float(np.mean([p.pair_weight for p in pairs])),
            "min_w_ci_low": float(min(p.w_ci_low for p in pairs)),
        }
    return summary


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    args = parse_args()

    ppw_path, topic_id, step = _find_ppw(args.search_dir, args.topic_id, args.step)
    with open(ppw_path) as f:
        ppw = json.load(f)
    attrs = list(ppw["attrs"])
    cfg = _load_config(args.search_dir)

    # resolve params: CLI overrides config defaults
    N = args.N if args.N is not None else int(_cfg_get(cfg, "bon_amplified", "N", default=2))
    reward_name = args.reward or _cfg_get(cfg, "models", "reward_model", "name", default="pickscore")
    manifest = args.manifest or Path(_cfg_get(cfg, "data", "baseline_manifest", default=""))
    baseline_root_str = args.baseline_root if args.baseline_root is not None \
        else _cfg_get(cfg, "data", "baseline_root", default="")
    baseline_root = Path(baseline_root_str) if baseline_root_str else None
    det_path = args.detection_cache or Path(
        _cfg_get(cfg, "bon_amplified", "detection_cache_path",
                 default="outputs/detection_cache/mjhq/black-forest-labs-FLUX.1-dev.json"))
    det_key = args.detector_key or (
        f"{_cfg_get(cfg, 'models', 'detector', 'model', default='Qwen/Qwen3.5-9B')}"
        f"::{_cfg_get(cfg, 'models', 'detector', 'image_detail', default='auto')}")

    print(f"[ppw] {ppw_path.name}  topic={topic_id} step={step} K={len(attrs)} attrs "
          f"P={len(ppw['per_prompt_W'])} prompts")
    print(f"[cfg] N={N}  reward={reward_name}  detector_key={det_key}")
    print(f"[cfg] manifest={manifest}")

    with open(det_path) as f:
        detection = json.load(f).get(det_key, {})
    if not detection:
        raise ValueError(f"detector_key {det_key!r} not in {det_path}")
    with open(manifest) as f:
        baselines = json.load(f).get("baselines", {})

    # ── reconstruct regression inputs + validate ──
    ri = build_regression_inputs(
        ppw=ppw, detection=detection, baselines=baselines,
        reward_name=reward_name, N=N, baseline_root=baseline_root,
    )
    print(f"[recon] {len(ri.G_per_prompt)} prompts reconstructed "
          f"({sum(len(v) for v in ri.image_ids_per_prompt.values())} images)")
    max_diff = validate_reconstruction(ri, ppw)
    flag = "OK" if max_diff < 1e-6 else "!! MISMATCH — check --reward / --N"
    print(f"[recon] max|W_recomputed - W_saved| = {max_diff:.2e}  [{flag}]")

    # ── bootstrap reliability (once per prompt) ──
    rng = np.random.default_rng(args.seed)
    boot_stats = {
        prompt: bootstrap_W_all_attrs(G_c, ri.u_per_prompt[prompt], args.n_boot, rng)
        for prompt, G_c in ri.G_per_prompt.items()
    }

    target_attrs = ([int(x) for x in args.target_attrs.split(",")]
                    if args.target_attrs else list(range(len(attrs))))

    pairs_per_attr = build_counterfactual_pairs(
        ri=ri, boot_stats=boot_stats, target_attributes=target_attrs,
        max_hamming=args.max_hamming,
        max_reward_gap_quantile=args.max_reward_gap_quantile,
        max_pairs_per_prompt=args.max_pairs_per_prompt,
    )

    # ── render + write ──
    out_dir = args.out_dir or Path(f"outputs/cf_pairs/topic{topic_id}_step{step}")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[CounterfactualPair] = []
    for attr_k in target_attrs:
        pairs = pairs_per_attr.get(attr_k, [])
        if not pairs:
            continue
        attr_dir = out_dir / f"attr_{attr_k:03d}_{_short(attrs[attr_k])}"
        attr_dir.mkdir(parents=True, exist_ok=True)
        n_rendered = 0
        prompt_rank: dict[str, int] = {}
        for pair in pairs:
            rank = prompt_rank.get(pair.prompt, 0)
            prompt_rank[pair.prompt] = rank + 1
            png = attr_dir / f"{_short(pair.prompt)}_r{rank}_h{pair.hamming_distance}_gap{pair.reward_gap:.3f}.png"
            if render_pair(png, pair, ri, reward_name):
                pair.png_path = str(png)
                n_rendered += 1
            all_records.append(pair)
        print(f"  attr[{attr_k:>3}] {attrs[attr_k][:52]!r:<54} "
              f"reliable→{len({p.prompt for p in pairs})}p  pairs={len(pairs)}  png={n_rendered}")

    summary = summarize_pairs(pairs_per_attr, attrs)
    with open(out_dir / "pairs.json", "w") as f:
        json.dump({
            "topic_id": topic_id, "step": step, "reward": reward_name, "N": N,
            "detector_key": det_key, "n_boot": args.n_boot,
            "max_hamming": args.max_hamming,
            "max_reward_gap_quantile": args.max_reward_gap_quantile,
            "max_pairs_per_prompt": args.max_pairs_per_prompt,
            "recon_max_diff": max_diff,
            "n_pairs": len(all_records),
            "summary": {str(k): v for k, v in summary.items()},
            "pairs": [asdict(r) for r in all_records],
        }, f, indent=2, ensure_ascii=False)
    if all_records:
        with open(out_dir / "pairs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(all_records[0]).keys()))
            w.writeheader()
            for r in all_records:
                w.writerow(asdict(r))

    n_reliable_pa = sum(
        len({p.prompt for p in pairs_per_attr.get(k, [])}) for k in target_attrs
    )
    print(f"\n[done] target attrs={len(target_attrs)}  "
          f"reliable (prompt,attr) with pairs={n_reliable_pa}  "
          f"total pairs={len(all_records)}\n[out] {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # inputs (most default from the run's config_effective.yaml)
    p.add_argument("--search_dir", required=True, type=Path,
                   help="Dir with per_prompt_W_step{N}_topic{T}.json + config_effective.yaml")
    p.add_argument("--topic_id", type=int, default=None, help="Required only if multi-topic.")
    p.add_argument("--step", type=int, default=None, help="Which step. Default: latest.")
    p.add_argument("--reward", default=None, help="Reward key. Default: config reward_model.name")
    p.add_argument("--N", type=int, default=None, help="BoN N for U^(N-1). Default: config.")
    p.add_argument("--manifest", type=Path, default=None, help="Default: config data.baseline_manifest")
    p.add_argument("--baseline_root", default=None, help="Default: config data.baseline_root")
    p.add_argument("--detection_cache", type=Path, default=None,
                   help="Default: config bon_amplified.detection_cache_path")
    p.add_argument("--detector_key", default=None,
                   help="Default: '<detector.model>::<detector.image_detail>' from config")
    # reliability
    p.add_argument("--n_boot", type=int, default=500, help="Bootstrap resamples per prompt.")
    p.add_argument("--seed", type=int, default=0)
    # pair construction
    p.add_argument("--target_attrs", default=None,
                   help="Comma-sep attr indices. Default: all K.")
    p.add_argument("--max_hamming", type=int, default=2,
                   help="Max Hamming distance over OTHER attrs (0 = strict counterfactual).")
    p.add_argument("--max_reward_gap_quantile", type=float, default=0.75,
                   help="Gap cap = this quantile of within-prompt positive gaps.")
    p.add_argument("--max_pairs_per_prompt", type=int, default=5)
    # output
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Default: outputs/cf_pairs/topic{T}_step{N}")
    return p.parse_args()


if __name__ == "__main__":
    main()
