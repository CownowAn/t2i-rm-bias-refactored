"""Compute per-prompt R² from the detection cache + reward manifest.

Usage:
    python analysis/per_prompt_r2.py \
        --manifest /nfs/data/sohyun/projects/t2i-rm-bias/data/baselines/mjhq/topic_0/black-forest-labs-FLUX.1-dev/manifest.json \
        --cache outputs/detection_cache/mjhq/black-forest-labs-FLUX.1-dev.json \
        --model_key "Qwen/Qwen3.5-9B::auto" \
        --reward_name imagereward \
        --attrs "Eyes show exaggerated brightness" "High global contrast/HDR-like tonemapping" \
        --N 2

For each given attribute pool, this:
  1. Loads detection results and reward scores
  2. Computes within-prompt centered G_x, u_x
  3. Fits OLS per prompt → R²_x
  4. Reports distribution of R²_x and global var_explained
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="Path to manifest.json (with reward_scores)")
    p.add_argument("--cache",    required=True, help="Path to detection cache JSON")
    p.add_argument("--model_key", default="Qwen/Qwen3.5-9B::auto",
                   help="Detector model key inside the detection cache")
    p.add_argument("--reward_name", default="imagereward")
    p.add_argument("--N", type=int, default=2, help="BoN N (uses U^{N-1})")
    p.add_argument("--attrs", nargs="+", default=None,
                   help="Specific attrs to include in OLS (default: 8 attrs from step 1 S_t)")
    p.add_argument("--n_top_print", type=int, default=10,
                   help="Print top-N prompts by R²")
    return p.parse_args()


# Default attrs = the 8 surviving attrs in step 1 S_t (from log)
DEFAULT_ATTRS = [
    "Eyes show exaggerated brightness (strong catchlights or unnatural glow).",
    "High global contrast/HDR-like tonemapping with very bright highlights and deep shadows.",
    "Fur/skin reads as synthetic ‘plush/velvet’ material with overly uniform fiber direction/clumping, rather than irregular natural hair/skin variation.",
    "Internal pattern boundaries on the subject (e.g., stripes/patch edges) have unnaturally crisp, vector-like edges with no natural feathering/texture transition.",
    "Over-saturated, neon/iridescent-looking coloration on the subject.",
    "The animal/character is depicted with its mouth/beak open showing a saturated red mouth interior and/or tongue in a posed “smile” rather than a neutral closed mouth.",
    "Aggressive micro-detail and sharpening visible in textures (fur/skin/foliage) with little or no film grain.",
    "Heavy vignette where the corners/edges are noticeably darker than the center.",
]


def load_data(manifest_path: str, cache_path: str, model_key: str, reward_name: str):
    """Build per-prompt arrays: prompt → {image_ids, rewards}, detection[img_id] → dict[attr → 0/1]."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(cache_path) as f:
        cache_all = json.load(f)
    detection = cache_all[model_key]

    by_prompt: dict[str, dict] = {}
    for prompt, entries in manifest["baselines"].items():
        scored = [e for e in entries if reward_name in e.get("reward_scores", {})
                  and e["image_id"] in detection]
        if len(scored) < 2:
            continue
        by_prompt[prompt] = {
            "image_ids": [e["image_id"] for e in scored],
            "rewards":   np.array([e["reward_scores"][reward_name] for e in scored]),
        }
    return by_prompt, detection


def build_G(image_ids: list[str], attrs: list[str], detection: dict) -> np.ndarray | None:
    """Returns G ∈ {0,1}^(M, K) or None if any attr missing."""
    G = np.zeros((len(image_ids), len(attrs)), dtype=np.float64)
    for i, img_id in enumerate(image_ids):
        d = detection.get(img_id, {})
        for k, attr in enumerate(attrs):
            if attr not in d:
                return None
            G[i, k] = float(d[attr])
    return G


def compute_u(rewards: np.ndarray, N: int) -> np.ndarray:
    """Compute U^{N-1} where U_i = rank_i / M."""
    M = len(rewards)
    ranks = np.argsort(np.argsort(rewards)) + 1     # 1..M
    U = ranks / M
    return U ** (N - 1)


def per_prompt_r2(G: np.ndarray, u: np.ndarray) -> tuple[float, float, float]:
    """Within-prompt OLS R². Returns (R²_x, corr_max, n_active_attrs)."""
    G_c = G - G.mean(axis=0, keepdims=True)
    u_c = u - u.mean()
    var_u = float(np.var(u_c))
    if var_u < 1e-12:
        return 0.0, 0.0, 0
    # Drop zero-variance columns (attr always 0 or always 1)
    col_var = G_c.var(axis=0)
    active = col_var > 1e-12
    if not active.any():
        return 0.0, 0.0, 0
    G_active = G_c[:, active]
    W, *_ = np.linalg.lstsq(G_active, u_c, rcond=None)
    residuals = u_c - G_active @ W
    r2 = 1.0 - float(np.var(residuals)) / var_u
    # Max single-attr correlation
    corr = np.array([
        np.corrcoef(G_active[:, k], u_c)[0, 1] if G_active[:, k].std() > 1e-12 else 0.0
        for k in range(G_active.shape[1])
    ])
    return r2, float(np.max(np.abs(corr))), int(active.sum())


def compute_global_var_explained(
    by_prompt: dict, attrs: list[str], detection: dict, N: int
) -> float:
    """Global var_explained: stack all (centered G, centered u) and fit single W."""
    G_all, u_all = [], []
    for info in by_prompt.values():
        G = build_G(info["image_ids"], attrs, detection)
        if G is None:
            continue
        u = compute_u(info["rewards"], N)
        G_c = G - G.mean(axis=0, keepdims=True)
        u_c = u - u.mean()
        G_all.append(G_c)
        u_all.append(u_c)
    if not G_all:
        return 0.0
    G_all = np.vstack(G_all)
    u_all = np.concatenate(u_all)
    W, *_ = np.linalg.lstsq(G_all, u_all, rcond=None)
    residuals = u_all - G_all @ W
    return 1.0 - float(np.var(residuals)) / float(np.var(u_all))


def main() -> None:
    args = parse_args()
    attrs = args.attrs or DEFAULT_ATTRS

    print(f"Loading data ...")
    by_prompt, detection = load_data(args.manifest, args.cache, args.model_key, args.reward_name)
    print(f"  → {len(by_prompt)} prompts loaded")
    print(f"  → {len(detection)} images in detection cache")
    print(f"  → using K={len(attrs)} attrs, N={args.N}")
    print()

    # Per-prompt R²
    results = []
    for prompt, info in by_prompt.items():
        G = build_G(info["image_ids"], attrs, detection)
        if G is None:
            print(f"  SKIP (missing attr detection): '{prompt[:60]}'")
            continue
        u = compute_u(info["rewards"], args.N)
        r2, corr_max, n_active = per_prompt_r2(G, u)
        results.append({
            "prompt": prompt,
            "r2": r2,
            "corr_max": corr_max,
            "n_active": n_active,
            "n_imgs": len(info["image_ids"]),
        })

    results.sort(key=lambda r: r["r2"], reverse=True)

    print(f"=== Per-prompt R² (sorted, top {args.n_top_print}) ===")
    print(f"{'R²':>8}  {'corr_max':>8}  {'n_act':>5}  prompt")
    for r in results[:args.n_top_print]:
        print(f"{r['r2']:>8.4f}  {r['corr_max']:>8.4f}  {r['n_active']:>5d}  '{r['prompt'][:80]}'")
    print(f"  ... ({len(results) - args.n_top_print} more)")
    print()

    print(f"=== Per-prompt R² (bottom {args.n_top_print}) ===")
    for r in results[-args.n_top_print:]:
        print(f"{r['r2']:>8.4f}  {r['corr_max']:>8.4f}  {r['n_active']:>5d}  '{r['prompt'][:80]}'")
    print()

    r2_vals = np.array([r["r2"] for r in results])
    print(f"=== Distribution (n={len(r2_vals)} prompts) ===")
    print(f"  mean    R²  = {r2_vals.mean():.4f}")
    print(f"  median  R²  = {np.median(r2_vals):.4f}")
    print(f"  max     R²  = {r2_vals.max():.4f}")
    print(f"  R² > 0.05   = {(r2_vals > 0.05).sum()} / {len(r2_vals)}")
    print(f"  R² > 0.10   = {(r2_vals > 0.10).sum()} / {len(r2_vals)}")
    print(f"  R² > 0.20   = {(r2_vals > 0.20).sum()} / {len(r2_vals)}")
    print()

    # Global var_explained for comparison
    global_ve = compute_global_var_explained(by_prompt, attrs, detection, args.N)
    print(f"Global var_explained (single W across all prompts): {global_ve:.4f}")
    print(f"Average per-prompt R² (separate W per prompt):      {r2_vals.mean():.4f}")
    print(f"  Ratio (per-prompt / global): {r2_vals.mean() / max(global_ve, 1e-9):.1f}x")


if __name__ == "__main__":
    main()
