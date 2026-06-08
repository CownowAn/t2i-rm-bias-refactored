"""Post-hoc leave-one-out (LOO) cleanup of a finished BoN-amplified search run.

Implements POSTHOC_CLEANUP.md. The admission criterion guarantees that each g_k
contributes uniquely *at the moment it is admitted*, but an attribute admitted
later may render an earlier one retrospectively redundant. We therefore re-test
every member of the final pool S_T against *all the others*:

    A_LOO(g_k) := A_partial(g_k | S_T \\ {g_k})
                = (N / |X|) * Σ_x Cov_x(g_k, e^{-k}_x),

where e^{-k}_x is the per-prompt residual of U^{N-1} regressed on S_T \\ {g_k}.
We prune sequentially: remove the member with the smallest A_LOO whenever it is
<= tau_p, recompute A_LOO for the remainder, and repeat until every member
exceeds tau_p (the surviving pool is then Pareto-optimal — no member is
dominated by the rest).

This reuses the exact estimators the search itself used:
  - search.pipeline.bon_amplified_evo._compute_bon_residuals   (per-prompt OLS)
  - search.pipeline.bon_amplified_evo._compute_partial_a_hat   (N·E_x[Cov_x(g,e)])

so the cleanup is consistent with how attributes were admitted in the first place.

Usage
-----
    python -m search.posthoc_cleanup.cleanup outputs/search/20260601-154143
    python -m search.posthoc_cleanup.cleanup <run_dir> --step 12 --topic 0
    python -m search.posthoc_cleanup.cleanup <run_dir> --tau-p 0.0

`--step K` cleans the pool recorded at ba_expand step K (default: the final step).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from random import Random

import numpy as np
from loguru import logger

from search.config import SearchConfig
from search.pipeline.baselines import load_baselines_from_manifest, load_topic_states
from search.pipeline.bon_amplified_evo import (
    _compute_bon_residuals,
    _compute_partial_a_hat,
)


# ── Run reconstruction ────────────────────────────────────────────────────────


def load_config(run_dir: Path) -> SearchConfig:
    """Load the effective config that the run was executed with."""
    cfg_path = run_dir / "configs" / "config_effective.yaml"
    if not cfg_path.exists():
        legacy = run_dir / "config_effective.yaml"
        if legacy.exists():
            cfg_path = legacy
        else:
            raise FileNotFoundError(
                f"No config_effective.yaml in {run_dir}/configs (or legacy {run_dir})"
            )
    return SearchConfig.from_yaml(cfg_path)


def load_pool(
    run_dir: Path, topic_id: int, step: int | None = None
) -> tuple[list[str], int, dict]:
    """Return (pool S_step, step_idx, ba_expand_dict) for a topic.

    The pool is the acc_pool recorded in ba_expand_step{step}_topic{topic_id}.json.
    If `step` is None, the highest-numbered (final) step is used.
    """
    if step is None:
        files = list(run_dir.glob(f"ba_expand_step*_topic{topic_id}.json"))
        if not files:
            raise FileNotFoundError(
                f"No ba_expand_step*_topic{topic_id}.json files in {run_dir}"
            )
        files.sort(key=lambda p: int(re.search(r"step(\d+)", p.name).group(1)))
        path = files[-1]
    else:
        path = run_dir / f"ba_expand_step{step}_topic{topic_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"No {path.name} in {run_dir}")
    data = json.loads(path.read_text())
    return list(data["acc_pool"]), int(data["step_idx"]), data


def reconstruct_run_data(
    config: SearchConfig, topic_id: int
) -> tuple[dict[str, list], dict[str, dict[str, int]], str]:
    """Rebuild the exact fixed-baseline set + detection cache the run used.

    Mirrors BonAmplifiedEvolutionEngine SETUP: same RNG seed/offset, same
    prompt/image sampling, same detection-cache model key. Reward scores come
    straight from the manifest (no reward model is loaded).

    Returns (fixed_baselines, detection_cache, reward_model_name).
    """
    rm_name = config.models.reward_model.name

    states = load_topic_states(
        prompts_dir=config.data.prompts_dir,
        topic_ids=[topic_id],
        val_split_size=config.data.val_split_size,
        random_seed=config.run.random_seed,
    )
    ts = states[0]
    load_baselines_from_manifest(ts, config.data.baseline_manifest, config.data.baseline_root)

    # Same sampling as SETUP (bon_amplified_evo.run): seed + topic_id + 77777.
    rng = Random(config.run.random_seed + topic_id + 77777)
    train_prompts = [
        p for p in ts.train_prompts() if p in ts.baselines and ts.baselines[p]
    ]
    n_prompts = min(config.evaluation.amp_n_prompts, len(train_prompts))
    sample_prompts = rng.sample(train_prompts, n_prompts)
    n_per = config.evaluation.amp_n_images_per_prompt
    fixed_baselines = {
        p: rng.sample(ts.baselines[p], min(n_per, len(ts.baselines[p])))
        for p in sample_prompts
    }

    # Detection cache: same model key + same fixed-image filtering as SETUP.
    detection_cache: dict[str, dict[str, int]] = {}
    cache_path = config.bon_amplified.detection_cache_path
    if cache_path:
        fixed_ids = {
            img.image_id for imgs in fixed_baselines.values() for img in imgs
        }
        model_key = (
            f"{config.models.detector.model}::{config.models.detector.image_detail}"
        )
        path = Path(cache_path)
        if path.exists():
            all_saved = json.loads(path.read_text())
            for image_id, attr_vals in all_saved.get(model_key, {}).items():
                if image_id in fixed_ids:
                    detection_cache.setdefault(image_id, {}).update(attr_vals)
        else:
            logger.warning(f"Detection cache not found at {path}")

    n_imgs = sum(len(v) for v in fixed_baselines.values())
    n_missing_score = sum(
        1
        for imgs in fixed_baselines.values()
        for img in imgs
        if rm_name not in img.reward_scores
    )
    logger.info(
        f"Topic {topic_id}: reconstructed {len(fixed_baselines)} prompts × "
        f"≤{n_per} imgs = {n_imgs} images ({len(detection_cache)} in detection cache)"
    )
    if n_missing_score:
        logger.warning(
            f"Topic {topic_id}: {n_missing_score} images lack reward_scores"
            f"['{rm_name}'] in the manifest — they will be dropped by the OLS."
        )
    return fixed_baselines, detection_cache, rm_name


# ── LOO core ──────────────────────────────────────────────────────────────────


def _centered_u_residuals(
    fixed_baselines: dict[str, list], rm_name: str, N: int
) -> dict[tuple[str, str], float]:
    """Residual of U^{N-1} when regressed on the empty set: just centered U^{N-1}.

    Used for the LOO of a singleton pool, where the complement S_T\\{g_k} is empty
    and A_LOO(g_k) reduces to the marginal amplification N·E_x[Cov_x(g_k, U^{N-1})].
    Mirrors the U-quantile computation in _compute_bon_residuals.
    """
    out: dict[tuple[str, str], float] = {}
    for prompt_text, images in fixed_baselines.items():
        scored = [img for img in images if rm_name in img.reward_scores]
        if len(scored) < 2:
            continue
        rewards = np.array([img.reward_scores[rm_name] for img in scored], dtype=float)
        sorted_r = np.sort(rewards)
        u = np.searchsorted(sorted_r, rewards, side="right") / len(scored)
        u_pow = u ** (N - 1)
        u_c = u_pow - u_pow.mean()
        for img, val in zip(scored, u_c):
            out[(prompt_text, img.image_id)] = float(val)
    return out


def _residuals_for(
    pool: list[str],
    detection_cache: dict[str, dict[str, int]],
    fixed_baselines: dict[str, list],
    rm_name: str,
    N: int,
    ols_mode: str,
) -> dict[tuple[str, str], float]:
    """Per-prompt residuals of U^{N-1} regressed on `pool` (handles empty pool)."""
    if not pool:
        return _centered_u_residuals(fixed_baselines, rm_name, N)
    residuals, *_ = _compute_bon_residuals(
        detection_cache, fixed_baselines, list(pool), rm_name, N, mode=ols_mode
    )
    return residuals


def loo_scores(
    pool: list[str],
    detection_cache: dict[str, dict[str, int]],
    fixed_baselines: dict[str, list],
    rm_name: str,
    N: int,
    ols_mode: str,
) -> dict[str, float]:
    """A_LOO(g_k) = A_partial(g_k | pool\\{g_k}) for every g_k in `pool`."""
    scores: dict[str, float] = {}
    for g_k in pool:
        complement = [a for a in pool if a != g_k]
        residuals = _residuals_for(
            complement, detection_cache, fixed_baselines, rm_name, N, ols_mode
        )
        scores[g_k] = _compute_partial_a_hat(
            [g_k], detection_cache, residuals, fixed_baselines, N
        )[g_k]
    return scores


def sequential_prune(
    pool: list[str],
    detection_cache: dict[str, dict[str, int]],
    fixed_baselines: dict[str, list],
    rm_name: str,
    N: int,
    ols_mode: str,
    tau_p: float,
) -> tuple[list[str], list[dict], dict[str, float]]:
    """Greedy LOO pruning until every survivor has A_LOO > tau_p.

    Returns (kept_pool, removed[list of dicts in removal order], final_loo_scores).
    Stops at a singleton pool (a lone member cannot be "dominated by the others").
    """
    kept = list(pool)
    removed: list[dict] = []
    rank = 0
    while len(kept) > 1:
        scores = loo_scores(kept, detection_cache, fixed_baselines, rm_name, N, ols_mode)
        g_min = min(scores, key=lambda g: scores[g])
        if scores[g_min] > tau_p:
            break
        rank += 1
        removed.append(
            {
                "removal_rank": rank,
                "attribute": g_min,
                "loo_score": scores[g_min],
                "pool_size_before": len(kept),
            }
        )
        logger.info(
            f"  [-] remove (A_LOO={scores[g_min]:+.6f} ≤ {tau_p}) "
            f"pool {len(kept)}→{len(kept) - 1}: {g_min[:70]}"
        )
        kept.remove(g_min)

    final_scores = loo_scores(
        kept, detection_cache, fixed_baselines, rm_name, N, ols_mode
    )
    return kept, removed, final_scores


# ── Orchestration ─────────────────────────────────────────────────────────────


def cleanup_topic(
    run_dir: Path,
    config: SearchConfig,
    topic_id: int,
    tau_p: float,
    ols_mode: str,
    step: int | None = None,
) -> dict:
    """Run the full LOO cleanup for one topic/step and return a result dict.

    `step` selects which ba_expand step's acc_pool to clean (None → final step).
    """
    N = config.bon_amplified.N
    pool, step_idx, expand = load_pool(run_dir, topic_id, step)
    logger.info(
        f"Topic {topic_id}: pool S_{step_idx} has {len(pool)} attrs "
        f"(from step {step_idx}); tau_p={tau_p}, ols_mode={ols_mode}, N={N}"
    )

    fixed_baselines, detection_cache, rm_name = reconstruct_run_data(config, topic_id)

    # Sanity check against what the run recorded.
    n_imgs = sum(len(v) for v in fixed_baselines.values())
    recorded = expand.get("n_residual_images")
    if recorded is not None and recorded != n_imgs:
        logger.warning(
            f"Topic {topic_id}: reconstructed {n_imgs} images but the run recorded "
            f"{recorded} — fixed-baseline sampling may not match exactly."
        )

    initial_scores = loo_scores(
        pool, detection_cache, fixed_baselines, rm_name, N, ols_mode
    )
    kept, removed, final_scores = sequential_prune(
        pool, detection_cache, fixed_baselines, rm_name, N, ols_mode, tau_p
    )

    return {
        "run_dir": str(run_dir),
        "topic_id": topic_id,
        "reward_model": rm_name,
        "N": N,
        "ols_mode": ols_mode,
        "tau_p": tau_p,
        "step": step_idx,
        "n_prompts": len(fixed_baselines),
        "n_images": n_imgs,
        "n_initial": len(pool),
        "n_kept": len(kept),
        "n_removed": len(removed),
        "initial_pool": pool,
        "initial_loo_scores": initial_scores,
        "removed": removed,
        "kept_pool": kept,
        "kept_loo_scores": final_scores,
    }


def _print_summary(result: dict) -> None:
    t = result["topic_id"]
    print(f"\n{'=' * 78}")
    print(
        f"LOO cleanup — topic {t} step {result['step']} | "
        f"{result['n_initial']} → {result['n_kept']} attrs "
        f"({result['n_removed']} removed) | tau_p={result['tau_p']} "
        f"| ols={result['ols_mode']} N={result['N']}"
    )
    print(f"{'=' * 78}")
    if result["removed"]:
        print("Removed (in order):")
        for r in result["removed"]:
            print(f"  {r['removal_rank']:>2}. A_LOO={r['loo_score']:+.6f}  {r['attribute'][:66]}")
    else:
        print("Removed: none — every member already exceeds tau_p.")
    print("\nKept (final Pareto pool, by A_LOO):")
    for attr, score in sorted(
        result["kept_loo_scores"].items(), key=lambda kv: kv[1], reverse=True
    ):
        print(f"  A_LOO={score:+.6f}  {attr[:66]}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc leave-one-out cleanup of a BoN-amplified search run."
    )
    parser.add_argument("run_dir", type=Path, help="Search run output directory")
    parser.add_argument(
        "--topic", type=int, default=None,
        help="Topic id to clean (default: all topic_ids in the run config)",
    )
    parser.add_argument(
        "--step", type=int, default=None,
        help="ba_expand step whose acc_pool to clean (default: final step)",
    )
    parser.add_argument(
        "--tau-p", type=float, default=None,
        help="LOO pruning threshold (default: bon_amplified.tau_partial from config)",
    )
    parser.add_argument(
        "--ols-mode", choices=["per_prompt", "global"], default=None,
        help="OLS residual mode (default: match the run's use_per_prompt_ols)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output JSON path (default: <run_dir>/posthoc_cleanup_topic{T}.json)",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    config = load_config(run_dir)

    tau_p = args.tau_p if args.tau_p is not None else config.bon_amplified.tau_partial
    ols_mode = args.ols_mode or (
        "per_prompt" if config.bon_amplified.use_per_prompt_ols else "global"
    )
    topics = [args.topic] if args.topic is not None else list(config.data.topic_ids)

    for topic_id in topics:
        result = cleanup_topic(run_dir, config, topic_id, tau_p, ols_mode, args.step)
        out_path = args.out or (
            run_dir / f"posthoc_cleanup_topic{topic_id}_step{result['step']}.json"
        )
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        _print_summary(result)
        logger.info(f"Topic {topic_id}: wrote cleanup result → {out_path}")


if __name__ == "__main__":
    main()
