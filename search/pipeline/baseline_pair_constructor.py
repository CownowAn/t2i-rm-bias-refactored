"""Constructs counterfactual pairs from baseline images.

Two modes:
  BaselinePairConstructor          — gap/hamming scoring; high/low assigned by RM score
                                     → D[:,k] is random w.r.t. attribute presence
  AttributeStratifiedPairConstructor — stratified by attribute presence; high=G1 (attr present),
                                     low=G0 (attr absent) for each (attr_k, prompt) stratum
                                     → D[:,k] = 1 by construction; cov(D[:,k], delta_rm) > 0
                                        for attributes the RM is biased toward
                                     → global quota per attribute + density-bonus scoring
                                        to prevent sparse D columns as K grows
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
import math

from search.data.baseline_pair_types import BaselinePair

if TYPE_CHECKING:
    from search.data.types import BaselineImage


def _hamming(vec_a: dict[str, int], vec_b: dict[str, int], attrs: list[str]) -> int:
    return sum(vec_a.get(a, 0) != vec_b.get(a, 0) for a in attrs)


class BaselinePairConstructor:
    """Constructs baseline-image counterfactual pairs within each prompt.

    Pairs are selected by score = reward_gap / hamming_distance, which naturally
    prefers near-counterfactual pairs (small hamming) with strong signal (large gap).
    This adapts automatically as the attribute pool K grows — no hard hamming threshold
    needed.
    """

    def __init__(self, n_pairs_per_prompt: int = 20):
        self.n_pairs_per_prompt = n_pairs_per_prompt

    def construct(
        self,
        baselines_by_prompt: dict[str, list["BaselineImage"]],
        detection: dict[str, dict[str, int]],
        attr_pool: list[str],
        reward_model_name: str,
    ) -> list[BaselinePair]:
        """
        For each prompt, enumerate all image pairs with hamming > 0, score each by
        gap / hamming, and keep the top n_pairs_per_prompt by score.
        """
        all_pairs: list[BaselinePair] = []

        for images in baselines_by_prompt.values():
            scored = [
                img for img in images
                if reward_model_name in img.reward_scores
                and img.image_id in detection
            ]
            if len(scored) < 2:
                continue

            prompt_pairs: list[tuple[float, BaselinePair]] = []  # (score, pair)
            for i in range(len(scored)):
                for j in range(i + 1, len(scored)):
                    a, b = scored[i], scored[j]
                    h = _hamming(detection[a.image_id], detection[b.image_id], attr_pool)
                    if h == 0:
                        continue  # identical attribute vectors — not a useful counterfactual
                    r_a = a.reward_scores[reward_model_name]
                    r_b = b.reward_scores[reward_model_name]
                    gap = abs(r_a - r_b)
                    score = gap / h  # reward gap per unit Hamming distance
                    high, low = (a, b) if r_a >= r_b else (b, a)
                    prompt_pairs.append((score, BaselinePair(
                        high_reward=high, low_reward=low, delta_rm=gap
                    )))

            # Take top n by score (small hamming + large gap float to top naturally)
            prompt_pairs.sort(key=lambda x: x[0], reverse=True)
            all_pairs.extend(p for _, p in prompt_pairs[: self.n_pairs_per_prompt])

        logger.info(
            f"BaselinePairConstructor: {len(all_pairs)} pairs from "
            f"{len(baselines_by_prompt)} prompts  "
            f"(K={len(attr_pool)}, score=gap/hamming, top-{self.n_pairs_per_prompt}/prompt)"
        )
        return all_pairs


class AttributeStratifiedPairConstructor:
    """Constructs pairs explicitly stratified by attribute presence.

    For each attribute k, globally collects all valid (G1, G0) pairs across all prompts
    where RM(G1) > RM(G0), then selects the top-n_quota using a density-aware score:

      score = (gap, extra_positive_cols)

    where extra_positive_cols counts how many OTHER attributes also have D[:,j]=+1 for
    this pair (detection(G1,j)=1 and detection(G0,j)=0). This makes D-matrix rows denser
    across multiple columns simultaneously, preventing any single column from being sparse
    as K grows.

    A per-prompt cap limits over-representation of any single prompt.
    Global quota n_quota = n_pairs_per_stratum × P // K keeps total N ≈ constant as K grows,
    ensuring Lasso always has a consistent N/K ratio.
    """

    def __init__(self, n_pairs_per_stratum: int = 4):
        self.n_pairs_per_stratum = n_pairs_per_stratum

    def construct(
        self,
        baselines_by_prompt: dict[str, list["BaselineImage"]],
        detection: dict[str, dict[str, int]],
        attr_pool: list[str],
        reward_model_name: str,
    ) -> list["BaselinePair"]:
        """
        For each attr_k: globally collect valid (G1, G0) pairs, score by
        (gap, extra_positive_cols), select top n_quota with per-prompt cap.
        D[i, k] = 1 by construction; denser D rows reduce column sparsity.
        """
        all_pairs: list[BaselinePair] = []
        K = len(attr_pool)
        P = len(baselines_by_prompt)

        # Global quota per attribute: keeps total N ≈ n_pairs_per_stratum × P
        n_quota = max(
            self.n_pairs_per_stratum * P // max(K, 1),
            self.n_pairs_per_stratum,
        )
        per_prompt_cap = self.n_pairs_per_stratum  # max pairs from one prompt per attr

        logger.info(
            f"AttributeStratifiedPairConstructor: K={K} P={P} "
            f"n_quota={n_quota}/attr per_attr_cap={per_prompt_cap}/prompt"
        )

        # Per-attribute stats collected during selection, logged at the end
        attr_stats: list[dict] = []

        for attr_k in attr_pool:
            k_idx = attr_pool.index(attr_k)
            # (gap, extra_cols, prompt_text, pair)
            attr_candidates: list[tuple[float, int, str, BaselinePair]] = []
            n_valid_strata = 0  # prompts with at least one valid (G1,G0) pair

            for prompt_text, images in baselines_by_prompt.items():
                scored = [
                    img for img in images
                    if reward_model_name in img.reward_scores
                    and img.image_id in detection
                    and attr_k in detection[img.image_id]
                ]
                if len(scored) < 2:
                    continue

                G1 = [img for img in scored if detection[img.image_id][attr_k] == 1]
                G0 = [img for img in scored if detection[img.image_id][attr_k] == 0]
                if not G1 or not G0:
                    continue

                # Vectorised gap and extra_cols computation
                G1_rm = np.array([img.reward_scores[reward_model_name] for img in G1])
                G0_rm = np.array([img.reward_scores[reward_model_name] for img in G0])
                gaps = G1_rm[:, np.newaxis] - G0_rm[np.newaxis, :]  # (|G1|, |G0|)
                valid = gaps > 0
                if not valid.any():
                    continue

                n_valid_strata += 1

                # Detection vectors for extra_cols: D[:,j]=+1 count for j ≠ k_idx
                G1_vecs = np.array(
                    [[detection.get(img.image_id, {}).get(a, 0) for a in attr_pool]
                     for img in G1], dtype=np.int8
                )  # (|G1|, K)
                G0_vecs = np.array(
                    [[detection.get(img.image_id, {}).get(a, 0) for a in attr_pool]
                     for img in G0], dtype=np.int8
                )  # (|G0|, K)
                other = np.arange(K) != k_idx
                # extra_cols[i,j] = Σ_{j'≠k} [(G1[i,j']>G0[j,j'])]
                D_other = (G1_vecs[:, np.newaxis, :] - G0_vecs[np.newaxis, :, :])[:, :, other]
                extra_mat = (D_other > 0).sum(axis=2)  # (|G1|, |G0|)

                g1_idxs, g0_idxs = np.where(valid)
                for i, j in zip(g1_idxs.tolist(), g0_idxs.tolist()):
                    attr_candidates.append((
                        float(gaps[i, j]),
                        int(extra_mat[i, j]),
                        prompt_text,
                        BaselinePair(
                            high_reward=G1[i],   # has attr_k → D[pair,k] = 1
                            low_reward=G0[j],    # no attr_k
                            delta_rm=float(gaps[i, j]),
                        ),
                    ))

            if not attr_candidates:
                attr_stats.append({
                    "attr": attr_k, "n_candidates": 0, "n_valid_strata": n_valid_strata,
                    "selected": 0, "n_prompts_used": 0,
                    "avg_gap": 0.0, "avg_extra": 0.0,
                })
                continue

            # Sort: gap desc primary, extra_cols desc secondary
            attr_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

            # Adapt per-prompt cap based on the number of valid strata.
            # If only a few prompts contain valid pairs for this attribute,
            # allow each valid prompt to contribute more pairs so we can fill n_quota.
            adaptive_cap = max(
                per_prompt_cap,
                math.ceil(n_quota / max(n_valid_strata, 1)),
            )

            # Select top-n_quota with per-prompt cap; track stats
            prompt_count: dict[str, int] = {}
            selected_gaps: list[float] = []
            selected_extras: list[int] = []
            prompts_used: set[str] = set()

            for gap, extra_cols, prompt_text, pair in attr_candidates:
                if len(selected_gaps) >= n_quota:
                    break
                if prompt_count.get(prompt_text, 0) >= adaptive_cap:
                    continue
                all_pairs.append(pair)
                prompt_count[prompt_text] = prompt_count.get(prompt_text, 0) + 1
                selected_gaps.append(gap)
                selected_extras.append(extra_cols)
                prompts_used.add(prompt_text)

            attr_stats.append({
                "attr": attr_k,
                "n_candidates": len(attr_candidates),
                "n_valid_strata": n_valid_strata,
                "selected": len(selected_gaps),
                "n_prompts_used": len(prompts_used),
                "adaptive_cap": adaptive_cap,
                "avg_gap": float(np.mean(selected_gaps)) if selected_gaps else 0.0,
                "avg_extra": float(np.mean(selected_extras)) if selected_extras else 0.0,
            })

        # ── Final D-column density (computed once over the full pair set) ──────
        N = len(all_pairs)
        col_densities: list[float] = []
        for attr_k in attr_pool:
            pos = sum(
                1 for p in all_pairs
                if detection.get(p.high_reward.image_id, {}).get(attr_k, 0)
                 - detection.get(p.low_reward.image_id, {}).get(attr_k, 0) > 0
            )
            col_densities.append(pos / N if N > 0 else 0.0)

        # ── Per-attribute log lines ───────────────────────────────────────────
        for stats, density in zip(attr_stats, col_densities):
            status = "✓" if stats["selected"] > 0 else "✗"
            logger.info(
                f"  {status} [{stats['attr'][:45]:45s}] "
                f"sel={stats['selected']:3d}/{n_quota:3d}  "
                f"cands={stats['n_candidates']:5d}  "
                f"strata={stats['n_valid_strata']:2d}/{P:2d}  "
                f"cap={stats.get('adaptive_cap', per_prompt_cap):2d}  "
                f"prompts_used={stats['n_prompts_used']:2d}  "
                f"avg_gap={stats['avg_gap']:.3f}  "
                f"avg_extra={stats['avg_extra']:.1f}  "
                f"D_col={density:.0%}"
            )

        # ── Summary ───────────────────────────────────────────────────────────
        n_attrs_contributed = sum(1 for s in attr_stats if s["selected"] > 0)
        if col_densities:
            logger.info(
                f"  → D column density: "
                f"min={min(col_densities):.0%}  "
                f"mean={float(np.mean(col_densities)):.0%}  "
                f"max={max(col_densities):.0%}"
            )
        logger.info(
            f"  → {N} pairs total from {n_attrs_contributed}/{K} attrs"
        )
        return all_pairs


class AllPairConstructor:
    """Uses ALL valid (high, low) image pairs within each prompt where at least one
    attr in acc_pool differs between the two images (D[i,:] ≠ 0).

    No quota or stratum-based selection — every informative pair is included.
    Pairs with identical detection vectors (D[i,:] = 0) are excluded since they
    contribute no information to the regression.

    Deduplication is by (high_image_id, low_image_id); the same image can appear
    in many pairs but each directed pair is counted once.
    """

    def construct(
        self,
        baselines_by_prompt: dict[str, list["BaselineImage"]],
        detection: dict[str, dict[str, int]],
        attr_pool: list[str],
        reward_model_name: str,
    ) -> list["BaselinePair"]:
        all_pairs: list[BaselinePair] = []
        seen: set[tuple[str, str]] = set()
        K = len(attr_pool)
        n_skipped_no_diff = 0

        for images in baselines_by_prompt.values():
            scored = [
                img for img in images
                if reward_model_name in img.reward_scores
                and img.image_id in detection
            ]
            if len(scored) < 2:
                continue

            rm = {img.image_id: img.reward_scores[reward_model_name] for img in scored}
            det = {img.image_id: detection[img.image_id] for img in scored}

            for i in range(len(scored)):
                for j in range(i + 1, len(scored)):
                    a, b = scored[i], scored[j]
                    r_a, r_b = rm[a.image_id], rm[b.image_id]
                    if r_a == r_b:
                        continue
                    high, low = (a, b) if r_a > r_b else (b, a)
                    pair_key = (high.image_id, low.image_id)
                    if pair_key in seen:
                        continue

                    # Keep only pairs where at least one attr differs (D[i,:] ≠ 0)
                    det_h = det[high.image_id]
                    det_l = det[low.image_id]
                    has_diff = any(
                        det_h.get(attr, 0) != det_l.get(attr, 0)
                        for attr in attr_pool
                    )
                    if not has_diff:
                        n_skipped_no_diff += 1
                        continue

                    seen.add(pair_key)
                    all_pairs.append(BaselinePair(
                        high_reward=high,
                        low_reward=low,
                        delta_rm=abs(r_a - r_b),
                    ))

        logger.info(
            f"AllPairConstructor: {len(all_pairs)} pairs from "
            f"{len(baselines_by_prompt)} prompts  "
            f"(K={K}, skipped {n_skipped_no_diff} zero-diff pairs)"
        )
        return all_pairs
