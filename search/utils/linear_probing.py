"""Lasso-based linear probing: build D = X2 - X1, fit regression, return per-pair residuals."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from search.data.state import EvoStep
from search.data.types import CounterfactualPair

try:
    from sklearn.linear_model import LassoCV
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def pair_key(attr: str, prompt_text: str, pair: CounterfactualPair) -> str:
    return f"{attr}|{prompt_text}|{pair.baseline.image_path.stem}"


@dataclass
class LinearProbingResult:
    attribute_weights: dict[str, float]  # attr_name → W_RM[k] from Lasso
    residuals: dict[str, float]          # pair_key → r_i = delta_rm - predicted
    variance_explained: float            # 1 - Var(residuals) / Var(delta_rm)
    lasso_alpha: float                   # regularization lambda chosen by CV
    n_pairs: int
    fallback: bool = False               # True when Lasso couldn't run


def compute_lasso_residuals(
    last_step: EvoStep,
    *,
    min_pairs: int = 5,
) -> LinearProbingResult | None:
    """
    Build D = X2 − X1 (simplified: only applied attribute column changes),
    fit LassoCV(fit_intercept=False), and return per-pair residuals.

    D_{i, k_applied} = 1 - X1_{i, k_applied}  (from VLM detection stored in baseline_detected)
    D_{i, k != k_applied} = 0

    Returns None if scikit-learn is not installed.
    Returns LinearProbingResult(fallback=True) if total valid pairs < min_pairs.
    """
    if not _SKLEARN_AVAILABLE:
        logger.warning(
            "scikit-learn not installed; residual context mode unavailable. "
            "Install with: pip install scikit-learn>=1.3.0"
        )
        return None

    attr_names = list(last_step.attributes.keys())
    K = len(attr_names)
    attr_idx = {a: i for i, a in enumerate(attr_names)}

    # Collect all valid pairs across all attributes
    rows: list[tuple[int, CounterfactualPair, str, str]] = []  # (k, pair, prompt_text, pk)
    for attr, stats in last_step.attributes.items():
        k = attr_idx[attr]
        for prompt_text, pairs in stats.pairs.items():
            for p in pairs:
                if (
                    p.delta_rm is not None
                    and p.edited_image_path.exists()
                    and p.baseline.image_path.exists()
                ):
                    rows.append((k, p, prompt_text, pair_key(attr, prompt_text, p)))

    N = len(rows)
    if N < min_pairs:
        logger.warning(
            f"Only {N} valid pairs (< min_pairs={min_pairs}); skipping Lasso for residual context"
        )
        return LinearProbingResult(
            attribute_weights={},
            residuals={},
            variance_explained=0.0,
            lasso_alpha=0.0,
            n_pairs=N,
            fallback=True,
        )

    # Build D matrix using X1 from amplification-score VLM detection
    D = np.zeros((N, K), dtype=np.float32)
    delta_rm_vec = np.zeros(N, dtype=np.float32)

    for i, (k, pair, _prompt_text, _pk) in enumerate(rows):
        attr = attr_names[k]
        stats = last_step.attributes[attr]
        # X1_{i,k} = VLM detection of attr on baseline image (0 if unknown → conservative)
        x1 = stats.baseline_detected.get(pair.baseline.image_id, 0)
        # Simplified X2: applied attribute forced to 1, other attributes unchanged
        # D_{i,k} = X2_{i,k} - X1_{i,k} = 1 - x1
        D[i, k] = 1.0 - float(x1)
        delta_rm_vec[i] = float(pair.delta_rm)  # type: ignore[arg-type]

    var_total = float(np.var(delta_rm_vec))
    if var_total < 1e-10:
        logger.warning("Near-zero variance in delta_rm across pairs; Lasso residuals are degenerate")
        return LinearProbingResult(
            attribute_weights={a: 0.0 for a in attr_names},
            residuals={pk: 0.0 for _, _, _, pk in rows},
            variance_explained=1.0,
            lasso_alpha=0.0,
            n_pairs=N,
            fallback=True,
        )

    lasso = LassoCV(cv=min(5, N), fit_intercept=False, max_iter=10_000, n_jobs=1)
    lasso.fit(D, delta_rm_vec)

    W = lasso.coef_
    residuals_vec = delta_rm_vec - D @ W
    var_explained = float(max(0.0, 1.0 - np.var(residuals_vec) / var_total))

    return LinearProbingResult(
        attribute_weights={attr_names[k]: float(W[k]) for k in range(K)},
        residuals={pk: float(residuals_vec[i]) for i, (_, _, _, pk) in enumerate(rows)},
        variance_explained=var_explained,
        lasso_alpha=float(lasso.alpha_),
        n_pairs=N,
    )
