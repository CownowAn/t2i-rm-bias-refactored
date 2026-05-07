"""Linear probing for pair-based reward-model bias detection.

Supported regression models (all wrapped in StandardScaler pipeline):
  "lasso"      — LassoCV     (L1 only,     l1_ratio fixed at 1.0)
  "ridge"      — RidgeCV     (L2 only,     l1_ratio fixed at 0.0)
  "elasticnet" — ElasticNetCV (L1+L2 mix,  l1_ratio searched by CV)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from search.data.state import EvoStep
from search.data.types import CounterfactualPair

try:
    from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

_VALID_REGRESSION_MODELS = ("lasso", "ridge", "elasticnet")


def _make_regression_pipeline(
    regression_model: str,
    l1_ratio: "float | list[float]",
    n_alphas: int,
    cv: int,
    fit_intercept: bool,
) -> "make_pipeline":
    """Build a StandardScaler + CV regression pipeline for the requested model."""
    if regression_model == "lasso":
        estimator = LassoCV(
            alphas=n_alphas,
            cv=cv,
            fit_intercept=fit_intercept,
            max_iter=10_000,
            n_jobs=1,
        )
    elif regression_model == "ridge":
        # RidgeCV needs an explicit array of alphas (not an integer count)
        ridge_alphas = np.logspace(-4, 4, n_alphas)
        estimator = RidgeCV(
            alphas=ridge_alphas,
            cv=cv,
            fit_intercept=fit_intercept,
            scoring="neg_mean_squared_error",
        )
    elif regression_model == "elasticnet":
        estimator = ElasticNetCV(
            l1_ratio=l1_ratio,
            alphas=n_alphas,
            cv=cv,
            fit_intercept=fit_intercept,
            max_iter=10_000,
            n_jobs=1,
        )
    else:
        raise ValueError(
            f"Unknown regression_model {regression_model!r}. "
            f"Choose from: {_VALID_REGRESSION_MODELS}"
        )
    return make_pipeline(StandardScaler(), estimator)


def pair_key(attr: str, prompt_text: str, pair: CounterfactualPair) -> str:
    return f"{attr}|{prompt_text}|{pair.baseline.image_path.stem}"


def compute_regression_residuals_from_matrix(
    D: "np.ndarray",
    delta_rm_vec: "np.ndarray",
    attr_names: list[str],
    pair_keys: list[str],
    min_pairs: int = 5,
    fit_intercept: bool = True,
    regression_model: str = "elasticnet",
    l1_ratio: "float | list[float]" = (0.1, 0.5, 0.9, 1.0),
    n_alphas: int = 100,
    cv: int = 5,
) -> "LinearProbingResult":
    """
    Fit a regularised regression (StandardScaler + CV estimator) on a pre-computed
    D matrix and return per-pair residuals.

    D: {-1,0,1}^{N×K} — attribute difference matrix
    delta_rm_vec: ℝ^N — reward differences (always positive: r(high) - r(low))
    attr_names: K attribute names (column labels for D)
    pair_keys: N pair identifiers (row labels for residuals)
    fit_intercept: whether to fit a bias term (True for baseline-pairs).
    regression_model: "lasso" | "ridge" | "elasticnet"
    l1_ratio: ElasticNet only — L1/(L1+L2) mix. 0=Ridge, 1=Lasso.
              Scalar or list (list → CV searches over all values).
    n_alphas: number of alpha candidates per l1_ratio (Lasso/ElasticNet);
              number of logspace alpha values searched for Ridge.
    cv: number of cross-validation folds.

    StandardScaler normalises D columns so that sparse columns (rare attributes)
    are not systematically penalised relative to dense ones.
    Coefficients are returned in the original D scale.
    """
    if not _SKLEARN_AVAILABLE:
        logger.warning(
            "scikit-learn not installed; regression residuals unavailable. "
            "Install with: pip install scikit-learn>=1.3.0"
        )
        return None  # type: ignore[return-value]

    _default_l1 = float(l1_ratio) if not hasattr(l1_ratio, "__len__") else 1.0

    N, K = D.shape
    if N < min_pairs:
        logger.warning(
            f"Only {N} pairs (< min_pairs={min_pairs}); skipping {regression_model}"
        )
        return LinearProbingResult(
            attribute_weights={a: 0.0 for a in attr_names},
            residuals={pk: 0.0 for pk in pair_keys},
            variance_explained=0.0,
            reg_alpha=0.0,
            l1_ratio=_default_l1,
            reg_intercept=0.0,
            n_pairs=N,
            fallback=True,
        )

    var_total = float(np.var(delta_rm_vec))
    if var_total < 1e-10:
        logger.warning(f"Near-zero variance in delta_rm; {regression_model} residuals are degenerate")
        return LinearProbingResult(
            attribute_weights={a: 0.0 for a in attr_names},
            residuals={pk: 0.0 for pk in pair_keys},
            variance_explained=1.0,
            reg_alpha=0.0,
            l1_ratio=_default_l1,
            reg_intercept=0.0,
            n_pairs=N,
            fallback=True,
        )

    model = _make_regression_pipeline(
        regression_model=regression_model,
        l1_ratio=l1_ratio,
        n_alphas=n_alphas,
        cv=min(cv, N),
        fit_intercept=fit_intercept,
    )
    model.fit(D, delta_rm_vec)

    scaler: StandardScaler = model[0]
    estimator = model[-1]  # LassoCV, RidgeCV, or ElasticNetCV

    # Transform coefficients from scaled space back to original D scale
    W = estimator.coef_ / scaler.scale_

    # Intercept in original space
    intercept = float(estimator.intercept_) - float(np.dot(W, scaler.mean_)) if fit_intercept else 0.0

    # l1_ratio: 1.0 for Lasso, 0.0 for Ridge, CV-chosen for ElasticNet
    chosen_l1 = float(getattr(estimator, "l1_ratio_", None) or
                      (1.0 if regression_model == "lasso" else 0.0))

    residuals_vec = delta_rm_vec - model.predict(D)
    var_explained = float(max(0.0, 1.0 - np.var(residuals_vec) / var_total))

    logger.debug(
        f"{regression_model}: alpha={estimator.alpha_:.4f}  "
        f"l1_ratio={chosen_l1:.2f}  var_explained={var_explained:.4f}"
    )

    return LinearProbingResult(
        attribute_weights={attr_names[k]: float(W[k]) for k in range(K)},
        residuals={pair_keys[i]: float(residuals_vec[i]) for i in range(N)},
        variance_explained=var_explained,
        reg_alpha=float(estimator.alpha_),
        l1_ratio=chosen_l1,
        reg_intercept=intercept,
        n_pairs=N,
    )


@dataclass
class LinearProbingResult:
    attribute_weights: dict[str, float]  # attr_name → W_RM[k] (original D scale)
    residuals: dict[str, float]          # pair_key → r_i = delta_rm - predicted
    variance_explained: float            # 1 - Var(residuals) / Var(delta_rm)
    reg_alpha: float                   # regularization alpha chosen by CV
    l1_ratio: float                      # L1/(L1+L2) chosen by CV (0=Ridge, 1=Lasso)
    reg_intercept: float               # bias term in original scale; 0.0 when fit_intercept=False
    n_pairs: int
    fallback: bool = False               # True when ElasticNet couldn't run


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
            reg_alpha=0.0,
            reg_intercept=0.0,
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
            reg_alpha=0.0,
            reg_intercept=0.0,
            n_pairs=N,
            fallback=True,
        )

    model = make_pipeline(
        StandardScaler(),
        ElasticNetCV(alphas=100, cv=min(5, N), fit_intercept=False, max_iter=10_000, n_jobs=1),
    )
    model.fit(D, delta_rm_vec)

    scaler: StandardScaler = model.named_steps["standardscaler"]
    enet: ElasticNetCV = model.named_steps["elasticnetcv"]
    W = enet.coef_ / scaler.scale_
    residuals_vec = delta_rm_vec - model.predict(D)
    var_explained = float(max(0.0, 1.0 - np.var(residuals_vec) / var_total))

    return LinearProbingResult(
        attribute_weights={attr_names[k]: float(W[k]) for k in range(K)},
        residuals={pk: float(residuals_vec[i]) for i, (_, _, _, pk) in enumerate(rows)},
        variance_explained=var_explained,
        reg_alpha=float(enet.alpha_),
        l1_ratio=float(enet.l1_ratio_),
        reg_intercept=0.0,
        n_pairs=N,
    )
