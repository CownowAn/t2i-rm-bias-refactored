"""Data types for the baseline-pairs pipeline mode."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from search.data.types import BaselineImage


@dataclass
class BaselinePair:
    """A pair of baseline images for the same prompt used for linear probing."""
    high_reward: "BaselineImage"   # higher-reward image
    low_reward: "BaselineImage"    # lower-reward image
    delta_rm: float                # r(high) - r(low) > 0
    delta_j: float | None = None   # +1 = high preferred, 0 = tie, -1 = low preferred
    judge_reasoning: str | None = None


@dataclass
class BaselinePairStep:
    """Per-step state for the baseline-pairs pipeline.

    Lifecycle:
      EVALUATE fills: attribute_pool, detection (new attrs only), amp_scores, acc_pool_snapshot
      EXPAND   fills: pairs, D (full acc_pool cols), delta_rm_vec, W_rm, residuals

    Note: amp_baselines and detection_cache are stored engine-level (_fixed_baselines,
    _detection_cache) and NOT duplicated here to avoid memory bloat.
    """
    step_idx: int
    attribute_pool: list[str]              # new attrs selected this step (set by EVALUATE)
    acc_pool_snapshot: list[str]           # full accumulated pool at this step (set by EVALUATE)
    detection: dict[str, dict[str, int]]  # new-attrs-only detection (set by EVALUATE)
    amp_scores: dict[str, float] = field(default_factory=dict)  # attr → A(g) (set by EVALUATE)
    pairs: list[BaselinePair] = field(default_factory=list)      # (set by EXPAND)
    D: np.ndarray | None = None            # {-1,0,1}^{N×K_full} full acc_pool cols (set by EXPAND)
    delta_rm_vec: np.ndarray | None = None  # ℝ^N (set by EXPAND)
    W_rm: np.ndarray | None = None          # ℝ^K_full regression weights on full pool (set by EXPAND)
    reg_intercept: float = 0.0             # bias term from regression (set by EXPAND)
    reg_alpha: float = 0.0                 # regularization alpha chosen by CV (set by EXPAND)
    reg_l1_ratio: float = 1.0             # L1/(L1+L2) ratio chosen by CV (set by EXPAND)
    residuals: np.ndarray | None = None     # ℝ^N (set by EXPAND)
