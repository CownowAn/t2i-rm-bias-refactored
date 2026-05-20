"""Data types for the bon_amplified pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BonAmplifiedStep:
    """Per-step state for the bon_amplified pipeline."""

    step_idx: int
    N: int                                    # BoN parameter
    attribute_pool: list[str]                 # attrs selected this step
    acc_pool_snapshot: list[str]             # full acc pool after this step
    detection: dict[str, dict[str, int]]     # new detection results (image_id -> {attr: 0/1})
    amp_scores: dict[str, float]             # A_hat per attr (bon formula)
    # Residual fields — filled during EXPAND
    residuals: dict[str, float] = field(default_factory=dict)
    # key = "prompt_text||image_id", value = residual e_{x,i}
    W: list[float] = field(default_factory=list)   # OLS regression weights (K,)
    W_mode: str = "mean_per_prompt"                         # "global" or "mean_per_prompt"
    reg_var_explained: float = 0.0
    n_images: int = 0                             # total images used in residual regression
    # Per-prompt diagnostics (filled regardless of OLS mode)
    per_prompt_r2: dict[str, float] = field(default_factory=dict)
    # Per-prompt W (only populated when use_per_prompt_ols=True)
    per_prompt_W: dict[str, list[float]] = field(default_factory=dict)
