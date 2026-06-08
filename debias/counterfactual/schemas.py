"""Shared dataclasses for the counterfactual debiasing pipeline.

Single source of truth — all stages exchange these immutable objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PerPromptW:
    """Parsed `per_prompt_W_step{N}_topic{T}.json` artifact."""
    step_idx: int
    topic_id: int
    attrs: list[str]                              # K attributes (order matches W vectors)
    per_prompt_W: dict[str, list[float]]          # prompt_text → [W_0..W_{K-1}]
    per_prompt_r2: dict[str, float]               # prompt_text → R²


@dataclass(frozen=True)
class PromptAttrSelection:
    """One (prompt, attr) pair selected for editing."""
    prompt_text: str
    topic_id: int
    attr: str
    w_value: float
    rank_in_prompt: int                           # 0 = top-weight attr in this prompt
    is_undesirable: bool


@dataclass(frozen=True)
class SourceImage:
    """A baseline image picked as the source for an edit."""
    image_id: str
    image_path: Path
    prompt_text: str
    detected_attrs_snapshot: dict[str, int]       # detection cache snapshot for this image


@dataclass(frozen=True)
class EditTask:
    """A pending FLUX-Kontext edit job."""
    selection: PromptAttrSelection
    source: SourceImage
    instruction: str
    edited_output_path: Path


@dataclass(frozen=True)
class EditResult:
    """Outcome of one edit + post-edit detector / reward verification."""
    task: EditTask
    success: bool                                 # edited_attr_detected == 0
    edited_attr_detected: int | None              # 0/1 on edited (None on error)
    original_attr_detected: int | None            # 1 expected on original (sanity)
    side_effect_drift: dict[str, tuple[int, int]] | None  # attr → (before, after)
    error: str | None
    # Reward verification (populated only when --check_reward is on)
    orig_reward: float | None = None
    edited_reward: float | None = None
    reward_drop: float | None = None              # = orig_reward - edited_reward (>0 = good)
    reward_model_name: str | None = None


@dataclass(frozen=True)
class CFPair:
    """Validated counterfactual pair for BT-loss finetune."""
    prompt_text: str
    topic_id: int
    attr: str
    winner_path: Path                             # edited
    loser_path: Path                              # original
    winner_image_id: str
    loser_image_id: str
    meta: dict


@dataclass
class AttrSurveyRow:
    """Per-attribute summary in the PoC survey report."""
    attr: str
    n_attempted: int
    n_success: int
    success_rate: float
    sample_success: list[tuple[Path, Path]] = field(default_factory=list)   # (original, edited)
    sample_fail: list[tuple[Path, Path]] = field(default_factory=list)
    thumbnail_path: Path | None = None
    # Reward Δ over edits with both orig and edited scored (None if check_reward off)
    reward_n_scored: int = 0
    reward_drop_mean: float | None = None
    reward_drop_pct_positive: float | None = None
    reward_model_name: str | None = None


@dataclass
class PoCConfig:
    """User-tunable knobs for the PoC survey."""
    tau: float
    top_n_per_prompt: int
    n_prompts_per_attr: int
    n_images_per_prompt: int
    humanness_recheck: bool
    side_effect_check: bool
    make_thumbnails: bool
    seed: int
    run_id: str


@dataclass
class SurveyResult:
    """End-to-end PoC output bundle."""
    selections: list[PromptAttrSelection]
    edit_results: list[EditResult]
    rows: list[AttrSurveyRow]
