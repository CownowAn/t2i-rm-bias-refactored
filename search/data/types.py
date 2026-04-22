from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Prompt:
    text: str
    topic_id: int


@dataclass
class BaselineImage:
    image_path: Path
    image_id: str
    prompt: Prompt
    policy_model: str
    reward_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class CounterfactualPair:
    baseline: BaselineImage
    edited_image_path: Path
    edit_instruction: str
    delta_rm: float | None = None   # reward(edited) - reward(baseline)
    delta_j: float | None = None    # judge(edited vs baseline) - 0.5
    judge_reasoning: str | None = None
    step_idx: int | None = None     # evo step this pair was evaluated in
