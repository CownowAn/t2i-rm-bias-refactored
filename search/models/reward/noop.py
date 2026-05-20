"""No-op reward model: holds the model_name string only, never loads weights.

Used when every baseline in the manifest already carries reward_scores for the
configured model. Skips the expensive GPU load entirely.
"""
from __future__ import annotations

from typing import Any

from search.models.base import RewardModel, RatingResult


class NoOpRewardModel(RewardModel):
    def __init__(self, name: str):
        self._name = name

    @property
    def model_name(self) -> str:
        return self._name

    async def rate(
        self,
        image_paths: list[str],
        prompts: list[str],
    ) -> list[RatingResult]:
        raise RuntimeError(
            f"NoOpRewardModel({self._name}).rate() called — "
            "a baseline is missing pre-computed scores. Run "
            "baselines/score_baselines.sh first, or instantiate the real model."
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self._name, "noop": True}
