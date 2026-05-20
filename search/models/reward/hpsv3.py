from __future__ import annotations
from pathlib import Path
from typing import Any

from loguru import logger

from search.models.base import RewardModel, RatingResult


class HPSv3Model(RewardModel):
    """Reward model using HPSv3 (MizzenAI/HPSv3, based on Qwen2-VL-7B)."""

    def __init__(self, device: str = "cuda:0", hf_cache_dir: str | None = None):
        self._device = device
        self._hf_cache_dir = hf_cache_dir
        self._inferencer = None
        self._load()

    def _load(self) -> None:
        try:
            import os
            if self._hf_cache_dir:
                os.environ.setdefault("HF_HOME", self._hf_cache_dir)
            from hpsv3 import HPSv3RewardInferencer
            logger.info(f"Loading HPSv3 model on {self._device}")
            self._inferencer = HPSv3RewardInferencer(device=self._device)
            logger.info("HPSv3 model loaded successfully")
        except ImportError:
            raise ImportError("Install HPSv3: pip install hpsv3")

    @property
    def model_name(self) -> str:
        return "hpsv3"

    async def rate(
        self,
        image_paths: list[str],
        prompts: list[str],
    ) -> list[RatingResult]:
        if len(image_paths) != len(prompts):
            raise ValueError("image_paths and prompts must have the same length")

        results = []
        for img_path, prompt in zip(image_paths, prompts):
            try:
                # reward() returns [(mu, sigma), ...]; we use mu as the score
                reward = self._inferencer.reward([img_path], [prompt])
                score = float(reward[0][0])
                results.append(RatingResult(score=score))
                logger.debug(f"Rated {Path(img_path).name}: score={score:.3f}")
            except Exception as e:
                logger.error(f"Failed to rate {img_path}: {e}")
                results.append(RatingResult(score=None, reasoning=str(e)))
        return results

    def to_dict(self) -> dict[str, Any]:
        return {"name": "hpsv3", "device": self._device}