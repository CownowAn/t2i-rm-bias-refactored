from __future__ import annotations
from pathlib import Path
from typing import Any

from loguru import logger

from search.models.base import RewardModel, RatingResult


class ImageRewardModel(RewardModel):
    """Student reward model using ImageReward-v1.0 (local HuggingFace)."""

    def __init__(self, device: str = "cuda:0", hf_cache_dir: str | None = None):
        self._device = device
        self._hf_cache_dir = hf_cache_dir
        self._model = None
        self._load()

    def _load(self) -> None:
        try:
            import ImageReward as RM
            logger.info(f"Loading ImageReward model on {self._device}")
            kwargs: dict = {"device": self._device}
            if self._hf_cache_dir:
                kwargs["download_root"] = self._hf_cache_dir
            self._model = RM.load("ImageReward-v1.0", **kwargs)
            self._model.eval()
            logger.info("ImageReward model loaded successfully")
        except ImportError:
            raise ImportError("Install ImageReward: pip install image-reward")

    @property
    def model_name(self) -> str:
        return "imagereward"

    async def rate(
        self,
        image_paths: list[str],
        prompts: list[str],
    ) -> list[RatingResult]:
        import torch
        from PIL import Image

        if len(image_paths) != len(prompts):
            raise ValueError("image_paths and prompts must have the same length")

        results = []
        for img_path, prompt in zip(image_paths, prompts):
            try:
                image = Image.open(img_path).convert("RGB")
                with torch.no_grad():
                    score = self._model.score(prompt, image)
                results.append(RatingResult(score=float(score)))
                logger.debug(f"Rated {Path(img_path).name}: score={score:.3f}")
            except Exception as e:
                logger.error(f"Failed to rate {img_path}: {e}")
                results.append(RatingResult(score=None, reasoning=str(e)))
        return results

    def to_dict(self) -> dict[str, Any]:
        return {"name": "imagereward", "device": self._device}
