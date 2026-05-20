from __future__ import annotations
from pathlib import Path
from typing import Any

from loguru import logger

from search.models.base import RewardModel, RatingResult

_PROCESSOR_NAME = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
_MODEL_NAME     = "yuvalkirstain/PickScore_v1"


class PickScoreModel(RewardModel):
    """Reward model using PickScore v1 (yuvalkirstain/PickScore_v1)."""

    def __init__(self, device: str = "cuda:0", hf_cache_dir: str | None = None):
        self._device = device
        self._hf_cache_dir = hf_cache_dir
        self._processor = None
        self._model = None
        self._load()

    def _load(self) -> None:
        try:
            from transformers import AutoProcessor, AutoModel
            cache_kw = {"cache_dir": self._hf_cache_dir} if self._hf_cache_dir else {}
            logger.info(f"Loading PickScore processor and model on {self._device}")
            self._processor = AutoProcessor.from_pretrained(_PROCESSOR_NAME, **cache_kw)
            self._model = (
                AutoModel.from_pretrained(_MODEL_NAME, **cache_kw)
                .eval()
                .to(self._device)
            )
            logger.info("PickScore model loaded successfully")
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

    @property
    def model_name(self) -> str:
        return "pickscore"

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
                image_inputs = self._processor(
                    images=[image],
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(self._device)
                text_inputs = self._processor(
                    text=prompt,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(self._device)
                with torch.no_grad():
                    image_embs = self._model.get_image_features(**image_inputs)
                    image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
                    text_embs = self._model.get_text_features(**text_inputs)
                    text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
                    score = (self._model.logit_scale.exp() * (text_embs @ image_embs.T))[0, 0]
                results.append(RatingResult(score=float(score.cpu())))
                logger.debug(f"Rated {Path(img_path).name}: score={score:.3f}")
            except Exception as e:
                logger.error(f"Failed to rate {img_path}: {e}")
                results.append(RatingResult(score=None, reasoning=str(e)))
        return results

    def to_dict(self) -> dict[str, Any]:
        return {"name": "pickscore", "device": self._device}