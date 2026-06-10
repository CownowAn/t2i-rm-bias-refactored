"""VQAScore (image-text alignment) scorer via the t2v_metrics library.

Used as an extra baseline metric alongside the reward models. Because VQAScore has
many possible backbones (clip-flant5-xxl, qwen2.5-vl-7b, ...), scores are stored under
reward_scores['vqascore_{vqa_model}'] so multiple backbones coexist without clobbering.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger

from search.models.base import RewardModel, RatingResult


class VQAScoreModel(RewardModel):
    """Image-text alignment scorer (Lin et al. VQAScore) via t2v_metrics.

    Returns P(Yes) for "Does this image show {prompt}?" in [0, 1] (higher = better aligned).
    """

    def __init__(
        self,
        device: str = "cuda:0",
        hf_cache_dir: str | None = None,
        vqa_model: str = "clip-flant5-xxl",
    ):
        self._device = device
        self._hf_cache_dir = hf_cache_dir
        self._vqa_model = vqa_model
        self._model = None
        self._load()

    def _load(self) -> None:
        try:
            import t2v_metrics
        except ImportError:
            raise ImportError("Install t2v_metrics: pip install t2v-metrics")
        if self._hf_cache_dir:
            os.environ.setdefault("HF_HOME", self._hf_cache_dir)
        logger.info(f"Loading VQAScore ({self._vqa_model}) on {self._device}")
        kwargs: dict = {"model": self._vqa_model, "device": self._device}
        if self._hf_cache_dir:
            kwargs["cache_dir"] = self._hf_cache_dir
        self._model = t2v_metrics.VQAScore(**kwargs)
        logger.info("VQAScore model loaded successfully")

    @property
    def model_name(self) -> str:
        return f"vqascore_{self._vqa_model}"

    async def rate(
        self,
        image_paths: list[str],
        prompts: list[str],
    ) -> list[RatingResult]:
        import torch

        n = len(image_paths)
        if n == 0:
            return []

        # t2v_metrics.batch_forward(dataset=[{images, texts}, ...], batch_size=...) does
        # PAIRED scoring (each dataset sample is independent) while batching GPU work.
        # This avoids both the N×M blow-up of __call__ and the batch=1 inefficiency of
        # per-pair calls. Returns Tensor[n_sample, n_images_per_sample, n_texts_per_sample]
        # = [n, 1, 1] here, so flatten and take the single paired score per sample.
        dataset = [{"images": [img], "texts": [txt]}
                   for img, txt in zip(image_paths, prompts)]
        try:
            with torch.no_grad():
                scores = self._model.batch_forward(dataset=dataset, batch_size=n)
            flat = scores.reshape(n, -1)[:, 0]
            results = []
            for i in range(n):
                score = float(flat[i])
                logger.debug(f"Rated {Path(image_paths[i]).name}: score={score:.3f}")
                results.append(RatingResult(score=score))
            return results
        except Exception as e:
            logger.warning(f"VQAScore batch_forward failed ({e}); falling back to per-pair")

        # Per-pair fallback: slower (batch=1) but robust to per-image failures.
        results: list[RatingResult] = []
        for img, txt in zip(image_paths, prompts):
            try:
                with torch.no_grad():
                    s = self._model(images=[img], texts=[txt])  # Tensor[1, 1]
                results.append(RatingResult(score=float(s.reshape(-1)[0])))
            except Exception as e:
                logger.warning(f"VQAScore failed for {img}: {e}")
                results.append(RatingResult(score=None, reasoning=str(e)))
        return results

    def to_dict(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "vqa_model": self._vqa_model}
