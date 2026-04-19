from __future__ import annotations
import threading
from pathlib import Path

from loguru import logger


class FluxKontextApplier:
    """Applies natural language edit instructions via FLUX Kontext.

    Lazy-loads the pipeline on first use. One instance per GPU.
    Thread-safe: _global_load_lock serializes loading across all instances
    to prevent accelerate meta-tensor conflicts on concurrent from_pretrained calls.
    """

    _global_load_lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-Kontext-dev",
        device: str = "auto",
        guidance_scale: float = 2.5,
        hf_cache_dir: str | None = None,
    ):
        import torch
        self.model_name = model_name
        self.guidance_scale = guidance_scale
        self.hf_cache_dir = hf_cache_dir
        self._pipeline = None
        self._lock = threading.Lock()

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def _load_pipeline(self) -> None:
        """Must be called while holding self._lock."""
        if self._pipeline is not None:
            return
        import torch
        from diffusers import FluxKontextPipeline

        logger.info(f"Loading FluxKontextPipeline: {self.model_name} on {self.device}")
        with self.__class__._global_load_lock:
            if self._pipeline is not None:
                return
            kwargs: dict = {"torch_dtype": torch.bfloat16}
            if self.hf_cache_dir:
                kwargs["cache_dir"] = self.hf_cache_dir
            pipe = FluxKontextPipeline.from_pretrained(self.model_name, **kwargs)
            pipe.to(self.device)
            self._pipeline = pipe
        logger.info(f"FluxKontextPipeline loaded on {self.device}")

    def apply(self, image_path: str, instruction: str, output_path: str) -> str:
        """Apply instruction to image. Returns output_path (skips if already exists)."""
        if Path(output_path).exists():
            logger.debug(f"Cache hit: {output_path}")
            return output_path

        from diffusers.utils import load_image

        with self._lock:
            self._load_pipeline()
            image = load_image(image_path)
            result = self._pipeline(
                prompt=instruction,
                image=image,
                guidance_scale=self.guidance_scale,
            ).images[0]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        logger.debug(f"Saved edited image: {output_path}")
        return output_path
