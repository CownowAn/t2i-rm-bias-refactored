from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from search.config import DetectorConfig
    from search.models.base import DetectorModel
    from caller.cache import CacheConfig


def build_detector(cfg: "DetectorConfig", cache_config: "Optional[CacheConfig]" = None) -> "DetectorModel":
    """Instantiate DetectorModel from config.

    API mode (default): cfg.vllm_base_url is None → AutoCaller routes to cloud API.
    vLLM serving mode:  cfg.vllm_base_url set    → LocalCaller routes to local vLLM server.
    """
    from search.models.judge.vlm_judge import VisionLLMDetector
    return VisionLLMDetector(
        model_name=cfg.model,
        max_tokens=cfg.max_tokens,
        max_parallel=cfg.max_parallel,
        image_detail=cfg.image_detail,
        use_batch_api=cfg.use_batch_api,
        vllm_base_url=cfg.vllm_base_url,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        presence_penalty=cfg.presence_penalty,
        extra_body=cfg.extra_body,
        use_prompt=cfg.use_prompt,
        use_reasoning=cfg.use_reasoning,
        use_applicability=getattr(cfg, "use_applicability", False),
        cache_config=cache_config,
    )
