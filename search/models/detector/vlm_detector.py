"""Attribute-presence detector via a vision LLM.

Wraps either the cloud OpenAI-compatible API (via :class:`AutoCaller`) or a
local vLLM server (via :class:`LocalCaller`) behind a uniform :meth:`detect`
interface that returns one int per image:

  ``1``  → attribute present
  ``0``  → attribute not present (or parse failed)
  ``-1`` → attribute does NOT apply to this image (only when
           ``use_applicability=True`` and the model said ``applicable=false``).

Downstream OLS / amplification code coerces ``-1`` to absent (``0``) when
``DetectorConfig.not_applicable_as_absent`` is set, but the cache keeps ``-1``
so the raw signal is preserved on disk.
"""
from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from caller import AutoCaller, LocalCaller, ChatHistory, ChatMessage
from caller.cache import CacheConfig
from search.models.base import DetectorModel
from search.prompts.detection import ATTRIBUTE_DETECTION_SYSTEM, build_detection_prompt


_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


class VisionLLMDetector(DetectorModel):
    """Attribute presence detector via Vision LLM."""

    def __init__(
        self,
        model_name: str = "openai/gpt-4o-mini",
        max_tokens: int = 50000,
        max_parallel: int = 32,
        image_detail: str = "auto",
        use_batch_api: bool = False,
        vllm_base_url: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        extra_body: dict | None = None,
        use_prompt: bool = True,
        use_reasoning: bool = True,
        use_applicability: bool = False,
        not_applicable_as_absent: bool = False,
        cache_config: CacheConfig | None = None,
    ):
        self._model_name = model_name
        self.max_tokens = max_tokens
        self.max_parallel = max_parallel
        self.image_detail = image_detail
        self.use_batch_api = use_batch_api
        self.vllm_base_url = vllm_base_url
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.extra_body = extra_body
        self.use_prompt = use_prompt
        self.use_reasoning = use_reasoning
        self.use_applicability = use_applicability
        self.not_applicable_as_absent = not_applicable_as_absent
        self.caller = (
            LocalCaller(base_url=vllm_base_url, cache_config=cache_config)
            if vllm_base_url
            else AutoCaller(dotenv_path=".env", cache_config=cache_config)
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    async def detect(
        self,
        image_paths: list[str],
        prompts: list[str],
        attribute: str,
    ) -> list[int]:
        if len(image_paths) != len(prompts):
            raise ValueError("image_paths and prompts must have the same length")

        chats = self._build_chats(image_paths, prompts, attribute)
        responses = await self._call_caller(chats)
        return self._parse_responses(responses, n=len(image_paths))

    def _build_chats(
        self,
        image_paths: list[str],
        prompts: list[str],
        attribute: str,
    ) -> list[ChatHistory | None]:
        """Build one ChatHistory per (image, prompt). Returns None for failures."""
        chats: list[ChatHistory | None] = []
        for img_path, prompt in zip(image_paths, prompts):
            try:
                img_url = ChatMessage.image_to_base64_url(img_path)
                user_text = build_detection_prompt(
                    attribute=attribute,
                    prompt=prompt,
                    use_prompt=self.use_prompt,
                    use_reasoning=self.use_reasoning,
                    use_applicability=self.use_applicability,
                )
                if self.vllm_base_url:
                    # vLLM (e.g. Qwen): separate system msg, image before text.
                    content = [
                        {"type": "input_image", "image_url": img_url},
                        {"type": "input_text", "text": user_text},
                    ]
                    history = ChatHistory(messages=[
                        ChatMessage(role="system", content=ATTRIBUTE_DETECTION_SYSTEM),
                        ChatMessage(role="user", content=content),
                    ])
                else:
                    content = [
                        {"type": "input_text",
                         "text": ATTRIBUTE_DETECTION_SYSTEM + "\n\n" + user_text + "\n\nImage:"},
                        {"type": "input_image", "image_url": img_url, "detail": self.image_detail},
                    ]
                    history = ChatHistory(messages=[ChatMessage(role="user", content=content)])
                chats.append(history)
            except Exception as e:
                logger.error(f"detector: failed to build chat for {img_path}: {e}")
                chats.append(None)
        return chats

    async def _call_caller(self, chats: list[ChatHistory | None]) -> dict[int, Any]:
        """Send valid chats through the configured caller. Returns {orig_idx: resp}."""
        valid_idx = [i for i, c in enumerate(chats) if c is not None]
        if not valid_idx:
            return {}
        valid_chats = [chats[i] for i in valid_idx]
        if self.use_batch_api:
            responses = await self.caller.call_batch(
                messages=valid_chats,
                model=self._model_name,
                max_tokens=self.max_tokens,
            )
        else:
            responses = await self.caller.call(
                messages=valid_chats,
                model=self._model_name,
                max_parallel=self.max_parallel,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                presence_penalty=self.presence_penalty,
                extra_body=self.extra_body,
            )
        return dict(zip(valid_idx, responses))

    def _parse_responses(self, resp_map: dict[int, Any], n: int) -> list[int]:
        """Convert per-image JSON responses into 0/1/-1 ints. Errors → 0."""
        results: list[int] = []
        for idx in range(n):
            resp = resp_map.get(idx)
            if resp is None or not resp.has_response:
                results.append(0)
                continue
            try:
                m = _JSON_OBJ_RE.search(resp.first_response)
                if not m:
                    logger.warning(
                        "detector: no JSON object in response — "
                        f"first 200 chars: {resp.first_response[:200]!r}"
                    )
                    results.append(0)
                    continue
                data = json.loads(m.group())
            except Exception as e:
                logger.warning(
                    f"detector: parse failure ({type(e).__name__}: {e}); "
                    f"first 200 chars: {resp.first_response[:200]!r}"
                )
                results.append(0)
                continue
            if self.use_applicability and data.get("applicable") is False:
                # Always store -1; downstream coerces to 0 when
                # `not_applicable_as_absent` is set in DetectorConfig.
                results.append(-1)
            else:
                results.append(1 if data.get("present", False) else 0)
        return results

    def to_dict(self) -> dict[str, Any]:
        return {"model_name": self._model_name, "type": "vision_llm_detector"}
