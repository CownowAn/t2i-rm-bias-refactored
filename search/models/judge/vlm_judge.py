from __future__ import annotations
import json
import re
import random
from pathlib import Path
from typing import Any

from loguru import logger

from caller import AutoCaller, LocalCaller, ChatHistory, ChatMessage
from caller.cache import CacheConfig
from search.models.base import JudgeModel, DetectorModel, ComparisonResult
from search.prompts.judging import IMAGE_JUDGE_SYSTEM, IMAGE_JUDGE_PROMPT
from search.prompts.detection import ATTRIBUTE_DETECTION_SYSTEM, build_detection_prompt


class VisionLLMJudge(JudgeModel):
    """Pairwise image quality judge via Vision LLM."""

    def __init__(
        self,
        model_name: str = "openai/gpt-4o-mini",
        max_tokens: int = 50000,
        max_parallel: int = 32,
        random_seed: int = 42,
        image_detail: str = "auto",
        use_batch_api: bool = False,
        cache_config: CacheConfig | None = None,
    ):
        self._model_name = model_name
        self.max_tokens = max_tokens
        self.max_parallel = max_parallel
        self.image_detail = image_detail
        self.use_batch_api = use_batch_api
        self.rng = random.Random(random_seed)
        self.caller = AutoCaller(dotenv_path=".env", cache_config=cache_config)

    @property
    def model_name(self) -> str:
        return self._model_name

    async def compare(
        self,
        image_A_paths: list[str],
        image_B_paths: list[str],
        prompts: list[str],
    ) -> list[ComparisonResult]:
        """Compare pairs of images for overall quality (A=edited, B=baseline)."""
        if not (len(image_A_paths) == len(image_B_paths) == len(prompts)):
            raise ValueError("All input lists must have the same length")

        judge_template = IMAGE_JUDGE_SYSTEM + "\n\n" + IMAGE_JUDGE_PROMPT

        # Random A/B flip to eliminate position bias
        flips = [self.rng.choice([False, True]) for _ in range(len(image_A_paths))]

        chats: list[ChatHistory | None] = []
        for img_A, img_B, prompt, flip in zip(image_A_paths, image_B_paths, prompts, flips):
            try:
                first_url  = ChatMessage.image_to_base64_url(img_B if flip else img_A)
                second_url = ChatMessage.image_to_base64_url(img_A if flip else img_B)
                content = [
                    {"type": "input_text", "text": judge_template.format(prompt=prompt) + "\n\nImage A:"},
                    {"type": "input_image", "image_url": first_url, "detail": self.image_detail},
                    {"type": "input_text", "text": "Image B:"},
                    {"type": "input_image", "image_url": second_url, "detail": self.image_detail},
                ]
                chats.append(ChatHistory(messages=[ChatMessage(role="user", content=content)]))
            except Exception as e:
                logger.error(f"Failed to build judge chat: {e}")
                chats.append(None)

        valid_idx = [i for i, c in enumerate(chats) if c is not None]
        resp_map: dict[int, Any] = {}
        if valid_idx:
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
                )
            for i, r in zip(valid_idx, responses):
                resp_map[i] = r

        results: list[ComparisonResult] = []
        for idx in range(len(image_A_paths)):
            flip = flips[idx]
            if idx not in resp_map:
                results.append(ComparisonResult(winner=None, score_diff=None, reasoning="Build error"))
                continue
            resp = resp_map[idx]
            if resp is None or not resp.has_response:
                results.append(ComparisonResult(winner=None, score_diff=None, reasoning="Empty response"))
                continue
            try:
                m = re.search(r"\{[\s\S]*\}", resp.first_response)
                if not m:
                    results.append(ComparisonResult(winner=None, score_diff=None, reasoning="Parse error"))
                    continue
                data = json.loads(m.group())
                raw_winner = data.get("judgment")
                reasoning = data.get("reasoning")
                winner = ({"A": "B", "B": "A", "Tie": "Tie"}.get(raw_winner, raw_winner)
                          if flip else raw_winner)
                score_diff = {"A": 1.0, "B": -1.0, "Tie": 0.0}.get(winner)
                results.append(ComparisonResult(winner=winner, score_diff=score_diff, reasoning=reasoning))
            except Exception as e:
                results.append(ComparisonResult(winner=None, score_diff=None, reasoning=str(e)))

        return results

    def to_dict(self) -> dict[str, Any]:
        return {"model_name": self._model_name, "type": "vision_llm_judge"}


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
        """Detect whether an attribute is present in each image.

        Returns one int per image:
          1  → attribute present
          0  → attribute not present (or parse failed)
          -1 → attribute does NOT apply to the image (only when
               `use_applicability=True` and the model said `applicable=false`).
        """
        if len(image_paths) != len(prompts):
            raise ValueError("image_paths and prompts must have the same length")

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
                    # vLLM (e.g. Qwen): separate system msg, image before text
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
                        {"type": "input_text", "text": ATTRIBUTE_DETECTION_SYSTEM + "\n\n" + user_text + "\n\nImage:"},
                        {"type": "input_image", "image_url": img_url, "detail": self.image_detail},
                    ]
                    history = ChatHistory(messages=[ChatMessage(role="user", content=content)])
                chats.append(history)
            except Exception as e:
                logger.error(f"Failed to build detection chat for {img_path}: {e}")
                chats.append(None)

        valid_idx = [i for i, c in enumerate(chats) if c is not None]
        resp_map: dict[int, Any] = {}
        if valid_idx:
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
            for i, r in zip(valid_idx, responses):
                resp_map[i] = r

        results: list[int] = []
        for idx in range(len(image_paths)):
            if idx not in resp_map:
                results.append(0)
                continue
            resp = resp_map[idx]
            if resp is None or not resp.has_response:
                results.append(0)
                continue
            try:
                m = re.search(r"\{[\s\S]*\}", resp.first_response)
                if not m:
                    results.append(0)
                    continue
                data = json.loads(m.group())
                if self.use_applicability and data.get("applicable") is False:
                    results.append(-1)
                else:
                    results.append(1 if data.get("present", False) else 0)
            except Exception:
                results.append(0)

        return results

    def to_dict(self) -> dict[str, Any]:
        return {"model_name": self._model_name, "type": "vision_llm_detector"}
