"""
Main Caller class.
"""

import io
import json
import os
import time
import random
import asyncio
import requests
from pathlib import Path
from loguru import logger
from typing import Sequence, Optional, Callable, Type
from json import JSONDecodeError
from abc import ABC, abstractmethod
from tqdm.asyncio import tqdm_asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

import openai
import anthropic
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from .types import (
    Tool,
    ToolChoice,
    NonStreamingChoice,
    ResponseFormat,
    ChatMessage,
    ChatHistory,
    InferenceConfig,
    Request,
    Response,
)
from .cache import CacheConfig, Cache


class CriteriaNotSatisfiedError(Exception):
    def __init__(self, message: str = "Criteria provided is not satisfied"):
        super().__init__(message)


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    raise_when_exhausted: bool = True  
    # raise an exception when all retry attempts are exhausted
    # the alternate is to return the last response obtained,
    # or None if all of the errors were API exceptions

    max_attempts: int = 8  # Maximum number of retry attempts
    min_wait_seconds: float = 1.0  # Minimum wait time between retries
    max_wait_seconds: float = 30.0  # Maximum wait time between retries
    multiplier: float = 2.0  # Exponential backoff multiplier

    criteria: Optional[Callable[[Response], bool]]=None  # criteria that must be satisfied
    retryable_exceptions: tuple[Type[Exception], ...] = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
        openai.PermissionDeniedError,
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic._exceptions.OverloadedError,
        JSONDecodeError,
        ValidationError,
    )


class CallerBaseClass(ABC):
    def __init__(
        self, cache_config: Optional[CacheConfig] = None, retry_config: Optional[RetryConfig] = None
    ) -> None:
        self.cache_config = cache_config or CacheConfig()
        self.retry_config = retry_config or RetryConfig()

        if self.cache_config.base_path is not None:
            self.cache_dir = Path(self.cache_config.base_path)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Each model has its own Cache
            self.model_caches: dict[str, Cache[Response]] = dict()
            self._cache_lock = asyncio.Lock()  # Lock for cache creation
        else:
            self.cache_dir = None  # caching disabled

    async def shutdown(self):
        """Flush pending cache writes and close connections."""
        if self.cache_dir is not None:
            for cache in self.model_caches.values():
                await cache.shutdown()

    async def _get_cache(self, model: str) -> Cache[Response]:
        """Get or create cache for a model."""
        if model in self.model_caches:
            return self.model_caches[model]
        async with self._cache_lock:  # Ensure only one cache is created per model
            if model not in self.model_caches:
                safe_model_name = model.replace("/", "_")
                self.model_caches[model] = Cache(
                    safe_model_name=safe_model_name,
                    response_type=Response,
                    cache_config=self.cache_config,
                )
            return self.model_caches[model]

    @abstractmethod
    async def _call(self, request: Request) -> Response:
        pass

    async def _call_with_retry(self, request: Request) -> Response|None:
        """
        Wraps _call() with automatic retry on transient errors.
        Uses jittered exponential backoff configured via retry_config.
        """
        wait_time = self.retry_config.min_wait_seconds

        for attempt in range(self.retry_config.max_attempts):
            logger.debug(
                f"Attempt {attempt + 1}/{self.retry_config.max_attempts} to call {request.model}"
            )
            response = None
            try:
                response = await self._call(request)
                if self.retry_config.criteria is not None:
                    if not self.retry_config.criteria(response):
                        finish_reason = None
                        error = None
                        if response.choices and len(response.choices) > 0:
                            finish_reason = response.choices[0].finish_reason
                            error = response.choices[0].error
                        raise CriteriaNotSatisfiedError(
                            f"Criteria provided is not satisfied for response. "
                            f"Reason: {finish_reason}; "
                            f"Error: {error}"
                        ) 
                return response

            except (*self.retry_config.retryable_exceptions, CriteriaNotSatisfiedError) as e:
                if attempt < self.retry_config.max_attempts - 1:
                    logger.warning(
                        f"Retryable error on attempt {attempt + 1}/{self.retry_config.max_attempts}: "
                        f"{type(e).__name__}: {str(e)}. Waiting {wait_time:.1f}s before retry."
                    )
                    await asyncio.sleep(wait_time + random.uniform(0, 1))
                    wait_time = min(
                        wait_time * self.retry_config.multiplier, self.retry_config.max_wait_seconds
                    )
                else:
                    logger.error(f"All {self.retry_config.max_attempts} retry attempts exhausted")
                    if self.retry_config.raise_when_exhausted:
                        raise
                    else:
                        return response

    async def call_one(
        self,
        messages: ChatHistory | Sequence[ChatMessage] | str,
        model: str,
        enable_cache: bool = True,
        response_format: Optional[dict] = None,  # pass in the desired json schema
        stop: Optional[list[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        reasoning: Optional[str | int] = None,
        tools: Optional[list[Tool]] = None,
        tool_choice: Optional[ToolChoice] = None,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        min_p: Optional[float] = None,
        top_a: Optional[float] = None,
        logit_bias: Optional[dict[int, float]] = None,
        top_logprobs: Optional[int] = None,
        extra_body: Optional[dict] = None,
    ) -> Response|None:
        """
        Make a single async API call.
        """
        if isinstance(messages, str):
            messages = ChatHistory.from_user(messages)
        elif not isinstance(messages, ChatHistory):
            messages = ChatHistory(messages=messages)

        config = InferenceConfig(
            response_format=(
                ResponseFormat(type="json_schema", json_schema=response_format)
                if response_format is not None
                else None
            ),
            stop=stop,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            seed=seed,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_a=top_a,
            logit_bias=logit_bias,
            top_logprobs=top_logprobs,
            reasoning=reasoning,
            extra_body=extra_body,
        )

        should_cache = (
            enable_cache
            and (model not in self.cache_config.no_cache_models)
            and (self.cache_dir is not None)
        )

        if should_cache:
            cache = await self._get_cache(model)
            cached_response = await cache.get_entry(messages=messages, config=config)
            if cached_response:
                logger.debug(f"Cache hit for model {model}")
                return cached_response

        response = await self._call_with_retry(
            Request(
                model=model,
                messages=messages,
                config=config,
            )
        )

        if should_cache and response is not None and response.has_response and response.finish_reason == "stop":
            assert self.cache_dir is not None
            cache = await self._get_cache(model)
            await cache.put_entry(
                messages=messages,
                config=config,
                response=response,
            )

        return response

    async def call(
        self,
        messages: Sequence[ChatHistory | Sequence[ChatMessage] | str],
        model: str | list[str],
        max_parallel: int,
        desc: Optional[str] = None,
        **kwargs,
    ) -> list[Response|None]:
        """
        Make multiple async API calls in parallel.

        Satisfies: len(output) == len(messages)

        See call_one for possible kwargs.
        """
        if not messages:
            return []

        if isinstance(model, str):
            tasks = [{"messages": msg, "model": model, **kwargs} for msg in messages]
        else:
            assert len(model) == len(messages), "Number of models must match number of messages"
            tasks = [{"messages": msg, "model": model_name, **kwargs} for msg, model_name in zip(messages, model)]
        sem = asyncio.Semaphore(max_parallel)

        async def call_one_with_sem(task: dict) -> Response|None:
            async with sem:
                return await self.call_one(**task)

        if desc is not None:
            responses = await tqdm_asyncio.gather(*[call_one_with_sem(task) for task in tasks], desc=desc)
        else:
            responses = await asyncio.gather(*[call_one_with_sem(task) for task in tasks])
        return responses



class OpenRouterCaller(CallerBaseClass):

    def __init__(
        self,
        api_key: Optional[str] = None,
        dotenv_path: Optional[str | Path] = None,
        cache_config: Optional[CacheConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        super().__init__(cache_config=cache_config, retry_config=retry_config)
        load_dotenv(dotenv_path)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=1800.0,   # 30 min — reasoning models with many images can be slow
        )
        if self.api_key is None:
            raise ValueError("api_key not provided and OPENROUTER_API_KEY not found in .env")

    def check_model_support(self, model_name: str, property: str) -> bool:
        """
        Check if a model supports various parameters.
        e.g. reasoning, tools, structured responses.
        """

        split_model_name = model_name.split("/")
        assert len(split_model_name) == 2, "Model name must be in the format of author/slug"
        author, slug = split_model_name

        url = f"https://openrouter.ai/api/v1/parameters/{author}/{slug}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(url, headers=headers)

        assert response.json()["data"]["model"] == model_name, "Model name does not match"
        return property in response.json()["data"]["supported_parameters"]

    async def _call(self, request: Request) -> Response:
        request_body = request.to_openrouter_request()
        if request_body["extra_body"] is None:
            request_body["extra_body"] = {}

        # Ask for the provider to support all parameters specified in the request
        request_body["extra_body"].setdefault("provider", {}).setdefault("require_parameters", True)

        # Provider-specific routing (to avoid unreliable providers)
        if "order" not in request_body["extra_body"]["provider"]:
            if request.model == "meta-llama/llama-3.1-8b-instruct":
                request_body["extra_body"]["provider"].update(
                    {
                        "order": ["groq", "deepinfra/turbo", "novita/fp8"],
                        "allow_fallbacks": False,
                    }
                )
            elif request.model == "meta-llama/llama-3.1-70b-instruct":
                request_body["extra_body"]["provider"].update(
                    {
                        "order": ["hyperbolic/fp8", "together/fp8"],
                        "allow_fallbacks": False,
                    }
                )
            elif request.model == "meta-llama/llama-3.2-3b-instruct":
                request_body["extra_body"]["provider"].update(
                    {
                        "order": ["cloudflare", "together/fp8", "hyperbolic/fp8"],
                        "allow_fallbacks": False,
                    }
                )
            elif request.model == "qwen/qwen-2.5-7b-instruct":
                request_body["extra_body"]["provider"].update(
                    {
                        "order": ["together/fp8", "phala"],
                        "allow_fallbacks": False,
                    }
                )
            elif request.model == "qwen/qwen-2.5-72b-instruct":
                request_body["extra_body"]["provider"].update(
                    {
                        "order": ["deepinfra/fp8", "together/fp8"],
                        "allow_fallbacks": False,
                    }
                )

        request_body_to_pass = {k: v for k, v in request_body.items() if v is not None}
        try:
            # logger.info(f"Sent an API call to model: {request.model}")
            chat_completion = await self.client.chat.completions.create(**request_body_to_pass)
        except Exception as e:
            raise RuntimeError(f"Model: {request.model}. OpenRouter API error: {e}") from e

        return Response.model_validate(chat_completion.model_dump())


class OpenAICaller(CallerBaseClass):
    def __init__(
        self,
        api_key: Optional[str] = None,
        dotenv_path: Optional[str | Path] = None,
        cache_config: Optional[CacheConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        super().__init__(cache_config=cache_config, retry_config=retry_config)
        load_dotenv(dotenv_path)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            timeout=1800.0,   # 30 min — reasoning models with many images can be slow
        )
        if self.api_key is None:
            raise ValueError("api_key not provided and OPENAI_API_KEY not found in .env")

    @staticmethod
    def openai_response_to_unified(openai_resp: dict) -> Response:
        choices = []
        reasoning_details = None

        for idx, output_item in enumerate(openai_resp.get("output", [])):
            if output_item.get("type") not in ["message", "reasoning"]:
                raise ValueError(f"Unexpected output type: {output_item.get('type')}")

            if output_item.get("type") == "reasoning":
                if not output_item.get("summary", None):
                    continue
                output_item["summary"][0]["type"] = "reasoning.text"
                if reasoning_details is None:
                    reasoning_details = []
                reasoning_details.append(output_item["summary"][0])
                # print(reasoning_details)
                
            if output_item.get("type") == "message":
                # Extract the text content from the message
                content_items = output_item.get("content", [])
                text = ""
                for content in content_items:
                    if content.get("type") == "output_text":
                        text = content.get("text", "")
                        break  # Use first text content

                # Normalize to OpenRouter's finish_reason values
                native_status = output_item.get("status")     
                finish_reason_map = {
                    "completed": "stop",
                    "incomplete": "length",
                    "failed": "error",
                    "cancelled": "error"
                }
                finish_reason = finish_reason_map.get(native_status, "error")
                
                choice = NonStreamingChoice(
                    message={
                        "role": output_item.get("role"),
                        "content": text,
                    },
                    finish_reason=finish_reason,
                    native_finish_reason=native_status,
                    error=openai_resp.get("error"),
                )
                if reasoning_details is not None:
                    choice.message["reasoning_details"] = reasoning_details
                choices.append(choice)
        
        return Response(
            id=openai_resp["id"],
            choices=choices,
            created=openai_resp["created_at"],
            model=openai_resp["model"],
            system_fingerprint=openai_resp.get("system_fingerprint"),
            usage=openai_resp.get("usage", {}),
            **{k: v for k, v in openai_resp.items() if k not in {"id", "created_at", "model", "output", "usage", "error"}}
        )
    

    async def _call(self, request: Request) -> Response:
        request_body = request.to_openai_request()
        request_body_to_pass = {k: v for k, v in request_body.items() if v is not None}
        try:
            # logger.info(f"Sent an API call to model: {request.model}")
            response = await self.client.responses.create(**request_body_to_pass)
        except Exception as e:
            logger.debug(f"Model: {request.model}. OpenAI API error: {e}")
            raise
        
        return self.openai_response_to_unified(response.model_dump())


    async def call_batch(
        self,
        messages: "Sequence[ChatHistory]",
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        poll_interval: int = 60,
        **kwargs,
    ) -> "list[Response | None]":
        """Submit requests to the OpenAI Batch API (50% price discount, async).

        The method blocks (with asyncio.sleep) until the batch completes or fails.
        Only Chat-Completions-compatible parameters are forwarded; reasoning and
        other Responses-API-only params are intentionally ignored.

        Args:
            messages:      One ChatHistory per request.
            model:         Model name WITHOUT the "openai/" prefix.
            max_tokens:    Max output tokens per request.
            temperature:   Sampling temperature.
            poll_interval: Seconds between status-poll requests (default 60).

        Returns:
            list[Response | None] in the same order as *messages*.
            None for individual requests that errored inside the batch.
        """
        if not messages:
            return []

        # ── 1. Build JSONL ────────────────────────────────────────────────────
        cfg_kwargs: dict = {}
        if max_tokens is not None:
            cfg_kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            cfg_kwargs["temperature"] = temperature

        lines: list[str] = []
        for i, msg in enumerate(messages):
            request = Request(
                model=model,
                messages=msg if isinstance(msg, ChatHistory) else ChatHistory(messages=msg),
                config=InferenceConfig(**cfg_kwargs),
            )
            lines.append(json.dumps(request.to_batch_item(f"req-{i}")))

        # ── 2. Split into chunks ≤ MAX_FILE_BYTES (OpenAI limit: 100 MB) ──────
        _MAX_FILE_BYTES = 90 * 1024 * 1024  # 90 MB safety margin
        chunks: list[list[str]] = []
        current: list[str] = []
        current_bytes = 0
        for line in lines:
            lb = len(line.encode("utf-8")) + 1  # +1 for newline
            if current and current_bytes + lb > _MAX_FILE_BYTES:
                chunks.append(current)
                current, current_bytes = [line], lb
            else:
                current.append(line)
                current_bytes += lb
        if current:
            chunks.append(current)

        logger.info(
            f"Batch: {len(lines)} requests split into {len(chunks)} chunk(s) "
            f"(limit={_MAX_FILE_BYTES // 1024 // 1024} MB each)"
        )

        # ── 3. Submit + poll + parse each chunk (concurrently) ────────────────
        async def _run_chunk(chunk_lines: list[str]) -> dict[str, "Response | None"]:
            jsonl_bytes = "\n".join(chunk_lines).encode("utf-8")
            file_obj = await self.client.files.create(
                file=("batch_input.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
                purpose="batch",
            )
            batch = await self.client.batches.create(
                input_file_id=file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            logger.info(f"Batch chunk {batch.id}: {len(chunk_lines)} requests submitted")

            terminal = {"completed", "failed", "expired", "cancelled"}
            while batch.status not in terminal:
                await asyncio.sleep(poll_interval)
                batch = await self.client.batches.retrieve(batch.id)
                counts = batch.request_counts
                logger.info(
                    f"Batch {batch.id}: status={batch.status}  "
                    f"completed={counts.completed}/{counts.total}  failed={counts.failed}"
                )

            partial: dict[str, "Response | None"] = {}
            if batch.status != "completed":
                logger.error(f"Batch {batch.id} ended with status={batch.status!r}")
                for line in chunk_lines:
                    cid = json.loads(line).get("custom_id", "")
                    partial[cid] = None
            else:
                output_text = (await self.client.files.content(batch.output_file_id)).text
                for line in output_text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    cid = item.get("custom_id", "")
                    if item.get("error") or not item.get("response"):
                        logger.warning(f"Batch request {cid} failed: {item.get('error')}")
                        partial[cid] = None
                    else:
                        try:
                            partial[cid] = Response.model_validate(item["response"]["body"])
                        except Exception as exc:
                            logger.warning(f"Batch {cid}: parse error: {exc}")
                            partial[cid] = None

            for fid in (file_obj.id, getattr(batch, "output_file_id", None)):
                if fid:
                    try:
                        await self.client.files.delete(fid)
                    except Exception:
                        pass
            return partial

        chunk_results = await asyncio.gather(*[_run_chunk(chunk) for chunk in chunks])

        result_map: dict[str, "Response | None"] = {}
        for partial in chunk_results:
            result_map.update(partial)

        return [result_map.get(f"req-{i}") for i in range(len(messages))]


class AnthropicCaller(CallerBaseClass):
    def __init__(
        self,
        api_key: Optional[str] = None,
        dotenv_path: Optional[str | Path] = None,
        cache_config: Optional[CacheConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        super().__init__(cache_config=cache_config, retry_config=retry_config)
        load_dotenv(dotenv_path)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = AsyncAnthropic(
            api_key=self.api_key,
            timeout=1800.0,   # 30 min — reasoning models with many images can be slow
        )
        if self.api_key is None:
            raise ValueError("api_key not provided and ANTHROPIC_API_KEY not found in .env")

    @staticmethod
    def anthropic_response_to_unified(anthropic_resp: dict) -> Response:
        choices = []
        reasoning_details = None

        for idx, output_item in enumerate(anthropic_resp.get("content", [])):
            if output_item.get("type") not in ["text", "thinking", "redacted_thinking"]:
                raise ValueError(f"Unexpected output type: {output_item.get('type')}")

            if output_item.get("type") == "thinking":
                output_item["type"] = "reasoning.summary"
                if not output_item.get("thinking", None):
                    continue
                output_item["summary"] = output_item["thinking"]
                if reasoning_details is None:
                    reasoning_details = []
                reasoning_details.append(output_item)
                # print(reasoning_details)
                
            if output_item.get("type") == "text":
                # Extract the text content from the message
                text = output_item.get("text", "")

                # Normalize to OpenRouter's finish_reason values
                finish_reason_map = {
                    "end_turn": "stop",
                    "stop_sequence": "stop",
                    "max_tokens": "length",
                    "tool_use": "tool_calls",
                    "refusal": "content_filter",
                }
                finish_reason = finish_reason_map.get(anthropic_resp.get("stop_reason", ""), "error")
                
                choice = NonStreamingChoice(
                    message={
                        "role": anthropic_resp.get("role"),
                        "content": text,
                    },
                    finish_reason=finish_reason,
                    native_finish_reason=anthropic_resp.get("stop_reason"),
                    error=None,
                )
                if reasoning_details is not None:
                    choice.message["reasoning_details"] = reasoning_details
                choices.append(choice)
        
        return Response(
            id=anthropic_resp["id"],
            choices=choices,
            created=int(time.time()),  # possibly should default to 0
            model=anthropic_resp["model"],
            system_fingerprint=anthropic_resp.get("system_fingerprint"),
            usage=anthropic_resp.get("usage", {}),
            **{k: v for k, v in anthropic_resp.items() if k not in {"id", "model", "usage"}}
        )
    

    async def _call(self, request: Request) -> Response:
        request_body = request.to_anthropic_request()
        request_body_to_pass = {k: v for k, v in request_body.items() if v is not None}
        try:
            # logger.info(f"Sent an API call to model: {request.model}")
            response = await self.client.messages.create(**request_body_to_pass)
        except Exception as e:
            logger.debug(f"Model: {request.model}. Anthropic API error: {e}")
            raise
        
        return self.anthropic_response_to_unified(response.model_dump())


class AutoCaller:
    """
    Exposes only the :meth:`call` and :meth:`call_one` methods, 
    which automatically selects the appropriate caller to use.
    """
    def __init__(self,
        dotenv_path: Optional[str | Path] = None,
        cache_config: Optional[CacheConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        force_caller: Optional[str] = None,
    ):
        self.dotenv_path = dotenv_path
        self.cache_config = cache_config
        self.retry_config = retry_config
        self.force_caller = force_caller

        self._openai_caller = None
        self._anthropic_caller = None
        self._openrouter_caller = None
        self._tried_openai = False
        self._tried_anthropic = False
        self._tried_openrouter = False

        self.anthropic_model_mapping = {
            "anthropic/claude-sonnet-4.5": "claude-sonnet-4-5",
            "anthropic/claude-opus-4.1": "claude-opus-4-1",
            "anthropic/claude-haiku-4.5": "claude-haiku-4-5",
            "anthropic/claude-opus-4": "claude-opus-4-20250514",
            "anthropic/claude-sonnet-4": "claude-sonnet-4-20250514",
            "anthropic/claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
            "anthropic/claude-3.5-haiku": "claude-3-5-haiku-20241022",
            "anthropic/claude-3-haiku": "claude-3-haiku-20240307",
        }

    def _get_caller(self, name: str):
        """Lazily initialize and return the requested caller, or None if unavailable."""
        if self.force_caller is not None and name != self.force_caller:
            return None

        attr = f"_{name}_caller"
        sentinel = f"_tried_{name}"

        if getattr(self, sentinel, False):
            return getattr(self, attr)

        setattr(self, sentinel, True)

        caller_cls = {
            "openai": OpenAICaller,
            "anthropic": AnthropicCaller,
            "openrouter": OpenRouterCaller,
        }[name]

        try:
            caller = caller_cls(
                dotenv_path=self.dotenv_path,
                cache_config=self.cache_config,
                retry_config=self.retry_config,
            )
            setattr(self, attr, caller)
            return caller
        except ValueError as e:
            logger.warning(f"Could not initialize {name} caller: {e}")
            return None

    async def call(
        self,
        messages: Sequence[ChatHistory | Sequence[ChatMessage] | str],
        model: str | list[str],
        max_parallel: int,
        desc: Optional[str] = None,
        **kwargs,
    ) -> list[Response|None]:
        """Satisfies: len(output) == len(messages)"""
        if not messages:
            return []

        if isinstance(model, str):
            tasks = [{"messages": msg, "model": model, **kwargs} for msg in messages]
        else:
            assert len(model) == len(messages), "Number of models must match number of messages"
            tasks = [{"messages": msg, "model": model_name, **kwargs} for msg, model_name in zip(messages, model)]
        sem = asyncio.Semaphore(max_parallel)

        async def call_one_with_sem(task: dict) -> Response|None:
            async with sem:
                return await self.call_one(**task)

        if desc is not None:
            responses = await tqdm_asyncio.gather(*[call_one_with_sem(task) for task in tasks], desc=desc)
        else:
            responses = await asyncio.gather(*[call_one_with_sem(task) for task in tasks])
        return responses

    async def call_one(
        self,
        messages: ChatHistory | Sequence[ChatMessage] | str,
        model: str,
        **kwargs,
    ) -> Response|None:
        if model.startswith("openai/"):
            caller = self._get_caller("openai")
            if caller is not None:
                model_stripped = model.removeprefix("openai/")
                return await caller.call_one(messages=messages, model=model_stripped, **kwargs)
            else:
                caller = self._get_caller("openrouter")
                if caller is not None:
                    return await caller.call_one(messages=messages, model=model, **kwargs)
                else:
                    raise ValueError(f"No caller was found that supports the given model {model}")

        elif model.startswith("anthropic/"):
            caller = self._get_caller("anthropic")
            if caller is not None:
                model_stripped = self.anthropic_model_mapping[model]
                return await caller.call_one(messages=messages, model=model_stripped, **kwargs)
            else:
                caller = self._get_caller("openrouter")
                if caller is not None:
                    return await caller.call_one(messages=messages, model=model, **kwargs)
                else:
                    raise ValueError(f"No caller was found that supports the given model {model}")

        else:
            caller = self._get_caller("openrouter")
            if caller is not None:
                return await caller.call_one(messages=messages, model=model, **kwargs)
            else:
                raise ValueError(f"No caller was found that supports the given model {model}")

    async def call_batch(
        self,
        messages: "Sequence[ChatHistory]",
        model: str,
        **kwargs,
    ) -> "list[Response | None]":
        """Submit requests via the OpenAI Batch API (50% discount).

        Only supported for models with the "openai/" prefix. Raises ValueError otherwise.
        See OpenAICaller.call_batch for full parameter documentation.
        """
        if not model.startswith("openai/"):
            raise ValueError(
                f"Batch API is only supported for OpenAI models (prefix 'openai/'). Got: {model!r}"
            )
        caller = self._get_caller("openai")
        if caller is None:
            raise ValueError("OpenAI caller is unavailable; check OPENAI_API_KEY in .env")
        model_stripped = model.removeprefix("openai/")
        return await caller.call_batch(messages=messages, model=model_stripped, **kwargs)

    async def shutdown(self) -> None:
        """Flush cache and close connections for all initialized sub-callers."""
        for name in ("openai", "anthropic", "openrouter"):
            caller = getattr(self, f"_{name}_caller")
            if caller is not None:
                await caller.shutdown()


class LocalCaller(CallerBaseClass):
    def __init__(
        self,
        base_url: str,
        cache_config: Optional[CacheConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        super().__init__(cache_config=cache_config, retry_config=retry_config)
        self.base_url = base_url
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="EMPTY",
            timeout=1800.0,   # 30 min — large batched detection requests can be slow
        )

    async def _call(self, request: Request) -> Response:
        request_body = request.to_openrouter_request()
        # vLLM uses Chat Completions format ("text"/"image_url"), not Responses API format
        # ("input_text"/"input_image"). to_cc_content() handles this conversion.
        if isinstance(request.messages, ChatHistory):
            request_body["messages"] = request.messages.to_chat_completions_messages()
        request_body_to_pass = {k: v for k, v in request_body.items() if v is not None}
        try:
            chat_completion = await self.client.chat.completions.create(**request_body_to_pass)
        except Exception as e:
            logger.debug(f"Model: {request.model}. OpenRouter API error: {e}")
            raise

        # print(chat_completion.model_dump_json())
        return Response.model_validate(chat_completion.model_dump())
