"""Initial attribute planner: generates visual attribute hypotheses from scored baseline images."""
from __future__ import annotations
import json
from random import Random
from typing import Any

from loguru import logger

from caller import AutoCaller, ChatHistory, ChatMessage
from caller.cache import CacheConfig
from search.data.state import TopicState, AttributeStats, AttributeMeta, EvoStep
from search.data.types import BaselineImage
from search.prompts.planning import (
    PLANNER_SYSTEM, LIST_PROMPT_PRE, LIST_PROMPT_POST,
    LIST_PROMPT_PRE_MULTI, LIST_PROMPT_POST_MULTI,
    BIAS_NUDGE, BIAS_CHECK,
    EDITABLE_LABEL, EDITABLE_DESC, EDITABLE_CHECK,
    MEASURABLE_LABEL, MEASURABLE_DESC, MEASURABLE_CHECK,
)
from search.utils.io import parse_json_response


class InitialPlanner:
    """Generates initial attribute hypotheses from scored baseline images (step 0)."""

    def __init__(
        self,
        model_name: str = "openai/gpt-5",
        reasoning: str | None = "high",
        max_tokens: int = 50000,
        max_parallel: int = 64,
        n_attrs_per_prompt: int = 4,
        n_per_user_prompt: int = 1,
        n_context_imgs: int = 16,
        n_initial_plan_prompts: int | None = None,
        initial_context_sampling: str = "random",
        use_cluster_summary: bool = True,
        direction: str = "plus",
        order: str = "descending",
        random_seed: int = 42,
        require_editable: bool = True,
        n_prompts_per_plan_call: int = 1,
        cache_config: CacheConfig | None = None,
    ):
        self.model_name = model_name
        self.reasoning = reasoning
        self.max_tokens = max_tokens
        self.max_parallel = max_parallel
        self.n_attrs_per_prompt = n_attrs_per_prompt
        self.n_per_user_prompt = n_per_user_prompt
        self.n_context_imgs = n_context_imgs
        self.n_initial_plan_prompts = n_initial_plan_prompts
        self.initial_context_sampling = initial_context_sampling
        self.use_cluster_summary = use_cluster_summary
        self.direction = direction
        self.order = order
        self.random_seed = random_seed
        self.require_editable = require_editable
        self.n_prompts_per_plan_call = n_prompts_per_plan_call
        self.caller = AutoCaller(dotenv_path=".env", cache_config=cache_config)

    async def plan(
        self,
        topic_states: list[TopicState],
        reward_model_name: str,
        fixed_prompts: "dict[int, list[str]] | None" = None,
    ) -> None:
        """Populate history[0] on each TopicState with initial AttributeStats.

        fixed_prompts: if provided, {topic_id: [prompt_text, ...]} overrides the
        random sampling of train_prompts so InitialPlanner sees the exact same
        prompts used for OLS evaluation (fixed baselines).
        """
        await self._plan_impl(
            topic_states, reward_model_name,
            target_step_idx=None, fixed_prompts=fixed_prompts,
        )

    async def plan_into_step(
        self,
        topic_states: list[TopicState],
        step_idx: int,
        reward_model_name: str,
    ) -> None:
        """Add freshly planned attributes into an existing step (no new EvoStep created)."""
        await self._plan_impl(topic_states, reward_model_name, target_step_idx=step_idx)

    async def _plan_impl(
        self,
        topic_states: list[TopicState],
        reward_model_name: str,
        target_step_idx: int | None,
        fixed_prompts: "dict[int, list[str]] | None" = None,
    ) -> None:
        """Shared implementation for plan() and plan_into_step().

        target_step_idx=None: create a new EvoStep at index 0 (initial planning).
        target_step_idx=N:    add candidates into ts.history[N] (replan fallback).
        """
        to_send: list[ChatHistory] = []
        metas: list[dict[str, Any]] = []

        for topic_state in topic_states:
            rng = Random(self.random_seed + topic_state.topic_id)
            if target_step_idx is None:
                step = EvoStep(step_idx=0)
                topic_state.history.append(step)
            else:
                step = topic_state.history[target_step_idx]

            if fixed_prompts is not None and topic_state.topic_id in fixed_prompts:
                train_prompts = fixed_prompts[topic_state.topic_id]
            else:
                train_prompts = [p.text for p in topic_state.prompts]
                if self.n_initial_plan_prompts is not None:
                    train_prompts = rng.sample(train_prompts, min(self.n_initial_plan_prompts, len(train_prompts)))

            higher_lower = "higher-scoring" if self.direction == "plus" else "lower-scoring"
            reverse = (self.order == "descending")
            if self.use_cluster_summary and topic_state.cluster_summary:
                general_constraint_block = (
                    "the feature must be applicable to images from ANY sensible text prompt "
                    "in this cluster:\n"
                    "      <user_prompt_cluster_summary>\n"
                    f"      {topic_state.cluster_summary}\n"
                    "      </user_prompt_cluster_summary>"
                )
                general_check_block = "applies to ANY image in this cluster"
            else:
                general_constraint_block = (
                    "the feature must be applicable to any image, "
                    "regardless of its specific subject or scene."
                )
                general_check_block = "applies to any image regardless of subject or scene"
            em_label = EDITABLE_LABEL if self.require_editable else MEASURABLE_LABEL
            em_desc  = EDITABLE_DESC  if self.require_editable else MEASURABLE_DESC
            em_check = EDITABLE_CHECK if self.require_editable else MEASURABLE_CHECK

            if self.n_prompts_per_plan_call > 1:
                # ── Multi-prompt mode ────────────────────────────────────────
                chunk_size = self.n_prompts_per_plan_call
                chunks = [train_prompts[i:i + chunk_size] for i in range(0, len(train_prompts), chunk_size)]
                for chunk in chunks:
                    pre_text = (
                        PLANNER_SYSTEM + "\n\n"
                        + LIST_PROMPT_PRE_MULTI.format(
                            n_groups=len(chunk),
                            n_attrs_per_prompt=self.n_attrs_per_prompt,
                            higher_lower=higher_lower,
                            general_constraint_block=general_constraint_block,
                            order=self.order,
                            bias_nudge=BIAS_NUDGE[self.direction],
                            editable_or_measurable_label=em_label,
                            editable_or_measurable_desc=em_desc,
                        )
                    )
                    content: list[dict] = [{"type": "input_text", "text": pre_text}]
                    for g_idx, prompt_text in enumerate(chunk, 1):
                        baselines = topic_state.baselines.get(prompt_text, [])
                        scored = [b for b in baselines if reward_model_name in b.reward_scores]
                        if not scored:
                            continue
                        n = min(self.n_context_imgs, len(scored))
                        if self.initial_context_sampling == "stratified":
                            sorted_scored = sorted(scored, key=lambda b: b.reward_scores[reward_model_name])
                            n_bottom = n // 2
                            sampled = sorted_scored[:n_bottom] + sorted_scored[-(n - n_bottom):]
                        else:
                            sampled = rng.sample(scored, n)
                        sampled.sort(key=lambda b: b.reward_scores[reward_model_name], reverse=reverse)
                        content.append({"type": "input_text",
                                        "text": f'\n## Group {g_idx}\n**Prompt:** "{prompt_text}"\n'})
                        for baseline in sampled:
                            score = round(baseline.reward_scores[reward_model_name], 2)
                            try:
                                img_url = ChatMessage.image_to_base64_url(str(baseline.image_path))
                            except Exception as e:
                                logger.warning(f"Failed to load image {baseline.image_path}: {e}")
                                continue
                            content.append({"type": "input_image", "image_url": img_url, "detail": "auto"})
                            content.append({"type": "input_text", "text": f"Score: {score}"})
                    post_text = LIST_PROMPT_POST_MULTI.format(
                        n_attrs_per_prompt=self.n_attrs_per_prompt,
                        bias_check=BIAS_CHECK[self.direction],
                        editable_or_measurable_check=em_check,
                    )
                    content.append({"type": "input_text", "text": post_text})
                    to_send.append(ChatHistory(messages=[ChatMessage(role="user", content=content)]))
                    metas.append({
                        "topic_id": topic_state.topic_id,
                        "prompt_text": chunk,
                        "planner_prompt": pre_text + "[images]" + post_text,
                        "planner_model": self.model_name,
                        "reasoning_effort": str(self.reasoning),
                    })
            else:
                # ── Single-prompt mode (original behaviour) ──────────────────
                for prompt_text in train_prompts:
                    baselines = topic_state.baselines.get(prompt_text, [])
                    scored = [b for b in baselines if reward_model_name in b.reward_scores]
                    if not scored:
                        continue
                    for _ in range(self.n_per_user_prompt):
                        n = min(self.n_context_imgs, len(scored))
                        if self.initial_context_sampling == "stratified":
                            sorted_scored = sorted(scored, key=lambda b: b.reward_scores[reward_model_name])
                            n_bottom = n // 2
                            sampled = sorted_scored[:n_bottom] + sorted_scored[-(n - n_bottom):]
                        else:
                            sampled = rng.sample(scored, n)
                        sampled.sort(key=lambda b: b.reward_scores[reward_model_name], reverse=reverse)
                        display_scores = [round(b.reward_scores[reward_model_name], 2) for b in sampled]
                        pre_text = (
                            PLANNER_SYSTEM + "\n\n"
                            + LIST_PROMPT_PRE.format(
                                n_attrs_per_prompt=self.n_attrs_per_prompt,
                                higher_lower=higher_lower,
                                general_constraint_block=general_constraint_block,
                                order=self.order,
                                bias_nudge=BIAS_NUDGE[self.direction],
                                editable_or_measurable_label=em_label,
                                editable_or_measurable_desc=em_desc,
                            )
                            + f"\n\nUser prompt: {prompt_text}\n"
                        )
                        content: list[dict] = [{"type": "input_text", "text": pre_text}]
                        for baseline, score in zip(sampled, display_scores):
                            try:
                                img_url = ChatMessage.image_to_base64_url(str(baseline.image_path))
                            except Exception as e:
                                logger.warning(f"Failed to load image {baseline.image_path}: {e}")
                                continue
                            content.append({"type": "input_image", "image_url": img_url, "detail": "auto"})
                            content.append({"type": "input_text", "text": f"Score: {score}"})
                        post_text = LIST_PROMPT_POST.format(
                            n_attrs_per_prompt=self.n_attrs_per_prompt,
                            higher_lower=higher_lower,
                            bias_nudge=BIAS_NUDGE[self.direction],
                            bias_check=BIAS_CHECK[self.direction],
                            general_check_block=general_check_block,
                            editable_or_measurable_check=em_check,
                        )
                        content.append({"type": "input_text", "text": post_text})
                        to_send.append(ChatHistory(messages=[ChatMessage(role="user", content=content)]))
                        metas.append({
                            "topic_id": topic_state.topic_id,
                            "prompt_text": prompt_text,
                            "planner_prompt": pre_text + "[images]" + post_text,
                            "planner_model": self.model_name,
                            "reasoning_effort": str(self.reasoning),
                        })

        if not to_send:
            logger.warning("InitialPlanner: no messages to send (no scored baselines?)")
            return

        responses = await self.caller.call(
            messages=to_send,
            model=self.model_name,
            max_parallel=self.max_parallel,
            max_tokens=self.max_tokens,
            reasoning=self.reasoning,
            enable_cache=False,
            desc="Initial planning",
        )

        # Parse and store attributes into the target step
        operation = "initial" if target_step_idx is None else "replan"
        desc = "initial attributes" if target_step_idx is None else "replan attributes"
        topic_state_by_id = {ts.topic_id: ts for ts in topic_states}
        for i, resp in enumerate(responses):
            if resp is None:
                continue
            meta = metas[i]
            attributes, reasoning = parse_json_response(resp)
            prompt_label = (
                ", ".join(f'"{p[:40]}"' for p in meta["prompt_text"])
                if isinstance(meta["prompt_text"], list)
                else meta["prompt_text"]
            )
            if reasoning is not None:
                logger.info(f"InitialPlanner reasoning -- topic {meta['topic_id']} (index {i}):\n{reasoning}")
            if not isinstance(attributes, list):
                logger.warning(f"InitialPlanner: response not a list (topic={meta['topic_id']})")
                continue
            attributes = [str(a).strip() for a in attributes if a]
            if attributes:
                logger.info(f"InitialPlanner: topic {meta['topic_id']} (index {i}) [{prompt_label}] proposed attributes: {attributes}")

            ts = topic_state_by_id[meta["topic_id"]]
            sidx = 0 if target_step_idx is None else target_step_idx
            step = ts.history[sidx]
            for attr in attributes:
                if attr not in step.attributes:
                    step.attributes[attr] = AttributeStats(
                        attribute=attr,
                        meta=AttributeMeta(
                            time_step=sidx,
                            parent=None,
                            parent_time_step=None,
                            operation=operation,
                            planner_model=meta["planner_model"],
                            reasoning_effort=meta["reasoning_effort"],
                            planner_prompt=meta["planner_prompt"],
                            planner_reasoning=str(reasoning),
                        ),
                    )

        for ts in topic_states:
            sidx = 0 if target_step_idx is None else target_step_idx
            n = len(ts.history[sidx].attributes) if ts.history else 0
            logger.info(f"Topic {ts.topic_id}: {n} {desc} generated")
