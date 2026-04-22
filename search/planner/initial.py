"""Initial attribute planner: generates visual attribute hypotheses from scored baseline images."""
from __future__ import annotations
import json
from random import Random
from typing import Any

from loguru import logger

from caller import AutoCaller, ChatHistory, ChatMessage
from search.data.state import TopicState, AttributeStats, AttributeMeta, EvoStep
from search.data.types import BaselineImage
from search.prompts.planning import PLANNER_SYSTEM, LIST_PROMPT_PRE, LIST_PROMPT_POST, BIAS_NUDGE
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
        self.caller = AutoCaller(dotenv_path=".env")

    async def plan(
        self,
        topic_states: list[TopicState],
        reward_model_name: str,
    ) -> None:
        """Populate history[0] on each TopicState with initial AttributeStats."""
        to_send: list[ChatHistory] = []
        metas: list[dict[str, Any]] = []

        for topic_state in topic_states:
            rng = Random(self.random_seed + topic_state.topic_id)
            step = EvoStep(step_idx=0)
            topic_state.history.append(step)

            train_prompts = [p.text for p in topic_state.prompts]
            if self.n_initial_plan_prompts is not None:
                train_prompts = rng.sample(train_prompts, min(self.n_initial_plan_prompts, len(train_prompts)))
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
                        n_top = n - n_bottom
                        sampled = sorted_scored[:n_bottom] + sorted_scored[-n_top:]
                    else:
                        sampled = rng.sample(scored, n)
                    higher_lower = "higher-scoring" if self.direction == "plus" else "lower-scoring"
                    reverse = (self.order == "descending")
                    sampled.sort(key=lambda b: b.reward_scores[reward_model_name], reverse=reverse)
                    display_scores = [round(b.reward_scores[reward_model_name], 2) for b in sampled]

                    if self.use_cluster_summary and topic_state.cluster_summary:
                        general_constraint_block = (
                            "the feature must apply to images from ANY sensible text prompt "
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
                    pre_text = (
                        PLANNER_SYSTEM + "\n\n"
                        + LIST_PROMPT_PRE.format(
                            n_attrs_per_prompt=self.n_attrs_per_prompt,
                            higher_lower=higher_lower,
                            general_constraint_block=general_constraint_block,
                            order=self.order,
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
                        general_check_block=general_check_block,
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

        # Parse and store attributes into history[0]
        topic_state_by_id = {ts.topic_id: ts for ts in topic_states}
        for i, resp in enumerate(responses):
            if resp is None:
                continue
            meta = metas[i]
            attributes, reasoning = parse_json_response(resp)
            if i < 3:
                logger.info(f"InitialPlanner reasoning:\n{reasoning}")
            if not isinstance(attributes, list):
                logger.warning(f"InitialPlanner: response not a list (topic={meta['topic_id']})")
                continue
            attributes = [str(a).strip() for a in attributes if a]

            ts = topic_state_by_id[meta["topic_id"]]
            step = ts.history[0]
            for attr in attributes:
                if attr not in step.attributes:
                    step.attributes[attr] = AttributeStats(
                        attribute=attr,
                        meta=AttributeMeta(
                            time_step=0,
                            parent=None,
                            parent_time_step=None,
                            operation="initial",
                            planner_model=meta["planner_model"],
                            reasoning_effort=meta["reasoning_effort"],
                            planner_prompt=meta["planner_prompt"],
                            planner_reasoning=str(reasoning),
                        ),
                    )

        for ts in topic_states:
            n = len(ts.history[0].attributes) if ts.history else 0
            logger.info(f"Topic {ts.topic_id}: {n} initial attributes generated")
