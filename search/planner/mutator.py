"""Attribute mutator: generates variations of surviving attributes at each evo step."""
from __future__ import annotations
import json
from random import Random
from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger

from caller import AutoCaller, ChatHistory, ChatMessage
from search.data.state import TopicState, AttributeStats, AttributeMeta, EvoStep
from search.prompts.planning import PLANNER_SYSTEM
from search.prompts.mutation import (
    DIRECTION_GOAL, BIAS_NUDGE, MUTATE_PRE, MUTATE_POST_HEAD, get_post_tail,
)
from search.utils.io import parse_json_response
from search.utils.stats import remove_outliers


class AttributeMutator:
    """Generates mutations of surviving attributes using image context (step N > 0)."""

    def __init__(
        self,
        model_name: str = "openai/gpt-5",
        reasoning: str | None = "high",
        max_tokens: int = 50000,
        max_parallel: int = 64,
        n_mutations: int = 1,
        context: Literal["all", "ancestry", "vanilla"] = "ancestry",
        n_rollouts_in_context: int = 4,
        n_neighbors: int = 8,
        direction: str = "plus",
        random_seed: int = 42,
    ):
        self.model_name = model_name
        self.reasoning = reasoning
        self.max_tokens = max_tokens
        self.max_parallel = max_parallel
        self.n_mutations = n_mutations
        self.context = context
        self.n_rollouts_in_context = n_rollouts_in_context
        self.n_neighbors = n_neighbors
        self.direction = direction
        self.random_seed = random_seed
        self.caller = AutoCaller(dotenv_path=".env")

    # ──────────────────────────────────────────────────────────────────────────

    def _get_ancestry_content(
        self,
        attribute: str,
        time_step: int,
        topic_state: TopicState,
    ) -> tuple[list[str], list[dict]]:
        """Collect ancestor attributes and build multimodal ancestry content blocks."""
        ancestor_attrs: list[str] = []
        content_blocks: list[dict] = []

        current_attr = attribute
        current_step = time_step
        visited: set[str] = set()

        while True:
            step = topic_state.history[current_step]
            stats = step.attributes.get(current_attr)
            if stats is None:
                break
            parent = stats.meta.parent
            if parent is None or parent in visited:
                break
            visited.add(parent)

            # Find which step the parent was in
            parent_step_idx = stats.meta.parent_time_step
            if parent_step_idx is None or parent_step_idx >= len(topic_state.history):
                break

            parent_stats = topic_state.history[parent_step_idx].attributes.get(parent)
            if parent_stats is None:
                break

            ancestor_attrs.append(parent)

            # Add text summary of the parent
            sw = parent_stats.delta_rm()
            tw = parent_stats.delta_j()
            summary = (
                f"Ancestor: {parent}\n"
                f"Metric A uplift: {sw:.3f if sw is not None else 'N/A'}\n"
                f"Metric B uplift: {tw:.3f if tw is not None else 'N/A'}"
            )
            content_blocks.append({"type": "input_text", "text": summary})

            # Add up to 2 example image pairs from the ancestor
            all_pairs = [
                (p, prompt)
                for prompt, pairs in parent_stats.pairs.items()
                for p in pairs
                if p.delta_rm is not None and p.edited_image_path.exists() and p.baseline.image_path.exists()
            ]
            for pair, _ in all_pairs[:2]:
                try:
                    b_url = ChatMessage.image_to_base64_url(str(pair.baseline.image_path))
                    e_url = ChatMessage.image_to_base64_url(str(pair.edited_image_path))
                    content_blocks.append({"type": "input_text", "text": "Baseline:"})
                    content_blocks.append({"type": "input_image", "image_url": b_url, "detail": "auto"})
                    content_blocks.append({"type": "input_text", "text": "Edited:"})
                    content_blocks.append({"type": "input_image", "image_url": e_url, "detail": "auto"})
                    content_blocks.append({
                        "type": "input_text",
                        "text": f"Metric A: {round(pair.delta_rm, 3)}  Metric B: {round(pair.delta_j, 3) if pair.delta_j is not None else 'N/A'}",
                    })
                except Exception as e:
                    logger.warning(f"Failed to load ancestry image pair: {e}")

            current_attr = parent
            current_step = parent_step_idx

        return ancestor_attrs, content_blocks

    # ──────────────────────────────────────────────────────────────────────────

    async def mutate(
        self,
        topic_states: list[TopicState],
    ) -> None:
        """Build mutations of surviving attributes, append new EvoStep to each TopicState."""
        to_send: list[ChatHistory] = []
        msg_info: list[dict[str, Any]] = []

        for topic_state in topic_states:
            if not topic_state.history or not topic_state.surviving:
                logger.warning(f"Topic {topic_state.topic_id}: no survivors to mutate")
                continue

            rng = Random(self.random_seed + topic_state.topic_id)
            step_idx = len(topic_state.history)

            # All attributes evaluated in the last step (for neighbor data)
            last_step = topic_state.history[-1]
            last_step_attrs = [
                {"attribute": a, "delta_rm": s.delta_rm(), "delta_j": s.delta_j()}
                for a, s in last_step.attributes.items()
                if s.delta_rm() is not None and s.delta_j() is not None
            ]

            new_step = EvoStep(step_idx=step_idx)
            topic_state.history.append(new_step)

            for attribute, orig_step_idx in topic_state.surviving.items():
                orig_stats = topic_state.history[orig_step_idx].attributes.get(attribute)
                if orig_stats is None:
                    continue

                # Ancestry context
                if self.context in ("all", "ancestry"):
                    ancestor_attrs, ancestry_blocks = self._get_ancestry_content(
                        attribute, orig_step_idx, topic_state
                    )
                    exclude_set = {attribute} | set(ancestor_attrs)
                else:
                    ancestry_blocks = []
                    exclude_set = {attribute}

                # Neighbor data (only for "all")
                if self.context == "all":
                    others = [a for a in last_step_attrs if a["attribute"] not in exclude_set]
                    rng.shuffle(others)
                    neighbor_lines = [
                        f"{i+1}. {a['attribute']}\n"
                        f"   Metric A: {a['delta_rm']:.3f}  Metric B: {a['delta_j']:.3f}"
                        for i, a in enumerate(others[:self.n_neighbors])
                    ]
                    neighbor_data = "\n".join(neighbor_lines) or "No other attributes available."
                else:
                    neighbor_data = None

                # Collect valid rollout pairs for this attribute
                sw = orig_stats.delta_rm()
                tw = orig_stats.delta_j()
                current_summary = (
                    f"Attribute: {attribute}\n"
                    f"Metric A average uplift: {f'{sw:.3f}' if sw is not None else 'N/A'}\n"
                    f"Metric B average uplift: {f'{tw:.3f}' if tw is not None else 'N/A'}"
                )

                valid_pairs = [
                    (p, prompt)
                    for prompt, pairs in orig_stats.pairs.items()
                    for p in pairs
                    if p.delta_rm is not None
                    and p.edited_image_path.exists()
                    and p.baseline.image_path.exists()
                ]

                if not valid_pairs:
                    logger.warning(f"No valid pairs for attribute '{attribute}' (topic {topic_state.topic_id})")
                    continue

                # IQR outlier removal then pick top/bottom
                scores = np.array([p.delta_rm for p, _ in valid_pairs])
                if len(scores) > 1:
                    q1, q3 = np.percentile(scores, [25, 75])
                    iqr = q3 - q1
                    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    valid_pairs = [(p, pr) for (p, pr), s in zip(valid_pairs, scores) if lo <= s <= hi]

                if not valid_pairs:
                    continue

                valid_pairs.sort(key=lambda x: x[0].delta_rm)
                n = len(valid_pairs)
                half = self.n_rollouts_in_context // 2
                chosen = valid_pairs[:half] + valid_pairs[-half:] if n > self.n_rollouts_in_context else valid_pairs
                representative_prompt = chosen[0][1]

                # Build multimodal content
                pre_text = (
                    PLANNER_SYSTEM + "\n\n"
                    + MUTATE_PRE.format(
                        attribute=attribute,
                        num_plans=self.n_mutations,
                        direction_goal=DIRECTION_GOAL[self.direction],
                        bias_nudge=BIAS_NUDGE[self.direction],
                        cluster_summary=topic_state.cluster_summary,
                        current_attr_summary=current_summary,
                    )
                )
                content: list[dict] = [{"type": "input_text", "text": pre_text}]

                for pair, prompt_text in chosen:
                    metric_a = round(pair.delta_rm, 3)
                    metric_b = round(pair.delta_j, 3) if pair.delta_j is not None else "N/A"
                    try:
                        b_url = ChatMessage.image_to_base64_url(str(pair.baseline.image_path))
                        e_url = ChatMessage.image_to_base64_url(str(pair.edited_image_path))
                    except Exception as e:
                        logger.warning(f"Failed to load image pair: {e}")
                        continue
                    content.append({"type": "input_text", "text": f"Prompt: {prompt_text}\nBaseline:"})
                    content.append({"type": "input_image", "image_url": b_url, "detail": "auto"})
                    content.append({"type": "input_text", "text": "Edited:"})
                    content.append({"type": "input_image", "image_url": e_url, "detail": "auto"})
                    content.append({"type": "input_text", "text": f"Metric A: {metric_a}  Metric B: {metric_b}"})

                # Post section
                if self.context in ("all", "ancestry"):
                    content.append({"type": "input_text", "text": MUTATE_POST_HEAD})
                    content.extend(ancestry_blocks)

                post_tail = get_post_tail(self.context)
                fmt_kwargs: dict[str, Any] = {
                    "attribute": attribute,
                    "num_plans": self.n_mutations,
                    "direction_goal": DIRECTION_GOAL[self.direction],
                }
                if self.context == "all" and neighbor_data:
                    fmt_kwargs["neighbor_data"] = neighbor_data
                content.append({"type": "input_text", "text": post_tail.format(**fmt_kwargs)})

                to_send.append(ChatHistory(messages=[ChatMessage(role="user", content=content)]))
                msg_info.append({
                    "topic_id": topic_state.topic_id,
                    "parent": attribute,
                    "parent_time_step": orig_step_idx,
                    "step_idx": step_idx,
                    "pre_text": pre_text,
                    "representative_prompt": representative_prompt,
                })

            # Re-add surviving attributes to new step for re-evaluation
            for attr, orig_idx in topic_state.surviving.items():
                orig_meta = topic_state.history[orig_idx].attributes[attr].meta
                new_step.attributes[attr] = AttributeStats(
                    attribute=attr,
                    meta=AttributeMeta(
                        time_step=step_idx,
                        parent=orig_meta.parent,
                        parent_time_step=orig_meta.parent_time_step,
                        operation=orig_meta.operation,
                        planner_model=orig_meta.planner_model,
                        reasoning_effort=orig_meta.reasoning_effort,
                    ),
                )

        if not to_send:
            logger.warning("AttributeMutator: no messages to send")
            return

        responses = await self.caller.call(
            messages=to_send,
            model=self.model_name,
            max_parallel=self.max_parallel,
            max_tokens=self.max_tokens,
            reasoning=self.reasoning,
            enable_cache=False,
            desc="Mutating attributes",
        )

        topic_state_by_id = {ts.topic_id: ts for ts in topic_states}
        for i, resp in enumerate(responses):
            if resp is None:
                continue
            info = msg_info[i]
            attributes_list, reasoning = parse_json_response(resp)
            if i < 3:
                logger.info(f"Mutation reasoning:\n{reasoning}")

            if not isinstance(attributes_list, list):
                logger.warning(f"Mutation response not a list (topic={info['topic_id']})")
                continue
            attributes_list = [str(a).strip() for a in attributes_list if a]

            ts = topic_state_by_id[info["topic_id"]]
            new_step = ts.history[info["step_idx"]]
            for attr in attributes_list:
                if attr not in new_step.attributes:
                    new_step.attributes[attr] = AttributeStats(
                        attribute=attr,
                        meta=AttributeMeta(
                            time_step=info["step_idx"],
                            parent=info["parent"],
                            parent_time_step=info["parent_time_step"],
                            operation="mutate",
                            planner_model=self.model_name,
                            reasoning_effort=str(self.reasoning),
                            planner_prompt=info["pre_text"],
                            planner_reasoning=str(reasoning),
                        ),
                    )

        for ts in topic_states:
            if ts.history:
                n = len(ts.history[-1].attributes)
                logger.info(f"Topic {ts.topic_id}: {n} attributes after mutation (step {len(ts.history)-1})")
