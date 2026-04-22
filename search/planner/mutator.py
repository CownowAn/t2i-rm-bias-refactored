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
    MUTATE_POST_HEAD_RESIDUAL, MUTATE_POST_TAIL_RESIDUAL,
    MUTATE_PRE_GENERAL_WITH_CLUSTER, MUTATE_PRE_GENERAL_NO_CLUSTER,
    CLUSTER_SUMMARY_BLOCK_TEMPLATE,
    MUTATE_POST_GENERAL_APPLIES_WITH_CLUSTER, MUTATE_POST_GENERAL_APPLIES_NO_CLUSTER,
)
from search.utils.io import parse_json_response
from search.utils.stats import remove_outliers
from search.utils.linear_probing import (
    compute_lasso_residuals, LinearProbingResult, pair_key as lp_pair_key,
)


class AttributeMutator:
    """Generates mutations of surviving attributes using image context (step N > 0)."""

    def __init__(
        self,
        model_name: str = "openai/gpt-5",
        reasoning: str | None = "high",
        max_tokens: int = 50000,
        max_parallel: int = 64,
        n_mutations: int = 1,
        context: Literal["all", "ancestry", "vanilla", "residual"] = "ancestry",
        n_rollouts_in_context: int = 4,
        n_neighbors: int = 8,
        direction: str = "plus",
        random_seed: int = 42,
        n_residual_high: int = 3,
        n_residual_low: int = 1,
        n_residual_explained: int = 2,
        lasso_min_pairs: int = 5,
        mutation_context_source: str = "origin",
        use_cluster_summary: bool = False,
        use_outlier_removal: bool = False,
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
        self.n_residual_high = n_residual_high
        self.n_residual_low = n_residual_low
        self.n_residual_explained = n_residual_explained
        self.lasso_min_pairs = lasso_min_pairs
        self.mutation_context_source = mutation_context_source
        self.use_cluster_summary = use_cluster_summary
        self.use_outlier_removal = use_outlier_removal
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

    def _get_residual_pairs(
        self,
        attribute: str,
        ctx_stats: AttributeStats,
        probing_result: LinearProbingResult,
    ) -> tuple[list[tuple], list[tuple], list[tuple]]:
        """
        Partition valid pairs for `attribute` into high-residual, low-residual, and
        well-explained groups using pre-computed Lasso residuals.

        Returns:
            high_pairs:      (pair, prompt, residual) — highest positive residuals first
            low_pairs:       (pair, prompt, residual) — highest negative residuals first
            explained_pairs: (pair, prompt, residual) — smallest |residual| first
        """
        valid_pairs = [
            (p, prompt)
            for prompt, pairs in ctx_stats.pairs.items()
            for p in pairs
            if p.delta_rm is not None
            and p.edited_image_path.exists()
            and p.baseline.image_path.exists()
        ]

        if not valid_pairs:
            return [], [], []

        # IQR outlier removal on delta_rm
        scores = [p.delta_rm for p, _ in valid_pairs]
        cleaned = remove_outliers(scores)
        cleaned_set = set(cleaned)
        valid_pairs = [(p, pr) for (p, pr), s in zip(valid_pairs, scores) if s in cleaned_set]

        # Look up residuals
        residual_triples: list[tuple] = []
        for pair, prompt_text in valid_pairs:
            pk = lp_pair_key(attribute, prompt_text, pair)
            if pk in probing_result.residuals:
                residual_triples.append((pair, prompt_text, probing_result.residuals[pk]))

        if not residual_triples:
            return [], [], []

        sorted_asc = sorted(residual_triples, key=lambda x: x[2])

        high_pairs = list(reversed(sorted_asc[-self.n_residual_high:]))
        low_pairs = sorted_asc[:self.n_residual_low]

        sorted_by_abs = sorted(residual_triples, key=lambda x: abs(x[2]))
        explained_pairs = sorted_by_abs[:self.n_residual_explained]

        return high_pairs, low_pairs, explained_pairs

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

            # Compute Lasso residuals once per topic for residual context mode
            probing_result: LinearProbingResult | None = None
            if self.context == "residual":
                probing_result = compute_lasso_residuals(last_step, min_pairs=self.lasso_min_pairs)
                if probing_result is None:
                    logger.warning(
                        f"Topic {topic_state.topic_id}: sklearn unavailable, "
                        "falling back to vanilla context for this topic"
                    )
                elif probing_result.fallback:
                    logger.warning(
                        f"Topic {topic_state.topic_id}: too few pairs "
                        f"({probing_result.n_pairs} < {self.lasso_min_pairs}), "
                        "falling back to vanilla context for this topic"
                    )
                else:
                    logger.info(
                        f"Topic {topic_state.topic_id}: Lasso fit — "
                        f"R²={probing_result.variance_explained:.3f}, "
                        f"α={probing_result.lasso_alpha:.4f}, "
                        f"N={probing_result.n_pairs}"
                    )

            use_residual = (
                self.context == "residual"
                and probing_result is not None
                and not probing_result.fallback
            )
            # Effective context for non-residual path (or fallback from residual)
            effective_context = "vanilla" if (self.context == "residual" and not use_residual) else self.context

            for attribute, orig_step_idx in topic_state.surviving.items():
                if self.mutation_context_source == "accumulated":
                    ctx_stats = topic_state.get_accumulated_stats(attribute)
                elif self.mutation_context_source == "latest":
                    ctx_stats = topic_state.get_latest_stats(attribute)
                else:  # "origin"
                    ctx_stats = topic_state.history[orig_step_idx].attributes.get(attribute)
                if ctx_stats is None:
                    continue

                sw = ctx_stats.delta_rm()
                tw = ctx_stats.delta_j()
                current_summary = (
                    f"Attribute: {attribute}\n"
                    f"Metric A average uplift: {f'{sw:.3f}' if sw is not None else 'N/A'}\n"
                    f"Metric B average uplift: {f'{tw:.3f}' if tw is not None else 'N/A'}"
                )

                if self.use_cluster_summary:
                    cluster_summary_block = CLUSTER_SUMMARY_BLOCK_TEMPLATE.format(
                        cluster_summary=topic_state.cluster_summary
                    )
                    general_requirement = MUTATE_PRE_GENERAL_WITH_CLUSTER
                else:
                    cluster_summary_block = ""
                    general_requirement = MUTATE_PRE_GENERAL_NO_CLUSTER

                pre_text = (
                    PLANNER_SYSTEM + "\n\n"
                    + MUTATE_PRE.format(
                        attribute=attribute,
                        num_plans=self.n_mutations,
                        direction_goal=DIRECTION_GOAL[self.direction],
                        bias_nudge=BIAS_NUDGE[self.direction],
                        cluster_summary_block=cluster_summary_block,
                        general_requirement=general_requirement,
                        current_attr_summary=current_summary,
                    )
                )

                if use_residual:
                    # ── Residual context path ──────────────────────────────────────
                    high_pairs, low_pairs, explained_pairs = self._get_residual_pairs(
                        attribute, ctx_stats, probing_result  # type: ignore[arg-type]
                    )

                    if not high_pairs and not explained_pairs:
                        logger.warning(
                            f"No residual pairs found for '{attribute}' "
                            f"(topic {topic_state.topic_id}), skipping"
                        )
                        continue

                    w_k = probing_result.attribute_weights.get(attribute, 0.0)  # type: ignore[union-attr]
                    content: list[dict] = [{"type": "input_text", "text": pre_text}]

                    weight_note = f"Lasso weight for this attribute: {w_k:+.3f}"
                    if abs(w_k) < 1e-8:
                        weight_note += (
                            "\nNote: Lasso assigned ZERO weight — the reward change in these "
                            "pairs is entirely unexplained by the current attribute pool."
                        )
                    content.append({"type": "input_text", "text": f"===== HIGH-RESIDUAL PAIRS =====\n{weight_note}"})

                    for pair, prompt_text, residual in high_pairs:
                        model_pred = pair.delta_rm - residual  # type: ignore[operator]
                        try:
                            b_url = ChatMessage.image_to_base64_url(str(pair.baseline.image_path))
                            e_url = ChatMessage.image_to_base64_url(str(pair.edited_image_path))
                        except Exception as e:
                            logger.warning(f"Failed to load residual image pair: {e}")
                            continue
                        content.append({"type": "input_text", "text": (
                            f"Prompt: {prompt_text}\n"
                            f"Actual Δ_RM: {pair.delta_rm:+.3f}  "
                            f"Model prediction: {model_pred:+.3f}  "
                            f"Residual: {residual:+.3f}\nBaseline:"
                        )})
                        content.append({"type": "input_image", "image_url": b_url, "detail": "auto"})
                        content.append({"type": "input_text", "text": "Edited:"})
                        content.append({"type": "input_image", "image_url": e_url, "detail": "auto"})

                    # Residual post-head with stats
                    mean_residual = float(np.mean([r for _, _, r in high_pairs])) if high_pairs else 0.0
                    actual_mean = float(np.mean([p.delta_rm for p, _, _ in high_pairs])) if high_pairs else 0.0
                    model_pred_mean = actual_mean - mean_residual
                    content.append({"type": "input_text", "text": MUTATE_POST_HEAD_RESIDUAL.format(
                        n_total_attrs=len(last_step.attributes),
                        model_pred_mean=model_pred_mean,
                        actual_mean=actual_mean,
                        mean_residual=mean_residual,
                        attribute=attribute,
                    )})

                    for pair, prompt_text, residual in explained_pairs:
                        model_pred = pair.delta_rm - residual  # type: ignore[operator]
                        try:
                            b_url = ChatMessage.image_to_base64_url(str(pair.baseline.image_path))
                            e_url = ChatMessage.image_to_base64_url(str(pair.edited_image_path))
                        except Exception as e:
                            logger.warning(f"Failed to load contrast image pair: {e}")
                            continue
                        content.append({"type": "input_text", "text": (
                            f"Prompt: {prompt_text}\n"
                            f"Actual Δ_RM: {pair.delta_rm:+.3f}  "
                            f"Model prediction: {model_pred:+.3f}  "
                            f"Residual: {residual:+.3f} (well-explained)\nBaseline:"
                        )})
                        content.append({"type": "input_image", "image_url": b_url, "detail": "auto"})
                        content.append({"type": "input_text", "text": "Edited:"})
                        content.append({"type": "input_image", "image_url": e_url, "detail": "auto"})

                    # Residual post-tail
                    known_attrs = list(last_step.attributes.keys())
                    known_summary = ", ".join(f'"{a}"' for a in known_attrs[:10])
                    if len(known_attrs) > 10:
                        known_summary += f" (and {len(known_attrs) - 10} more)"
                    content.append({"type": "input_text", "text": MUTATE_POST_TAIL_RESIDUAL.format(
                        num_plans=self.n_mutations,
                        n_known_attrs=len(known_attrs),
                        known_attrs_summary=known_summary,
                        general_applies=(
                            MUTATE_POST_GENERAL_APPLIES_WITH_CLUSTER
                            if self.use_cluster_summary
                            else MUTATE_POST_GENERAL_APPLIES_NO_CLUSTER
                        ),
                    )})

                    representative_prompt = (
                        high_pairs[0][1] if high_pairs else explained_pairs[0][1]
                    )

                else:
                    # ── Existing context path (all / ancestry / vanilla / residual-fallback) ──
                    # Ancestry context
                    if effective_context in ("all", "ancestry"):
                        ancestor_attrs, ancestry_blocks = self._get_ancestry_content(
                            attribute, orig_step_idx, topic_state
                        )
                        exclude_set = {attribute} | set(ancestor_attrs)
                    else:
                        ancestry_blocks = []
                        exclude_set = {attribute}

                    # Neighbor data (only for "all")
                    if effective_context == "all":
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
                    valid_pairs = [
                        (p, prompt)
                        for prompt, pairs in ctx_stats.pairs.items()
                        for p in pairs
                        if p.delta_rm is not None
                        and p.edited_image_path.exists()
                        and p.baseline.image_path.exists()
                    ]

                    if not valid_pairs:
                        logger.warning(f"No valid pairs for attribute '{attribute}' (topic {topic_state.topic_id})")
                        continue

                    # Optionally remove IQR outliers before picking top/bottom context pairs
                    if self.use_outlier_removal:
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
                    content = [{"type": "input_text", "text": pre_text}]

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
                    if effective_context in ("all", "ancestry"):
                        content.append({"type": "input_text", "text": MUTATE_POST_HEAD})
                        content.extend(ancestry_blocks)

                    post_tail = get_post_tail(effective_context)
                    fmt_kwargs: dict[str, Any] = {
                        "attribute": attribute,
                        "num_plans": self.n_mutations,
                        "direction_goal": DIRECTION_GOAL[self.direction],
                        "general_applies": (
                            MUTATE_POST_GENERAL_APPLIES_WITH_CLUSTER
                            if self.use_cluster_summary
                            else MUTATE_POST_GENERAL_APPLIES_NO_CLUSTER
                        ),
                    }
                    if effective_context == "all" and neighbor_data:
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
                        operation="carry_over",
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
