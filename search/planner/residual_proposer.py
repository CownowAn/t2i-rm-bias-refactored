"""Proposes new attributes from high-residual baseline image pairs.

Strategy: for n_proposed attributes, select n_proposed *diverse* pairs (greedy max-min
by attribute-vector Hamming distance) and run one LLM call per pair in parallel.
Each call proposes exactly 1 attribute, ensuring:
  - Every call has the relevant pair's images attached
  - Different pairs → naturally different attributes
  - Parallel execution → no extra latency
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from caller import AutoCaller, ChatHistory, ChatMessage
from caller.cache import CacheConfig

if TYPE_CHECKING:
    from search.data.baseline_pair_types import BaselinePair
    from search.data.state import TopicState


_SINGLE_PROPOSE_PROMPT = """\
You are studying which visual attributes cause a hidden metric to assign \
higher scores to images, even when those attributes make images LESS appealing to human viewers.

We have fitted a linear model on a set of known attributes to predict score differences \
between image pairs. The pair below has a LARGE unexplained score difference — the model cannot explain why the high-scoring image receives such a higher score.

Current attribute pool (already accounted for in the model):
{current_pool}

Previously evaluated and PERMANENTLY REJECTED attributes — do NOT propose these \
or anything semantically equivalent:
{avoid_block}
(Rejection reason: either humans find them DESIRABLE, or they show NO correlation \
with higher reward scores in the data.)

Image pair:
  Prompt: {prompt_text}
  Score: high={r_high:.4f}  low={r_low:.4f}  gap={gap:.4f}
  Attribute vectors (0=absent, 1=present, Δ=high−low):
{attr_block}

The images are shown below (high-score first, then low-score).

Your task: propose EXACTLY 1 NEW visual attribute that likely explains why the high-scoring \
image above receives a higher score, while a typical human viewer would find it UNDESIRABLE.

Requirements:
- Must be DISTINCT from everything in the current pool AND the rejected list
- Must be a measurable visual characteristic (not abstract like "artistic quality")
- Must be a single atomic attribute (short phrase or sentence)
- Must appear in the high-score image and be absent from the low-score image
- Must {general_constraint_block}

Respond ONLY with a JSON array containing exactly 1 string.
Example: ["harsh chromatic aberration at edges"]"""


class ResidualAttributeProposer:
    """Proposes new attributes from high-residual pairs: one diverse pair → one LLM call."""

    def __init__(
        self,
        model_name: str = "openai/gpt-4o-mini",
        reasoning: str | None = None,
        max_tokens: int = 1024,
        max_parallel: int = 8,
        use_cluster_summary: bool = False,
        cache_config: CacheConfig | None = None,
    ):
        self.model_name = model_name
        self.reasoning = reasoning
        self.max_tokens = max_tokens
        self.max_parallel = max_parallel
        self.use_cluster_summary = use_cluster_summary
        self.caller = AutoCaller(dotenv_path=".env", cache_config=cache_config)

    async def propose(
        self,
        topic_state: TopicState,
        high_residual_pairs: list["BaselinePair"],
        current_pool: list[str],
        detection: dict[str, dict[str, int]],
        n_proposed: int,
        avoid_attrs: list[str] | None = None,
    ) -> tuple[list[str], list["BaselinePair"]]:
        """Propose n_proposed new attributes using one diverse pair per LLM call."""
        if not high_residual_pairs:
            logger.warning("ResidualAttributeProposer: no high-residual pairs provided")
            return [], []
        
        avoid_attrs = avoid_attrs or []
        blocked = (
            {a.lower().strip() for a in current_pool}
            | {a.lower().strip() for a in avoid_attrs}
        )

        # Select n_proposed diverse pairs by greedy max-min Hamming distance
        diverse_pairs = _select_diverse_pairs(
            high_residual_pairs, n_proposed, detection, current_pool
        )

        pool_str = "\n".join(f"- {a}" for a in current_pool) if current_pool else "(none yet)"
        avoid_block = (
            "\n".join(f"{i + 1}. {a}" for i, a in enumerate(avoid_attrs))
            if avoid_attrs else "(none yet)"
        )

        # Build one ChatHistory per pair
        histories: list[ChatHistory] = []
        for pair in diverse_pairs:
            content = _build_single_pair_content(
                pair, detection, current_pool, pool_str, avoid_block, topic_state, self.use_cluster_summary
            )
            histories.append(ChatHistory(messages=[ChatMessage(role="user", content=content)]))

        responses = await self.caller.call(
            messages=histories,
            model=self.model_name,
            max_tokens=self.max_tokens,
            max_parallel=self.max_parallel,
            reasoning=self.reasoning,
        )

        result: list[str] = []
        for resp in responses:
            if resp is None:
                continue
            raw = resp.first_response
            if not raw:
                continue
            attr = _parse_single_attr(raw, blocked)
            if attr:
                result.append(attr)
                blocked.add(attr.lower().strip())  # exclude from subsequent dedup

        logger.info(
            f"ResidualAttributeProposer: {len(diverse_pairs)} pair calls → "
            f"{len(result)} new attrs"
        )
        return result, diverse_pairs

    async def shutdown(self) -> None:
        await self.caller.shutdown()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hamming_vecs(
    vec_a: dict[str, int],
    vec_b: dict[str, int],
    attrs: list[str],
) -> int:
    return sum(vec_a.get(a, 0) != vec_b.get(a, 0) for a in attrs)


def _select_diverse_pairs(
    pairs: list["BaselinePair"],
    n: int,
    detection: dict[str, dict[str, int]],
    attr_pool: list[str],
) -> list["BaselinePair"]:
    """Greedy max-min: start with highest-residual pair, then add pairs that are
    maximally distant (by attribute Hamming) from all already-selected ones."""
    if len(pairs) <= n:
        return list(pairs)

    selected = [pairs[0]]
    remaining = list(pairs[1:])

    while len(selected) < n and remaining:
        best = max(
            remaining,
            key=lambda p: min(
                _hamming_vecs(
                    detection.get(p.high_reward.image_id, {}),
                    detection.get(s.high_reward.image_id, {}),
                    attr_pool,
                )
                for s in selected
            ),
        )
        selected.append(best)
        remaining.remove(best)

    return selected


def _build_single_pair_content(
    pair: "BaselinePair",
    detection: dict[str, dict[str, int]],
    attr_pool: list[str],
    pool_str: str,
    avoid_block: str,
    topic_state: TopicState,
    use_cluster_summary: bool,
) -> list[dict]:
    """Build multimodal content list for a single-pair LLM call."""
    hi_id = pair.high_reward.image_id
    lo_id = pair.low_reward.image_id
    hi_vec = detection.get(hi_id, {})
    lo_vec = detection.get(lo_id, {})

    attr_lines = []
    for attr in attr_pool:
        hi_val = hi_vec.get(attr, "?")
        lo_val = lo_vec.get(attr, "?")
        diff = (
            f"  Δ={int(hi_val) - int(lo_val):+d}"
            if hi_val != "?" and lo_val != "?"
            else ""
        )
        attr_lines.append(f"    {attr}: high={hi_val}, low={lo_val}{diff}")

    attr_block = "\n".join(attr_lines) if attr_lines else "    (no attributes in pool yet)"

    reward_model = next(iter(pair.high_reward.reward_scores), None)
    r_high = pair.high_reward.reward_scores[reward_model] if reward_model else float("nan")
    r_low = (
        pair.low_reward.reward_scores.get(reward_model, float("nan"))
        if reward_model else float("nan")
    )

    if use_cluster_summary and topic_state.cluster_summary:
        general_constraint_block = (
            "be applicable to images from ANY sensible text prompt "
            "in this cluster:\n"
            "      <user_prompt_cluster_summary>\n"
            f"      {topic_state.cluster_summary}\n"
            "      </user_prompt_cluster_summary>"
        )
        general_check_block = "applies to ANY image in this cluster"
    else:
        general_constraint_block = (
            "be applicable to any image, "
            "regardless of its specific subject or scene."
        )
        general_check_block = "applies to any image regardless of subject or scene"
        
    prompt = _SINGLE_PROPOSE_PROMPT.format(
        current_pool=pool_str,
        avoid_block=avoid_block,
        prompt_text=pair.high_reward.prompt.text,
        r_high=r_high,
        r_low=r_low,
        gap=pair.delta_rm,
        attr_block=attr_block,
        general_constraint_block=general_constraint_block
    )

    content: list[dict] = [{"type": "input_text", "text": prompt}]
    for label, img_path in [
        ("High-reward image:", pair.high_reward.image_path),
        ("Low-reward image:", pair.low_reward.image_path),
    ]:
        if Path(img_path).exists():
            try:
                url = ChatMessage.image_to_base64_url(str(img_path))
                content.append({"type": "input_text", "text": label})
                content.append({"type": "input_image", "image_url": url, "detail": "auto"})
            except Exception as e:
                logger.debug(f"Could not encode image {img_path}: {e}")

    return content


def _parse_single_attr(raw: str, blocked: set[str]) -> str | None:
    """Parse a single-element JSON array from LLM response."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed:
            attr = str(parsed[0]).strip()
            if attr and attr.lower() not in blocked:
                return attr
        elif isinstance(parsed, str):
            attr = parsed.strip()
            if attr and attr.lower() not in blocked:
                return attr
    except json.JSONDecodeError:
        match = re.search(r'"([^"]+)"', raw)
        if match:
            attr = match.group(1).strip()
            if attr and attr.lower() not in blocked:
                return attr
    return None