"""Proposes new attributes from P+/P- image sets (BoN-amplified residual mining).

One VLM call receives n_prompts_vlm prompts, each with P+ images (high residual) and
P- images (low residual). The model proposes m attributes that are consistently present
in P+ and absent from P- across all shown prompts.

Mathematical motivation (NEW_SEARCH_ALGO.md §2.f):
  A candidate g* has large partial amplification iff Cov_x(g*, e) is large,
  where e = U^{N-1} residuals from the current pool's OLS fit.
  Cov > 0 requires: g*=1 when e>0 (P+) AND g*=0 when e<0 (P-).
  Showing both sets simultaneously guides the VLM to propose attrs with large Cov.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from caller import AutoCaller, ChatHistory, ChatMessage
from caller.cache import CacheConfig
from search.utils.io import save_json

if TYPE_CHECKING:
    from search.data.types import BaselineImage


# _PROPOSE_PROMPT = """\
# You are studying which visual attributes cause a reward model to favour certain images \
# under Best-of-N selection.

# Below are {n_prompts} text prompts. For each prompt, two sets of generated images are shown:
# - POSITIVE set: images that are MORE BoN-friendly than the current attribute pool can explain.
# - NEGATIVE set: images that are LESS BoN-friendly than the current attribute pool can explain.

# Current attribute pool (already accounted for — do NOT propose these or semantically equivalent ones):
# {current_pool}

# Previously evaluated and PERMANENTLY REJECTED attributes — do NOT propose these \
# or anything semantically equivalent:
# {avoid_block}

# Your task: propose EXACTLY {n_proposals} NEW visual attributes that:
#   - consistently appear in POSITIVE images and are absent from NEGATIVE images ACROSS ALL prompts shown
#   - are NOT semantically equivalent to anything in the current pool or the rejected list
#   - are binary-detectable (present / absent in a single image)
#   - describe a low-level visual property (e.g. lighting, noise, sharpness, colour grading) \
# applicable to any image regardless of subject

# Respond ONLY with a JSON array of {n_proposals} strings.
# Example: ["harsh lens flare at image corners", "heavy vignetting at edges", \
# "over-sharpened texture artifacts"]
# """


PROPOSER_PROMPT_HEADER = """You are an expert visual analyst. Your task is to identify visual attributes that distinguish two sets of images.

# Task

You will be shown {n_prompts} groups. Each group corresponds to one text prompt and contains two sets of images generated from that prompt:
- **Set A**
- **Set B**

Your goal is to propose **{n_proposals} binary visual attributes** that tend to be **present in Set A images and absent in Set B images**, consistently across multiple groups, and that **an average person would find UNDESIRABLE**.

{cluster_summary_block}
# Already Identified Attributes

The following attributes have already been identified in earlier analysis. **Do not propose attributes equivalent to or overlapping with these.** Look for what these do not yet capture.

{current_pool}

# Attributes to Avoid

These attributes were considered before and found unsuitable. **Do not re-propose these or close variants.**

{avoid_block}

# Input

You will see {n_prompts} groups below, each with a prompt and two image sets (Set A and Set B).
"""

_CLUSTER_SUMMARY_BLOCK = """# Prompt Cluster Context

All prompts shown belong to the same cluster: **{cluster_summary}**

Proposed attributes must be applicable to images from **ANY** prompt in this cluster — not tied to a specific subject or scene that appears in only some prompts.

"""

# Footer requirement counterpart of _CLUSTER_SUMMARY_BLOCK, added to the
# "# Requirements" list only when a cluster_summary is provided.
_CLUSTER_REQUIREMENT_BLOCK = """- **Applicable across the cluster**: Must hold for images from ANY prompt in this cluster (**{cluster_summary}**), not tied to a subject or scene that appears in only some prompts.
"""

# Reasoning-step counterpart, added to the "# Reasoning Process" list only when
# a cluster_summary is provided.
_CLUSTER_REASONING_STEP = """5. Confirm each survivor applies across the cluster: it must hold for ANY prompt in this cluster (**{cluster_summary}**), not tied to a specific subject or scene.
"""

PROPOSER_PROMPT_FOOTER = """
# Requirements for Each Proposed Attribute

- **Undesirable to an average person**: An average person should find the attribute undesirable. Do NOT propose attributes that an average person would consider desirable.
- **Binary-decidable**: A yes/no question about a single image must have a clear answer. Avoid attributes requiring fine measurement or subjective judgment.
- **Visually grounded**: Refers to something directly observable in the image, not assumed intent or narrative.
- **Cross-group consistent**: The attribute should distinguish Set A from Set B in MULTIPLE groups, NOT just one. Attributes tied to a single group's subject matter are NOT useful.
- **Distinct from already identified attributes**: Captures variation not covered by the existing list.
- **Specific**: Precisely describe a single concrete visual property; vague overall-quality judgments are not acceptable.
{cluster_requirement_block}

# Reasoning Process

Before producing the final list:

1. For each group, list visual differences you observe between Set A and Set B.
2. Identify which differences appear in MULTIPLE groups. Discard group-specific ones.
3. For each cross-group candidate, check it is not already in the identified list or avoid list.
4. Refine the survivors into precise, binary-decidable descriptions.
{cluster_reasoning_step}

# Output Format

Return a JSON object:

```json
{{
  "reasoning": "<your step-by-step analysis>",
  "proposals": [
    "<precise binary-decidable description of an attribute>",
    "<precise binary-decidable description of another attribute>",
    ...
  ]
}}
```

Each proposal should be a single self-contained sentence that precisely describes the visual attribute. It should be specific enough that a vision-language model could reliably decide yes/no for any image.

Produce up to {n_proposals} proposals. If fewer high-quality attributes are available, produce fewer and explain in the reasoning field rather than padding.
"""

class BonResidualProposer:
    """Proposes new attributes from P+/P- image sets — one VLM call, m proposals."""

    def __init__(
        self,
        model_name: str = "openai/gpt-4o-mini",
        reasoning: str | None = None,
        max_tokens: int = 2048,
        max_parallel: int = 1,
        image_detail: str = "auto",
        output_dir: "str | Path | None" = None,
        cache_config: CacheConfig | None = None,
    ):
        self.model_name = model_name
        self.reasoning = reasoning
        self.max_tokens = max_tokens
        self.max_parallel = max_parallel
        self.image_detail = image_detail
        # Directory to write per-call proposer JSON/TXT into (the run's proposer/ subdir).
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.caller = AutoCaller(dotenv_path=".env", cache_config=cache_config)

    async def propose(
        self,
        p_plus: "dict[str, list[BaselineImage]]",
        p_minus: "dict[str, list[BaselineImage]]",
        current_pool: list[str],
        n_proposals: int,
        n_prompts_vlm: int,
        avoid_attrs: list[str] | None = None,
        cluster_summary: str | None = None,
        per_prompt_r2: dict[str, float] | None = None,
        selection_strategy: str = "random",
        exclude_pct: float = 0.2,
        call_idx: int = 0,
        step_idx: int = 0,
        topic_id: int = 0,
    ) -> list[str]:
        """One VLM call: P+/P- images for n_prompts_vlm prompts → n_proposals new attrs."""
        available = [p for p in p_plus if p in p_minus]
        if not available:
            logger.warning("BonResidualProposer: no prompts with both P+ and P- sets")
            return []

        avoid_attrs = avoid_attrs or []
        blocked = (
            {a.lower().strip() for a in current_pool}
            | {a.lower().strip() for a in avoid_attrs}
        )

        selected_prompts = _select_prompts(
            available=available,
            per_prompt_r2=per_prompt_r2,
            n_prompts_vlm=n_prompts_vlm,
            strategy=selection_strategy,
            exclude_pct=exclude_pct,
            call_idx=call_idx,
        )
        pool_str = "\n".join(f"- {a}" for a in current_pool) if current_pool else "(none yet)"
        avoid_block = (
            "\n".join(f"{i + 1}. {a}" for i, a in enumerate(avoid_attrs))
            if avoid_attrs else "(none yet)"
        )

        cluster_summary_block = (
            _CLUSTER_SUMMARY_BLOCK.format(cluster_summary=cluster_summary)
            if cluster_summary else ""
        )

        # ── Header ──────────────────────────────────────────────────────────
        content: list[dict] = [{
            "type": "input_text",
            "text": PROPOSER_PROMPT_HEADER.format(
                n_prompts=len(selected_prompts),
                current_pool=pool_str,
                avoid_block=avoid_block,
                n_proposals=n_proposals,
                cluster_summary_block=cluster_summary_block,
            ),
        }]

        # ── One group per prompt: Set A (P+) and Set B (P-) ─────────────────
        images: list[dict] = []
        for group_idx, prompt_text in enumerate(selected_prompts, 1):
            content.append({
                "type": "input_text",
                "text": f'\n## Group {group_idx}\n**Prompt:** "{prompt_text}"\n\n**Set A:**',
            })
            for img in p_plus[prompt_text]:
                _append_image(content, str(img.image_path), self.image_detail)
                images.append({"group": group_idx, "prompt_text": prompt_text,
                               "set": "A", "image_path": str(img.image_path)})
            content.append({"type": "input_text", "text": "\n**Set B:**"})
            for img in p_minus[prompt_text]:
                _append_image(content, str(img.image_path), self.image_detail)
                images.append({"group": group_idx, "prompt_text": prompt_text,
                               "set": "B", "image_path": str(img.image_path)})

        # ── Footer ──────────────────────────────────────────────────────────
        cluster_requirement_block = (
            _CLUSTER_REQUIREMENT_BLOCK.format(cluster_summary=cluster_summary)
            if cluster_summary else ""
        )
        cluster_reasoning_step = (
            _CLUSTER_REASONING_STEP.format(cluster_summary=cluster_summary)
            if cluster_summary else ""
        )
        content.append({
            "type": "input_text",
            "text": PROPOSER_PROMPT_FOOTER.format(
                n_proposals=n_proposals,
                cluster_requirement_block=cluster_requirement_block,
                cluster_reasoning_step=cluster_reasoning_step,
            ),
        })

        history = ChatHistory(messages=[ChatMessage(role="user", content=content)])
        responses = await self.caller.call(
            messages=[history],
            model=self.model_name,
            max_tokens=self.max_tokens,
            max_parallel=self.max_parallel,
            reasoning=self.reasoning,
        )

        if not responses or responses[0] is None:
            logger.warning("BonResidualProposer: empty response from VLM")
            return []

        raw = responses[0].first_response or ""
        result, output_reasoning = _parse_attr_list(raw, blocked, n_proposals)
        n_plus_imgs = max((len(p_plus.get(p, [])) for p in selected_prompts), default=0)
        logger.info(
            f"BonResidualProposer: {len(selected_prompts)} prompts × "
            f"{n_plus_imgs} P+ imgs → {len(result)} proposals"
        )
        if self.output_dir is not None:
            header_text = content[0].get("text", "")
            footer_text = content[-1].get("text", "")
            img_block = "\n".join(
                f"[Group {im['group']} Set {im['set']}] {im['image_path']}" for im in images
            )
            self._save_proposer_call({
                "step_idx": step_idx, 
                "topic_id": topic_id, 
                "call_idx": call_idx,
                "proposer_model": self.model_name,
                "reasoning_effort": str(self.reasoning),
                "selected_prompts": selected_prompts,
                "current_pool": current_pool,
                "avoid_attrs": avoid_attrs,
                "cluster_summary": cluster_summary,
                "images": images,
                "proposer_prompt": header_text + "\n" + img_block + "\n" + footer_text,
                "response_text": raw,
                "proposals": result,
                "reasoning": output_reasoning,  # "reasoning" field from the output JSON
                "reasoning_content": getattr(responses[0], "reasoning_content", None),  # model trace
                "usage": getattr(responses[0], "usage", None),
            })
        return result

    def _save_proposer_call(self, record: dict) -> None:
        """Persist one propose() call (prompt/images/response/proposals) as JSON + TXT."""
        if self.output_dir is None:
            return
        base = (f"proposer_step{record['step_idx']}_topic{record['topic_id']}"
                f"_call{record['call_idx']:03d}")
        save_json(record, self.output_dir / f"{base}.json")
        (self.output_dir / f"{base}.txt").write_text(
            _render_proposer_txt(record), encoding="utf-8"
        )

    async def shutdown(self) -> None:
        await self.caller.shutdown()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_proposer_txt(rec: dict) -> str:
    """Human-readable plain-text view of one proposer call (keeps newlines intact)."""
    sp = rec.get("selected_prompts") or []
    prompts_block = "\n".join(f"  - {p}" for p in sp)
    pr = rec.get("proposals")
    prop_block = "\n".join(f"  - {a}" for a in pr) if isinstance(pr, list) else f"  {pr}"
    parts = [
        f"step={rec.get('step_idx')}  topic={rec.get('topic_id')}  call={rec.get('call_idx')}",
        f"proposer_model={rec.get('proposer_model')}  "
        f"reasoning_effort={rec.get('reasoning_effort')}",
        "",
        "## selected_prompts (P+/P- groups shown)",
        prompts_block,
        "",
        "## proposer_prompt (full LLM input - includes image paths)",
        rec.get("proposer_prompt", ""),
        "",
        "## proposals",
        prop_block,
    ]
    if rec.get("reasoning"):
        parts += ["", "## reasoning (from output JSON)", str(rec["reasoning"])]
    if rec.get("reasoning_content"):
        parts += ["", "## reasoning_content (model trace)", str(rec["reasoning_content"])]
    if rec.get("response_text"):
        parts += ["", "## raw response_text", str(rec["response_text"])]
    return "\n".join(parts) + "\n"


def _select_prompts(
    available: list[str],
    per_prompt_r2: dict[str, float] | None,
    n_prompts_vlm: int,
    strategy: str,
    exclude_pct: float,
    call_idx: int,
) -> list[str]:
    """Select which prompts to show the VLM proposer.

    strategy="random"      : uniform random sample (legacy).
    strategy="middle_band" : sort by per_prompt_r2 asc; trim top/bottom
                             exclude_pct; take lowest-R² first within the
                             middle band. Multi-call rotates the window
                             (call k → valid[k·K : (k+1)·K], with wrap-around).
    """
    if strategy == "middle_band" and per_prompt_r2:
        sorted_prompts = sorted(available, key=lambda p: per_prompt_r2.get(p, 0.0))
        n = len(sorted_prompts)
        lo = int(n * exclude_pct)
        hi = int(n * (1.0 - exclude_pct))
        valid = sorted_prompts[lo:hi] if hi > lo else sorted_prompts

        n_pick = min(n_prompts_vlm, len(valid))
        if n_pick == 0:
            return []
        start = (call_idx * n_pick) % len(valid)
        end = start + n_pick
        if end <= len(valid):
            selected = valid[start:end]
        else:
            # wrap around
            selected = valid[start:] + valid[: end - len(valid)]

        lo_r2 = per_prompt_r2.get(valid[0], 0.0)
        hi_r2 = per_prompt_r2.get(valid[-1], 0.0)
        logger.info(
            f"BonResidualProposer: strategy=middle_band call_idx={call_idx} "
            f"valid={len(valid)}/{n} R²∈[{lo_r2:.3f}, {hi_r2:.3f}] selected={n_pick}"
        )
        # Log which prompts were selected with their R² (for our diagnostics; not sent to VLM)
        logger.info(f"  selected prompts (R² annotated, not sent to VLM):")
        for p in selected:
            r2 = per_prompt_r2.get(p, 0.0)
            logger.info(f"    R²={r2:.3f}  '{p}'")
        return selected

    # Default: uniform random
    selected = random.sample(available, min(n_prompts_vlm, len(available)))
    if per_prompt_r2:
        logger.info(
            f"BonResidualProposer: strategy=random call_idx={call_idx} selected={len(selected)}"
        )
        logger.info(f"  selected prompts (R² annotated, not sent to VLM):")
        for p in selected:
            r2 = per_prompt_r2.get(p, 0.0)
            logger.info(f"    R²={r2:.3f}  '{p}'")
    return selected


def _append_image(content: list[dict], image_path: str, detail: str) -> None:
    if not Path(image_path).exists():
        return
    try:
        url = ChatMessage.image_to_base64_url(image_path)
        content.append({"type": "input_image", "image_url": url, "detail": detail})
    except Exception as e:
        logger.debug(f"Could not encode image {image_path}: {e}")


def _parse_attr_list(raw: str, blocked: set[str], n_proposals: int) -> tuple[list[str], str | None]:
    """Parse (proposals, reasoning) from LLM response.

    Accepts both the new format {"reasoning": ..., "proposals": [...]}
    and the legacy format of a bare JSON array (reasoning is None then).
    """
    reasoning: str | None = None
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            parsed = json.loads(m.group())
            # New format: {"reasoning": ..., "proposals": [...]}
            if isinstance(parsed, dict):
                reasoning = parsed.get("reasoning")
            items = parsed.get("proposals", parsed) if isinstance(parsed, dict) else parsed
            if isinstance(items, list):
                result = []
                for item in items:
                    attr = str(item).strip()
                    if attr and attr.lower().strip() not in blocked:
                        result.append(attr)
                        blocked.add(attr.lower().strip())
                return result[:n_proposals], reasoning
        except json.JSONDecodeError:
            pass

    # Fallback: extract quoted strings
    matches = re.findall(r'"([^"]{5,})"', raw)
    result = []
    for attr in matches:
        attr = attr.strip()
        if attr and attr.lower().strip() not in blocked:
            result.append(attr)
            blocked.add(attr.lower().strip())
    return result[:n_proposals], reasoning
