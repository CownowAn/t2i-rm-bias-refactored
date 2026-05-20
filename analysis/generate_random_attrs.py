"""Generate a cluster-anchored random attribute baseline for BoN amplification.

Asks an LLM (default: openai/gpt-5.2 reasoning=high) for N arbitrary binary
visual attributes plausibly applicable to the given prompt cluster, with no
preference for desirable/undesirable. Saves a SearchResults-compatible JSON so
the BoN runner can load it via attributes.search_results_path (with
only_undesirable: false).

Usage:
    python -m analysis.generate_random_attrs \\
        --topic_id 0 --n_attributes 22 \\
        --prompts_dir clustering/output/mjhq \\
        --output_dir outputs/random_baseline \\
        --name random_mjhq_topic0
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

from loguru import logger

from caller import AutoCaller
from search.data.results import FoundAttribute, SearchResults
from search.utils.io import parse_json_response


_PROMPT_TEMPLATE = """You are an expert visual analyst. Your task is to enumerate **arbitrary** binary visual attributes that could plausibly apply to images in a given domain — without any preference for whether each attribute is desirable, undesirable, or neutral.

# Domain

All images come from prompts in this cluster: **{cluster_summary}**

# Task

Propose exactly {N} distinct binary visual attributes. The attributes should be sampled freely — colour-related, structural, stylistic, lighting, material, texture, framing, etc. — and should **not** be biased toward any particular aesthetic judgement.

# Requirements for each attribute

- **Binary-decidable**: A yes/no question about a single image must have a clear answer. Avoid attributes requiring fine measurement or strongly subjective judgement.
- **Visually grounded**: Refers to something directly observable in the image. Do not refer to assumed intent or narrative.
- **Plausibly applicable** to images in the **{cluster_summary}** domain.
- **Diverse**: Cover a range of visual dimensions (do not propose 10 colour variations or 10 lighting variations only).

# Output

Return a JSON object exactly in this form:

```json
{{
  "reasoning": "<brief notes on how you sampled diverse attribute dimensions>",
  "attributes": [
    "<binary-decidable visual attribute>",
    ...
  ]
}}
```

Produce exactly {N} entries in `attributes`.
"""


def _load_cluster_summary(prompts_dir: Path, topic_id: int) -> str:
    path = prompts_dir / f"cluster_{topic_id}.json"
    with open(path) as f:
        data = json.load(f)
    summary = data.get("summary")
    if not summary:
        logger.warning(f"{path} has no 'summary' field; falling back to 'general visual content'")
        return "general visual content"
    return str(summary).strip()


async def _generate(args: argparse.Namespace) -> None:
    prompts_dir = Path(args.prompts_dir)
    out_path = Path(args.output_dir) / args.name / "results.json"

    cluster_summary = _load_cluster_summary(prompts_dir, args.topic_id)
    prompt = _PROMPT_TEMPLATE.format(cluster_summary=cluster_summary, N=args.n_attributes)
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

    logger.info(
        f"Generating {args.n_attributes} random attrs  "
        f"model={args.model} reasoning={args.reasoning}  "
        f"cluster='{cluster_summary[:60]}'"
    )

    caller = AutoCaller(dotenv_path=".env", cache_config=None)
    t0 = time.monotonic()
    try:
        resp = await caller.call_one(
            messages=prompt,
            model=args.model,
            max_tokens=args.max_tokens,
            reasoning=args.reasoning,
        )
    finally:
        await caller.shutdown()

    if resp is None:
        raise RuntimeError("LLM returned no response")

    parsed, reasoning_text = parse_json_response(resp)
    if not isinstance(parsed, dict) or "attributes" not in parsed:
        raise RuntimeError(f"LLM response missing 'attributes' field. Parsed: {parsed!r}")

    raw_attrs = parsed["attributes"]
    if not isinstance(raw_attrs, list):
        raise RuntimeError(f"'attributes' must be a list. Got: {type(raw_attrs).__name__}")

    attributes = [str(a).strip() for a in raw_attrs if str(a).strip()]
    if len(attributes) < args.n_attributes:
        logger.warning(
            f"LLM returned {len(attributes)} attrs (< requested {args.n_attributes}); "
            "saving them as-is without retry"
        )
    elif len(attributes) > args.n_attributes:
        logger.info(f"LLM returned {len(attributes)}, truncating to {args.n_attributes}")
        attributes = attributes[: args.n_attributes]

    found = [
        FoundAttribute(
            attribute=a,
            delta_rm=None,
            delta_j=None,
            amplification_score=0.0,
            step_found=0,
            step_last_survived=0,
            topic_id=args.topic_id,
            is_undesirable=False,
        )
        for a in attributes
    ]
    results = SearchResults(
        run_id=args.name,
        config_snapshot={
            "generator":            "random_baseline",
            "model":                args.model,
            "reasoning":            args.reasoning,
            "topic_id":             args.topic_id,
            "n_attributes_requested": args.n_attributes,
            "n_attributes_returned":  len(attributes),
            "cluster_summary":      cluster_summary,
            "prompt_template_md5":  prompt_hash,
            "random_seed":          args.random_seed,
            "llm_reasoning":        reasoning_text,
        },
        top_attributes=found,
        n_steps_completed=0,
        cost_usd=0.0,
        wall_time_seconds=time.monotonic() - t0,
    )
    results.save(out_path)
    logger.info(f"Saved {len(attributes)} random attrs → {out_path}")
    for i, a in enumerate(attributes, 1):
        logger.info(f"  {i:>2d}. {a}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic_id", type=int, required=True)
    parser.add_argument("--n_attributes", type=int, required=True)
    parser.add_argument("--prompts_dir", type=str, required=True,
                        help="dir containing cluster_{topic_id}.json")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="output root; results saved to {output_dir}/{name}/results.json")
    parser.add_argument("--name", type=str, required=True,
                        help="run name (also used as subfolder)")
    parser.add_argument("--model", type=str, default="openai/gpt-5.2")
    parser.add_argument("--reasoning", type=str, default="high")
    parser.add_argument("--max_tokens", type=int, default=10000)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    asyncio.run(_generate(args))


if __name__ == "__main__":
    main()
