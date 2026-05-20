"""Shared LLM summary generation for prompt clusters."""
from __future__ import annotations

import os


def generate_summary(
    cluster_prompts: list[str],
    model: str,
    n_sample: int | None = None,
) -> str:
    """Generate a general one-sentence summary for a cluster of prompts.

    Args:
        cluster_prompts: All prompts in the cluster (will be shuffled externally).
        model: OpenAI model name.
        n_sample: If set, only the first n_sample prompts are shown to the LLM.
                  None means all prompts are used.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return "(summary generation requires openai package)"

    client = OpenAI()
    sample = cluster_prompts if n_sample is None else cluster_prompts[:n_sample]
    joined = "\n".join(f"- {p}" for p in sample)

    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": (
                "The following are text prompts from the same cluster.\n"
                "Summarize this cluster in one short phrase (not a full sentence). "
                "Use only the broadest category-level description — like a topic label. "
                "Do NOT mention specific styles, settings, lighting, aesthetics, render techniques, "
                "or any detail that only applies to some prompts. "
                "Examples of good summaries: 'Portrait photography', 'Fantasy creature art', "
                "'Food and drink photography', 'Urban landscape scenes'.\n\n"
                + joined
            ),
        }],
        max_completion_tokens=256,
    )
    return resp.choices[0].message.content.strip()