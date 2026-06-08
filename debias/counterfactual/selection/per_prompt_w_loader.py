"""Load and iterate a `per_prompt_W_step{N}_topic{T}.json` artifact."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from debias.counterfactual.schemas import PerPromptW


def load_per_prompt_w(path: Path | str) -> PerPromptW:
    """Parse a per_prompt_W JSON written by search.pipeline.bon_amplified_evo."""
    with open(path) as f:
        d = json.load(f)
    return PerPromptW(
        step_idx=int(d["step_idx"]),
        topic_id=int(d["topic_id"]),
        attrs=list(d["attrs"]),
        per_prompt_W={p: list(w) for p, w in d["per_prompt_W"].items()},
        per_prompt_r2={p: float(r) for p, r in d.get("per_prompt_r2", {}).items()},
    )


def iter_attr_columns(ppw: PerPromptW) -> Iterator[tuple[str, dict[str, float]]]:
    """Yield (attr, {prompt_text: W_{x,k}}) for each attribute column."""
    for k, attr in enumerate(ppw.attrs):
        col = {p: w_vec[k] for p, w_vec in ppw.per_prompt_W.items() if k < len(w_vec)}
        yield attr, col


def limit_attrs(ppw: PerPromptW, n: int) -> PerPromptW:
    """Return a new PerPromptW restricted to the first `n` attrs.

    Slices both the attribute list and the corresponding W vector columns.
    `n <= 0` or `n >= len(ppw.attrs)` returns the original object unchanged.
    """
    if n <= 0 or n >= len(ppw.attrs):
        return ppw
    return PerPromptW(
        step_idx=ppw.step_idx,
        topic_id=ppw.topic_id,
        attrs=ppw.attrs[:n],
        per_prompt_W={p: list(w_vec[:n]) for p, w_vec in ppw.per_prompt_W.items()},
        per_prompt_r2=dict(ppw.per_prompt_r2),
    )
