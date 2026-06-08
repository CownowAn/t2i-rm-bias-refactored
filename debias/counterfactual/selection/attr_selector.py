"""Per-prompt selection: B_x = {k : W_{x,k} > tau ∧ undesirable(k)}, top-N."""
from __future__ import annotations

from debias.counterfactual.schemas import PerPromptW, PromptAttrSelection


def select_per_prompt(
    ppw: PerPromptW,
    undesirable: set[str],
    tau: float,
    top_n: int,
) -> list[PromptAttrSelection]:
    """Return all PromptAttrSelection entries from B_x ∩ top-N per prompt.

    Rank is assigned within each prompt: 0 = largest W among undesirable attrs.
    """
    out: list[PromptAttrSelection] = []
    K = len(ppw.attrs)
    for prompt_text, w_vec in ppw.per_prompt_W.items():
        # (attr, w) for undesirable attrs with W > tau
        candidates: list[tuple[str, float, int]] = []  # (attr, w, k_idx)
        for k, attr in enumerate(ppw.attrs):
            if k >= len(w_vec):
                continue
            if attr not in undesirable:
                continue
            w = float(w_vec[k])
            if w <= tau:
                continue
            candidates.append((attr, w, k))
        # rank by W desc
        candidates.sort(key=lambda t: t[1], reverse=True)
        for rank, (attr, w, _k) in enumerate(candidates[:top_n]):
            out.append(PromptAttrSelection(
                prompt_text=prompt_text,
                topic_id=ppw.topic_id,
                attr=attr,
                w_value=w,
                rank_in_prompt=rank,
                is_undesirable=True,
            ))
    return out


def group_by_attr(
    selections: list[PromptAttrSelection],
) -> dict[str, list[PromptAttrSelection]]:
    """Group selections by attribute text. Order within each list = input order."""
    out: dict[str, list[PromptAttrSelection]] = {}
    for s in selections:
        out.setdefault(s.attr, []).append(s)
    return out
