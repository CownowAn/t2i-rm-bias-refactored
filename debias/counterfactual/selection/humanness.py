"""Resolve the set of undesirable attrs — from search artifacts or live re-check.

The attrs that appear in `per_prompt_W.attrs` (== `ba_expand.acc_pool`) have
already passed search-time humanness filtering. They ARE the undesirable set.

`ba_expand_step{N}_topic{T}.json["humanness_rejected"]` lists attrs that were
removed at search time as desirable — we never want them. (They're already
absent from acc_pool, so listing them is for transparency, not subtraction.)

`recheck=True` re-runs `AttributeUndesirabilityFilter.filter_by_humanness` over
the current attrs as a stricter safety net.
"""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger


def load_undesirable_from_search(
    ba_expand_path: Path | None,
    attrs: list[str],
) -> set[str]:
    """Default policy: every attr in the per_prompt_W list is undesirable.

    If `ba_expand_path` is provided, log which attrs were rejected at search
    time so the user can see they're already excluded.
    """
    undesirable = set(attrs)
    if ba_expand_path is not None and Path(ba_expand_path).exists():
        try:
            with open(ba_expand_path) as f:
                ba = json.load(f)
            rejected = list(ba.get("humanness_rejected", []))
            logger.info(
                f"ba_expand.humanness_rejected size = {len(rejected)} "
                f"(already absent from acc_pool; for transparency only)"
            )
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"could not read ba_expand for humanness diagnostics: {e}")
    logger.info(f"undesirable set (search default) = {len(undesirable)} attrs")
    return undesirable


async def recheck_humanness(
    attrs: list[str],
    *,
    model_name: str,
    max_parallel: int = 16,
    max_tokens: int = 16,
) -> set[str]:
    """Re-run the humanness filter against the current attr list (stricter)."""
    from search.pipeline.attribute_filter import AttributeUndesirabilityFilter

    filt = AttributeUndesirabilityFilter(
        model_name=model_name,
        max_tokens=max_tokens,
        max_parallel=max_parallel,
        cache_config=None,
    )
    try:
        passed = await filt.filter_by_humanness(list(attrs))
    finally:
        await filt.shutdown()
    out = set(passed)
    logger.info(
        f"undesirable set (recheck via {model_name}) = "
        f"{len(out)} / {len(attrs)} attrs remained"
    )
    return out


async def resolve_undesirable_set(
    attrs: list[str],
    ba_expand_path: Path | None,
    *,
    recheck: bool,
    humanness_model: str,
    max_parallel: int = 16,
) -> set[str]:
    """Entry point used by the PoC orchestrator."""
    base = load_undesirable_from_search(ba_expand_path, attrs)
    if not recheck:
        return base
    rechecked = await recheck_humanness(
        list(base), model_name=humanness_model, max_parallel=max_parallel
    )
    return rechecked
