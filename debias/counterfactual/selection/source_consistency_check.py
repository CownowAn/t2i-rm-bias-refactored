"""Strict source image filtering via repeated detector queries.

The VLM detector (Qwen3.5-9B with temperature > 0) is not deterministic — the
cached g=1 might be a one-off lucky answer. To make sure a baseline image
*really* has the target attribute, we re-query the detector `n_repeats` times
and keep ONLY images where every single query returns 1.

This filters out cache entries driven by detector noise — without this step,
"success = attr removed" can be a false positive simply because the edited
image is being judged by an inconsistent detector.

Cost: `n_repeats` × (#sources × #attrs) extra detection calls. With local vLLM
this is cheap (a few minutes at most).
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from loguru import logger

from debias.counterfactual.schemas import SourceImage

if TYPE_CHECKING:
    from search.models.base import DetectorModel


async def filter_consistent_sources(
    sources_by_key: dict[tuple[str, str], list[SourceImage]],
    detector: "DetectorModel",
    n_repeats: int = 3,
) -> dict[tuple[str, str], list[SourceImage]]:
    """Keep only sources whose attribute detection is g=1 across `n_repeats` calls.

    Args:
        sources_by_key: {(prompt_text, attr): [SourceImage, ...]}
        detector:       VLM detector (already configured)
        n_repeats:      number of independent detector calls per (image, attr).
                        An image survives iff EVERY call returns 1.

    Returns a new dict (same keys, possibly with fewer SourceImages each).
    Empty keys are dropped from the result.
    """
    if not sources_by_key or n_repeats <= 0:
        return dict(sources_by_key)

    # Group by attr to share detector batches: attr → [(prompt, src, key)]
    by_attr: dict[str, list[tuple[str, SourceImage]]] = defaultdict(list)
    for (prompt, attr), srcs in sources_by_key.items():
        for s in srcs:
            by_attr[attr].append((prompt, s))

    n_total = sum(len(v) for v in by_attr.values())
    logger.info(
        f"source consistency check: re-querying detector {n_repeats}× on "
        f"{n_total} (image, attr) pairs across {len(by_attr)} attrs"
    )

    survived_by_key: dict[tuple[str, str], list[SourceImage]] = defaultdict(list)
    kept = 0
    dropped = 0

    for attr, pairs in by_attr.items():
        prompts = [p for p, _ in pairs]
        image_paths = [str(s.image_path) for _, s in pairs]
        # Count positives per image across `n_repeats` rounds
        positives = [0] * len(pairs)
        rounds_done = 0
        for _ in range(n_repeats):
            try:
                results = await detector.detect(image_paths, prompts, attr)
            except Exception as e:
                logger.exception(
                    f"  consistency check: detector failed on "
                    f"attr={attr[:60]!r}: {e}"
                )
                continue                                # treat as "no positive this round"
            rounds_done += 1
            for i, r in enumerate(results):
                if r is not None and int(r) == 1:
                    positives[i] += 1
        # An image survives only if ALL completed rounds returned 1.
        # Rounds that crashed don't count as positives → image still has to win
        # the same `n_repeats` total. (We deliberately treat detector errors as
        # negatives so noisy attributes aren't silently waved through.)
        for i, (prompt, src) in enumerate(pairs):
            if positives[i] == n_repeats:
                survived_by_key[(prompt, attr)].append(src)
                kept += 1
            else:
                dropped += 1
        logger.info(
            f"  consistency check  attr={attr[:60]!r}  "
            f"{rounds_done}/{n_repeats} rounds completed  "
            f"kept={sum(1 for p in positives if p == n_repeats)}/{len(pairs)}"
        )

    logger.info(
        f"  consistency filter: kept {kept}/{n_total} sources "
        f"({100 * kept / max(n_total, 1):.1f}% retention)"
    )
    return dict(survived_by_key)
