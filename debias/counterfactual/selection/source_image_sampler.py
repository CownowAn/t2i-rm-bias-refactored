"""Pick source baseline images for a given (prompt, attr) selection."""
from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING

from loguru import logger

from debias.counterfactual.schemas import PromptAttrSelection, SourceImage

if TYPE_CHECKING:
    from search.data.types import BaselineImage


def sample_source_images(
    sel: PromptAttrSelection,
    baselines_for_prompt: "list[BaselineImage]",
    detection: dict[str, dict[str, int]],
    k_img: int,
    rng_seed: int,
) -> list[SourceImage]:
    """Choose up to `k_img` baselines for this prompt that have attr=1 in cache.

    Deterministic given `rng_seed`. If fewer than k_img qualify, returns what we have.
    """
    eligible = [
        img for img in baselines_for_prompt
        if img.image_id in detection
        and detection[img.image_id].get(sel.attr) == 1
    ]
    if not eligible:
        logger.warning(
            f"  no eligible source for prompt={sel.prompt_text[:40]!r} "
            f"attr={sel.attr[:50]!r}"
        )
        return []
    # seed mixes prompt + attr for per-(prompt,attr) determinism
    seed_mix = rng_seed ^ (hash(sel.prompt_text) & 0xFFFF) ^ (hash(sel.attr) & 0xFFFF)
    rng = Random(seed_mix)
    n = min(k_img, len(eligible))
    chosen = rng.sample(eligible, n)
    return [
        SourceImage(
            image_id=img.image_id,
            image_path=img.image_path,
            prompt_text=sel.prompt_text,
            detected_attrs_snapshot=dict(detection.get(img.image_id, {})),
        )
        for img in chosen
    ]
