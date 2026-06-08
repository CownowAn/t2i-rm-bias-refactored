"""Detection cache access — `{model_key: {image_id: {attr: 0|1}}}`."""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger


def load_detection_cache(
    path: Path | str,
    detector_key: str,
) -> dict[str, dict[str, int]]:
    """Load `{image_id: {attr: 0|1}}` for one model key from the global cache file."""
    path = Path(path)
    if not path.exists():
        logger.warning(f"detection cache missing: {path}")
        return {}
    with open(path) as f:
        all_saved = json.load(f)
    saved = all_saved.get(detector_key, {})
    out: dict[str, dict[str, int]] = {}
    for image_id, attrs in saved.items():
        out[image_id] = {a: int(bool(v)) for a, v in attrs.items()}
    logger.info(
        f"detection cache[{detector_key}] loaded: {len(out)} images from {path}"
    )
    return out


def find_image_ids_with_attr(
    detection: dict[str, dict[str, int]],
    attr: str,
    value: int = 1,
) -> set[str]:
    """Return image_ids where the given attribute was detected with the given value."""
    return {
        image_id
        for image_id, attrs in detection.items()
        if attrs.get(attr) == value
    }
