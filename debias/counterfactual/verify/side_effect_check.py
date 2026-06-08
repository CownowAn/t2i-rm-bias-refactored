"""Detect whether other previously-present attrs drifted after editing."""
from __future__ import annotations

from typing import TYPE_CHECKING

from debias.counterfactual.schemas import EditTask

if TYPE_CHECKING:
    from search.models.base import DetectorModel


async def detect_side_effect_drift(
    task: EditTask,
    other_present_attrs: list[str],
    detector: "DetectorModel",
) -> dict[str, tuple[int, int]]:
    """For each other_attr (originally 1), detect on edited image.

    Returns `{attr: (before, after)}` with before=1 (assumed) and after in {0,1,None_as_-1}.
    """
    drift: dict[str, tuple[int, int]] = {}
    for attr in other_present_attrs:
        try:
            after = await detector.detect(
                [str(task.edited_output_path)], [task.selection.prompt_text], attr,
            )
            after_v = int(after[0]) if after and after[0] is not None else -1
        except Exception:
            after_v = -1
        drift[attr] = (1, after_v)
    return drift
