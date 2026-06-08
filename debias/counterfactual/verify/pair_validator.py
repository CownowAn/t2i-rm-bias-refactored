"""Use the VLM detector to confirm the target attr is gone in the edited image."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from debias.counterfactual.schemas import EditResult, EditTask
from debias.counterfactual.verify.side_effect_check import detect_side_effect_drift

if TYPE_CHECKING:
    from search.models.base import DetectorModel


def _group_by_attr(tasks: list[EditTask]) -> dict[str, list[int]]:
    """Map attr → indices into tasks, so we batch-detect once per attr."""
    out: dict[str, list[int]] = {}
    for i, t in enumerate(tasks):
        out.setdefault(t.selection.attr, []).append(i)
    return out


async def validate_edits(
    tasks: list[EditTask],
    detector: "DetectorModel",
    *,
    check_original: bool = True,
    side_effect: bool = False,
) -> list[EditResult]:
    """Run detector on (edited [, original]) per attr; build EditResult list.

    For each task we want:
      - edited_attr_detected: detector(edited, attr)  → expect 0
      - original_attr_detected: detector(original, attr) → expect 1 (sanity)
    Tasks are batched by attribute so the detector call count is minimal.
    """
    results: list[EditResult | None] = [None] * len(tasks)
    by_attr = _group_by_attr(tasks)
    logger.info(
        f"  validating {len(tasks)} edits across {len(by_attr)} attrs"
    )

    for attr, idxs in by_attr.items():
        edited_paths: list[str] = []
        orig_paths: list[str] = []
        prompts: list[str] = []
        edited_exists: list[bool] = []
        for i in idxs:
            t = tasks[i]
            ep = Path(t.edited_output_path)
            edited_exists.append(ep.exists())
            edited_paths.append(str(ep) if ep.exists() else str(t.source.image_path))
            orig_paths.append(str(t.source.image_path))
            prompts.append(t.selection.prompt_text)

        try:
            edited_det = await detector.detect(edited_paths, prompts, attr)
        except Exception as e:
            logger.exception(f"  detector failed on edited (attr={attr[:50]!r}): {e}")
            edited_det = [None] * len(idxs)

        if check_original:
            try:
                orig_det = await detector.detect(orig_paths, prompts, attr)
            except Exception as e:
                logger.exception(f"  detector failed on original (attr={attr[:50]!r}): {e}")
                orig_det = [None] * len(idxs)
        else:
            orig_det = [None] * len(idxs)

        for local, i in enumerate(idxs):
            edt = edited_det[local] if local < len(edited_det) else None
            ort = orig_det[local] if local < len(orig_det) else None
            if not edited_exists[local]:
                results[i] = EditResult(
                    task=tasks[i],
                    success=False,
                    edited_attr_detected=None,
                    original_attr_detected=int(ort) if ort is not None else None,
                    side_effect_drift=None,
                    error="edited image missing on disk",
                )
                continue
            edt_int = int(edt) if edt is not None else None
            success = (edt_int == 0)
            results[i] = EditResult(
                task=tasks[i],
                success=success,
                edited_attr_detected=edt_int,
                original_attr_detected=int(ort) if ort is not None else None,
                side_effect_drift=None,
                error=None,
            )

    # Optional side-effect probe per surviving task
    if side_effect:
        for i, r in enumerate(results):
            if r is None or not r.success:
                continue
            other_present = [
                a for a, v in tasks[i].source.detected_attrs_snapshot.items()
                if v == 1 and a != tasks[i].selection.attr
            ]
            if not other_present:
                continue
            try:
                drift = await detect_side_effect_drift(tasks[i], other_present, detector)
                results[i] = EditResult(
                    task=r.task,
                    success=r.success,
                    edited_attr_detected=r.edited_attr_detected,
                    original_attr_detected=r.original_attr_detected,
                    side_effect_drift=drift,
                    error=r.error,
                )
            except Exception as e:
                logger.exception(f"  side-effect check failed: {e}")

    # Should be fully populated at this point
    return [r for r in results if r is not None]
