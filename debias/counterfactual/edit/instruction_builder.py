"""Convert an attribute description into a FLUX-Kontext edit instruction.

Two static framings are provided. They differ only in how the attribute is
described to the editor — neither needs an extra LLM/VLM call.

Failure mode that motivated the `correct` framing:
    "remove" phrasing makes FLUX-Kontext erase the object itself
    (e.g. "Eyes are exaggerated" → eyes replaced with white blobs)
The `correct` framing reframes the edit as "replace with a natural-looking
alternative while preserving everything else." Subject identity + composition
must stay, only the specific issue is corrected.
"""
from __future__ import annotations


def _clean(attr: str) -> str:
    cleaned = " ".join(attr.strip().split())
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    return cleaned


def build_remove_instruction(attr: str) -> str:
    """Legacy `remove` phrasing. Use sparingly — prone to erasing the object."""
    return f"Edit the image to remove the following visual property: {_clean(attr)}."


def build_correction_instruction(attr: str) -> str:
    """Reframe the edit as `modify the specific feature in place`.

    Rationale: plain `remove` makes FLUX-Kontext erase the carrier of the feature
    (e.g. "Eyes are exaggerated…" → eyes wiped out). The earlier `replace with
    natural` framing dragged stylised images (cyberpunk, anime, illustration)
    toward photorealism. This version drops both `remove` and `natural`, and
    tells the editor explicitly to *adjust the existing feature* — not delete,
    not replace, not change style — so the description no longer applies.
    """
    cleaned = _clean(attr)
    return (
        "Edit only the specific visual feature described below so this description "
        "no longer applies to the image. Modify that feature in place — do not delete "
        "or swap out any object. "
        f"Description: {cleaned}."
    )


_BUILDERS = {
    "remove":  build_remove_instruction,
    "correct": build_correction_instruction,
}

# Public list of available modes (used by argparse `choices=`).
INSTRUCTION_MODES = tuple(_BUILDERS.keys())


def build_instruction(attr: str, mode: str = "correct") -> str:
    """Dispatch to the appropriate builder. Defaults to `correct`."""
    try:
        return _BUILDERS[mode](attr)
    except KeyError as e:
        raise ValueError(
            f"Unknown instruction_mode: {mode!r} (must be one of {INSTRUCTION_MODES})"
        ) from e
