"""Thin re-export of the path helpers used by the edit stage."""
from __future__ import annotations

from debias.counterfactual.io_utils import (
    edited_image_path,
    edits_root,
)

__all__ = ["edited_image_path", "edits_root"]
