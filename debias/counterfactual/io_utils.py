"""Path conventions for the counterfactual debiasing pipeline.

The edited-image cache is **global and run-id free**: any process editing the
same (topic_id, attr, image_id) tuple writes to the same file. Concurrent
safety is handled in `edit/editor_runner.py` via a sidecar lockfile +
atomic rename.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


# ── Roots ─────────────────────────────────────────────────────────────────────
DEFAULT_CF_ROOT = Path("/nfs/data/sohyun/projects/t2i-rm-bias/counterfactuals")
DEFAULT_POC_REPORT_ROOT = Path("outputs/counterfactual_poc")
DEFAULT_PAIRS_ROOT = Path("outputs/counterfactual")

EDITS_SUBDIR = "edits"      # shared cache under cf_root, no run_id


# ── Hashing ───────────────────────────────────────────────────────────────────

def attr_hash(attr: str, n: int = 12) -> str:
    """Stable short hash of an attribute text for use in filenames/dirnames."""
    return hashlib.md5(attr.encode("utf-8")).hexdigest()[:n]


# ── Edit-cache layout (shared across runs) ────────────────────────────────────

def edits_root(cf_root: Path | None = None) -> Path:
    """The global edited-image cache root.

    Defaults to `<DEFAULT_CF_ROOT>/edits`. All runs/processes write here.
    """
    return (cf_root or DEFAULT_CF_ROOT) / EDITS_SUBDIR


def edited_image_path(
    topic_id: int,
    attr: str,
    image_id: str,
    cf_root: Path | None = None,
    instruction_mode: str | None = None,
) -> Path:
    """`<cf_root>/edits/[<instruction_mode>/]topic_<T>/<attr_hash>/<image_id>.png`.

    The path is determined by (topic_id, attr, image_id, instruction_mode).
    Different instruction modes produce different PNGs, so they get separate
    cache namespaces. Re-running the PoC reuses prior edits transparently.

    For backward compatibility, `instruction_mode=None` writes to the legacy
    no-mode subtree.
    """
    base = edits_root(cf_root)
    if instruction_mode:
        base = base / instruction_mode
    return (
        base
        / f"topic_{topic_id}"
        / attr_hash(attr)
        / f"{image_id}.png"
    )


def edited_metadata_path(
    topic_id: int,
    attr: str,
    image_id: str,
    cf_root: Path | None = None,
    instruction_mode: str | None = None,
) -> Path:
    """Sidecar JSON next to the edited PNG."""
    return edited_image_path(
        topic_id, attr, image_id, cf_root, instruction_mode,
    ).with_suffix(".json")


def poc_report_dir(run_id: str, report_root: Path | None = None) -> Path:
    return (report_root or DEFAULT_POC_REPORT_ROOT) / run_id


def thumb_path(run_id: str, attr: str, report_root: Path | None = None) -> Path:
    return poc_report_dir(run_id, report_root) / "thumbs" / f"{attr_hash(attr)}.png"


# ── Inference helpers ─────────────────────────────────────────────────────────

_PPW_PAT = re.compile(r"per_prompt_W_step(\d+)_topic(\d+)\.json$")


def infer_ba_expand_path(per_prompt_w_path: Path) -> Path | None:
    """Derive the ba_expand_step{N}_topic{T}.json sibling, if it exists."""
    per_prompt_w_path = Path(per_prompt_w_path)
    m = _PPW_PAT.search(per_prompt_w_path.name)
    if not m:
        return None
    step, topic = m.group(1), m.group(2)
    candidate = per_prompt_w_path.parent / f"ba_expand_step{step}_topic{topic}.json"
    return candidate if candidate.exists() else None
