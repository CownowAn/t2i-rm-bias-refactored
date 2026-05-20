"""Print attributes from search results.json (or bon results), markdown-formatted.

Sorts each file's attributes by `--sort` key:
  - a_hat     (default): amplification_score from search results.json
  - bon_delta: prev[n=last] - prev[n=1] from a matched bon results JSON
              (auto-discovered under --bon_root by search_results_path field)

Output is markdown-friendly (numbered list, bold metrics) — paste-ready for Notion.

Usage:
    python -m analysis.top_attrs --paths PATH [PATH ...] [--k K] [--sort KEY]

Examples:
    # All attrs sorted by A_hat
    python -m analysis.top_attrs --paths outputs/search/20260519-003512/results.json

    # All attrs sorted by BoN delta (needs matching bon run)
    python -m analysis.top_attrs --paths outputs/search/20260519-003512/results.json \
        --sort bon_delta

    # Top-5 only
    python -m analysis.top_attrs --paths outputs/search/<run>/results.json --k 5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _find_matching_bon(search_path: Path, bon_root: Path) -> Path | None:
    """Find the most recent bon results JSON whose search_results_path == search_path.

    bon results live at {bon_root}/bon_*/results_topic*.json. We scan and pick
    the most recently modified match (mtime).
    """
    target = str(search_path.resolve())
    candidates = []
    for p in glob.glob(str(bon_root / "bon_*" / "results_topic*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        srp = d.get("search_results_path")
        if not srp:
            continue
        if os.path.realpath(srp) == target:
            candidates.append(Path(p))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _bon_delta_map(bon_path: Path, bon_n: int | None = None) -> tuple[dict[str, float], int | None]:
    """attr -> prev[n=bon_n] - prev[n=1]. If bon_n is None, use last n in n_values.

    Returns (delta_map, n_used). delta_map is empty if requirements not met.
    """
    d = _load(bon_path)
    n_values = d.get("n_values", [])
    prevalence = d.get("prevalence", {})
    if 1 not in n_values or len(n_values) < 2:
        return {}, None
    i_first = n_values.index(1)
    target = bon_n if bon_n is not None else n_values[-1]
    if target not in n_values:
        return {}, None
    i_target = n_values.index(target)
    out = {}
    for attr, ps in prevalence.items():
        if len(ps) > max(i_first, i_target):
            out[attr] = ps[i_target] - ps[i_first]
    return out, target


def _print_section(
    search_path: Path,
    k: int | None,
    sort_by: str,
    bon_root: Path,
    bon_n: int | None = None,
) -> None:
    try:
        data = _load(search_path)
    except (OSError, json.JSONDecodeError) as e:
        print(f"\n[ERROR] {search_path}: {e}", file=sys.stderr)
        return

    attrs = data.get("top_attributes", [])
    n_total = len(attrs)
    if n_total == 0:
        print(f"\n### `{search_path}`  *(empty)*")
        return

    bon_path = None
    bon_delta: dict[str, float] = {}
    n_used: int | None = None
    if sort_by == "bon_delta" or sort_by == "both":
        bon_path = _find_matching_bon(search_path, bon_root)
        if bon_path is not None:
            bon_delta, n_used = _bon_delta_map(bon_path, bon_n=bon_n)

    if sort_by == "bon_delta":
        if not bon_delta:
            print(f"\n[ERROR] {search_path}: no matching bon results found under {bon_root}; "
                  "cannot sort by bon_delta", file=sys.stderr)
            return
        ranked = sorted(
            attrs,
            key=lambda a: bon_delta.get(a.get("attribute", ""), float("-inf")),
            reverse=True,
        )
    else:
        ranked = sorted(attrs, key=lambda a: a.get("amplification_score", 0.0), reverse=True)

    top = ranked if k is None else ranked[:k]

    topics = sorted({a.get("topic_id") for a in attrs if a.get("topic_id") is not None})
    header_topic = ", ".join(str(t) for t in topics) if topics else "?"
    run_id = data.get("run_id", "?")
    shown = f"{len(top)}/{n_total}" if k is not None and k < n_total else str(n_total)

    print()
    print(f"### `{search_path}`")
    parts = [f"**run_id**: {run_id}",
             f"**shown**: {shown}",
             f"**topic(s)**: {header_topic}",
             f"**sort**: {sort_by}"]
    if bon_path is not None:
        parts.append(f"**bon**: `{bon_path}`")
    print(" · ".join(parts))
    print()

    for i, a in enumerate(top, 1):
        ahat = a.get("amplification_score", 0.0)
        sf = a.get("step_found", "?")
        attr_text = a.get("attribute", "")
        metric_parts = [f"A_hat={ahat:+.4f}"]
        if attr_text in bon_delta and n_used is not None:
            metric_parts.append(f"BoN Δ(n=1→{n_used})={bon_delta[attr_text]:+.4f}")
        metric_parts.append(f"found={sf}")
        prefix = " · ".join(metric_parts)
        print(f"{i}. **{prefix}** — {attr_text}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paths", nargs="+", required=True,
                   help="One or more search results.json paths")
    p.add_argument("--k", type=int, default=None,
                   help="How many top attrs to print per file (default: all)")
    p.add_argument("--sort", choices=["a_hat", "bon_delta"], default="a_hat",
                   help="Sort key (default: a_hat)")
    p.add_argument("--bon_root", type=str, default="outputs/bon",
                   help="Root directory to scan for matching bon runs "
                        "(default: outputs/bon)")
    p.add_argument("--bon_n", type=int, default=None,
                   help="BoN delta reference: use prev[n=BON_N] - prev[n=1] "
                        "(default: largest n in bon's n_values)")
    args = p.parse_args()

    bon_root = Path(args.bon_root)
    for path_str in args.paths:
        _print_section(Path(path_str), k=args.k, sort_by=args.sort,
                       bon_root=bon_root, bon_n=args.bon_n)


if __name__ == "__main__":
    main()