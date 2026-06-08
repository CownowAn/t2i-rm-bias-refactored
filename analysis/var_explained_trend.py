"""Plot var_explained vs step for one or more search runs.

For each given search run directory, scans `ba_expand_step{N}_topic{T}.json`,
extracts `reg_var_explained` per step, and saves one plot per run inside that
run's own directory (`{run_dir}/var_explained_trend.png`). Optionally also
saves a combined overlay PNG across all runs.

Usage:
    python -m analysis.var_explained_trend --runs RUN_DIR [RUN_DIR ...] [--combined_out PATH]

Examples:
    # Per-run PNG in each run's directory
    python -m analysis.var_explained_trend \
        --runs outputs/search/20260514-020231 \
               outputs/search/20260518-222320 \
               outputs/search/20260519-152519

    # Also save a combined overlay
    python -m analysis.var_explained_trend --runs outputs/search/* \
        --combined_out outputs/var_explained_trend_combined.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


_PAT = re.compile(r"ba_expand_step(\d+)_topic(\d+)\.json$")


def collect_run(run_dir: Path) -> dict[int, list[tuple[int, float, int]]]:
    """Return {topic_id: [(step_idx, var_explained, K), ...]} sorted by step."""
    by_topic: dict[int, list[tuple[int, float, int]]] = {}
    for p in run_dir.glob("ba_expand_step*_topic*.json"):
        m = _PAT.search(p.name)
        if not m:
            continue
        step = int(m.group(1))
        topic = int(m.group(2))
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        ve = float(d.get("reg_var_explained", 0.0))
        K = len(d.get("acc_pool", []))
        by_topic.setdefault(topic, []).append((step, ve, K))
    for topic in by_topic:
        by_topic[topic].sort()
    return by_topic


def _print_markdown_table(per_run: list[tuple[str, int, list[tuple[int, float, int]]]]) -> None:
    if not per_run:
        return
    max_steps = max(len(traj) for _, _, traj in per_run)
    print()
    print("### var_explained per step")
    header = "| run / topic |" + "".join(f" s{i} |" for i in range(max_steps))
    sep    = "|---|" + "---|" * max_steps
    print(header)
    print(sep)
    for name, topic, traj in per_run:
        cells = []
        for i in range(max_steps):
            if i < len(traj):
                step, ve, K = traj[i]
                cells.append(f" {ve:.3f} (K={K}) ")
            else:
                cells.append(" — ")
        print(f"| `{name}` / t{topic} |" + "|".join(cells) + "|")


def _plot_single(name: str, topic: int, traj: list[tuple[int, float, int]],
                  out_path: Path) -> None:
    xs = [s for s, _, _ in traj]
    ys = [ve for _, ve, _ in traj]
    Ks = [K for _, _, K in traj]
    if not xs:
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    line_color = "#1f77b4"
    ax.plot(xs, ys, marker="o", linewidth=2.5, color=line_color, zorder=2)

    # K labels above each marker (small font, in plot area)
    for x, y, K in zip(xs, ys, Ks):
        ax.annotate(
            f"K={K}",
            xy=(x, y), xytext=(0, 9), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, color="#666", zorder=3,
        )

    # Emphasize first and last with bigger markers
    ax.scatter([xs[0], xs[-1]], [ys[0], ys[-1]], s=140, color=line_color, zorder=4,
               edgecolor="white", linewidth=1.5)

    # First-value annotation: just below the marker, offset DOWN-RIGHT
    ax.annotate(
        f"init: {ys[0]:.3f}",
        xy=(xs[0], ys[0]), xytext=(8, -18), textcoords="offset points",
        ha="left", va="top", fontsize=12, color=line_color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=line_color, alpha=0.95),
        zorder=5,
    )
    # Final-value annotation: below the line, offset DOWN-LEFT, pushed further down
    ax.annotate(
        f"final: {ys[-1]:.3f}",
        xy=(xs[-1], ys[-1]), xytext=(-8, -44), textcoords="offset points",
        ha="right", va="top", fontsize=12, color=line_color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=line_color, alpha=0.95),
        zorder=5,
    )

    ax.set_xlabel("EXPAND step")
    ax.set_ylabel("var_explained  (= mean per-prompt R²)")
    ax.set_title(f"{name}   topic={topic}", pad=10)
    ax.grid(True, alpha=0.3)

    ax.set_xticks(xs)
    # Tight vertical limits — just enough room for K labels above
    # and init label below (which extends ~30 px below the first marker).
    y_min, y_max = min(ys), max(ys)
    y_range = y_max - y_min
    top_pad = max(0.012, y_max * 0.06)
    bot_pad = max(0.025, y_range * 0.30)
    ax.set_ylim(bottom=max(0.0, y_min - bot_pad), top=y_max + top_pad)
    ax.set_xlim(left=xs[0] - 0.5, right=xs[-1] + 0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_combined(per_run, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    cmap = plt.get_cmap("tab10")
    for i, (name, topic, traj) in enumerate(per_run):
        xs = [s for s, _, _ in traj]
        ys = [ve for _, ve, _ in traj]
        ax.plot(xs, ys, marker="o", linewidth=2.0, color=cmap(i % 10),
                label=f"{name} (t{topic})")
        if xs:
            ax.text(xs[-1] + 0.05, ys[-1], f"{ys[-1]:.3f}",
                    va="center", fontsize=10, color=cmap(i % 10))
    ax.set_xlabel("EXPAND step")
    ax.set_ylabel("var_explained  (= mean per-prompt R² in per_prompt mode)")
    ax.set_title("Pool explanatory power across steps — multiple search runs", pad=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", framealpha=0.85)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True,
                   help="One or more search run directories (each contains ba_expand_step*_topic*.json)")
    p.add_argument("--filename", type=str, default="var_explained_trend.png",
                   help="Filename used inside each run dir (default: var_explained_trend.png)")
    p.add_argument("--combined_out", type=str, default=None,
                   help="If set, also save a combined overlay PNG here")
    p.add_argument("--no_plot", action="store_true", help="Skip plot, only print table")
    args = p.parse_args()

    per_run: list[tuple[str, int, list[tuple[int, float, int]]]] = []
    per_run_with_dir: list[tuple[Path, str, int, list[tuple[int, float, int]]]] = []
    for r in args.runs:
        run_dir = Path(r)
        if not run_dir.is_dir():
            print(f"[WARN] not a directory: {run_dir}")
            continue
        by_topic = collect_run(run_dir)
        if not by_topic:
            print(f"[WARN] no ba_expand_step*_topic*.json in {run_dir}")
            continue
        for topic, traj in sorted(by_topic.items()):
            per_run.append((run_dir.name, topic, traj))
            per_run_with_dir.append((run_dir, run_dir.name, topic, traj))

    if not per_run:
        print("[ERROR] no runs with data found")
        return

    # ── Print markdown table ─────────────────────────────────────────────────
    _print_markdown_table(per_run)

    if args.no_plot:
        return

    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10,
    })

    # ── One plot per run ────────────────────────────────────────────────────
    for run_dir, name, topic, traj in per_run_with_dir:
        # If a run has multiple topics, suffix with topic id to avoid overwrite
        topics_in_run = {t for r2, _, t, _ in per_run_with_dir if r2 == run_dir}
        if len(topics_in_run) > 1:
            stem = Path(args.filename).stem
            suffix = Path(args.filename).suffix
            fname = f"{stem}_topic{topic}{suffix}"
        else:
            fname = args.filename
        out_path = run_dir / fname
        _plot_single(name, topic, traj, out_path)
        print(f"Saved per-run plot: {out_path}")

    # ── Optional combined overlay ───────────────────────────────────────────
    if args.combined_out:
        _plot_combined(per_run, Path(args.combined_out))
        print(f"\nSaved combined plot: {args.combined_out}")


if __name__ == "__main__":
    main()