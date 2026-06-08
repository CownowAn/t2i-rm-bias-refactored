"""Plot adjusted R² vs step for search runs (penalises pool size growth).

Background
──────────
The search log already reports `reg_var_explained = mean per-prompt R²`. Per-prompt
OLS fits a K-dim weight vector W_x on M_x centered observations per prompt; R² is
non-decreasing in K, so the headline trend just confirms "more attrs ⇒ better fit"
without telling us whether the marginal attr is worth its degree of freedom.

Adjusted R² ([Wherry 1931]) is the standard correction:

    R²_adj = 1 - (1 - R²) · (n - 1) / (n - K)

with n = M_x observations per prompt and K = pool size. We adjust per-prompt
(matching how the search itself fits), then average across prompts.

This script reads only on-disk artifacts of a completed search run:
  • per_prompt_r2_history_topic{T}.json — step-by-step per-prompt R²
  • ba_expand_step{N}_topic{T}.json     — K (= |acc_pool|) and n_residual_images
…and writes `{run_dir}/adjusted_r2_trend.png` plus a markdown table.

Usage:
    python -m analysis.adjusted_r2_trend --runs outputs/search/20260601-154143
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


_BA_PAT = re.compile(r"ba_expand_step(\d+)_topic(\d+)\.json$")


# ── Data ──────────────────────────────────────────────────────────────────────


def _load_pp_history(run_dir: Path, topic_id: int) -> list[dict]:
    path = run_dir / f"per_prompt_r2_history_topic{topic_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    with open(path) as f:
        return json.load(f)["history"]


def _collect_step_meta(run_dir: Path) -> dict[int, list[tuple[int, int, int]]]:
    """Return {topic_id: [(step_idx, K, n_residual_images), ...] sorted by step}."""
    by_topic: dict[int, list[tuple[int, int, int]]] = {}
    for p in run_dir.glob("ba_expand_step*_topic*.json"):
        m = _BA_PAT.search(p.name)
        if not m:
            continue
        step, topic = int(m.group(1)), int(m.group(2))
        with open(p) as f:
            d = json.load(f)
        K = len(d.get("acc_pool", []))
        n_res = int(d.get("n_residual_images", 0))
        by_topic.setdefault(topic, []).append((step, K, n_res))
    for t in by_topic:
        by_topic[t].sort()
    return by_topic


# ── Adjusted-R² math ──────────────────────────────────────────────────────────


def _adjusted_r2(r2: float, n: int, k: int) -> float:
    """Wherry's adjusted R² for a single prompt. Returns NaN-safe finite value.

    For centered (no-intercept) OLS, dof_total = n − 1 (mean is subtracted),
    dof_residual = n − K. If K ≥ n, return -inf-clamped value (unfittable).
    """
    dof_resid = n - k
    if dof_resid <= 0:
        return float("nan")
    return 1.0 - (1.0 - r2) * (n - 1) / dof_resid


def _step_adjusted_mean(
    pp_r2: dict[str, float],
    K: int,
    n_per_prompt: int,
) -> tuple[float, int]:
    """Average adjusted R² across prompts, ignoring NaN entries."""
    vals = []
    for r2 in pp_r2.values():
        v = _adjusted_r2(float(r2), n_per_prompt, K)
        if v == v:                              # filter NaN
            vals.append(v)
    if not vals:
        return float("nan"), 0
    return sum(vals) / len(vals), len(vals)


# ── Aggregation per run ───────────────────────────────────────────────────────


def collect_run(run_dir: Path) -> dict[int, list[dict]]:
    """{topic_id: [{step,K,n_per_prompt,raw,adj,P}, ...]} sorted by step."""
    meta = _collect_step_meta(run_dir)
    out: dict[int, list[dict]] = {}
    for topic_id, step_rows in meta.items():
        try:
            hist = _load_pp_history(run_dir, topic_id)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue
        # key history by step for fast lookup
        hist_by_step = {int(entry["step"]): entry for entry in hist}
        rows: list[dict] = []
        for step, K, n_res in step_rows:
            entry = hist_by_step.get(step)
            if entry is None:
                continue
            pp_r2 = {p: v for p, v in entry.items() if p != "step"}
            P = len(pp_r2)
            if P == 0 or n_res == 0:
                continue
            n_per_prompt = n_res // P             # 128 in the canonical setup
            raw = sum(float(v) for v in pp_r2.values()) / P
            adj, _ = _step_adjusted_mean(pp_r2, K, n_per_prompt)
            rows.append({
                "step": step, "K": K,
                "n_per_prompt": n_per_prompt, "P": P,
                "raw": raw, "adj": adj,
            })
        if rows:
            out[topic_id] = rows
    return out


# ── Plot ──────────────────────────────────────────────────────────────────────


def _plot_delta(run_name: str, topic_id: int, rows: list[dict], out_path: Path) -> None:
    """Trend (line) plot of ΔR²_t = R²(S_t) − R²(S_{t-1})  (raw)."""
    if len(rows) < 2:
        return
    xs    = [r["step"] for r in rows[1:]]                    # Δ starts at step 1
    d_raw = [rows[i]["raw"] - rows[i - 1]["raw"] for i in range(1, len(rows))]
    dK    = [rows[i]["K"]   - rows[i - 1]["K"]   for i in range(1, len(rows))]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    raw_color = "#1f77b4"
    ax.plot(xs, d_raw, marker="o", linewidth=2.5, color=raw_color,
            label="Δ R² (raw)", zorder=2)

    # ΔK labels above each marker
    for x, y, k_add in zip(xs, d_raw, dK):
        ax.annotate(
            f"+{k_add}" if k_add else "0",
            xy=(x, y), xytext=(0, 9), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, color="#666", zorder=3,
        )

    ax.scatter([xs[0], xs[-1]], [d_raw[0], d_raw[-1]], s=120,
               color=raw_color, zorder=4, edgecolor="white", linewidth=1.5)
    ax.annotate(f"first: {d_raw[0]:+.3f}",
                xy=(xs[0], d_raw[0]), xytext=(8, 18),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=11, color=raw_color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=raw_color, alpha=0.95), zorder=5)
    ax.annotate(f"last: {d_raw[-1]:+.3f}",
                xy=(xs[-1], d_raw[-1]), xytext=(-8, 18),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=11, color=raw_color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=raw_color, alpha=0.95), zorder=5)

    n_per = rows[0]["n_per_prompt"]
    P     = rows[0]["P"]
    ax.set_xlabel("EXPAND step  (Δ between t and t−1, ΔK above markers)")
    ax.set_ylabel("Δ R²")
    ax.set_title(
        f"{run_name}   topic={topic_id}   "
        f"P={P} × n={n_per} — incremental R² per step",
        pad=10,
    )
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_run(run_name: str, topic_id: int, rows: list[dict], out_path: Path) -> None:
    xs = [r["step"] for r in rows]
    raws = [r["raw"] for r in rows]
    adjs = [r["adj"] for r in rows]
    Ks = [r["K"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    train_color = "#1f77b4"
    adj_color   = "#d62728"

    ax.plot(xs, raws, marker="o", linewidth=2.5, color=train_color,
            label="R² (raw)", zorder=2)
    ax.plot(xs, adjs, marker="s", linewidth=2.5, color=adj_color,
            label="R²_adj (Wherry)", zorder=2, linestyle="--")

    # K labels above the higher curve
    upper = [max(r, a) for r, a in zip(raws, adjs)]
    for x, y, K in zip(xs, upper, Ks):
        ax.annotate(f"K={K}", xy=(x, y), xytext=(0, 9),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, color="#666", zorder=3)

    for series, color in ((raws, train_color), (adjs, adj_color)):
        ax.scatter([xs[0], xs[-1]], [series[0], series[-1]], s=120,
                   color=color, zorder=4, edgecolor="white", linewidth=1.5)

    ax.annotate(f"raw final: {raws[-1]:.3f}",
                xy=(xs[-1], raws[-1]), xytext=(-8, 18),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=11, color=train_color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=train_color, alpha=0.95), zorder=5)
    ax.annotate(f"adj final: {adjs[-1]:.3f}",
                xy=(xs[-1], adjs[-1]), xytext=(-8, -18),
                textcoords="offset points", ha="right", va="top",
                fontsize=11, color=adj_color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=adj_color, alpha=0.95), zorder=5)

    # Argmax marker (best generalisation step under the penalty)
    if any(a == a for a in adjs):
        i_star = max(range(len(adjs)), key=lambda i: adjs[i])
        ax.axvline(xs[i_star], color=adj_color, alpha=0.25, linestyle=":")
        ax.text(xs[i_star], min(min(adjs), min(raws)),
                f" peak adj @ step {xs[i_star]} (K={Ks[i_star]}, adj={adjs[i_star]:.3f})",
                color=adj_color, fontsize=10, va="bottom", ha="left")

    n_per = rows[0]["n_per_prompt"]
    P     = rows[0]["P"]
    ax.set_xlabel("EXPAND step")
    ax.set_ylabel("R²  (mean per-prompt)")
    ax.set_title(
        f"{run_name}   topic={topic_id}   "
        f"P={P} prompts × n={n_per} imgs/prompt — raw vs adjusted R²",
        pad=10,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.85)
    ax.set_xticks(xs)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ── Markdown summary ──────────────────────────────────────────────────────────


def _print_markdown(per_run: list[tuple[str, int, list[dict]]]) -> None:
    if not per_run:
        return
    max_steps = max(len(rows) for _, _, rows in per_run)
    print()
    print("### R² per step  (raw → adjusted)")
    header = "| run / topic |" + "".join(f" s{i} |" for i in range(max_steps))
    sep    = "|---|" + "---|" * max_steps
    print(header)
    print(sep)
    for name, topic, rows in per_run:
        cells = []
        for i in range(max_steps):
            if i < len(rows):
                r = rows[i]
                cells.append(f" {r['raw']:.3f}→{r['adj']:.3f} ")
            else:
                cells.append(" — ")
        print(f"| `{name}` / t{topic} |" + "|".join(cells) + "|")

    print()
    print("### Δ R²  per step  (ΔK in parens)")
    header = "| run / topic |" + "".join(f" Δs{i} |" for i in range(1, max_steps))
    sep    = "|---|" + "---|" * (max_steps - 1)
    print(header)
    print(sep)
    for name, topic, rows in per_run:
        cells = []
        for i in range(1, max_steps):
            if i < len(rows):
                r, p = rows[i], rows[i - 1]
                d_raw = r["raw"] - p["raw"]
                dK    = r["K"]   - p["K"]
                cells.append(f" {d_raw:+.3f} (+{dK}) ")
            else:
                cells.append(" — ")
        print(f"| `{name}` / t{topic} |" + "|".join(cells) + "|")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--runs", nargs="+", required=True,
                   help="One or more search run directories")
    p.add_argument("--filename", default="adjusted_r2_trend.png",
                   help="Filename inside each run dir (level plot)")
    p.add_argument("--delta_filename", default="delta_r2_trend.png",
                   help="Filename for the per-step Δ R² bar plot")
    p.add_argument("--no_plot", action="store_true")
    args = p.parse_args()

    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    })

    per_run: list[tuple[str, int, list[dict]]] = []
    per_run_dir: list[tuple[Path, str, int, list[dict]]] = []
    for r in args.runs:
        run_dir = Path(r)
        if not run_dir.is_dir():
            print(f"[WARN] not a directory: {run_dir}")
            continue
        topic_rows = collect_run(run_dir)
        if not topic_rows:
            print(f"[WARN] no rows for {run_dir}")
            continue
        for topic_id, rows in sorted(topic_rows.items()):
            print(f"\n=== {run_dir.name}  topic {topic_id} ===")
            for r_ in rows:
                print(f"  step {r_['step']:>2}  K={r_['K']:>3}  "
                      f"raw={r_['raw']:.3f}  adj={r_['adj']:+.3f}")
            per_run.append((run_dir.name, topic_id, rows))
            per_run_dir.append((run_dir, run_dir.name, topic_id, rows))

    _print_markdown(per_run)

    if args.no_plot:
        return

    for run_dir, name, topic, rows in per_run_dir:
        topics_in_run = {t for r2, _, t, _ in per_run_dir if r2 == run_dir}
        if len(topics_in_run) > 1:
            stem      = Path(args.filename).stem
            suffix    = Path(args.filename).suffix
            stem_d    = Path(args.delta_filename).stem
            suffix_d  = Path(args.delta_filename).suffix
            fname   = f"{stem}_topic{topic}{suffix}"
            fname_d = f"{stem_d}_topic{topic}{suffix_d}"
        else:
            fname   = args.filename
            fname_d = args.delta_filename

        out_path   = run_dir / fname
        out_path_d = run_dir / fname_d
        _plot_run(name, topic, rows, out_path)
        print(f"Saved level plot → {out_path}")
        _plot_delta(name, topic, rows, out_path_d)
        print(f"Saved delta plot → {out_path_d}")


if __name__ == "__main__":
    main()