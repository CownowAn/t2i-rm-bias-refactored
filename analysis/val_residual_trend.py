"""Plot val_var_explained (held-out) vs step for one or more search runs.

For each given search run directory:
  1. Reads `config_effective.yaml` to recover the baseline manifest, val split,
     reward model, BoN N, OLS mode, and detector model key.
  2. Loads val prompts (`load_val_topic_state`) and val baseline images
     (`load_baselines_from_manifest`) — same seed as training, so this matches
     the held-out 40 prompts that search never touched.
  3. Loads detection caches from BoN (`outputs/bon/cache/<data>/topic{T}.json`)
     and search (`bon_amplified.detection_cache_path`) and merges them.
  4. For each `ba_expand_step{N}_topic{T}.json`, takes the recorded `acc_pool`
     and re-runs `_compute_bon_residuals` on the val baselines to get a
     held-out var_explained per step.
  5. Saves one plot per run inside that run's directory
     (`{run_dir}/val_residual_trend.png`) overlaying train (from JSON) and val
     (newly computed) trajectories.

Usage:
    python -m analysis.val_residual_trend \
        --runs outputs/search/20260514-020231 outputs/search/20260519-152519

If val detections are not fully cached for a step's acc_pool, that step is
reported as incomplete (n_val_prompts shrinks) but still plotted using only
val images with complete detections — exactly the same filter the training
residual code uses.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from search.pipeline.baselines import (  # noqa: E402
    load_baselines_from_manifest,
    load_val_topic_state,
)
from search.pipeline.bon_amplified_evo import _compute_bon_residuals  # noqa: E402


_PAT = re.compile(r"ba_expand_step(\d+)_topic(\d+)\.json$")


# ── Loaders ───────────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _dataset_name_from_manifest(manifest_path: str) -> str:
    """Parse '.../data/baselines/<DATASET>/topic_<N>/<POLICY>/manifest.json'."""
    p = Path(manifest_path)
    # parents: [POLICY, topic_X, DATASET, baselines, data, ...]
    return p.parent.parent.parent.name


def _load_detection_caches(
    cfg: dict, topic_id: int, extra_paths: list[Path],
) -> dict[str, dict[str, int]]:
    """Merge detection entries from BoN per-topic cache and the search cache,
    keyed by detector model_key. Returns {image_id: {attr: 0/1}}."""
    det_cfg = cfg["models"]["detector"]
    model_key = f"{det_cfg['model']}::{det_cfg['image_detail']}"

    candidates: list[Path] = []
    # BoN per-topic global cache (populated by bon/runner.py on val baselines)
    manifest = cfg["data"]["baseline_manifest"]
    data_name = _dataset_name_from_manifest(manifest)
    bon_cache = REPO_ROOT / "outputs" / "bon" / "cache" / data_name / f"topic{topic_id}.json"
    if bon_cache.exists():
        candidates.append(bon_cache)
    # BoN per-run snapshots — each contains the attrs for the specific search
    # results it was invoked with. The global cache can be partially overwritten
    # when multiple bon runs hit the same topic, so per-run snapshots provide
    # the most complete attribute coverage.
    per_run_snaps = sorted(
        (REPO_ROOT / "outputs" / "bon").glob(
            f"bon_*/detection_cache_topic{topic_id}.json"
        )
    )
    candidates.extend(per_run_snaps)
    # Search cache (won't contain val images, but harmless to merge)
    sc = cfg.get("bon_amplified", {}).get("detection_cache_path")
    if sc:
        sc_path = Path(sc)
        if not sc_path.is_absolute():
            sc_path = REPO_ROOT / sc_path
        if sc_path.exists():
            candidates.append(sc_path)
    candidates.extend(extra_paths)

    merged: dict[str, dict[str, int]] = {}
    for path in candidates:
        try:
            with open(path) as f:
                blob = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[WARN] could not read detection cache {path}: {e}")
            continue
        saved = blob.get(model_key, {})
        for image_id, attrs in saved.items():
            merged.setdefault(image_id, {}).update(
                {a: int(bool(v)) for a, v in attrs.items()}
            )
        print(f"  merged {len(saved)} images from {path}")
    return merged


def _collect_step_files(run_dir: Path) -> dict[int, list[tuple[int, Path]]]:
    """Return {topic_id: [(step_idx, json_path), ...] sorted by step}."""
    by_topic: dict[int, list[tuple[int, Path]]] = {}
    for p in run_dir.glob("ba_expand_step*_topic*.json"):
        m = _PAT.search(p.name)
        if not m:
            continue
        step, topic = int(m.group(1)), int(m.group(2))
        by_topic.setdefault(topic, []).append((step, p))
    for k in by_topic:
        by_topic[k].sort()
    return by_topic


# ── Core computation ──────────────────────────────────────────────────────────


def _diagnose_coverage(
    val_baselines: dict,
    detection_cache: dict[str, dict[str, int]],
    acc_pool: list[str],
) -> tuple[int, int, list[str]]:
    """Return (n_images_full_coverage, n_images_total, attrs_completely_missing).

    `attrs_completely_missing` are acc_pool attrs that no val image has a
    detection entry for — usually means BoN runner never detected them on val.
    """
    all_imgs = [img for imgs in val_baselines.values() for img in imgs]
    n_total = len(all_imgs)
    full = 0
    attr_present_anywhere: dict[str, int] = {a: 0 for a in acc_pool}
    for img in all_imgs:
        cache_row = detection_cache.get(img.image_id, {})
        if all(a in cache_row for a in acc_pool):
            full += 1
        for a in acc_pool:
            if a in cache_row:
                attr_present_anywhere[a] += 1
    missing = [a for a, c in attr_present_anywhere.items() if c == 0]
    return full, n_total, missing


def _val_var_explained(
    val_baselines: dict,
    detection_cache: dict[str, dict[str, int]],
    acc_pool: list[str],
    reward_model_name: str,
    N: int,
    mode: str,
) -> tuple[float, int, int]:
    """Run _compute_bon_residuals on val baselines. Return (var_exp, n_prompts_used, n_images_used)."""
    if not acc_pool:
        return 0.0, 0, 0
    _residuals, _W, var_exp, _ma, _mx, per_prompt_r2, _pp_W = _compute_bon_residuals(
        detection_cache, val_baselines, acc_pool,
        reward_model_name, N, mode=mode,
    )
    n_prompts = len(per_prompt_r2)
    n_images = len(_residuals)
    return float(var_exp), n_prompts, n_images


def _run_for_topic(
    run_dir: Path,
    topic_id: int,
    step_files: list[tuple[int, Path]],
    cfg: dict,
    extra_cache_paths: list[Path],
) -> list[dict]:
    """Compute val var_explained for every (step, topic) in this run."""
    data_cfg = cfg["data"]
    ba_cfg = cfg["bon_amplified"]
    reward_name = cfg["models"]["reward_model"]["name"]
    N = int(ba_cfg["N"])
    mode = "per_prompt" if ba_cfg.get("use_per_prompt_ols", False) else "global"

    val_state = load_val_topic_state(
        prompts_dir=data_cfg["prompts_dir"],
        topic_id=topic_id,
        val_split_size=int(data_cfg.get("val_split_size", 40)),
        random_seed=int(cfg["run"]["random_seed"]),
        summary_field=data_cfg.get("cluster_summary_field", "summary"),
    )
    load_baselines_from_manifest(
        topic_state=val_state,
        manifest_path=data_cfg["baseline_manifest"],
        baseline_root=data_cfg.get("baseline_root", ""),
    )
    n_val_prompts_loaded = len(val_state.baselines)
    if n_val_prompts_loaded == 0:
        print(f"[WARN] topic {topic_id}: no val baselines could be loaded — skipping")
        return []

    detection_cache = _load_detection_caches(cfg, topic_id, extra_cache_paths)

    # If the run was made with `not_applicable_as_absent=True`, the cache has
    # -1 entries marking "not applicable". Treat them as absent (=0) so the
    # OLS sees the same values the live search did.
    if cfg.get("models", {}).get("detector", {}).get("not_applicable_as_absent", False):
        n_collapsed = 0
        for image_id, attr_vals in detection_cache.items():
            for attr, v in list(attr_vals.items()):
                if v == -1:
                    attr_vals[attr] = 0
                    n_collapsed += 1
        if n_collapsed:
            print(
                f"  topic {topic_id}: not_applicable_as_absent=True → "
                f"collapsed {n_collapsed} '-1' cache entries to 0"
            )

    print(
        f"  topic {topic_id}: val prompts={n_val_prompts_loaded}, "
        f"detection-cached images={len(detection_cache)}, ols_mode={mode}"
    )

    rows: list[dict] = []
    for step_idx, json_path in step_files:
        with open(json_path) as f:
            d = json.load(f)
        acc_pool = list(d.get("acc_pool", []))
        train_var = float(d.get("reg_var_explained", 0.0))
        K = len(acc_pool)
        full, total, missing = _diagnose_coverage(
            val_state.baselines, detection_cache, acc_pool,
        )
        val_var, n_p, n_i = _val_var_explained(
            val_state.baselines, detection_cache, acc_pool,
            reward_name, N, mode,
        )
        rows.append({
            "step": step_idx,
            "K": K,
            "train_var": train_var,
            "val_var": val_var,
            "val_n_prompts": n_p,
            "val_n_images": n_i,
            "val_coverage": (full, total),
            "val_attrs_missing": missing,
        })
        cov_str = f"cov={full}/{total}"
        miss_str = f"  ⚠ {len(missing)} acc_pool attrs not in val cache" if missing else ""
        print(
            f"    step {step_idx:>2}  K={K:>2}  "
            f"train={train_var:.3f}  val={val_var:.3f}  "
            f"({cov_str}, {n_p} prompts){miss_str}"
        )
        if missing and step_idx == step_files[-1][0]:
            # On the final step, print the missing attrs so the user can re-detect.
            for a in missing[:6]:
                print(f"      missing: {a[:90]}")
            if len(missing) > 6:
                print(f"      ... and {len(missing) - 6} more")
    return rows


# ── Plotting ──────────────────────────────────────────────────────────────────


def _plot_run(name: str, topic: int, rows: list[dict], out_path: Path) -> None:
    xs = [r["step"] for r in rows]
    train_ys = [r["train_var"] for r in rows]
    val_ys = [r["val_var"] for r in rows]
    Ks = [r["K"] for r in rows]
    if not xs:
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    train_color = "#1f77b4"
    val_color = "#d62728"

    ax.plot(xs, train_ys, marker="o", linewidth=2.5, color=train_color,
            label="train", zorder=2)
    ax.plot(xs, val_ys, marker="s", linewidth=2.5, color=val_color,
            label="val (held-out)", zorder=2, linestyle="--")

    # K labels above markers
    y_max_pair = [max(t, v) for t, v in zip(train_ys, val_ys)]
    for x, y, K in zip(xs, y_max_pair, Ks):
        ax.annotate(
            f"K={K}", xy=(x, y), xytext=(0, 9), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, color="#666", zorder=3,
        )

    # Emphasize first / last on both series
    for series, color in ((train_ys, train_color), (val_ys, val_color)):
        ax.scatter([xs[0], xs[-1]], [series[0], series[-1]], s=120,
                   color=color, zorder=4, edgecolor="white", linewidth=1.5)

    # Val final annotation
    ax.annotate(
        f"val final: {val_ys[-1]:.3f}",
        xy=(xs[-1], val_ys[-1]), xytext=(-8, -18), textcoords="offset points",
        ha="right", va="top", fontsize=11, color=val_color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=val_color, alpha=0.95),
        zorder=5,
    )
    ax.annotate(
        f"train final: {train_ys[-1]:.3f}",
        xy=(xs[-1], train_ys[-1]), xytext=(-8, 18), textcoords="offset points",
        ha="right", va="bottom", fontsize=11, color=train_color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=train_color, alpha=0.95),
        zorder=5,
    )

    ax.set_xlabel("EXPAND step")
    ax.set_ylabel("var_explained  (= mean per-prompt R²)")
    ax.set_title(f"{name}   topic={topic}   train vs val", pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.85)

    ax.set_xticks(xs)
    all_ys = train_ys + val_ys
    y_min, y_max = min(all_ys), max(all_ys)
    y_range = max(y_max - y_min, 1e-6)
    ax.set_ylim(bottom=max(0.0, y_min - 0.25 * y_range),
                top=y_max + 0.18 * y_range + 0.02)
    ax.set_xlim(left=xs[0] - 0.5, right=xs[-1] + 0.5)

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
    print("### var_explained per step  (train → val)")
    header = "| run / topic |" + "".join(f" s{i} |" for i in range(max_steps))
    sep = "|---|" + "---|" * max_steps
    print(header)
    print(sep)
    for name, topic, rows in per_run:
        cells: list[str] = []
        for i in range(max_steps):
            if i < len(rows):
                r = rows[i]
                cells.append(f" {r['train_var']:.3f}→{r['val_var']:.3f} ")
            else:
                cells.append(" — ")
        print(f"| `{name}` / t{topic} |" + "|".join(cells) + "|")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runs", nargs="+", required=True,
                   help="One or more search run directories")
    p.add_argument("--filename", type=str, default="val_residual_trend.png",
                   help="Plot filename inside each run dir")
    p.add_argument("--extra_cache", nargs="*", default=[],
                   help="Optional extra detection-cache JSON paths to merge")
    p.add_argument("--no_plot", action="store_true",
                   help="Skip plotting, only print table")
    args = p.parse_args()

    extra_cache_paths = [Path(c) for c in args.extra_cache]
    per_run_rows: list[tuple[str, int, list[dict]]] = []
    per_run_with_dir: list[tuple[Path, str, int, list[dict]]] = []

    for r in args.runs:
        run_dir = Path(r)
        if not run_dir.is_dir():
            print(f"[WARN] not a directory: {run_dir}")
            continue
        cfg_path = run_dir / "configs" / "config_effective.yaml"
        if not cfg_path.exists():
            legacy = run_dir / "config_effective.yaml"
            if legacy.exists():
                cfg_path = legacy
            else:
                print(f"[WARN] no config_effective.yaml in {run_dir}/configs — skipping")
                continue
        cfg = _load_yaml(cfg_path)
        by_topic = _collect_step_files(run_dir)
        if not by_topic:
            print(f"[WARN] no ba_expand_step*_topic*.json in {run_dir}")
            continue

        print(f"\n=== {run_dir.name} ===")
        for topic_id, step_files in sorted(by_topic.items()):
            rows = _run_for_topic(
                run_dir, topic_id, step_files, cfg, extra_cache_paths
            )
            if not rows:
                continue
            per_run_rows.append((run_dir.name, topic_id, rows))
            per_run_with_dir.append((run_dir, run_dir.name, topic_id, rows))

    if not per_run_rows:
        print("[ERROR] no runs with data found")
        return

    _print_markdown(per_run_rows)

    if args.no_plot:
        return

    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
    })

    for run_dir, name, topic, rows in per_run_with_dir:
        topics_in_run = {t for r2, _, t, _ in per_run_with_dir if r2 == run_dir}
        if len(topics_in_run) > 1:
            stem = Path(args.filename).stem
            suffix = Path(args.filename).suffix
            fname = f"{stem}_topic{topic}{suffix}"
        else:
            fname = args.filename
        out_path = run_dir / fname
        _plot_run(name, topic, rows, out_path)
        print(f"Saved val/train trend plot: {out_path}")


if __name__ == "__main__":
    main()
