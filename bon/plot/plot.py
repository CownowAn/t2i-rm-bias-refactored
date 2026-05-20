import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt


def slugify(text: str, max_len: int = 80) -> str:
    # Create a filesystem-safe filename fragment
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:max_len] or "attribute"


def plot_one_attribute(json_path: Path, data: dict, attr: str, attr_idx: int):
    n_values = data["n_values"]
    prevalence = data["prevalence"][attr]

    x0, y0 = n_values[0], prevalence[0]
    x1, y1 = n_values[-1], prevalence[-1]
    delta = y1 - y0

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(n_values, prevalence, marker="o", linewidth=2)
    ax.set_xscale("log", base=2)

    # Draw horizontal reference lines for n=1 and max n
    ax.axhline(y0, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axhline(y1, linestyle="--", linewidth=1.5, alpha=0.8)

    # Highlight the first and last points
    ax.scatter([x0, x1], [y0, y1], s=120, zorder=3)

    # Draw an arrow showing the difference
    arrow_x = x1 * 0.82
    ax.annotate(
        "",
        xy=(arrow_x, y1),
        xytext=(arrow_x, y0),
        arrowprops=dict(arrowstyle="<->", linewidth=2),
    )

    # Add labels for the first and last values
    ax.text(x0, y0, f"  n={x0}: {y0:.3f}", va="bottom", fontsize=14)
    ax.text(x1, y1, f"  n={x1}: {y1:.3f}", va="bottom", ha="right", fontsize=14)

    # Add delta annotation
    pct_delta = (delta / y0 * 100) if y0 != 0 else float("nan")
    pct_text = f"{pct_delta:.1f}%" if y0 != 0 else "n/a"

    ax.text(
        arrow_x * 0.93,
        (y0 + y1) / 2,
        f"Δ = {delta:.3f}\n({pct_text})",
        va="center",
        ha="right",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.4", alpha=0.2),
    )

    wrapped_title = "\n".join(textwrap.wrap(attr, width=50)) or attr
    ax.set_title(wrapped_title, pad=12, fontsize=14)
    ax.set_xlabel("n_values")
    ax.set_ylabel("Prevalence")

    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.tick_params(axis="both", which="major", length=6, width=1.5)
    ax.grid(True, which="both", alpha=0.3)

    # Add vertical margin for readability
    y_min, y_max = min(prevalence), max(prevalence)
    y_margin = max(0.01, (y_max - y_min) * 0.25)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # Use a fixed layout area so figure size stays constant across titles.
    # `top` leaves room for up to ~3 wrapped title lines.
    fig.subplots_adjust(left=0.1, right=0.97, top=0.78, bottom=0.12)

    # Save one plot per attribute in the same directory as the input JSON file
    attr_slug = slugify(attr)
    output_path = json_path.with_name(
        f"{json_path.stem}_attr{attr_idx:02d}_{attr_slug}_prevalence_by_n.png"
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_summary(json_path: Path, data: dict, mode: str = "subtracted"):
    """Mean prevalence curve across all attributes (with ±1 std band).

    mode="subtracted": y = mean over attrs of (prev[n] - prev[n=1]).
                       Each attr starts at 0; isolates BoN-induced amplification.
                       Saved as {stem}_summary.png
    mode="raw":        y = mean over attrs of prev[n]. Absolute prevalence
                       averaged across attrs. Saved as {stem}_summary_raw.png
    """
    import numpy as np

    n_values = data["n_values"]
    prevalence = data["prevalence"]
    attrs = [a for a in data["attributes"] if a in prevalence]
    if not attrs:
        print(f"plot_summary[{mode}]: no attributes with prevalence — skipping")
        return None

    if mode == "subtracted":
        if 1 not in n_values:
            print("plot_summary[subtracted]: n=1 not in subset — skipping")
            return None
        base_idx = n_values.index(1)
        mat = np.array([
            [prevalence[a][i] - prevalence[a][base_idx] for i in range(len(n_values))]
            for a in attrs
        ], dtype=float)
        ylabel   = "mean Δ prevalence  (prev[n] − prev[n=1])"
        title    = (f"Summary across {len(attrs)} attributes\n"
                    f"(baseline-subtracted mean, ±1 std band)")
        ref_line = 0.0
        suffix   = "_summary"
        show_band = True
    elif mode == "raw":
        mat = np.array([prevalence[a] for a in attrs], dtype=float)
        ylabel   = "mean prevalence  P(attr=1 | BoN-n)"
        title    = (f"Summary across {len(attrs)} attributes\n"
                    f"(raw mean prevalence)")
        ref_line = None
        suffix   = "_summary_raw"
        show_band = False
    else:
        raise ValueError(f"mode must be 'subtracted' or 'raw', got {mode!r}")

    mean = mat.mean(axis=0)
    std  = mat.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    if ref_line is not None:
        ax.axhline(ref_line, linestyle="--", linewidth=1.2, color="gray", alpha=0.6)
    if show_band:
        ax.fill_between(n_values, mean - std, mean + std, alpha=0.25, label="±1 std")
    ax.plot(n_values, mean, marker="o", linewidth=2.5, label="mean")
    ax.scatter([n_values[0], n_values[-1]], [mean[0], mean[-1]], s=120, zorder=3)
    fmt = "{:+.3f}" if mode == "subtracted" else "{:.3f}"
    ax.text(n_values[0], mean[0],
            f"  n={n_values[0]}: " + fmt.format(mean[0]),
            va="bottom", ha="left", fontsize=14)
    ax.text(n_values[-1], mean[-1],
            f"  n={n_values[-1]}: " + fmt.format(mean[-1]),
            va="bottom", ha="right", fontsize=14)

    ax.set_xscale("log", base=2)
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_xlabel("n_values")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12, fontsize=14)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=14)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.85, bottom=0.12)

    out_path = json_path.with_name(f"{json_path.stem}{suffix}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved summary plot [{mode}]: {out_path}")
    return out_path


def plot_prevalence(json_path: str, n_values: list[int] | None = None):
    json_path = Path(json_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    if n_values is not None:
        available = set(data["n_values"])
        invalid = [n for n in n_values if n not in available]
        if invalid:
            raise ValueError(f"n_values {invalid} not found in data. Available: {sorted(available)}")
        n_to_idx = {n: i for i, n in enumerate(data["n_values"])}
        selected = sorted(set(n_values), key=lambda n: n_to_idx[n])
        for attr in data.get("prevalence", {}):
            orig_prev = data["prevalence"][attr]
            data["prevalence"][attr] = [orig_prev[n_to_idx[n]] for n in selected]
        data["n_values"] = selected

    # Increase default font sizes for readability
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    output_paths = []

    # Plot every attribute listed in the JSON
    for attr_idx, attr in enumerate(data["attributes"]):
        if attr not in data["prevalence"]:
            print(f"Skipping missing prevalence values for attribute: {attr}")
            continue

        output_path = plot_one_attribute(json_path, data, attr, attr_idx)
        output_paths.append(output_path)

    for mode in ("subtracted", "raw"):
        summary_path = plot_summary(json_path, data, mode=mode)
        if summary_path is not None:
            output_paths.append(summary_path)

    print("Saved plots:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the JSON result file.",
    )
    parser.add_argument(
        "--n_values",
        type=int,
        nargs="+",
        default=None,
        help="Subset of n values to plot (e.g. --n_values 1 4 16 64). Defaults to all.",
    )
    args = parser.parse_args()

    plot_prevalence(args.json_path, n_values=args.n_values)