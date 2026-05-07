import argparse
import json
import re
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

    ax.set_title(attr, pad=16)
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

    fig.tight_layout()

    # Save one plot per attribute in the same directory as the input JSON file
    attr_slug = slugify(attr)
    output_path = json_path.with_name(
        f"{json_path.stem}_attr{attr_idx:02d}_{attr_slug}_prevalence_by_n.png"
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path


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