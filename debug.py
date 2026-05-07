import argparse
import json
from pathlib import Path


def get_top_k_attributes(json_path: str, k: int):
    json_path = Path(json_path)

    # Load JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract attribute list
    attributes = data["top_attributes"]

    # Sort by amplification_score (descending)
    sorted_attrs = sorted(
        attributes,
        key=lambda x: x["amplification_score"],
        reverse=True,
    )

    # Select top-k
    top_k = sorted_attrs[:k]

    return top_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()

    top_k = get_top_k_attributes(args.json_path, args.k)

    print(f"\nTop-{args.k} attributes by amplification_score:\n")
    for i, item in enumerate(top_k):
        print(f"[{i+1}] score={item['amplification_score']:.6f}")
        print(f"     attr: {item['attribute']}")
        print(f"     step_found: {item['step_found']}")
        print()


if __name__ == "__main__":
    main()