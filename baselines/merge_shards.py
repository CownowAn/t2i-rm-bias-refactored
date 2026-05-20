"""Merge manifest_shard_*.json files into a single manifest.json.

Usage:
    python baselines/merge_shards.py --topic_dir <path>

    # Topic 0 example:
    python baselines/merge_shards.py \
        --topic_dir /nfs/data/sohyun/projects/t2i-rm-bias/data/baselines/mjhq/topic_0/black-forest-labs-FLUX.1-dev

    # All topics under an output_dir:
    python baselines/merge_shards.py \
        --output_dir /nfs/data/sohyun/projects/t2i-rm-bias/data/baselines/mjhq \
        --model_id black-forest-labs/FLUX.1-dev \
        --topics 0 1 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def merge_topic_dir(topic_dir: Path) -> None:
    shard_files = sorted(topic_dir.glob("manifest_shard_*.json"))
    if not shard_files:
        print(f"  WARNING: no shard files found in {topic_dir}", file=sys.stderr)
        return

    print(f"  Found {len(shard_files)} shard(s): {[p.name for p in shard_files]}")

    merged: dict = {}
    metadata: dict = {}
    for p in shard_files:
        d = json.loads(p.read_text())
        metadata = d.get("metadata", metadata)
        merged.update(d.get("baselines", {}))

    manifest_path = topic_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"metadata": metadata, "baselines": merged}, indent=2, ensure_ascii=False)
    )
    print(f"  Merged {len(merged)} prompts → {manifest_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Merge manifest shards into manifest.json")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--topic_dir", help="Direct path to the topic model directory")
    group.add_argument("--output_dir", help="Root output directory (use with --model_id and --topics)")
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev",
                   help="Model ID used during generation (needed with --output_dir)")
    p.add_argument("--topics", type=int, nargs="+", default=list(range(10)),
                   help="Topic IDs to merge (default: 0–9, used with --output_dir)")
    args = p.parse_args()

    if args.topic_dir:
        merge_topic_dir(Path(args.topic_dir))
    else:
        model_dir_name = args.model_id.replace("/", "-")
        for tid in args.topics:
            topic_dir = Path(args.output_dir) / f"topic_{tid}" / model_dir_name
            if not topic_dir.exists():
                print(f"  WARNING: directory not found: {topic_dir}", file=sys.stderr)
                continue
            print(f"--- Topic {tid} ---")
            merge_topic_dir(topic_dir)


if __name__ == "__main__":
    main()
