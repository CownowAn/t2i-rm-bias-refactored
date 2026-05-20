"""Create cluster_N.json files from MJHQ-30K dataset using its predefined categories.

MJHQ-30K has 10 categories (animals, art, fashion, food, indoor, landscape,
logo, people, plants, vehicles), each with ~3,000 prompts. Each category becomes
one cluster, so no embedding or K-means is needed.

Usage:
    python clustering/cluster_mjhq.py \
        --meta_data_path /nfs/data/sohyun/data/MJHQ-30K_meta_data.json \
        --output_dir clustering/output/mjhq_30k

Output format (same as cluster_prompts.py):
    {"prompts": [...], "summary": "..."}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from clustering.summary import generate_summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build cluster JSONs from MJHQ-30K categories"
    )
    p.add_argument("--meta_data_path",
                   default="/nfs/data/sohyun/data/MJHQ-30K_meta_data.json",
                   help="Path to MJHQ-30K meta_data.json")
    p.add_argument("--output_dir", default="clustering/output/mjhq",
                   help="Output directory for cluster_N.json files")
    p.add_argument("--summary_model", default="gpt-5.2",
                   help="OpenAI model for cluster summary generation")
    p.add_argument("--summary_n_sample", type=int, default=None,
                   help="Max prompts shown to LLM per cluster (None = all)")
    p.add_argument("--dotenv_path", default=".env",
                   help="Path to .env file containing OPENAI_API_KEY")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true",
                   help="Skip LLM summary generation (use placeholder)")
    p.add_argument("--test", action="store_true",
                   help="Process only the first cluster (cluster_0) for testing")
    # ── Sub-sampling options ──────────────────────────────────────────────────
    p.add_argument("--n_representative", type=int, default=200,
                   help="Representative prompts per category via K-means (0 = keep all, default: 250)")
    p.add_argument("--embed_provider", default="openai", choices=["local", "openai"],
                   help="Embedding provider for sub-sampling (default: openai)")
    p.add_argument("--embed_model", default="text-embedding-3-large",
                   help="Embedding model name (local default: all-MiniLM-L6-v2, openai default: text-embedding-3-large)")
    p.add_argument("--openai_batch_size", type=int, default=512,
                   help="Batch size for OpenAI embeddings API calls (default: 512)")
    p.add_argument("--cache_dir", default="clustering/.cache/mjhq",
                   help="Directory to cache embeddings (default: clustering/.cache/mjhq)")
    return p.parse_args()


# ── Step 1: Load & group by category ─────────────────────────────────────────

def load_by_category(meta_data_path: str) -> dict[str, list[str]]:
    print(f"[1/3] Loading {meta_data_path} ...")
    with open(meta_data_path, encoding="utf-8") as f:
        meta = json.load(f)

    by_category: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    skipped = 0

    for entry in meta.values():
        p = entry.get("prompt", "").strip()
        cat = entry.get("category", "unknown")
        if len(p) < 10 or p in seen:
            skipped += 1
            continue
        seen.add(p)
        by_category[cat].append(p)

    categories = sorted(by_category.keys())
    print(f"  → {len(seen)} unique prompts across {len(categories)} categories "
          f"({skipped} skipped)")
    for cat in categories:
        print(f"     {cat}: {len(by_category[cat])} prompts")
    return dict(by_category)


# ── Step 2: Sub-sampling via K-means ─────────────────────────────────────────

def subsample_prompts(
    prompts: list[str],
    n_representative: int,
    provider: str,
    model_name: str | None,
    openai_batch_size: int,
    cache_dir: str | None,
    seed: int,
) -> list[str]:
    """Reduce prompts to n_representative medoids via K-means."""
    if n_representative <= 0 or len(prompts) <= n_representative:
        return prompts

    from clustering.cluster_diffusiondb import embed_prompts
    import numpy as np
    from sklearn.cluster import KMeans

    embeddings = embed_prompts(prompts, provider, model_name, openai_batch_size, cache_dir)

    km = KMeans(n_clusters=n_representative, random_state=seed, n_init="auto")
    labels = km.fit_predict(embeddings)

    representatives = []
    for k in range(n_representative):
        members = np.where(labels == k)[0]
        if len(members) == 0:
            continue
        sub_embs = embeddings[members]
        center = km.cluster_centers_[k]
        dists = np.linalg.norm(sub_embs - center, axis=1)
        representatives.append(prompts[members[np.argmin(dists)]])

    return representatives


# ── Step 3 & 4: Generate summaries and save ───────────────────────────────────

def save_clusters(
    by_category: dict[str, list[str]],
    output_dir: Path,
    summary_model: str,
    summary_n_sample: int | None,
    seed: int,
    dry_run: bool,
) -> None:
    categories = sorted(by_category.keys())
    print(f"\n[3/4] Generating summaries and saving {len(categories)} clusters ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    for cluster_id, cat in enumerate(categories):
        prompts = list(by_category[cat])
        rng.shuffle(prompts)

        if dry_run:
            summary = f"Category '{cat}' — {len(prompts)} prompts (dry run)"
        else:
            print(f"  [{cluster_id + 1}/{len(categories)}] Summarising '{cat}' "
                  f"({len(prompts)} prompts) ...")
            summary = generate_summary(prompts, summary_model, n_sample=summary_n_sample)

        out = {"category": cat, "prompts": prompts, "summary": summary}
        path = output_dir / f"cluster_{cluster_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        print(f"  cluster_{cluster_id}.json  [{cat}]  {len(prompts):5d} prompts — {summary}")

    print(f"\n[4/4] Done. Files saved to: {output_dir.resolve()}")
    print("Next: update data.prompts_dir in bon_amplified.yaml to point to this directory.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # Load OPENAI_API_KEY from .env
    dotenv_path = Path(args.dotenv_path)
    if dotenv_path.exists():
        for line in dotenv_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
        print(f"Loaded environment from {dotenv_path}")

    if not Path(args.meta_data_path).exists():
        sys.exit(f"ERROR: meta_data.json not found at {args.meta_data_path}")

    by_category = load_by_category(args.meta_data_path)

    if args.test:
        first_cat = sorted(by_category.keys())[0]
        by_category = {first_cat: by_category[first_cat]}
        print(f"[test mode] Processing only '{first_cat}' (cluster_0)")

    if args.n_representative > 0:
        print(f"\n[2/4] Sub-sampling to {args.n_representative} prompts per category ...")
        for cat in sorted(by_category.keys()):
            original = len(by_category[cat])
            by_category[cat] = subsample_prompts(
                by_category[cat],
                n_representative=args.n_representative,
                provider=args.embed_provider,
                model_name=args.embed_model,
                openai_batch_size=args.openai_batch_size,
                cache_dir=args.cache_dir,
                seed=args.seed,
            )
            print(f"  {cat}: {original} → {len(by_category[cat])} prompts")

    save_clusters(
        by_category=by_category,
        output_dir=Path(args.output_dir),
        summary_model=args.summary_model,
        summary_n_sample=args.summary_n_sample,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
