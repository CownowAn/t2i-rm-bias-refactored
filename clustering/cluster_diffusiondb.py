"""Cluster DiffusionDB prompts semantically and save as cluster_N.json files.

Usage:
    python clustering/cluster_prompts.py \
        --subset 2m_random_1k \
        --output_dir clustering/output \
        --hf_cache_dir /nfs/data/sohyun/data \
        --max_k 10

Output format (same as existing cluster JSON consumed by search pipeline):
    {"prompts": [...], "summary": "..."}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cluster DiffusionDB prompts semantically")
    p.add_argument("--subset",       default="2m_random_1k",
                   help="DiffusionDB subset name (default: 2m_random_1k)")
    p.add_argument("--output_dir",   default="clustering/output/diffusiondb/train/2m_random_1k",
                   help="Output directory for cluster_N.json files")
    p.add_argument("--hf_cache_dir", default="/nfs/data/sohyun/data",
                   help="HuggingFace datasets cache directory")
    p.add_argument("--max_k",        type=int, default=10,
                   help="Maximum number of clusters to evaluate (default: 10)")
    p.add_argument("--min_k",        type=int, default=2,
                   help="Minimum number of clusters to evaluate (default: 2)")
    p.add_argument("--embed_provider", default="openai",
                   choices=["local", "openai"],
                   help="Embedding provider: 'local' (sentence-transformers) or 'openai' API")
    p.add_argument("--embed_model",  default="text-embedding-3-large",
                   help=("Embedding model name. "
                         "local default: 'all-MiniLM-L6-v2'. "
                         "openai default: 'text-embedding-3-small'"))
    p.add_argument("--openai_batch_size", type=int, default=512,
                   help="Batch size for OpenAI embeddings API calls (default: 512)")
    p.add_argument("--cache_dir", default="clustering/.cache/diffusiondb/train/2m_random_1k",
                   help="Directory to cache embedding results (default: clustering/.cache)")
    p.add_argument("--dotenv_path", default=".env",
                   help="Path to .env file containing OPENAI_API_KEY (default: .env)")
    p.add_argument("--summary_model", default="gpt-5.2",
                   help="OpenAI model for cluster summary generation")
    p.add_argument("--summary_n_sample", type=int, default=None,
                   help="Max prompts shown to LLM per cluster for summary (0 = all, default: 20)")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--silhouette_sample", type=int, default=1000,
                   help="Max sample size for silhouette score (for speed)")
    p.add_argument("--dry_run", action="store_true",
                   help="Skip LLM summary generation (use placeholder)")
    return p.parse_args()


# ── Step 1: Download & clean prompts ─────────────────────────────────────────

def load_prompts(subset: str, hf_cache_dir: str) -> list[str]:
    print(f"[1/5] Loading DiffusionDB subset '{subset}' ...")
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("ERROR: 'datasets' package not installed. Run: pip install datasets")

    ds = load_dataset(
        "poloclub/diffusiondb", subset,
        split="train",
        cache_dir=hf_cache_dir,
        trust_remote_code=True,
    )

    prompts: list[str] = []
    seen: set[str] = set()
    for row in ds:
        p = row["prompt"].strip()
        if len(p) < 10 or p in seen:
            continue
        seen.add(p)
        prompts.append(p)

    print(f"  → {len(prompts)} unique prompts (≥10 chars)")
    return prompts


# ── Step 2: Embed ─────────────────────────────────────────────────────────────

def _embedding_cache_key(prompts: list[str], provider: str, model_name: str) -> str:
    h = hashlib.sha256()
    h.update(f"{provider}::{model_name}".encode())
    for p in prompts:
        h.update(p.encode())
    return h.hexdigest()[:16]


def embed_prompts(
    prompts: list[str],
    provider: str,
    model_name: str | None,
    openai_batch_size: int = 512,
    cache_dir: str | None = None,
) -> "np.ndarray":
    import numpy as np

    resolved_model = model_name or ("text-embedding-3-small" if provider == "openai" else "all-MiniLM-L6-v2")
    cache_key = _embedding_cache_key(prompts, provider, resolved_model)

    # ── Cache lookup ──────────────────────────────────────────────────────────
    if cache_dir:
        cache_path = Path(cache_dir) / f"{cache_key}.npy"
        meta_path  = Path(cache_dir) / f"{cache_key}.json"
        if cache_path.exists():
            embeddings = np.load(str(cache_path))
            print(f"[2/5] Loaded embeddings from cache ({cache_path})")
            print(f"  → embeddings shape: {embeddings.shape}")
            return embeddings

    # ── Compute ───────────────────────────────────────────────────────────────
    if provider == "openai":
        embeddings = _embed_openai(prompts, resolved_model, openai_batch_size)
    else:
        embeddings = _embed_local(prompts, resolved_model)

    # ── Cache save ────────────────────────────────────────────────────────────
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), embeddings)
        meta_path.write_text(json.dumps({
            "provider": provider,
            "model": resolved_model,
            "n_prompts": len(prompts),
            "shape": list(embeddings.shape),
        }, indent=2))
        print(f"  → cached to {cache_path}")

    return embeddings


def _embed_local(prompts: list[str], model_name: str) -> "np.ndarray":
    print(f"[2/5] Embedding with local sentence-transformers '{model_name}' ...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("ERROR: 'sentence-transformers' not installed. Run: pip install sentence-transformers")
    import numpy as np

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        prompts,
        normalize_embeddings=True,
        batch_size=256,
        show_progress_bar=True,
    )
    print(f"  → embeddings shape: {embeddings.shape}")
    return embeddings


# OpenAI embedding pricing ($/1M tokens), May 2026
_OPENAI_EMBED_PRICING: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


def _estimate_embed_cost(prompts: list[str], model_name: str) -> None:
    """Print estimated OpenAI embedding cost before the API call."""
    # Rough token estimate: ~1 token per 4 characters (GPT tokenizer heuristic)
    total_chars = sum(len(p) for p in prompts)
    est_tokens = total_chars / 4
    price_per_1m = _OPENAI_EMBED_PRICING.get(model_name, 0.10)
    est_cost = est_tokens / 1_000_000 * price_per_1m
    print(f"  Estimated cost: ~{est_tokens:,.0f} tokens × ${price_per_1m}/1M = ${est_cost:.4f}")


def _embed_openai(prompts: list[str], model_name: str, batch_size: int) -> "np.ndarray":
    print(f"[2/5] Embedding with OpenAI API '{model_name}' (batch_size={batch_size}) ...")
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("ERROR: 'openai' not installed. Run: pip install openai")
    import numpy as np

    _estimate_embed_cost(prompts, model_name)

    client = OpenAI()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        resp = client.embeddings.create(model=model_name, input=batch)
        # API returns embeddings sorted by index
        batch_vecs = [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
        all_embeddings.extend(batch_vecs)
        print(f"  {min(i + batch_size, len(prompts))}/{len(prompts)} embedded", end="\r")

    print()
    embeddings = np.array(all_embeddings, dtype=np.float32)
    # L2-normalize (sentence-transformers does this; OpenAI does not by default)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1.0)
    print(f"  → embeddings shape: {embeddings.shape}")
    return embeddings


# ── Step 3: Find optimal K via silhouette score ───────────────────────────────

def find_best_k(
    embeddings: "np.ndarray",
    min_k: int,
    max_k: int,
    seed: int,
    sample_size: int,
) -> tuple[int, "np.ndarray"]:
    print(f"[3/5] Finding optimal K in [{min_k}, {max_k}] via silhouette score ...")
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        sys.exit("ERROR: 'scikit-learn' not installed. Run: pip install scikit-learn")

    n = len(embeddings)
    effective_sample = min(sample_size, n)

    scores: list[tuple[int, float]] = []
    label_cache: dict[int, "np.ndarray"] = {}

    for k in range(min_k, max_k + 1):
        if k >= n:
            print(f"  K={k}: skipped (K >= N={n})")
            break
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, sample_size=effective_sample,
                                 random_state=seed)
        scores.append((k, score))
        label_cache[k] = labels
        print(f"  K={k:2d}: silhouette={score:.4f}")

    best_k, best_score = max(scores, key=lambda x: x[1])
    print(f"  → Best K={best_k} (silhouette={best_score:.4f})")
    return best_k, label_cache[best_k]


from clustering.summary import generate_summary  # noqa: E402


# ── Step 5: Save cluster JSON files ──────────────────────────────────────────

def save_clusters(
    prompts: list[str],
    labels: "np.ndarray",
    best_k: int,
    output_dir: Path,
    summary_model: str,
    seed: int,
    dry_run: bool,
    summary_n_sample: int | None = 20,
) -> None:
    print(f"[5/5] Saving {best_k} cluster files to '{output_dir}/' ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    for cluster_id in range(best_k):
        cluster_prompts = [prompts[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
        rng.shuffle(cluster_prompts)

        if dry_run:
            summary = f"Cluster {cluster_id} — {len(cluster_prompts)} prompts (dry run, no LLM summary)"
        else:
            print(f"  [4/5] Generating summary for cluster {cluster_id} ({len(cluster_prompts)} prompts) ...")
            summary = generate_summary(cluster_prompts, summary_model, n_sample=summary_n_sample)

        out = {"prompts": cluster_prompts, "summary": summary}
        path = output_dir / f"cluster_{cluster_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        print(f"  cluster_{cluster_id}.json: {len(cluster_prompts):4d} prompts — {summary}")

    print(f"\nDone. Cluster files saved to: {output_dir.resolve()}")
    print("Next: update data.prompts_dir in bon_amplified.yaml to point to this directory.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # Load OPENAI_API_KEY from .env if present
    import os
    dotenv_path = Path(args.dotenv_path)
    if dotenv_path.exists():
        for line in dotenv_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
        print(f"Loaded environment from {dotenv_path}")

    prompts = load_prompts(args.subset, args.hf_cache_dir)
    if len(prompts) < args.min_k:
        sys.exit(f"ERROR: Only {len(prompts)} prompts — not enough for K={args.min_k}")

    embeddings = embed_prompts(
        prompts,
        provider=args.embed_provider,
        model_name=args.embed_model,
        openai_batch_size=args.openai_batch_size,
        cache_dir=args.cache_dir,
    )
    best_k, labels = find_best_k(
        embeddings,
        min_k=args.min_k,
        max_k=min(args.max_k, len(prompts) - 1),
        seed=args.seed,
        sample_size=args.silhouette_sample,
    )

    save_clusters(
        prompts=prompts,
        labels=labels,
        best_k=best_k,
        output_dir=Path(args.output_dir),
        summary_model=args.summary_model,
        seed=args.seed,
        dry_run=args.dry_run,
        summary_n_sample=args.summary_n_sample,
    )


if __name__ == "__main__":
    main()
