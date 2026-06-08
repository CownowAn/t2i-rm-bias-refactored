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
    p.add_argument("--output_dir", default="clustering/output/mjhq/100",
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
    p.add_argument("--n_representative", type=int, default=100,
                   help="Representative prompts per category via K-means (0 = keep all, default: 250)")
    p.add_argument("--embed_provider", default="openai", choices=["local", "openai"],
                   help="Embedding provider for sub-sampling (default: openai)")
    p.add_argument("--embed_model", default="text-embedding-3-large",
                   help="Embedding model name (local default: all-MiniLM-L6-v2, openai default: text-embedding-3-large)")
    p.add_argument("--openai_batch_size", type=int, default=512,
                   help="Batch size for OpenAI embeddings API calls (default: 512)")
    p.add_argument("--cache_dir", default="clustering/.cache/mjhq",
                   help="Directory to cache embeddings (default: clustering/.cache/mjhq)")
    # ── CLIP-length filter ────────────────────────────────────────────────────
    p.add_argument("--max_clip_tokens", type=int, default=75,
                   help="Drop prompts whose CLIP tokenization (excluding BOS/EOS) "
                        "exceeds this length. 0 disables the filter. Use 75 to "
                        "ensure every prompt fits CLIP's 77-token context.")
    p.add_argument("--clip_tokenizer", default="sd2-community/stable-diffusion-2-1",
                   help="HF id of the CLIP tokenizer used to count tokens. "
                        "Defaults to the public sd2-community SD 2.1 mirror so "
                        "the count matches the tokenizer the generation pipeline "
                        "uses without requiring HF authentication.")
    p.add_argument("--clip_tokenizer_subfolder", default="tokenizer",
                   help="Subfolder inside the HF repo holding the tokenizer "
                        "files. Empty string = repo root. SD 2.1 stores its "
                        "CLIP tokenizer under 'tokenizer/'.")
    # ── Resolution-tag filter ─────────────────────────────────────────────────
    p.add_argument("--filter_resolution_tags", action="store_true",
                   help="Drop prompts that mention resolution markers like "
                        "'4k', '8K', '1080p', 'HD', 'UHD'.")
    # ── Category-match LLM filter ─────────────────────────────────────────────
    p.add_argument("--filter_category_match", action="store_true",
                   help="LLM-classify each prompt as YES/NO for whether it really "
                        "depicts something in its category, and drop the NOs. "
                        "Runs just before sub-sampling so we only pay for prompts "
                        "that survived earlier filters.")
    p.add_argument("--category_match_model", default="gpt-4o-mini",
                   help="OpenAI model used for the YES/NO classification.")
    p.add_argument("--category_match_input_price", type=float, default=0.15,
                   help="USD per 1M input tokens (default: gpt-4o-mini rate).")
    p.add_argument("--category_match_output_price", type=float, default=0.60,
                   help="USD per 1M output tokens (default: gpt-4o-mini rate).")
    p.add_argument("--category_match_max_concurrency", type=int, default=50,
                   help="Max in-flight async OpenAI calls.")
    p.add_argument("--category_match_cache_path",
                   default="clustering/.cache/mjhq/category_match.json",
                   help="Path where YES/NO decisions are persisted across runs "
                        "so re-running doesn't re-pay for already-classified "
                        "(prompt, category, model) triples.")
    p.add_argument("--category_match_dry_run_cost", action="store_true",
                   help="Just estimate token count and dollar cost, then return "
                        "without calling the API. Useful for budgeting.")
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


# ── Step 1b: Drop prompts longer than CLIP's context ─────────────────────────

def filter_by_clip_tokens(
    by_category: dict[str, list[str]],
    max_tokens: int,
    tokenizer_id: str,
    tokenizer_subfolder: str = "",
) -> dict[str, list[str]]:
    """Drop prompts whose CLIP token count (excluding BOS/EOS) exceeds max_tokens."""
    from transformers import CLIPTokenizerFast

    src_label = (f"{tokenizer_id}#{tokenizer_subfolder}"
                 if tokenizer_subfolder else tokenizer_id)
    print(f"\n[1b] CLIP-length filter (tokenizer={src_label}, "
          f"max_content_tokens={max_tokens}) ...")
    from_pretrained_kwargs = {}
    if tokenizer_subfolder:
        from_pretrained_kwargs["subfolder"] = tokenizer_subfolder
    tok = CLIPTokenizerFast.from_pretrained(tokenizer_id, **from_pretrained_kwargs)
    # Silence "Token indices sequence length is longer than ..." spam — we
    # intentionally feed un-truncated long prompts to detect them.
    tok.model_max_length = int(1e9)
    out: dict[str, list[str]] = {}
    total_in = total_out = 0
    for cat in sorted(by_category.keys()):
        prompts = by_category[cat]
        # No padding/truncation: we need the true length to filter correctly.
        # add_special_tokens=True adds BOS+EOS so subtract 2 for content count.
        enc = tok(prompts, padding=False, truncation=False, add_special_tokens=True)
        keep: list[str] = []
        for p, ids in zip(prompts, enc["input_ids"]):
            n_content = max(0, len(ids) - 2)
            if n_content <= max_tokens:
                keep.append(p)
        out[cat] = keep
        total_in += len(prompts)
        total_out += len(keep)
        dropped = len(prompts) - len(keep)
        print(f"  {cat}: {len(prompts)} → {len(keep)}  (dropped {dropped})")
    print(f"  total: {total_in} → {total_out}  (dropped {total_in - total_out})")
    return out


# ── Step 1c: Drop prompts with explicit resolution markers ───────────────────

# Matches: 4k, 8K, 16k, 1080p, 2160p, HD, UHD, FHD, 4K-resolution, 8 k, etc.
# Uses word boundaries so "weekend" / "happy" don't trigger on 'hd'/'p'.
_RESOLUTION_REGEX = r"\b(\d+\s*k|\d+\s*p|uhd|fhd|qhd|hd)\b"


def filter_resolution_tags(by_category: dict[str, list[str]]) -> dict[str, list[str]]:
    """Drop prompts that contain a resolution marker like 4k/8k/1080p/HD/UHD."""
    import re

    rx = re.compile(_RESOLUTION_REGEX, re.IGNORECASE)
    print(f"\n[1c] Resolution-tag filter (pattern={_RESOLUTION_REGEX}) ...")
    out: dict[str, list[str]] = {}
    total_in = total_out = 0
    sample_dropped: list[str] = []
    for cat in sorted(by_category.keys()):
        prompts = by_category[cat]
        keep: list[str] = []
        for p in prompts:
            if rx.search(p):
                if len(sample_dropped) < 3:
                    sample_dropped.append(p)
            else:
                keep.append(p)
        out[cat] = keep
        total_in += len(prompts)
        total_out += len(keep)
        print(f"  {cat}: {len(prompts)} → {len(keep)}  (dropped {len(prompts) - len(keep)})")
    print(f"  total: {total_in} → {total_out}  (dropped {total_in - total_out})")
    for p in sample_dropped:
        truncated = p if len(p) <= 160 else p[:157] + "..."
        print(f"  [dropped sample] {truncated}")
    return out


# ── Step 1d: LLM category-match filter ───────────────────────────────────────

_CATEGORY_MATCH_INSTRUCTION = (
    "You are classifying text-to-image prompts. Decide whether the following "
    "prompt clearly depicts something belonging to the category \"{category}\".\n"
    "Reply with a single token: YES or NO. No explanation.\n\n"
    "Prompt:\n{prompt}"
)


def _category_match_key(prompt: str, category: str, model: str) -> str:
    import hashlib
    payload = f"{model}\n{category}\n{prompt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def filter_category_match(
    by_category: dict[str, list[str]],
    model: str,
    input_price_per_1m: float,
    output_price_per_1m: float,
    max_concurrency: int,
    cache_path: str,
    dry_run_cost: bool,
) -> dict[str, list[str]]:
    """Per-prompt LLM yes/no filter on whether the prompt fits its category."""
    import asyncio
    import json
    from pathlib import Path

    print(f"\n[1d] Category-match LLM filter (model={model}) ...")

    # 1. Load cache
    cache_file = Path(cache_path)
    cache: dict[str, str] = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
        except Exception as e:
            print(f"  WARN: failed to read cache at {cache_file}: {e}")
            cache = {}

    # 2. Identify pending (prompt, category) pairs
    pending: list[tuple[str, str, str]] = []  # (cat, prompt, key)
    for cat, prompts in by_category.items():
        for p in prompts:
            k = _category_match_key(p, cat, model)
            if k not in cache:
                pending.append((cat, p, k))

    n_total = sum(len(v) for v in by_category.values())
    n_cached = n_total - len(pending)
    print(f"  total prompts: {n_total}, cached: {n_cached}, pending: {len(pending)}")

    # 3. Token / cost estimate
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")  # gpt-4o family default
        def n_in_tok(prompt: str, category: str) -> int:
            return len(enc.encode(
                _CATEGORY_MATCH_INSTRUCTION.format(category=category, prompt=prompt)
            ))
    except ImportError:
        print("  WARN: tiktoken not installed; falling back to chars/4 approximation")
        def n_in_tok(prompt: str, category: str) -> int:
            text = _CATEGORY_MATCH_INSTRUCTION.format(category=category, prompt=prompt)
            return max(1, len(text) // 4)

    in_tokens = sum(n_in_tok(p, c) for c, p, _ in pending)
    out_tokens = len(pending) * 2  # YES/NO + EOS budget
    in_cost = in_tokens / 1_000_000 * input_price_per_1m
    out_cost = out_tokens / 1_000_000 * output_price_per_1m
    print(f"  estimated tokens : input {in_tokens:,}, output {out_tokens:,}")
    print(f"  estimated cost   : ${in_cost:.4f} (in) + ${out_cost:.4f} (out) = "
          f"${in_cost + out_cost:.4f}")

    if dry_run_cost:
        print("  --category_match_dry_run_cost set → skipping API calls and "
              "returning input unchanged.")
        return by_category

    if pending:
        # 4. Async classify
        from openai import AsyncOpenAI
        # Per-call timeout: keeps the tail from getting stuck on rate-limit
        # retries that ride retry-after headers, or on rare network hangs.
        client = AsyncOpenAI(timeout=30.0, max_retries=2)
        sem = asyncio.Semaphore(max_concurrency)
        progress = {"done": 0, "errors": 0, "timeouts": 0}
        flush_every = 100

        async def query_one(cat: str, prompt: str, k: str) -> None:
            async with sem:
                try:
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model,
                            messages=[{
                                "role": "user",
                                "content": _CATEGORY_MATCH_INSTRUCTION.format(
                                    category=cat, prompt=prompt
                                ),
                            }],
                            max_completion_tokens=4,
                        ),
                        timeout=60.0,
                    )
                    text = (resp.choices[0].message.content or "").strip().upper()
                    if text.startswith("YES"):
                        cache[k] = "YES"
                    elif text.startswith("NO"):
                        cache[k] = "NO"
                    else:
                        cache[k] = "UNKNOWN"
                except asyncio.TimeoutError:
                    cache[k] = "TIMEOUT"
                    progress["timeouts"] += 1
                except Exception:
                    cache[k] = "ERROR"
                    progress["errors"] += 1
                progress["done"] += 1
                if progress["done"] % flush_every == 0:
                    _flush_cache(cache_file, cache)
                    print(f"    {progress['done']}/{len(pending)} done "
                          f"({progress['errors']} err, {progress['timeouts']} timeout)")

        async def heartbeat(stop_evt: asyncio.Event, total: int) -> None:
            """Print progress every 10s so a stalled tail is visible."""
            last = -1
            while not stop_evt.is_set():
                try:
                    await asyncio.wait_for(stop_evt.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    pass
                if progress["done"] != last and progress["done"] < total:
                    print(f"    [heartbeat] {progress['done']}/{total} done "
                          f"({progress['errors']} err, {progress['timeouts']} timeout)")
                    last = progress["done"]

        async def run_all() -> None:
            stop_evt = asyncio.Event()
            hb = asyncio.create_task(heartbeat(stop_evt, len(pending)))
            try:
                await asyncio.gather(*(query_one(c, p, k) for c, p, k in pending))
            finally:
                stop_evt.set()
                await hb

        asyncio.run(run_all())
        _flush_cache(cache_file, cache)
        print(f"  done: {progress['done']} calls "
              f"({progress['errors']} errored, {progress['timeouts']} timed out "
              f"→ all treated as drops)")

    # 5. Apply decisions and report
    out: dict[str, list[str]] = {}
    total_in = total_out = 0
    decisions = {"YES": 0, "NO": 0, "UNKNOWN": 0, "ERROR": 0, "TIMEOUT": 0, "MISSING": 0}
    for cat in sorted(by_category.keys()):
        prompts = by_category[cat]
        keep: list[str] = []
        for p in prompts:
            k = _category_match_key(p, cat, model)
            d = cache.get(k, "MISSING")
            decisions[d] = decisions.get(d, 0) + 1
            if d == "YES":
                keep.append(p)
        out[cat] = keep
        total_in += len(prompts)
        total_out += len(keep)
        print(f"  {cat}: {len(prompts)} → {len(keep)}  "
              f"(dropped {len(prompts) - len(keep)})")
    print(f"  total: {total_in} → {total_out}  (dropped {total_in - total_out})")
    print(f"  decisions: {decisions}")
    return out


def _flush_cache(path, cache):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    tmp.replace(path)


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

    if args.max_clip_tokens > 0:
        by_category = filter_by_clip_tokens(
            by_category,
            max_tokens=args.max_clip_tokens,
            tokenizer_id=args.clip_tokenizer,
            tokenizer_subfolder=args.clip_tokenizer_subfolder,
        )

    if args.filter_resolution_tags:
        by_category = filter_resolution_tags(by_category)

    if args.filter_category_match:
        by_category = filter_category_match(
            by_category,
            model=args.category_match_model,
            input_price_per_1m=args.category_match_input_price,
            output_price_per_1m=args.category_match_output_price,
            max_concurrency=args.category_match_max_concurrency,
            cache_path=args.category_match_cache_path,
            dry_run_cost=args.category_match_dry_run_cost,
        )

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
