"""Generate baseline images from cluster_N.json prompt files using a T2I model.

Saves images and a manifest.json (without reward_scores).
Run score_baselines.py afterwards to add reward scores.

Usage:
    python baselines/generate_images.py \
        --cluster_dir clustering/output/mjhq \
        --topic_ids 0 1 2 \
        --output_dir data/baselines/mjhq \
        --model_id black-forest-labs/FLUX.1-dev \
        --images_per_prompt 128

Resumable: skips images that already exist on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate baseline images from cluster prompts")
    p.add_argument("--cluster_dir", required=True,
                   help="Directory containing cluster_N.json files")
    p.add_argument("--topic_ids", type=int, nargs="+", required=True,
                   help="Cluster IDs to process (e.g. 0 1 2)")
    p.add_argument("--output_dir", default="data/baselines/mjhq",
                   help="Root output directory")
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev",
                   help="HuggingFace T2I model ID")
    p.add_argument("--images_per_prompt", type=int, default=128)
    p.add_argument("--image_width", type=int, default=512)
    p.add_argument("--image_height", type=int, default=512)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--hf_cache_dir", default="/nfs/data/sohyun/models",
                   help="HuggingFace model cache directory")
    # If unset, defaults are chosen per pipeline kind (FLUX: 3.5/50, SD3: 4.5/28).
    p.add_argument("--guidance_scale", type=float, default=None,
                   help="Override pipeline-kind default")
    p.add_argument("--num_inference_steps", type=int, default=None,
                   help="Override pipeline-kind default")
    # ── Sharding: split prompts within a topic across multiple GPUs ──────────
    p.add_argument("--num_shards", type=int, default=1,
                   help="Number of GPU shards (default: 1 = no sharding)")
    p.add_argument("--shard_rank", type=int, default=0,
                   help="This shard's rank 0-indexed. Saves manifest_shard_{rank}.json")
    # ── Keep-alive: hold the pipeline on GPU after generation completes ──────
    p.add_argument("--keep_alive", action="store_true",
                   help="After all topics are done, idle in a sleep loop with "
                        "the pipeline still on GPU. Release with SIGINT/SIGTERM.")
    p.add_argument("--done_flag_dir", default=None,
                   help="If set, touch <dir>/shard_<rank>.done after the topic "
                        "loop (before any keep-alive sleep). The shell launcher "
                        "polls this so manifest merge can run even while pythons "
                        "are still idling on GPU.")
    return p.parse_args()


def load_prompts(cluster_dir: Path, topic_id: int) -> list[str]:
    path = cluster_dir / f"cluster_{topic_id}.json"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    prompts = [p.strip() for p in data.get("prompts", []) if p.strip()]
    print(f"  Topic {topic_id}: {len(prompts)} prompts from {path}")
    return prompts


def prompt_hash(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()[:10]


def model_dir_name(model_id: str) -> str:
    return model_id.replace("/", "-")


_PIPE_DEFAULTS = {
    "flux": {"guidance_scale": 3.5, "num_inference_steps": 50},
    "sd3":  {"guidance_scale": 4.5, "num_inference_steps": 28},
    # SD 2.1: leave guidance_scale unset so the HF pipeline default applies.
    "sd2":  {"guidance_scale": None, "num_inference_steps": 50},
}


def _pipe_kind(model_id: str) -> str:
    mid = model_id.lower()
    if "stable-diffusion-3" in mid or "sd3" in mid:
        return "sd3"
    if "stable-diffusion-2" in mid or "sd2" in mid:
        return "sd2"
    return "flux"


def load_pipe(model_id: str, hf_cache_dir: str):
    """Load and return a T2I pipeline on cuda. Call once per process.

    Dispatches on model_id: SD3.x → StableDiffusion3Pipeline, SD2.x →
    StableDiffusionPipeline (fp16, safety checker disabled), else FluxPipeline.
    """
    import torch

    kind = _pipe_kind(model_id)
    print(f"  Loading {model_id}  (pipeline kind: {kind}) ...")
    if kind == "sd3":
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, cache_dir=hf_cache_dir,
        ).to("cuda")
    elif kind == "sd2":
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir=hf_cache_dir,
            safety_checker=None,
            requires_safety_checker=False,
        )
        # HF SD 2.1 model card recommends DPM-Solver++ over the default scheduler.
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")
    else:
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, cache_dir=hf_cache_dir,
        ).to("cuda")
    pipe._pipe_kind = kind
    return pipe


def generate(
    prompts: list[str],
    out_dir: Path,
    model_id: str,
    images_per_prompt: int,
    width: int,
    height: int,
    seed: int,
    hf_cache_dir: str,
    guidance_scale: float,
    num_inference_steps: int,
    pipe=None,
) -> dict[str, list[dict]]:
    """Generate images and return baselines dict.

    If pipe is None, loads a new pipeline (FLUX or SD3 per model_id). Pass an
    existing pipe to reuse across topics within the same process.
    """
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)

    if pipe is None:
        pipe = load_pipe(model_id, hf_cache_dir)

    from tqdm import tqdm

    baselines: dict[str, list[dict]] = {}
    prompt_bar = tqdm(prompts, desc="Prompts", unit="prompt", position=0)

    for prompt in prompt_bar:
        phash = prompt_hash(prompt)
        prompt_bar.set_postfix_str(prompt[:50])
        entries = []
        all_exist = True

        for img_idx in range(images_per_prompt):
            img_name = f"baseline_{phash}_{img_idx:02d}.png"
            img_path = out_dir / img_name
            if not img_path.exists():
                all_exist = False
            entries.append({
                "image_path": str(img_path),
                "image_id": f"{phash}_{img_idx:02d}",
                "prompt": prompt,
                "policy_model": model_id,
                "reward_scores": {},
            })

        if all_exist:
            baselines[prompt] = entries
            continue

        generator = torch.Generator(device="cuda")
        img_bar = tqdm(range(images_per_prompt), desc="  Images", unit="img",
                       position=1, leave=False)
        for img_idx in img_bar:
            img_path = out_dir / f"baseline_{phash}_{img_idx:02d}.png"
            if img_path.exists():
                continue
            generator.manual_seed(seed + img_idx)
            call_kwargs = dict(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )
            if guidance_scale is not None:
                call_kwargs["guidance_scale"] = guidance_scale
            if getattr(pipe, "_pipe_kind", "flux") == "flux":
                call_kwargs["max_sequence_length"] = 512
            result = pipe(**call_kwargs)
            result.images[0].save(img_path)

        baselines[prompt] = entries

    return baselines


def merge_shard_manifests(out_dir: Path, num_shards: int, metadata: dict) -> None:
    """Merge manifest_shard_*.json into a single manifest.json."""
    merged_baselines: dict = {}
    for rank in range(num_shards):
        shard_path = out_dir / f"manifest_shard_{rank}.json"
        if not shard_path.exists():
            print(f"  WARNING: shard {rank} manifest missing: {shard_path}")
            continue
        with open(shard_path) as f:
            shard = json.load(f)
        merged_baselines.update(shard.get("baselines", {}))

    manifest = {"metadata": metadata, "baselines": merged_baselines}
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Merged {num_shards} shards → {manifest_path} ({len(merged_baselines)} prompts)")


def main() -> None:
    args = parse_args()
    cluster_dir = Path(args.cluster_dir)
    output_dir = Path(args.output_dir)
    mdir = model_dir_name(args.model_id)

    # Load pipeline once, reuse across all topics (the model stays on this GPU
    # even if a topic has no work to do for this shard).
    pipe = load_pipe(args.model_id, args.hf_cache_dir)

    # Resolve sampler defaults from pipeline kind unless user overrode them.
    defaults = _PIPE_DEFAULTS[pipe._pipe_kind]
    guidance_scale = (args.guidance_scale
                      if args.guidance_scale is not None else defaults["guidance_scale"])
    num_inference_steps = (args.num_inference_steps
                           if args.num_inference_steps is not None else defaults["num_inference_steps"])
    gs_str = "<pipeline default>" if guidance_scale is None else guidance_scale
    print(f"  Sampler: guidance_scale={gs_str}, num_inference_steps={num_inference_steps}")

    for topic_id in args.topic_ids:
        print(f"\n=== Topic {topic_id} (shard {args.shard_rank}/{args.num_shards}) ===")
        all_prompts = load_prompts(cluster_dir, topic_id)

        # Shard: this process handles every num_shards-th prompt starting at shard_rank
        prompts = all_prompts[args.shard_rank::args.num_shards]
        if args.num_shards > 1:
            print(f"  Shard {args.shard_rank}: {len(prompts)}/{len(all_prompts)} prompts")

        out_dir = output_dir / f"topic_{topic_id}" / mdir
        out_dir.mkdir(parents=True, exist_ok=True)

        baselines = generate(
            prompts=prompts,
            out_dir=out_dir,
            model_id=args.model_id,
            images_per_prompt=args.images_per_prompt,
            width=args.image_width,
            height=args.image_height,
            seed=args.random_seed,
            hf_cache_dir=args.hf_cache_dir,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            pipe=pipe,
        )

        metadata = {
            "model_id": args.model_id,
            "image_width": args.image_width,
            "image_height": args.image_height,
            "images_per_prompt": args.images_per_prompt,
            "random_seed": args.random_seed,
            "topic_ids": [topic_id],
        }

        if args.num_shards > 1:
            # Save partial manifest for this shard; merging done by run_mjhq.sh
            shard_path = out_dir / f"manifest_shard_{args.shard_rank}.json"
            with open(shard_path, "w", encoding="utf-8") as f:
                json.dump({"metadata": metadata, "baselines": baselines}, f,
                          indent=2, ensure_ascii=False)
            print(f"  Saved shard manifest: {shard_path}")
        else:
            # Single process: write final manifest directly
            manifest_path = out_dir / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump({"metadata": metadata, "baselines": baselines}, f,
                          indent=2, ensure_ascii=False)
            print(f"  Saved manifest: {manifest_path}")

    print("\nDone. Run score_baselines.py to add reward scores.")

    # Signal completion to the launcher BEFORE going idle, so the launcher
    # can start the manifest merge while we still hold the pipeline on GPU.
    if args.done_flag_dir:
        flag_dir = Path(args.done_flag_dir)
        flag_dir.mkdir(parents=True, exist_ok=True)
        flag_path = flag_dir / f"shard_{args.shard_rank}.done"
        flag_path.write_text("")
        print(f"  Touched done flag: {flag_path}")

    if args.keep_alive:
        import time
        print(f"[shard {args.shard_rank}] holding pipeline on GPU. "
              f"Send SIGINT/SIGTERM to release.", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print(f"[shard {args.shard_rank}] released.")


if __name__ == "__main__":
    main()
