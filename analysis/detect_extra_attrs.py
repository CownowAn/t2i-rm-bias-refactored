"""Run VLM detection for additional attributes (e.g. desirable ones) on existing baselines.

Reuses the detector configured in bon_amplified.yaml. Results are merged into
the existing detection cache so per_prompt_r2.py can immediately use them.

Usage:
    python analysis/detect_extra_attrs.py \
        --config search/configs/bon_amplified.yaml \
        --attrs_file analysis/desirable_attrs.txt \
        --manifest_path /nfs/data/sohyun/projects/t2i-rm-bias/data/baselines/mjhq/topic_0/black-forest-labs-FLUX.1-dev/manifest.json \
        --cache_path outputs/detection_cache/mjhq/black-forest-labs-FLUX.1-dev.json

The --attrs_file is plain text, one attribute per line (blank lines ignored).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from search.config import SearchConfig                              # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="search/configs/bon_amplified.yaml",
                   help="Path to YAML config (for detector model settings)")
    p.add_argument("--attrs_file", required=True,
                   help="Plain-text file with one attribute per line")
    p.add_argument("--manifest_path", required=True,
                   help="Path to baseline manifest.json")
    p.add_argument("--cache_path", required=True,
                   help="Path to detection cache JSON to update")
    p.add_argument("--dotenv_path", default=".env",
                   help="Path to .env (for OPENAI_API_KEY if used)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print plan but don't run detection")
    p.add_argument("--cache_images_only", action="store_true", default=True,
                   help="Only detect on images already present in the cache (i.e. the fixed baselines). "
                        "Default: True.")
    p.add_argument("--all_images", dest="cache_images_only", action="store_false",
                   help="Detect on ALL images in the manifest (overrides --cache_images_only)")
    return p.parse_args()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def load_attrs(attrs_file: str) -> list[str]:
    lines = Path(attrs_file).read_text().splitlines()
    attrs = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    # Dedup preserving order
    seen, out = set(), []
    for a in attrs:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def build_baseline_objs(manifest_path: str) -> dict[str, list]:
    """Reconstruct {prompt: [SimpleNamespace(image_id, image_path)]} from manifest."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    by_prompt = {}
    manifest_dir = Path(manifest_path).parent
    for prompt, entries in manifest["baselines"].items():
        imgs = []
        for e in entries:
            img_path = e["image_path"]
            # If relative, resolve against manifest dir
            if not Path(img_path).is_absolute():
                img_path = str(manifest_dir / img_path)
            imgs.append(SimpleNamespace(
                image_id=e["image_id"],
                image_path=img_path,
            ))
        by_prompt[prompt] = imgs
    return by_prompt


def cache_model_key(cfg) -> str:
    return f"{cfg.models.detector.model}::{cfg.models.detector.image_detail}"


def filter_missing_attrs(
    attrs: list[str], cache: dict, model_key: str, image_ids: set[str],
) -> list[str]:
    """Return attrs that are NOT fully covered in cache for these images."""
    det = cache.get(model_key, {})
    missing = []
    for a in attrs:
        covered = sum(1 for iid in image_ids if a in det.get(iid, {}))
        if covered < len(image_ids):
            missing.append(a)
            print(f"  '{a[:60]}...' → {covered}/{len(image_ids)} covered, will detect")
        else:
            print(f"  '{a[:60]}...' → already fully covered, skip")
    return missing


async def detect(
    cfg, attrs: list[str], baselines: dict, existing_cache_for_model: dict,
) -> dict[str, dict[str, int]]:
    from search.models.detector import build_detector
    from search.pipeline._shared import _detect_all_attributes

    cache_config = cfg.caller_cache.build()
    detector_model = build_detector(cfg.models.detector, cache_config=cache_config)

    detection = await _detect_all_attributes(
        detector_model,
        attrs,
        baselines,
        existing_cache=existing_cache_for_model,
        _retry=True,
    )
    return detection


def main() -> None:
    args = parse_args()
    load_dotenv(Path(args.dotenv_path))

    cfg = SearchConfig.from_yaml(args.config)
    model_key = cache_model_key(cfg)

    attrs = load_attrs(args.attrs_file)
    print(f"Loaded {len(attrs)} attrs from {args.attrs_file}")

    baselines = build_baseline_objs(args.manifest_path)
    n_imgs_total = sum(len(v) for v in baselines.values())
    print(f"Loaded {len(baselines)} prompts, {n_imgs_total} images from manifest")

    cache_path = Path(args.cache_path)
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"Loaded cache: {cache_path}")
    else:
        cache = {}
        print(f"No existing cache, will create: {cache_path}")

    # ── Restrict to fixed-baseline images (those already in cache) ────────────
    if args.cache_images_only and model_key in cache:
        cached_image_ids = set(cache[model_key].keys())
        filtered = {}
        for prompt, imgs in baselines.items():
            kept = [img for img in imgs if img.image_id in cached_image_ids]
            if kept:
                filtered[prompt] = kept
        baselines = filtered
        n_imgs = sum(len(v) for v in baselines.values())
        print(f"Restricted to images present in cache: "
              f"{len(baselines)} prompts, {n_imgs}/{n_imgs_total} images")
    elif args.cache_images_only:
        print(f"WARNING: --cache_images_only set but cache has no key '{model_key}'. "
              f"Falling back to all manifest images.")

    all_image_ids = {img.image_id for imgs in baselines.values() for img in imgs}

    print(f"\nDetector model: {cfg.models.detector.model} (key: {model_key})")
    print(f"Checking attr coverage in cache ...")
    missing = filter_missing_attrs(attrs, cache, model_key, all_image_ids)

    if not missing:
        print("\nAll attrs already fully covered. Nothing to do.")
        return

    print(f"\nWill run detection for {len(missing)} attrs × {n_imgs} images.")
    if args.dry_run:
        print("(dry_run, exiting)")
        return

    existing_cache_for_model = cache.setdefault(model_key, {})

    detection = asyncio.run(detect(cfg, missing, baselines, existing_cache_for_model))

    # Merge into cache
    for image_id, attr_vals in detection.items():
        existing_cache_for_model.setdefault(image_id, {}).update(attr_vals)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    print(f"\nUpdated cache saved to: {cache_path}")
    print(f"Now run: python analysis/per_prompt_r2.py "
          f"--manifest {args.manifest_path} --cache {args.cache_path} "
          f"--model_key '{model_key}' --attrs <your list>")


if __name__ == "__main__":
    main()
