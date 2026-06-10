"""Add reward scores to a baseline manifest.json.

Reads an existing manifest.json produced by generate_images.py,
scores each image with the specified reward model, and writes scores back.

Concurrency safety: uses a sidecar lockfile (O_CREAT|O_EXCL) around a
re-read-merge-atomic-rename cycle so multiple processes scoring different
reward models against the same manifest do not overwrite each other.

Usage:
    python baselines/score_baselines.py \
        --manifest_path data/baselines/mjhq/topic_0/black-forest-labs-FLUX.1-dev/manifest.json \
        --reward_model imagereward \
        --device cuda:0

Resumable: skips images that already have a score for the given reward model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score baseline images with a reward model")
    p.add_argument("--manifest_path", required=True,
                   help="Path to manifest.json to score")
    p.add_argument("--reward_model", default="imagereward",
                   choices=["imagereward", "pickscore", "hpsv3", "vqascore"],
                   help="Reward model to use (default: imagereward)")
    p.add_argument("--vqa_model", default="clip-flant5-xxl",
                   help="VQAScore backbone (only used when --reward_model vqascore); "
                        "e.g. clip-flant5-xxl, qwen2.5-vl-7b, gemma-3-27b-it, gpt-4o")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--hf_cache_dir", default="/nfs/data/sohyun/models")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Number of images to score per batch")
    p.add_argument("--lock_timeout", type=int, default=900,
                   help="Seconds to wait for manifest write-lock (default: 900)")
    return p.parse_args()


def load_reward_model(name: str, device: str, hf_cache_dir: str,
                      vqa_model: str = "clip-flant5-xxl"):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    if name == "imagereward":
        from search.models.reward.imagereward import ImageRewardModel
        return ImageRewardModel(device=device, hf_cache_dir=hf_cache_dir)
    elif name == "pickscore":
        from search.models.reward.pickscore import PickScoreModel
        return PickScoreModel(device=device, hf_cache_dir=hf_cache_dir)
    elif name == "hpsv3":
        from search.models.reward.hpsv3 import HPSv3Model
        return HPSv3Model(device=device, hf_cache_dir=hf_cache_dir)
    elif name == "vqascore":
        from search.models.reward.vqascore import VQAScoreModel
        return VQAScoreModel(device=device, hf_cache_dir=hf_cache_dir, vqa_model=vqa_model)
    else:
        sys.exit(f"Unknown reward model: {name}")


class FileLock:
    """Sidecar O_EXCL lockfile. Portable on local FS and most NFS setups.

    Stale-lock recovery: if the lockfile is older than `stale_after` seconds
    and the recorded PID no longer exists, the lock is force-released.
    """

    def __init__(self, target_path: Path, timeout: int = 900, stale_after: int = 1800):
        self.target = Path(target_path)
        self.lock_path = self.target.with_suffix(self.target.suffix + ".lock")
        self.timeout = timeout
        self.stale_after = stale_after
        self._fd: int | None = None

    def _maybe_clear_stale(self) -> None:
        try:
            age = time.time() - self.lock_path.stat().st_mtime
        except FileNotFoundError:
            return
        if age < self.stale_after:
            return
        try:
            pid = int(self.lock_path.read_text().strip().splitlines()[0])
        except (OSError, ValueError, IndexError):
            pid = -1
        if pid > 0:
            try:
                os.kill(pid, 0)
                return  # owner still alive
            except ProcessLookupError:
                pass
        print(f"  Removing stale lock {self.lock_path} (age={age:.0f}s)")
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> "FileLock":
        deadline = time.time() + self.timeout
        while True:
            try:
                self._fd = os.open(
                    self.lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
                os.write(self._fd, f"{os.getpid()}\n".encode())
                os.fsync(self._fd)
                return self
            except FileExistsError:
                if time.time() > deadline:
                    raise TimeoutError(
                        f"Could not acquire {self.lock_path} within {self.timeout}s"
                    )
                self._maybe_clear_stale()
                time.sleep(2.0)

    def __exit__(self, *exc) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass


def merge_scores_atomic(
    manifest_path: Path,
    new_scores: dict[str, float],
    reward_name: str,
    lock_timeout: int,
) -> int:
    """Re-read manifest under lock, merge new_scores (keyed by image_id),
    write atomically. Returns count of entries updated."""
    if not new_scores:
        return 0
    with FileLock(manifest_path, timeout=lock_timeout):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        updated = 0
        for entries in manifest.get("baselines", {}).values():
            for entry in entries:
                iid = entry.get("image_id")
                if iid in new_scores:
                    entry.setdefault("reward_scores", {})[reward_name] = new_scores[iid]
                    updated += 1
        # Atomic write: tempfile in same dir, then rename
        fd, tmp = tempfile.mkstemp(
            dir=str(manifest_path.parent),
            prefix=manifest_path.name + ".",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            os.replace(tmp, manifest_path)
        except Exception:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise
    return updated


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        sys.exit(f"ERROR: manifest not found: {manifest_path}")

    # Snapshot read (no lock): decide which images need scoring.
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    baselines = manifest.get("baselines", {})
    reward_name = args.reward_model
    if args.reward_model == "vqascore":
        # Include the backbone so multiple VQAScore models coexist (don't clobber).
        reward_name = f"vqascore_{args.vqa_model}"

    to_score: list[tuple[str, dict]] = []     # (prompt, entry)
    for prompt, entries in baselines.items():
        for entry in entries:
            if reward_name not in (entry.get("reward_scores") or {}):
                to_score.append((prompt, entry))

    if not to_score:
        print(f"All images already have '{reward_name}' scores. Nothing to do.")
        return

    print(f"Loading {reward_name} model on {args.device} ...")
    model = load_reward_model(args.reward_model, args.device, args.hf_cache_dir,
                              vqa_model=args.vqa_model)

    total = len(to_score)
    print(f"Scoring {total} images ...")

    import asyncio
    from tqdm import tqdm

    batch_size = args.batch_size
    pbar = tqdm(total=total, unit="img", desc="Scoring")

    # Accumulate scores keyed by image_id; do NOT mutate the in-memory manifest.
    # We re-read + merge atomically at the end so we never clobber other
    # processes' concurrent writes to the same manifest.
    new_scores: dict[str, float] = {}

    for batch_start in range(0, total, batch_size):
        batch = to_score[batch_start: batch_start + batch_size]
        img_paths = [str(manifest_path.parent / e["image_path"])
                     if not Path(e["image_path"]).is_absolute()
                     else e["image_path"]
                     for _, e in batch]
        prompts_b = [p for p, _ in batch]

        results = asyncio.run(model.rate(img_paths, prompts_b))

        for (prompt, entry), result in zip(batch, results):
            iid = entry.get("image_id")
            if iid is None:
                continue
            score = result.score if result.score is not None else float("nan")
            new_scores[iid] = score
        pbar.update(len(batch))

    pbar.close()

    print(f"\nMerging {len(new_scores)} '{reward_name}' scores into {manifest_path} ...")
    updated = merge_scores_atomic(
        manifest_path, new_scores, reward_name, lock_timeout=args.lock_timeout,
    )
    print(f"Done. Updated {updated} entries.")


if __name__ == "__main__":
    main()
