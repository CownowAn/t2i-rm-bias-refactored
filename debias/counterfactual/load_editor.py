"""Standalone: load FLUX-Kontext editor(s) on the requested GPU(s) and report
peak memory. Useful for:

  1. Verifying the env (diffusers/transformers compat) without running PoC.
  2. Measuring peak GPU memory before committing to a long run.
  3. Pre-warming the HF download cache (~24 GB).
  4. Keeping the model resident for ad-hoc inference (with `--keep_alive`).

Examples
────────
  # Quick env + memory check on GPU 2 (default uses /nfs HF cache).
  python -m debias.counterfactual.load_editor --devices cuda:0 \
      --test_inference

  # Load 4 GPUs, run a 1-image inference test, then wait for Ctrl+C.
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python -m debias.counterfactual.load_editor \
      --devices cuda:0,cuda:1,cuda:2,cuda:3 \
      --test_inference --keep_alive
"""
from __future__ import annotations

import argparse
import asyncio
import signal as _sig
import sys
import tempfile
from pathlib import Path

from loguru import logger

# Make project root importable when invoked as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from search.models.editor.flux_kontext import FluxKontextApplier  # noqa: E402


# ── Args ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--devices", default="cuda:0",
                   help="Comma-separated device list (e.g. 'cuda:0,cuda:1').")
    p.add_argument("--flux_model", default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--guidance_scale", type=float, default=2.5)
    p.add_argument("--hf_cache_dir", default="/nfs/data/sohyun/models")
    p.add_argument("--test_inference", action="store_true",
                   help="After loading, run one 512×512 dummy edit to verify GPU "
                        "memory is sufficient for actual inference (not just load).")
    p.add_argument("--test_image_size", type=int, default=512,
                   help="Side length for the dummy test image.")
    p.add_argument("--keep_alive", action="store_true",
                   help="Keep the model loaded until SIGINT/SIGTERM.")
    return p.parse_args()


# ── GPU memory reporting ──────────────────────────────────────────────────────


def _parse_idx(device: str) -> int:
    if ":" in device:
        return int(device.split(":")[1])
    return 0


def _report_memory(devices: list[str], header: str = "") -> None:
    import torch
    if header:
        logger.info(header)
    for d in devices:
        idx = _parse_idx(d)
        try:
            free, total = torch.cuda.mem_get_info(idx)
        except Exception as e:
            logger.warning(f"  {d}: mem_get_info failed: {e}")
            continue
        used = total - free
        # torch.cuda.max_memory_allocated returns the peak within this process.
        peak_alloc = torch.cuda.max_memory_allocated(idx)
        logger.info(
            f"  {d}: used={used / 1e9:>5.2f} GB / total={total / 1e9:>5.2f} GB "
            f"(free={free / 1e9:>5.2f} GB, this-process peak={peak_alloc / 1e9:>5.2f} GB)"
        )


# ── Main ──────────────────────────────────────────────────────────────────────


async def _wait_for_signal() -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for s in (_sig.SIGINT, _sig.SIGTERM):
        try:
            loop.add_signal_handler(s, stop.set)
        except (NotImplementedError, RuntimeError):
            pass
    await stop.wait()


def main() -> None:
    args = parse_args()
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if not devices:
        sys.exit("ERROR: --devices is empty")

    _report_memory(devices, header=f"Pre-load GPU memory ({len(devices)} device(s)):")

    logger.info(f"Loading {args.flux_model}  guidance_scale={args.guidance_scale} ...")
    appliers: list[FluxKontextApplier] = []
    for d in devices:
        a = FluxKontextApplier(
            model_name=args.flux_model,
            device=d,
            guidance_scale=args.guidance_scale,
            hf_cache_dir=args.hf_cache_dir,
        )
        try:
            a._load_pipeline()        # force eager load so we measure real memory
        except Exception as e:
            logger.exception(f"  ✗ load failed on {d}: {e}")
            sys.exit(1)
        logger.info(f"  ✓ loaded on {d}")
        appliers.append(a)

    _report_memory(devices, header="Post-load GPU memory:")

    if args.test_inference:
        logger.info(f"Running 1-step dummy inference on {devices[0]} "
                    f"({args.test_image_size}×{args.test_image_size}) ...")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            from PIL import Image
            src = td / "src.png"
            Image.new("RGB", (args.test_image_size, args.test_image_size),
                      (128, 128, 128)).save(src)
            out = td / "out.png"
            try:
                appliers[0].apply(
                    image_path=str(src),
                    instruction="Edit the image to add a small red dot in the centre.",
                    output_path=str(out),
                )
                size_kb = out.stat().st_size / 1024 if out.exists() else 0
                logger.info(f"  ✓ inference success ({size_kb:.0f} KB output)")
            except Exception as e:
                logger.exception(f"  ✗ inference failed: {e}")
                sys.exit(2)
        _report_memory(devices, header="Post-inference GPU memory (incl. peak):")

    if args.keep_alive:
        logger.info("Editor(s) loaded. Send SIGINT (Ctrl+C) or SIGTERM to release.")
        try:
            asyncio.run(_wait_for_signal())
        except KeyboardInterrupt:
            pass
        logger.info("→ exiting, GPU memory will be released.")
    else:
        logger.info("Done. (pass --keep_alive to keep the model loaded.)")


if __name__ == "__main__":
    main()