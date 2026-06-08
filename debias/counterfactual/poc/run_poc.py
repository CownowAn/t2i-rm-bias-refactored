"""PoC entry point: editor capability survey."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

# Make the project root importable when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from debias.counterfactual.edit.editor_runner import EditorRunner  # noqa: E402
from debias.counterfactual.io_utils import infer_ba_expand_path  # noqa: E402
from debias.counterfactual.poc.survey import run_survey  # noqa: E402
from debias.counterfactual.schemas import PoCConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Editor capability survey for counterfactual debiasing PoC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Inputs
    p.add_argument("--per_prompt_W_path", required=True, type=Path)
    p.add_argument("--ba_expand_path", type=Path, default=None,
                   help="Optional; auto-derived from per_prompt_W path if omitted.")
    p.add_argument("--topic_id", type=int, required=True)
    p.add_argument("--detection_cache_path", required=True, type=Path)
    p.add_argument("--detector_key", default="Qwen/Qwen3.5-9B::auto",
                   help="Top-level key in the detection cache JSON.")
    p.add_argument("--baseline_manifest", required=True, type=Path)
    p.add_argument("--baseline_root", default="",
                   help="Prefix for relative image_path in manifest. Empty = absolute paths.")

    # Run identity
    p.add_argument("--run_id", default=None,
                   help="Default: timestamp via search.utils.io.timestamp()")

    # Selection
    p.add_argument("--tau", type=float, default=0.0)
    p.add_argument("--top_n_per_prompt", type=int, default=3)
    p.add_argument("--n_prompts_per_attr", type=int, default=5)
    p.add_argument("--n_images_per_prompt", type=int, default=4)
    p.add_argument("--humanness_recheck", action="store_true")
    p.add_argument("--humanness_model", default="openai/gpt-5")
    p.add_argument("--source_consistency_n", type=int, default=0,
                   help="If >0, re-query the detector this many times on each candidate "
                        "source (image, attr). Keep only sources where every query returns "
                        "g=1. Filters out detector-noise-driven false positives in the "
                        "cache. 0 disables the check.")
    p.add_argument("--limit_n_attrs", type=int, default=0,
                   help="If >0, only run the PoC on the first N attrs of per_prompt_W "
                        "(in the order they appear in the file). 0 = no limit.")

    # Instruction phrasing
    from debias.counterfactual.edit.instruction_builder import INSTRUCTION_MODES
    p.add_argument("--instruction_mode", default="correct",
                   choices=INSTRUCTION_MODES,
                   help="'correct' (default) reframes as fix-with-natural to avoid the "
                        "'object erased' failure mode of plain 'remove' phrasing. "
                        "Each mode has its own cache namespace under <cf_root>/edits/<mode>/.")

    # Editor
    p.add_argument("--flux_model", default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--editor_devices", default="cuda:0",
                   help="Comma-separated device list (e.g. 'cuda:0,cuda:1,cuda:2,cuda:3'). "
                        "One FluxKontextApplier per device; coroutines round-robin via GpuApplierPool.")
    p.add_argument("--editor_max_parallel", type=int, default=None,
                   help="Concurrent in-flight edits. Default = number of editor devices.")
    p.add_argument("--guidance_scale", type=float, default=2.5)
    p.add_argument("--hf_cache_dir", default="/nfs/data/sohyun/models")

    # Detector
    p.add_argument("--detector_model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--detector_vllm_base_url", default=None)
    p.add_argument("--detector_image_detail", default="auto")
    p.add_argument("--detector_max_parallel", type=int, default=32)
    p.add_argument("--detector_max_tokens", type=int, default=1024)

    # Verification options
    p.add_argument("--side_effect_check", action="store_true")
    p.add_argument("--make_thumbnails", action="store_true")
    p.add_argument("--check_reward", action="store_true",
                   help="Score (orig, edited) with the reward model and report ΔR per attr.")
    p.add_argument("--reward_model", default="imagereward",
                   choices=["imagereward", "pickscore", "hpsv3"])
    p.add_argument("--reward_device", default=None,
                   help="Default: first editor device (same GPU as one of the FLUX appliers).")
    p.add_argument("--reward_hf_cache_dir", default="/nfs/data/sohyun/models")

    # Lifecycle
    p.add_argument("--keep_alive", action="store_true",
                   help="After the survey finishes, do NOT release GPU memory. "
                        "The editor (FLUX-Kontext) stays loaded on the chosen device(s); "
                        "send SIGINT (Ctrl+C) or SIGTERM to release and exit.")

    # Output roots
    p.add_argument("--cf_root", type=Path, default=None,
                   help="Default: /nfs/data/sohyun/projects/t2i-rm-bias/counterfactuals")
    p.add_argument("--report_root", type=Path, default=None,
                   help="Default: outputs/counterfactual_poc")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_detector(args: argparse.Namespace):
    from search.config import DetectorConfig
    from search.models.detector import build_detector as _build
    cfg = DetectorConfig(
        model=args.detector_model,
        max_tokens=args.detector_max_tokens,
        max_parallel=args.detector_max_parallel,
        image_detail=args.detector_image_detail,
        use_batch_api=False,
        vllm_base_url=args.detector_vllm_base_url,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        } if args.detector_vllm_base_url else None,
        use_prompt=False,
        use_reasoning=False,
    )
    return _build(cfg, cache_config=None)


async def amain() -> None:
    args = parse_args()
    from search.utils.io import timestamp
    run_id = args.run_id or timestamp()

    # Resolve ba_expand_path
    ba_expand_path = args.ba_expand_path
    if ba_expand_path is None:
        ba_expand_path = infer_ba_expand_path(args.per_prompt_W_path)
        if ba_expand_path is not None:
            logger.info(f"auto-derived ba_expand_path → {ba_expand_path}")

    cfg = PoCConfig(
        tau=args.tau,
        top_n_per_prompt=args.top_n_per_prompt,
        n_prompts_per_attr=args.n_prompts_per_attr,
        n_images_per_prompt=args.n_images_per_prompt,
        humanness_recheck=args.humanness_recheck,
        side_effect_check=args.side_effect_check,
        make_thumbnails=args.make_thumbnails,
        seed=args.seed,
        run_id=run_id,
    )
    logger.info(f"PoC config: {cfg}")

    devices = [d.strip() for d in args.editor_devices.split(",") if d.strip()]
    editor = EditorRunner(
        model_name=args.flux_model,
        devices=devices,
        guidance_scale=args.guidance_scale,
        hf_cache_dir=args.hf_cache_dir,
        max_parallel=args.editor_max_parallel,
    )
    detector = build_detector(args)

    reward_model = None
    reward_model_name: str | None = None
    if args.check_reward:
        rm_device = args.reward_device or devices[0]
        logger.info(f"loading reward model '{args.reward_model}' on {rm_device}")
        if args.reward_model == "imagereward":
            from search.models.reward.imagereward import ImageRewardModel
            reward_model = ImageRewardModel(
                device=rm_device, hf_cache_dir=args.reward_hf_cache_dir,
            )
        elif args.reward_model == "pickscore":
            from search.models.reward.pickscore import PickScoreModel
            reward_model = PickScoreModel(
                device=rm_device, hf_cache_dir=args.reward_hf_cache_dir,
            )
        elif args.reward_model == "hpsv3":
            from search.models.reward.hpsv3 import HPSv3Model
            reward_model = HPSv3Model(
                device=rm_device, hf_cache_dir=args.reward_hf_cache_dir,
            )
        reward_model_name = args.reward_model

    try:
        await run_survey(
            ppw_path=args.per_prompt_W_path,
            ba_expand_path=ba_expand_path,
            detection_cache_path=args.detection_cache_path,
            detector_key=args.detector_key,
            baseline_manifest_path=args.baseline_manifest,
            baseline_root=args.baseline_root,
            topic_id=args.topic_id,
            cfg=cfg,
            editor=editor,
            detector=detector,
            humanness_model=args.humanness_model,
            cf_root=args.cf_root,
            report_root=args.report_root,
            reward_model=reward_model,
            reward_model_name=reward_model_name,
            source_consistency_n=args.source_consistency_n,
            instruction_mode=args.instruction_mode,
            limit_n_attrs=args.limit_n_attrs,
        )
        if args.keep_alive:
            await _wait_for_signal(editor)
    finally:
        # Best-effort shutdown for caller-backed components (detector / API clients).
        # We deliberately do NOT touch the editor / reward model here — Python GC
        # will release GPU memory when the process exits.
        for obj in (detector,):
            if hasattr(obj, "shutdown"):
                try:
                    await obj.shutdown()
                except Exception as e:
                    logger.warning(f"shutdown failed for {obj}: {e}")


async def _wait_for_signal(editor) -> None:
    """Block until SIGINT/SIGTERM. Editor stays loaded on GPU until then."""
    import signal as _sig

    devices = getattr(editor, "_devices", ["<unknown>"])
    logger.info(
        f"✓ survey complete — editor (FLUX-Kontext) remains loaded on {devices}."
    )
    logger.info("  Send SIGINT (Ctrl+C) or SIGTERM to release GPU memory and exit.")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (_sig.SIGINT, _sig.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            # Windows / restricted env: fall back to default KeyboardInterrupt path
            pass
    try:
        await stop.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    logger.info("→ signal received, exiting and releasing GPU memory.")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
