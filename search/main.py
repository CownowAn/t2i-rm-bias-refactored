"""Entry point: python search/main.py --config search/configs/x.yaml [key=value ...]"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

import yaml
from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T2I reward-model bias evolutionary search")
    parser.add_argument("--config", default=None,
                        help="Path to YAML config file. Required unless --resume_from "
                             "is given, in which case the resumed run's "
                             "configs/config_effective.yaml is used.")
    parser.add_argument("--resume_from", type=Path, default=None,
                        help="Path to an existing run dir (e.g. outputs/search/20260609-140932). "
                             "Loads its config_effective.yaml, redirects output to the "
                             "same dir, auto-detects the highest completed "
                             "ba_expand_step{N}_topic*.json across topics, and continues "
                             "from step N+1.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dot-path overrides, e.g. evolution.n_steps=3 run.name=exp01",
    )
    args = parser.parse_args()
    if args.config is None and args.resume_from is None:
        parser.error("must supply --config or --resume_from")
    return args


async def main() -> None:
    args = parse_args()

    from search.config import SearchConfig

    # Resume mode: load config from the resumed run's configs/config_effective.yaml
    # (unless --config explicitly overrides) and force output back into the same dir.
    resume_from: Path | None = args.resume_from
    if resume_from is not None:
        resume_from = resume_from.resolve()
        if not resume_from.is_dir():
            raise SystemExit(f"--resume_from: not a directory: {resume_from}")
        cfg_source = args.config
        if cfg_source is None:
            cfg_source = resume_from / "configs" / "config_effective.yaml"
            if not cfg_source.exists():
                cfg_source = resume_from / "config_effective.yaml"  # legacy layout
            if not cfg_source.exists():
                raise SystemExit(
                    f"--resume_from: no config_effective.yaml under {resume_from}"
                )
        config = SearchConfig.from_yaml(cfg_source, overrides=args.overrides or [])
        # Force run output back into the existing dir.
        config.run.output_dir = str(resume_from.parent)
        config.run.name = resume_from.name
    else:
        config = SearchConfig.from_yaml(args.config, overrides=args.overrides or [])
    config.validate()

    # Configure loguru
    logger.remove()
    logger.add(sys.stderr, level=config.logging.console_level)
    logs_dir = Path(config.run_output_dir()) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    info_log_path = logs_dir / "run.info.log"
    logger.add(str(info_log_path), level="INFO", rotation="50 MB")
    debug_log_path = logs_dir / "run.debug.log"
    logger.add(str(debug_log_path), level="DEBUG", rotation="50 MB")

    # Save the original config yaml and the effective config (with CLI overrides applied)
    # In resume mode, preserve the original config_source.yaml and write the
    # post-override effective config to a timestamped sidecar instead of
    # overwriting the existing one.
    configs_dir = Path(config.run_output_dir()) / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    if resume_from is None:
        shutil.copy(args.config, configs_dir / "config_source.yaml")
        effective_config_path = configs_dir / "config_effective.yaml"
    else:
        from search.utils.io import timestamp as _ts
        effective_config_path = configs_dir / f"config_effective_resumed_{_ts()}.yaml"
        logger.info(f"Resuming run: {resume_from}")
    with open(effective_config_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Run: {config.run.name}")
    logger.info(f"Output dir: {config.run_output_dir()}")

    from search.utils.cost import log_cost_estimate
    log_cost_estimate(config)

    from search.logging.tracker import ExperimentTracker
    tracker = ExperimentTracker(
        config=config.logging,
        run_name=config.run.name,
        config_snapshot=config.to_dict(),
        output_dir=Path(config.run_output_dir()),
    )

    from search.pipeline.bon_amplified_evo import BonAmplifiedEvolutionEngine
    engine = BonAmplifiedEvolutionEngine.from_config(
        config, tracker=tracker, resume_from_dir=resume_from,
    )

    try:
        results = await engine.run()
    finally:
        await engine.shutdown()

    # Save results JSON
    output_path = config.run_output_dir() / "results.json"
    results.save(output_path)
    logger.info(f"Results saved to {output_path}")

    tracker.log_final(results)

    n_true_undesirable = sum(1 for fa in results.top_attributes if fa.is_undesirable)
    logger.info(
        f"Done. {n_true_undesirable} truly undesirable attributes found "
        f"({len(results.top_attributes)} total including padding) "
        f"in {results.wall_time_seconds:.0f}s, cost=${results.cost_usd:.2f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
