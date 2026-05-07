"""Entry point: python search/main.py --config search/configs/default.yaml [key=value ...]"""
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
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dot-path overrides, e.g. evolution.n_steps=3 run.name=exp01",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    from search.config import SearchConfig
    config = SearchConfig.from_yaml(args.config, overrides=args.overrides or [])
    config.validate()

    # Configure loguru
    logger.remove()
    logger.add(sys.stderr, level=config.logging.console_level)
    log_path = Path(config.run_output_dir()) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), level="DEBUG", rotation="50 MB")

    # Save the original config yaml and the effective config (with CLI overrides applied)
    shutil.copy(args.config, log_path.parent / "config_source.yaml")
    effective_config_path = log_path.parent / "config_effective.yaml"
    with open(effective_config_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Run: {config.run.name}")
    logger.info(f"Output dir: {config.run_output_dir()}")
    logger.info(f"n_steps={config.evolution.n_steps}, direction={config.evolution.direction}")

    from search.utils.cost import log_cost_estimate
    log_cost_estimate(config)

    from search.logging.tracker import ExperimentTracker
    tracker = ExperimentTracker(
        config=config.logging,
        run_name=config.run.name,
        config_snapshot=config.to_dict(),
        output_dir=Path(config.run_output_dir()),
    )

    if config.pipeline.mode == "edit":
        from search.pipeline.evolution import EvolutionEngine
        engine = EvolutionEngine.from_config(config, tracker=tracker)
    elif config.pipeline.mode == "baseline_pairs":
        from search.pipeline.baseline_evo import BaselinePairEvolutionEngine
        engine = BaselinePairEvolutionEngine.from_config(config, tracker=tracker)
    else:
        raise ValueError(f"Unknown pipeline.mode: {config.pipeline.mode!r}")

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
