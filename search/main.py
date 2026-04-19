"""Entry point: python search/main.py --config search/configs/default.yaml [key=value ...]"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

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

    logger.info(f"Run: {config.run.name}")
    logger.info(f"Output dir: {config.run_output_dir()}")
    logger.info(f"n_steps={config.evolution.n_steps}, direction={config.evolution.direction}")

    from search.logging.tracker import ExperimentTracker
    tracker = ExperimentTracker(
        config=config.logging,
        run_name=config.run.name,
        config_snapshot=config.to_dict(),
    )

    from search.pipeline.evolution import EvolutionEngine
    engine = EvolutionEngine.from_config(config, tracker=tracker)

    results = await engine.run()

    # Save results JSON
    output_path = config.run_output_dir() / "results.json"
    results.save(output_path)
    logger.info(f"Results saved to {output_path}")

    tracker.log_final(results)

    logger.info(
        f"Done. {len(results.pareto_front)} undesirable attributes found "
        f"in {results.wall_time_seconds:.0f}s, cost=${results.cost_usd:.2f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
