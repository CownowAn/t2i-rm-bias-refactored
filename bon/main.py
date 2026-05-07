"""Entry point: python -m bon.main --config bon/configs/default.yaml [key=value ...]"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T2I RM bias Best-of-N analysis")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dot-path overrides, e.g. sampling.n_trials=50 data.val_split_size=20",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    from bon.config import BonConfig
    config = BonConfig.from_yaml(args.config, overrides=args.overrides or [])
    config.validate()

    logger.remove()
    logger.add(sys.stderr, level=config.logging.console_level)
    log_path = Path(config.run_output_dir()) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), level="DEBUG", rotation="50 MB")

    logger.info(f"Run: {config.run.name}")
    logger.info(f"Output dir: {config.run_output_dir()}")
    logger.info(f"n_values={config.sampling.n_values}, n_trials={config.sampling.n_trials}")
    if config.attributes.attributes:
        logger.info(f"Attributes: explicit list ({len(config.attributes.attributes)}): {config.attributes.attributes}")
    else:
        logger.info(f"Attributes: from search results {config.attributes.search_results_path!r}  only_undesirable={config.attributes.only_undesirable}")

    from bon.utils.cost import log_bon_cost_estimate
    log_bon_cost_estimate(config)

    from bon.logging.tracker import BonTracker
    tracker = BonTracker(
        config=config.logging,
        run_name=config.run.name,
        config_snapshot=config.to_dict(),
        output_dir=Path(config.run_output_dir()),
    )

    from bon.runner import BonRunner
    runner = BonRunner.from_config(config, tracker=tracker)

    try:
        all_results = await runner.run()
    finally:
        await runner.shutdown()

    for results in all_results:
        out = config.run_output_dir() / f"results_topic{results.topic_id}.json"
        results.save(out)
        logger.info(f"Results saved → {out}")

    tracker.log_final(all_results)
    tracker.finish()


if __name__ == "__main__":
    asyncio.run(main())
