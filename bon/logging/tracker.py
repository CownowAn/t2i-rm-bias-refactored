"""BonTracker: WandB logging for Best-of-N analysis."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from bon.config import LoggingConfig
    from bon.results import BonResults


class BonTracker:
    def __init__(
        self,
        config: "LoggingConfig",
        run_name: str,
        config_snapshot: dict[str, Any],
        output_dir: Path | None = None,
    ):
        self.config = config
        self.run_name = run_name
        self.output_dir = output_dir
        self._wandb_run = None

        if config.wandb.enabled:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=config.wandb.project,
                    entity=config.wandb.entity or None,
                    name=run_name,
                    config=config_snapshot,
                    tags=config.wandb.tags or [],
                )
                logger.info(f"wandb run: {self._wandb_run.url}")
            except Exception as e:
                logger.warning(f"wandb init failed: {e} — continuing without wandb")
                self._wandb_run = None

    def log_curves(
        self,
        topic_id: int,
        attributes: list[str],
        curves: dict[str, list[float]],
        n_values: list[int],
    ) -> None:
        """Log per-attribute prevalence curves."""
        logger.info(f"BoN prevalence curves — topic {topic_id}:")
        col = max((len(a) for a in attributes), default=0) + 2
        for attr in attributes:
            pairs = "  ".join(f"N={n}:{v:.3f}" for n, v in zip(n_values, curves[attr]))
            logger.info(f"  {attr:{col}}{pairs}")

        if self._wandb_run is None:
            return

        import wandb

        # One table per attribute (columns: n, prevalence) for line-plot rendering
        for attr in attributes:
            rows = [[n, v] for n, v in zip(n_values, curves[attr])]
            safe = attr.replace("/", "_").replace(" ", "_")[:50]
            self._wandb_run.log({
                f"bon/topic{topic_id}/{safe}": wandb.Table(
                    columns=["n", "prevalence"],
                    data=rows,
                )
            })

        # Combined summary table for this topic
        rows = [
            [attr] + [round(v, 4) for v in curves[attr]]
            for attr in attributes
        ]
        self._wandb_run.log({
            f"bon/summary_topic{topic_id}": wandb.Table(
                columns=["attribute"] + [f"N={n}" for n in n_values],
                data=rows,
            )
        })

    def log_final(self, all_results: list["BonResults"]) -> None:
        """Log a cross-topic summary after all topics are processed."""
        for results in all_results:
            logger.info(
                f"[topic {results.topic_id}] "
                f"{results.n_val_prompts} val prompts  "
                f"{len(results.attributes)} attributes  "
                f"time={results.wall_time_seconds:.0f}s"
            )
            col = max((len(a) for a in results.attributes), default=0) + 2
            for attr in results.attributes:
                pairs = "  ".join(
                    f"N={n}:{v:.3f}"
                    for n, v in zip(results.n_values, results.prevalence[attr])
                )
                logger.info(f"  {attr:{col}}{pairs}")

        if self._wandb_run is None:
            return

        import wandb

        # Flat table: one row per (topic, attribute, N)
        rows = []
        for results in all_results:
            for attr in results.attributes:
                for n, v in zip(results.n_values, results.prevalence[attr]):
                    rows.append([results.topic_id, attr, n, round(v, 4)])
        if rows:
            self._wandb_run.log({
                "bon/all_results": wandb.Table(
                    columns=["topic_id", "attribute", "n", "prevalence"],
                    data=rows,
                )
            })

        total_time = sum(r.wall_time_seconds for r in all_results)
        self._wandb_run.log({
            "bon/n_topics": len(all_results),
            "bon/total_wall_time_seconds": total_time,
        })

    def finish(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()