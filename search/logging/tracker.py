"""ExperimentTracker: wandb metrics, images, and tables."""
from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from search.data.results import SearchResults, ParetoPoint
from search.data.state import AttributeStats

if TYPE_CHECKING:
    from search.config import LoggingConfig


class ExperimentTracker:
    """Wraps wandb for structured experiment logging."""

    def __init__(
        self,
        config: "LoggingConfig",
        run_name: str,
        config_snapshot: dict[str, Any],
    ):
        self.config = config
        self.run_name = run_name
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

    # ─── Step logging ─────────────────────────────────────────────────────────

    def log_step(
        self,
        step_idx: int,
        topic_id: int,
        stats_map: dict[str, AttributeStats],
        selected: list[ParetoPoint],
        cost_usd: float = 0.0,
    ) -> None:
        scored = [s for s in stats_map.values() if s.delta_rm() is not None]
        n_undesirable = sum(1 for s in scored if s.is_undesirable())
        mean_drm = float(np.mean([s.delta_rm() for s in scored])) if scored else 0.0
        dj_vals = [s.delta_j() for s in scored if s.delta_j() is not None]
        mean_dj = float(np.mean(dj_vals)) if dj_vals else 0.0

        metrics = {
            f"step/n_evaluated": len(scored),
            f"step/n_surviving": len(selected),
            f"step/mean_delta_rm": mean_drm,
            f"step/mean_delta_j": mean_dj,
            f"step/n_undesirable": n_undesirable,
            f"step/api_cost_usd": cost_usd,
        }

        logger.info(
            f"[step {step_idx} topic {topic_id}] "
            f"evaluated={len(scored)} surviving={len(selected)} "
            f"ΔRM={mean_drm:+.3f} ΔJ={mean_dj:+.3f} undesirable={n_undesirable}"
        )

        if self._wandb_run is None:
            return

        import wandb

        self._wandb_run.log(metrics, step=step_idx)

        # Attribute table
        rows = []
        for s in scored:
            rows.append([
                s.attribute,
                round(s.delta_rm() or 0.0, 4),
                round(s.delta_j() or 0.0, 4),
                round(s.amplification_score, 4),
                s.meta.parent or "",
                step_idx,
                topic_id,
            ])
        if rows:
            table = wandb.Table(
                columns=["attribute", "delta_rm", "delta_j", "amp_score", "parent", "step", "topic_id"],
                data=rows,
            )
            self._wandb_run.log({f"step/attributes_step{step_idx}": table}, step=step_idx)

    def log_image_pairs(
        self,
        step_idx: int,
        topic_id: int,
        stats_map: dict[str, AttributeStats],
        max_pairs: int = 8,
    ) -> None:
        if self._wandb_run is None:
            return

        import wandb
        from PIL import Image

        logged = 0
        images = []
        captions = []

        for attr, stats in stats_map.items():
            if logged >= max_pairs:
                break
            for prompt_text, pairs in stats.pairs.items():
                if logged >= max_pairs:
                    break
                for pair in pairs:
                    if pair.delta_rm is None:
                        continue
                    b_path = pair.baseline.image_path
                    e_path = pair.edited_image_path
                    if not b_path.exists() or not e_path.exists():
                        continue
                    try:
                        b_img = Image.open(b_path).convert("RGB")
                        e_img = Image.open(e_path).convert("RGB")
                        # Side-by-side
                        w = b_img.width + e_img.width + 4
                        h = max(b_img.height, e_img.height)
                        combined = Image.new("RGB", (w, h), (128, 128, 128))
                        combined.paste(b_img, (0, 0))
                        combined.paste(e_img, (b_img.width + 4, 0))
                        caption = f"{attr[:40]} | ΔRM={pair.delta_rm:+.3f}"
                        images.append(wandb.Image(combined, caption=caption))
                        captions.append(caption)
                        logged += 1
                    except Exception as e:
                        logger.warning(f"Failed to load image pair for logging: {e}")
                    if logged >= max_pairs:
                        break

        if images:
            self._wandb_run.log({f"step/pairs_step{step_idx}_topic{topic_id}": images}, step=step_idx)

    # ─── Final logging ────────────────────────────────────────────────────────

    def log_final(self, results: SearchResults) -> None:
        logger.info(
            f"Run complete — {len(results.pareto_front)} pareto points, "
            f"cost=${results.cost_usd:.2f}, time={results.wall_time_seconds:.0f}s"
        )

        if self._wandb_run is None:
            return

        import wandb

        rows = [
            [
                p.attribute,
                round(p.delta_rm, 4),
                round(p.delta_j, 4),
                round(p.amplification_score, 4),
                p.step_found,
                p.topic_id,
            ]
            for p in results.pareto_front
        ]
        if rows:
            table = wandb.Table(
                columns=["attribute", "delta_rm", "delta_j", "amp_score", "step_found", "topic_id"],
                data=rows,
            )
            self._wandb_run.log({"final/undesirable_attributes": table})

        self._wandb_run.log({
            "final/total_cost_usd": results.cost_usd,
            "final/n_pareto_points": len(results.pareto_front),
            "final/wall_time_seconds": results.wall_time_seconds,
        })

        # Save results artifact
        try:
            artifact = wandb.Artifact(name=f"results-{results.run_id}", type="results")
            with artifact.new_file("results.json") as f:
                import json
                json.dump(
                    {
                        "run_id": results.run_id,
                        "pareto_front": [p.to_dict() for p in results.pareto_front],
                        "n_steps_completed": results.n_steps_completed,
                        "cost_usd": results.cost_usd,
                    },
                    f,
                    indent=2,
                )
            self._wandb_run.log_artifact(artifact)
        except Exception as e:
            logger.warning(f"Failed to save wandb artifact: {e}")

        self._wandb_run.finish()

    def finish(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()
