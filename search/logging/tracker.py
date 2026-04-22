"""ExperimentTracker: wandb metrics, images, and tables."""
from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from search.data.results import SearchResults, FoundAttribute
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

    # ─── Step logging ─────────────────────────────────────────────────────────

    def log_step(
        self,
        step_idx: int,
        topic_id: int,
        stats_map: dict[str, AttributeStats],
        selected: list[FoundAttribute],
        cost_usd: float = 0.0,
        use_outlier_removal: bool = False,
    ) -> None:
        scored = [s for s in stats_map.values() if s.delta_rm() is not None]
        undesirables = [s for s in scored if s.is_undesirable()]
        n_undesirable = len(undesirables)
        undesirable_rate = n_undesirable / len(scored) if scored else 0.0

        # Overall means (all scored)
        mean_drm = float(np.mean([s.delta_rm(use_outlier_removal) for s in scored])) if scored else 0.0
        dj_vals = [s.delta_j() for s in scored if s.delta_j() is not None]
        mean_dj = float(np.mean(dj_vals)) if dj_vals else 0.0

        # Undesirable-only means
        mean_drm_und = float(np.mean([s.delta_rm(use_outlier_removal) for s in undesirables])) if undesirables else 0.0
        dj_und_vals = [s.delta_j() for s in undesirables if s.delta_j() is not None]
        mean_dj_und = float(np.mean(dj_und_vals)) if dj_und_vals else 0.0

        # A(g) stats among survivors
        amp_scores = [s.amplification_score for s in undesirables if s.amplification_score > 0]
        mean_amp = float(np.mean(amp_scores)) if amp_scores else 0.0
        max_amp = float(np.max(amp_scores)) if amp_scores else 0.0

        pfx = f"step/topic_{topic_id}"
        metrics = {
            f"{pfx}/n_evaluated": len(scored),
            f"{pfx}/n_surviving": len(selected),
            f"{pfx}/n_undesirable": n_undesirable,
            f"{pfx}/undesirable_rate": undesirable_rate,
            f"{pfx}/mean_drm_all": mean_drm,
            f"{pfx}/mean_dj_all": mean_dj,
            f"{pfx}/mean_drm_undesirable": mean_drm_und,
            f"{pfx}/mean_dj_undesirable": mean_dj_und,
            f"{pfx}/mean_amp_score_undesirable": mean_amp,
            f"{pfx}/max_amp_score": max_amp,
            f"{pfx}/api_cost_usd": cost_usd,
        }

        logger.info(
            f"[step {step_idx} topic {topic_id}] "
            f"evaluated={len(scored)} undesirable={n_undesirable} ({undesirable_rate:.0%}) "
            f"surviving={len(selected)} "
            f"ΔRM_und={mean_drm_und:+.3f} ΔJ_und={mean_dj_und:+.3f} "
            f"A(g)_max={max_amp:.4f}"
        )

        if self._wandb_run is None:
            return

        import wandb

        self._wandb_run.log(metrics, step=step_idx)

        # Attribute table
        rows = []
        for s in scored:
            n_rollouts = sum(len(pairs) for pairs in s.pairs.values())
            rows.append([
                s.attribute,
                round(s.delta_rm(use_outlier_removal) or 0.0, 4),
                round(s.delta_j() or 0.0, 4),
                round(s.amplification_score, 4),
                round(s.meta.amp_mean_p1, 4),
                round(s.meta.amp_mean_p0, 4),
                round(s.meta.amp_mean_mu1, 4),
                round(s.meta.amp_mean_mu0, 4),
                s.is_undesirable(),
                n_rollouts,
                s.meta.parent or "",
                step_idx,
                topic_id,
            ])
        if rows:
            table = wandb.Table(
                columns=["attribute", "delta_rm", "delta_j", "amp_score",
                         "amp_p1", "amp_p0", "amp_mu1", "amp_mu0",
                         "is_undesirable", "n_rollouts", "parent", "step", "topic_id"],
                data=rows,
            )
            self._wandb_run.log({f"step/attributes_step{step_idx}_topic{topic_id}": table}, step=step_idx)

    def log_image_pairs(
        self,
        step_idx: int,
        topic_id: int,
        stats_map: dict[str, AttributeStats],
        max_pairs: int = 8,
    ) -> None:
        # ── Local JSON log (all pairs, absolute paths) ────────────────────────
        if self.output_dir is not None:
            import json
            records = []
            for attr, stats in stats_map.items():
                for prompt_text, pairs in stats.pairs.items():
                    for pair in pairs:
                        detected = stats.baseline_detected.get(pair.baseline.image_id)
                        records.append({
                            "attribute": attr,
                            "prompt": prompt_text,
                            "edit_instruction": pair.edit_instruction,
                            "baseline_image": str(pair.baseline.image_path.resolve()),
                            "edited_image": str(pair.edited_image_path.resolve()),
                            "delta_rm": pair.delta_rm,
                            "delta_j": pair.delta_j,
                            "baseline_detected": detected,
                        })
            local_path = self.output_dir / f"pairs_step{step_idx}_topic{topic_id}.json"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "w") as f:
                json.dump(records, f, indent=2)
            logger.info(f"Saved {len(records)} pairs locally → {local_path}")

        # ── wandb image log (capped at max_pairs) ─────────────────────────────
        if self._wandb_run is None:
            return

        import wandb
        from PIL import Image

        logged = 0
        images = []

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
                        w = b_img.width + e_img.width + 4
                        h = max(b_img.height, e_img.height)
                        combined = Image.new("RGB", (w, h), (128, 128, 128))
                        combined.paste(b_img, (0, 0))
                        combined.paste(e_img, (b_img.width + 4, 0))
                        dj_str = f"{pair.delta_j:+.3f}" if pair.delta_j is not None else "n/a"
                        detected = stats.baseline_detected.get(pair.baseline.image_id)
                        detected_str = {1: "yes", 0: "no"}.get(detected, "n/a")
                        caption = (
                            f"{attr}\n"
                            f"prompt: {prompt_text}\n"
                            f"instruction: {pair.edit_instruction}\n"
                            f"ΔRM={pair.delta_rm:+.3f}  ΔJ={dj_str}  detected={detected_str}"
                        )
                        images.append(wandb.Image(combined, caption=caption))
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
            f"Run complete — {len(results.top_attributes)} pareto points, "
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
                p.step_last_survived,
                p.topic_id,
            ]
            for p in results.top_attributes
        ]
        if rows:
            table = wandb.Table(
                columns=["attribute", "delta_rm", "delta_j", "amp_score", "step_found", "step_last_survived", "topic_id"],
                data=rows,
            )
            self._wandb_run.log({"final/undesirable_attributes": table})

        self._wandb_run.log({
            "final/total_cost_usd": results.cost_usd,
            "final/n_top_attributes": len(results.top_attributes),
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
                        "top_attributes": [p.to_dict() for p in results.top_attributes],
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
