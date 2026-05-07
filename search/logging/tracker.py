"""ExperimentTracker: wandb metrics, images, and tables."""
from __future__ import annotations

import json
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from search.data.results import SearchResults, FoundAttribute
from search.data.state import AttributeStats

if TYPE_CHECKING:
    from search.config import LoggingConfig
    from search.data.baseline_pair_types import BaselinePairStep, BaselinePair


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
                round(s.meta.amp_mean_mu1, 4) if s.meta.amp_mean_mu1 is not None else None,
                round(s.meta.amp_mean_mu0, 4) if s.meta.amp_mean_mu0 is not None else None,
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

    # ─── Baseline-pairs mode logging ─────────────────────────────────────────

    def log_bp_evaluate(
        self,
        step_idx: int,
        topic_id: int,
        new_attrs: list[str],
        amp_scores: dict[str, float],
        acc_pool_size: int,
        humanness_failed: list[str] | None = None,
        mu_failed_stats: dict[str, tuple] | None = None,
    ) -> None:
        """Log EVALUATE phase results for baseline-pairs mode."""
        humanness_failed = humanness_failed or []
        mu_failed_stats = mu_failed_stats or {}
        n_new = len(new_attrs)
        max_amp = max(amp_scores.values(), default=0.0)
        mean_amp = float(np.mean(list(amp_scores.values()))) if amp_scores else 0.0

        all_names = new_attrs + humanness_failed + list(mu_failed_stats.keys())
        col = max((len(a) for a in all_names), default=0) + 2

        logger.info(f"[BP step {step_idx} topic {topic_id}] EVALUATE")

        # Humanness rejections
        if humanness_failed:
            logger.info(
                f"  ✗ Humanness rejected ({len(humanness_failed)}): "
                + ", ".join(humanness_failed)
            )

        # μ1>μ0 rejections with values
        if mu_failed_stats:
            logger.info(f"  ✗ μ1≤μ0 rejected ({len(mu_failed_stats)}):")
            for attr, stats in mu_failed_stats.items():
                mu1, mu0 = stats[0], stats[1]
                n1 = stats[2] if len(stats) > 2 else None
                n0 = stats[3] if len(stats) > 3 else None
                mu1_str = f"{mu1:.3f}" if mu1 is not None else "n/a"
                mu0_str = f"{mu0:.3f}" if mu0 is not None else "n/a"
                n = (n1 or 0) + (n0 or 0)
                p1_str = f"{n1/n:.2f}" if n1 is not None and n > 0 else "n/a"
                p0_str = f"{n0/n:.2f}" if n0 is not None and n > 0 else "n/a"
                count_str = f"g1={n1 if n1 is not None else '?'}  g0={n0 if n0 is not None else '?'}  p1={p1_str}  p0={p0_str}"
                logger.info(f"      {attr:{col}}  μ1={mu1_str}  μ0={mu0_str}  {count_str}")

        # Selected attrs
        if new_attrs:
            logger.info(f"  ✓ Selected {n_new} (acc_pool={acc_pool_size}):")
            for attr in new_attrs:
                logger.info(f"      {attr:{col}}  A(g)={amp_scores.get(attr, 0):.4f}")
        else:
            logger.warning(f"  No attrs selected (acc_pool={acc_pool_size})")

        if self._wandb_run is None:
            return

        import wandb

        pfx = f"bp/topic_{topic_id}"
        self._wandb_run.log({
            f"{pfx}/acc_pool_size":        acc_pool_size,
            f"{pfx}/n_new_attrs":          n_new,
            f"{pfx}/n_humanness_rejected": len(humanness_failed),
            f"{pfx}/n_mu_rejected":        len(mu_failed_stats),
            f"{pfx}/max_amp_score":        max_amp,
            f"{pfx}/mean_amp_score":       mean_amp,
        }, step=step_idx)

        # Selected attrs table
        if new_attrs:
            rows = [[a, round(amp_scores.get(a, 0), 4), step_idx] for a in new_attrs]
            self._wandb_run.log({
                f"bp/new_attrs_step{step_idx}_topic{topic_id}": wandb.Table(
                    columns=["attribute", "amp_score", "step"], data=rows,
                )
            }, step=step_idx)

        # Rejected attrs table
        rejected_rows = (
            [[a, "humanness", None, None, None, None, step_idx] for a in humanness_failed]
            + [[a, "mu_filter",
                stats[0], stats[1],
                stats[2] if len(stats) > 2 else None,
                stats[3] if len(stats) > 3 else None,
                step_idx]
               for a, stats in mu_failed_stats.items()]
        )
        if rejected_rows:
            self._wandb_run.log({
                f"bp/rejected_step{step_idx}_topic{topic_id}": wandb.Table(
                    columns=["attribute", "reason", "mu1", "mu0", "g1_count", "g0_count", "step"],
                    data=rejected_rows,
                )
            }, step=step_idx)

    def log_bp_expand(
        self,
        step_idx: int,
        topic_id: int,
        bp_step: "BaselinePairStep",
        acc_pool: list[str],
        proposed_attrs: list[str],
        high_residual_pairs: list["BaselinePair"] | None = None,
        diverse_pairs: list["BaselinePair"] | None = None,
        all_amp_scores: dict[str, float] | None = None,
        output_dir: Path | None = None,
    ) -> None:
        """Log EXPAND phase results for baseline-pairs mode."""
        n_pairs = len(bp_step.pairs)
        W_rm = bp_step.W_rm
        residuals = bp_step.residuals

        # Lasso stats
        var_explained: float | None = None
        if W_rm is not None and residuals is not None and n_pairs > 0:
            delta_rm = bp_step.delta_rm_vec
            if delta_rm is not None and np.var(delta_rm) > 1e-10:
                var_explained = float(max(0.0, 1.0 - np.var(residuals) / np.var(delta_rm)))

        # Residual stats
        res_mean_abs: float | None = None
        res_max_abs: float | None = None
        if residuals is not None and len(residuals) > 0:
            res_mean_abs = float(np.mean(np.abs(residuals)))
            res_max_abs = float(np.max(np.abs(residuals)))

        # ── Console ──────────────────────────────────────────────────────────
        reg_intercept = bp_step.reg_intercept if W_rm is not None else None
        reg_alpha = bp_step.reg_alpha if W_rm is not None else None
        reg_l1_ratio = getattr(bp_step, "reg_l1_ratio", None) if W_rm is not None else None
        var_str = f"{var_explained:.3f}" if var_explained is not None else "n/a"
        alpha_str = f"  alpha={reg_alpha:.4f}" if reg_alpha is not None else ""
        l1_str = f"  l1_ratio={reg_l1_ratio:.2f}" if reg_l1_ratio is not None else ""
        intercept_str = f"  intercept={reg_intercept:+.4f}" if reg_intercept is not None else ""
        col = max((len(a) for a in acc_pool), default=0) + 2

        logger.info(
            f"[BP step {step_idx} topic {topic_id}] EXPAND  "
            f"pairs={n_pairs}  ElasticNet: K={len(acc_pool)} N={n_pairs} "
            f"var_explained={var_str}{alpha_str}{l1_str}{intercept_str}"
        )

        # Attribute weights table
        if W_rm is not None:
            amp_lookup = all_amp_scores or bp_step.amp_scores
            for k, attr in enumerate(acc_pool):
                logger.info(
                    f"  {attr:{col}}  W_rm={W_rm[k]:+.4f}  A(g)={amp_lookup.get(attr, 0.0):.4f}"
                )

        # Residual summary
        if res_mean_abs is not None:
            logger.info(
                f"  Residuals: mean|r|={res_mean_abs:.4f}  max|r|={res_max_abs:.4f}"
            )

        # High-residual pairs shown to LLM
        if high_residual_pairs and bp_step.pairs and residuals is not None:
            pair_to_idx = {id(p): i for i, p in enumerate(bp_step.pairs)}
            logger.info(
                f"  High-residual pairs shown to LLM ({len(high_residual_pairs)}):"
            )
            for pi, pair in enumerate(high_residual_pairs, 1):
                idx = pair_to_idx.get(id(pair))
                res_val = float(residuals[idx]) if idx is not None else float("nan")
                prompt = pair.high_reward.prompt.text
                logger.info(
                    f"    Pair {pi}  residual={res_val:+.4f}  gap={pair.delta_rm:.4f}"
                    f"  '{prompt}'"
                )
                # Attr diff from D matrix row
                if idx is not None and bp_step.D is not None:
                    d_row = bp_step.D[idx]
                    diffs = [
                        f"{a}={d_row[k]:+.0f}"
                        for k, a in enumerate(acc_pool)
                        if d_row[k] != 0
                    ]
                    if diffs:
                        logger.info(f"      attr Δ (high−low): {', '.join(diffs)}")
                    else:
                        logger.info("      attr Δ: (all zero — pair has same attr vector)")
        
        # Diverse pairs shown to LLM
        if diverse_pairs and bp_step.pairs and residuals is not None:
            logger.info(
                f"  Diverse pairs shown to LLM ({len(diverse_pairs)}):"
            )
            for pi, pair in enumerate(diverse_pairs, 1):
                idx = pair_to_idx.get(id(pair))
                res_val = float(residuals[idx]) if idx is not None else float("nan")
                prompt = pair.high_reward.prompt.text
                logger.info(
                    f"    Pair {pi}  residual={res_val:+.4f}  gap={pair.delta_rm:.4f}"
                    f"  '{prompt}'"
                )
                # Attr diff from D matrix row
                if idx is not None and bp_step.D is not None:
                    d_row = bp_step.D[idx]
                    diffs = [
                        f"{a}={d_row[k]:+.0f}"
                        for k, a in enumerate(acc_pool)
                        if d_row[k] != 0
                    ]
                    if diffs:
                        logger.info(f"      attr Δ (high−low): {', '.join(diffs)}")
                    else:
                        logger.info("      attr Δ: (all zero — pair has same attr vector)")

        # Proposed attrs
        if proposed_attrs:
            logger.info(f"  Proposed {len(proposed_attrs)} → EvoStep[{step_idx + 1}]:")
            for i, attr in enumerate(proposed_attrs, 1):
                logger.info(f"    {i}. {attr}")
        else:
            logger.info("  No new attrs proposed")

        # Local JSON save — complete snapshot for offline analysis
        if output_dir is not None:
            amp_lookup = all_amp_scores or bp_step.amp_scores
            record = {
                "step_idx": step_idx,
                "topic_id": topic_id,
                "acc_pool": acc_pool,
                "new_attrs_this_step": bp_step.attribute_pool,
                "amp_scores": {a: round(amp_lookup.get(a, 0), 6) for a in acc_pool},
                "n_pairs": n_pairs,
                "regression": {
                    "variance_explained": var_explained,
                    "alpha": round(bp_step.reg_alpha, 6),
                    "l1_ratio": round(getattr(bp_step, "reg_l1_ratio", 1.0), 4),
                    "intercept": round(bp_step.reg_intercept, 6),
                    "K": len(acc_pool),
                    "N": n_pairs,
                    "W_rm": {attr: round(float(W_rm[k]), 6) for k, attr in enumerate(acc_pool)}
                    if W_rm is not None else {},
                    "residual_mean_abs": res_mean_abs,
                    "residual_max_abs": res_max_abs,
                },
                "high_residual_pairs": _serialize_pairs(
                    high_residual_pairs or [], bp_step
                ),
                "diverse_pairs": _serialize_pairs(
                    diverse_pairs or [], bp_step
                ),
                "proposed_attrs": proposed_attrs,
            }
            local_path = output_dir / f"bp_expand_step{step_idx}_topic{topic_id}.json"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "w") as f:
                json.dump(record, f, indent=2)
            logger.info(f"  Saved EXPAND snapshot → {local_path}")

            # Save regression matrices/vectors as .npz for offline analysis
            npz_arrays: dict[str, Any] = {
                "acc_pool": np.array(acc_pool, dtype=object),
            }
            if bp_step.D is not None:
                npz_arrays["D"] = bp_step.D                          # (N, K) int8
            if bp_step.delta_rm_vec is not None:
                npz_arrays["delta_rm_vec"] = bp_step.delta_rm_vec    # (N,) float32
            if bp_step.W_rm is not None:
                npz_arrays["W_rm"] = bp_step.W_rm                    # (K,) float32
            if bp_step.residuals is not None:
                npz_arrays["residuals"] = bp_step.residuals          # (N,) float32
            npz_arrays["reg_intercept"] = np.float32(bp_step.reg_intercept)
            npz_arrays["reg_alpha"] = np.float32(bp_step.reg_alpha)
            npz_arrays["reg_l1_ratio"] = np.float32(bp_step.reg_l1_ratio)
            npz_path = output_dir / f"bp_reg_step{step_idx}_topic{topic_id}.npz"
            np.savez_compressed(npz_path, **npz_arrays)
            logger.info(f"  Saved regression arrays → {npz_path}")

        if self._wandb_run is None:
            return

        import wandb

        pfx = f"bp/topic_{topic_id}"
        metrics: dict[str, Any] = {
            f"{pfx}/n_pairs":          n_pairs,
            f"{pfx}/n_proposed":       len(proposed_attrs),
        }
        if var_explained is not None:
            metrics[f"{pfx}/reg_var_explained"] = var_explained
        if res_mean_abs is not None:
            metrics[f"{pfx}/residual_mean_abs"] = res_mean_abs
            metrics[f"{pfx}/residual_max_abs"]  = res_max_abs
        self._wandb_run.log(metrics, step=step_idx)

        # Full attr pool table with A(g) and W_rm
        if W_rm is not None and acc_pool:
            amp_lookup = all_amp_scores or bp_step.amp_scores
            rows = [
                [attr, round(amp_lookup.get(attr, 0), 4),
                 round(float(W_rm[k]), 4), attr in bp_step.attribute_pool, step_idx]
                for k, attr in enumerate(acc_pool)
            ]
            self._wandb_run.log({
                f"bp/pool_step{step_idx}_topic{topic_id}": wandb.Table(
                    columns=["attribute", "amp_score", "W_rm", "new_this_step", "step"],
                    data=rows,
                )
            }, step=step_idx)

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
                round(p.delta_rm, 4) if p.delta_rm is not None else None,
                round(p.delta_j, 4) if p.delta_j is not None else None,
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


# ── Module-level helpers ──────────────────────────────────────────────────────

def _serialize_pairs(
    pairs: list["BaselinePair"],
    bp_step: "BaselinePairStep",
) -> list[dict]:
    if not pairs or bp_step.residuals is None:
        return []
    pair_to_idx = {id(p): i for i, p in enumerate(bp_step.pairs)}
    result = []
    for p in pairs:
        idx = pair_to_idx.get(id(p))
        result.append({
            "prompt": p.high_reward.prompt.text,
            "residual": round(float(bp_step.residuals[idx]), 6) if idx is not None else None,
            "delta_rm": round(p.delta_rm, 6),
            "delta_j": round(p.delta_j, 4) if p.delta_j is not None else None,
        })
    return result
