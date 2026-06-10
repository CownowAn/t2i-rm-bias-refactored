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
    from search.data.bon_amplified_types import BonAmplifiedStep


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

    def log_bon_evaluate(
        self,
        step_idx: int,
        topic_id: int,
        N: int,
        selected: list[str],
        amp_scores: dict[str, float],
        tau_rejected: list[str],
        tau: float,
        first_seen: dict[str, int] | None = None,
    ) -> None:
        """Log EVALUATE phase for bon_amplified mode (S_t selection by A_hat > tau)."""
        first_seen = first_seen or {}
        n_sel = len(selected)
        n_rej = len(tau_rejected)
        sel_vals = [amp_scores.get(a, 0.0) for a in selected]
        rej_vals = [amp_scores.get(a, 0.0) for a in tau_rejected]
        sel_max  = max(sel_vals, default=0.0)
        sel_mean = float(np.mean(sel_vals)) if sel_vals else 0.0

        col = max((len(a) for a in (selected + tau_rejected)), default=0) + 2
        logger.info(f"[BON step {step_idx} topic {topic_id}] EVALUATE  N={N}  tau={tau}")
        if tau_rejected:
            logger.info(f"  ✗ A_hat≤{tau} rejected ({n_rej}):")
            for a in tau_rejected:
                logger.info(f"      {a:{col}}  A_hat={amp_scores.get(a, 0.0):.4f}")
        if selected:
            logger.info(f"  ✓ S_t ({n_sel} attrs)  A_hat: max={sel_max:.4f}  mean={sel_mean:.4f}")
            # Sort by A_hat desc; show age = step_idx - first_seen[attr]
            sorted_sel = sorted(selected,
                                key=lambda a: amp_scores.get(a, 0.0), reverse=True)
            logger.info(f"      {'age':>4s} {'A_hat':>10s}  attribute")
            for a in sorted_sel:
                age = step_idx - first_seen.get(a, step_idx)
                logger.info(
                    f"      {age:>4d} {amp_scores.get(a, 0.0):>+10.4f}  {a}"
                )

        if self._wandb_run is None:
            return

        import wandb
        pfx = f"bon/topic_{topic_id}"
        self._wandb_run.log({
            f"{pfx}/n_selected":          n_sel,
            f"{pfx}/n_tau_rejected":      n_rej,
            f"{pfx}/selected_max_ahat":   sel_max,
            f"{pfx}/selected_mean_ahat":  sel_mean,
            f"{pfx}/tau":                 tau,
            f"{pfx}/N":                   N,
        }, step=step_idx)

        if selected:
            rows = [[a, round(amp_scores.get(a, 0.0), 4), step_idx] for a in selected]
            self._wandb_run.log({
                f"bon/selected_step{step_idx}_topic{topic_id}": wandb.Table(
                    columns=["attribute", "ahat", "step"], data=rows,
                )
            }, step=step_idx)

        if tau_rejected:
            rows_rej = [[a, round(amp_scores.get(a, 0.0), 4), step_idx] for a in tau_rejected]
            self._wandb_run.log({
                f"bon/tau_rejected_step{step_idx}_topic{topic_id}": wandb.Table(
                    columns=["attribute", "ahat", "step"], data=rows_rej,
                )
            }, step=step_idx)

    def log_bon_expand(
        self,
        step_idx: int,
        topic_id: int,
        ba_step: "BonAmplifiedStep",
        acc_pool: list[str],
        p_plus_n_prompts: int,
        raw_candidates: list[str],
        humanness_rejected: list[str],
        validated: list[str],
        tau_rejected_cands: list[str],
        candidate_ahat: dict[str, float],
        tau: float,
        output_dir: Path | None = None,
        candidate_partial_ahat: dict[str, float] | None = None,
    ) -> None:
        """Log EXPAND phase for bon_amplified mode (OLS residuals → P+/P- → proposal → validation)."""
        n_raw  = len(raw_candidates)
        n_hum  = len(humanness_rejected)
        n_val  = len(validated)
        n_rej  = len(tau_rejected_cands)
        var_exp = ba_step.reg_var_explained
        W       = ba_step.W
        W_mode  = getattr(ba_step, "W_mode", "global")
        pp_r2   = ba_step.per_prompt_r2 or {}
        pp_W    = ba_step.per_prompt_W or {}

        val_ahats = [candidate_ahat.get(a, 0.0) for a in validated]
        rej_ahats = [candidate_ahat.get(a, 0.0) for a in tau_rejected_cands]
        val_partials = [(candidate_partial_ahat or {}).get(a, 0.0) for a in validated]
        rej_partials = [(candidate_partial_ahat or {}).get(a, 0.0) for a in tau_rejected_cands]

        logger.info(
            f"[BON step {step_idx} topic {topic_id}] EXPAND  "
            f"N_imgs={ba_step.n_images}  K={len(acc_pool)}  "
            f"var_explained={var_exp:.4f}  W_mode={W_mode}"
        )

        # Funnel summary (#9)
        n_p1_rej = 0   # populated externally if needed; for now derive from current counters
        funnel_proposed = n_raw
        funnel_humanness = funnel_proposed - n_hum
        funnel_validated = n_val
        funnel_tau_rej   = n_rej
        if funnel_proposed > 0:
            logger.info(
                f"  EXPAND funnel: "
                f"proposed={funnel_proposed} → "
                f"humanness_pass={funnel_humanness} ({funnel_humanness / funnel_proposed:.0%}) → "
                f"validated={funnel_validated} ({funnel_validated / funnel_proposed:.0%})  "
                f"[tau/Top-K' rejected={funnel_tau_rej}]"
            )

        # Admitted partial_A_hat aggregates (#2)
        if val_partials:
            logger.info(
                f"  admitted partial_A_hat: "
                f"mean={float(np.mean(val_partials)):+.4f}  "
                f"max={float(np.max(val_partials)):+.4f}  "
                f"min={float(np.min(val_partials)):+.4f}"
            )

        # Per-prompt R² distribution
        if pp_r2:
            r2_vals = np.array(list(pp_r2.values()))
            logger.info(
                f"  per-prompt R²: mean={r2_vals.mean():.4f} "
                f"median={float(np.median(r2_vals)):.4f} "
                f"max={r2_vals.max():.4f}  "
                f"R²>0.05: {int((r2_vals > 0.05).sum())}/{len(r2_vals)}  "
                f"R²>0.10: {int((r2_vals > 0.10).sum())}/{len(r2_vals)}  "
                f"R²>0.20: {int((r2_vals > 0.20).sum())}/{len(r2_vals)}"
            )

        if W and acc_pool:
            for k, attr in enumerate(acc_pool):
                w_val = W[k] if k < len(W) else 0.0
                logger.info(f"  {attr[:55]:55s}  W={w_val:+.4f}")

        # Per-prompt W (all prompts, sorted by R² desc) — only when per_prompt mode is on
        if pp_W and pp_r2:
            sorted_prompts = sorted(pp_r2, key=pp_r2.get, reverse=True)
            logger.info(f"  per-prompt W (sorted by R²):")
            for p in sorted_prompts:
                w_str = ", ".join(f"{w:+.3f}" for w in pp_W[p])
                logger.info(f"    R²={pp_r2[p]:.3f}  W=[{w_str}]  '{p[:80]}'")

        logger.info(
            f"  P+/P- prompts={p_plus_n_prompts}  "
            f"proposed={n_raw}  humanness_failed={n_hum}  "
            f"validated={n_val}  tau_rejected={n_rej}"
        )

        # Partial A_hat (monotonic admission mode)
        if candidate_partial_ahat:
            logger.info(f"  partial A_hat (sorted desc):")
            sorted_cands = sorted(candidate_partial_ahat.items(),
                                  key=lambda kv: kv[1], reverse=True)
            validated_set = set(validated)
            for cand, pa in sorted_cands:
                mark = "✓" if cand in validated_set else "✗"
                a_hat = candidate_ahat.get(cand, 0.0)
                logger.info(
                    f"    {mark}  partial={pa:+.4f}  A_hat={a_hat:+.4f}  '{cand[:60]}'"
                )

        if validated:
            for a in validated:
                if candidate_partial_ahat:
                    logger.info(
                        f"  ✓ {a}  partial={candidate_partial_ahat.get(a, 0.0):+.4f}  "
                        f"A_hat={candidate_ahat.get(a, 0.0):+.4f}"
                    )
                else:
                    logger.info(f"  ✓ {a}  A_hat={candidate_ahat.get(a, 0.0):.4f}")

        # ── JSON snapshot ──────────────────────────────────────────────────────
        if output_dir is not None:
            record = {
                "step_idx": step_idx,
                "topic_id": topic_id,
                "acc_pool": acc_pool,
                "W": {attr: (W[k] if k < len(W) else 0.0) for k, attr in enumerate(acc_pool)},
                "W_mode": W_mode,
                "reg_var_explained": var_exp,
                "n_residual_images": ba_step.n_images,
                "p_plus_n_prompts": p_plus_n_prompts,
                "raw_candidates": raw_candidates,
                "humanness_rejected": humanness_rejected,
                "validated": validated,
                "tau_rejected_candidates": tau_rejected_cands,
                "candidate_ahat": {a: round(candidate_ahat.get(a, 0.0), 6)
                                   for a in raw_candidates},
                "candidate_partial_ahat": (
                    {a: round(candidate_partial_ahat.get(a, 0.0), 6) for a in raw_candidates}
                    if candidate_partial_ahat else None
                ),
                "tau": tau,
                "per_prompt_r2": pp_r2,
                "funnel": {
                    "proposed":        funnel_proposed,
                    "humanness_pass":  funnel_humanness,
                    "validated":       funnel_validated,
                    "tau_or_topk_rej": funnel_tau_rej,
                },
                "admitted_partial_ahat": (
                    {
                        "mean": float(np.mean(val_partials)) if val_partials else 0.0,
                        "max":  float(np.max(val_partials))  if val_partials else 0.0,
                        "min":  float(np.min(val_partials))  if val_partials else 0.0,
                    }
                    if candidate_partial_ahat else None
                ),
            }
            local_path = output_dir / f"ba_expand_step{step_idx}_topic{topic_id}.json"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "w") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            logger.info(f"  Saved EXPAND snapshot → {local_path}")

            # Per-prompt W as separate artifact when populated
            if pp_W:
                pp_W_path = output_dir / f"per_prompt_W_step{step_idx}_topic{topic_id}.json"
                with open(pp_W_path, "w") as f:
                    json.dump({
                        "step_idx": step_idx,
                        "topic_id": topic_id,
                        "attrs": acc_pool,
                        "per_prompt_W": pp_W,
                        "per_prompt_r2": pp_r2,
                    }, f, indent=2, ensure_ascii=False)
                logger.info(f"  Saved per-prompt W → {pp_W_path}")

        if self._wandb_run is None:
            return

        import wandb
        pfx = f"bon/topic_{topic_id}"
        metrics: dict[str, Any] = {
            f"{pfx}/n_residual_images":      ba_step.n_images,
            f"{pfx}/n_p_plus_prompts":       p_plus_n_prompts,
            f"{pfx}/n_raw_candidates":       n_raw,
            f"{pfx}/n_humanness_rejected":   n_hum,
            f"{pfx}/n_validated":            n_val,
            f"{pfx}/n_tau_rejected_cands":   n_rej,
            f"{pfx}/reg_var_explained":      var_exp,
        }
        if val_ahats:
            metrics[f"{pfx}/validated_mean_ahat"] = float(np.mean(val_ahats))
        if rej_ahats:
            metrics[f"{pfx}/tau_rejected_mean_ahat"] = float(np.mean(rej_ahats))

        # Funnel rates (#9)
        if funnel_proposed > 0:
            metrics[f"{pfx}/funnel_humanness_rate"] = funnel_humanness / funnel_proposed
            metrics[f"{pfx}/funnel_validated_rate"] = funnel_validated / funnel_proposed

        # Admitted partial_A_hat aggregates (#2) — monotonic mode only
        if val_partials:
            metrics[f"{pfx}/admitted_partial_ahat_mean"] = float(np.mean(val_partials))
            metrics[f"{pfx}/admitted_partial_ahat_max"]  = float(np.max(val_partials))
            metrics[f"{pfx}/admitted_partial_ahat_min"]  = float(np.min(val_partials))

        # Per-prompt R² distribution metrics
        if pp_r2:
            r2_vals = np.array(list(pp_r2.values()))
            metrics[f"{pfx}/per_prompt_r2_mean"]   = float(r2_vals.mean())
            metrics[f"{pfx}/per_prompt_r2_median"] = float(np.median(r2_vals))
            metrics[f"{pfx}/per_prompt_r2_max"]    = float(r2_vals.max())
            metrics[f"{pfx}/n_prompts_r2_above_0.05"] = int((r2_vals > 0.05).sum())
            metrics[f"{pfx}/n_prompts_r2_above_0.10"] = int((r2_vals > 0.10).sum())
            metrics[f"{pfx}/n_prompts_r2_above_0.20"] = int((r2_vals > 0.20).sum())

        self._wandb_run.log(metrics, step=step_idx)

        if W and acc_pool:
            rows_w = [
                [attr, round(W[k] if k < len(W) else 0.0, 6), step_idx]
                for k, attr in enumerate(acc_pool)
            ]
            self._wandb_run.log({
                f"bon/weights_step{step_idx}_topic{topic_id}": wandb.Table(
                    columns=["attribute", "W", "step"], data=rows_w,
                )
            }, step=step_idx)

        # Per-prompt W table (only in per_prompt mode)
        if pp_W and acc_pool:
            columns = ["prompt", "R²"] + [f"W[{i}]" for i in range(len(acc_pool))]
            rows_pp = []
            for prompt, W_x in pp_W.items():
                r2_x = pp_r2.get(prompt, 0.0)
                row = [prompt[:80], round(r2_x, 4)] + [
                    round(W_x[k] if k < len(W_x) else 0.0, 4)
                    for k in range(len(acc_pool))
                ]
                rows_pp.append(row)
            self._wandb_run.log({
                f"bon/per_prompt_W_step{step_idx}_topic{topic_id}":
                    wandb.Table(columns=columns, data=rows_pp),
            }, step=step_idx)

        if validated:
            rows_val = [[a, round(candidate_ahat.get(a, 0.0), 4), step_idx] for a in validated]
            self._wandb_run.log({
                f"bon/validated_expand_step{step_idx}_topic{topic_id}": wandb.Table(
                    columns=["attribute", "ahat", "step"], data=rows_val,
                )
            }, step=step_idx)

    # ─── Per-prompt R² history dump (#8) ──────────────────────────────────────

    def log_per_prompt_r2_history(
        self,
        topic_id: int,
        history: list[dict],
        output_dir: Path | None = None,
    ) -> None:
        """Dump per-prompt R² history (one entry per step) for offline analysis."""
        if not history:
            return
        target_dir = output_dir or self.output_dir
        if target_dir is None:
            return
        path = target_dir / f"per_prompt_r2_history_topic{topic_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"topic_id": topic_id, "history": history}, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved per-prompt R² history → {path}")

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

