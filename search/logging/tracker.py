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
        acc_pool_amp_scores: dict[str, float],
        acc_pool_size: int,
        humanness_failed: list[str] | None = None,
        mu_failed_stats: dict[str, tuple] | None = None,
    ) -> None:
        """Log EVALUATE phase results for baseline-pairs mode."""
        humanness_failed = humanness_failed or []
        mu_failed_stats = mu_failed_stats or {}
        n_new = len(new_attrs)

        # New attrs stats
        new_max_amp  = max(amp_scores.values(), default=0.0)
        new_mean_amp = float(np.mean(list(amp_scores.values()))) if amp_scores else 0.0

        # acc_pool stats (all attrs accumulated so far, including new ones)
        pool_max_amp  = max(acc_pool_amp_scores.values(), default=0.0)
        pool_mean_amp = float(np.mean(list(acc_pool_amp_scores.values()))) if acc_pool_amp_scores else 0.0

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

        # Selected attrs (new this step)
        if new_attrs:
            logger.info(f"  ✓ Selected {n_new} new attrs  A(g): max={new_max_amp:.4f}  mean={new_mean_amp:.4f}")
            for attr in new_attrs:
                logger.info(f"      {attr:{col}}  A(g)={amp_scores.get(attr, 0):.4f}")
        else:
            logger.warning(f"  No attrs selected (acc_pool={acc_pool_size})")

        # acc_pool summary
        logger.info(
            f"  acc_pool ({acc_pool_size} attrs)  A(g): max={pool_max_amp:.4f}  mean={pool_mean_amp:.4f}"
        )

        if self._wandb_run is None:
            return

        import wandb

        pfx = f"bp/topic_{topic_id}"
        self._wandb_run.log({
            f"{pfx}/acc_pool_size":        acc_pool_size,
            f"{pfx}/n_new_attrs":          n_new,
            f"{pfx}/n_humanness_rejected": len(humanness_failed),
            f"{pfx}/n_mu_rejected":        len(mu_failed_stats),
            f"{pfx}/new_max_amp_score":    new_max_amp,
            f"{pfx}/new_mean_amp_score":   new_mean_amp,
            f"{pfx}/pool_max_amp_score":   pool_max_amp,
            f"{pfx}/pool_mean_amp_score":  pool_mean_amp,
        }, step=step_idx)

        # New attrs table (with A(g))
        if new_attrs:
            rows = [[a, round(amp_scores.get(a, 0), 4), step_idx] for a in new_attrs]
            self._wandb_run.log({
                f"bp/new_attrs_step{step_idx}_topic{topic_id}": wandb.Table(
                    columns=["attribute", "amp_score", "step"], data=rows,
                )
            }, step=step_idx)

        # Full acc_pool A(g) table
        if acc_pool_amp_scores:
            pool_rows = [
                [a, round(v, 4), a in set(new_attrs), step_idx]
                for a, v in sorted(acc_pool_amp_scores.items(), key=lambda x: -x[1])
            ]
            self._wandb_run.log({
                f"bp/acc_pool_step{step_idx}_topic{topic_id}": wandb.Table(
                    columns=["attribute", "amp_score", "is_new", "step"], data=pool_rows,
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
        # All regression stats use full N pairs (before judge filtering).
        # bp_step.residuals may be filtered to M < N pairs after judge filter — do NOT use it here.
        var_explained: float | None = bp_step.reg_var_explained
        res_mean_abs: float | None = bp_step.reg_residual_mean_abs
        res_max_abs: float | None = bp_step.reg_residual_max_abs
        n_reg_pairs: int = bp_step.reg_n_pairs if bp_step.reg_n_pairs is not None else n_pairs

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
            f"pairs={n_pairs} (judge-filtered)  Regression: K={len(acc_pool)} N={n_reg_pairs} "
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

    # ─── BoN-Amplified logging ────────────────────────────────────────────────

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
