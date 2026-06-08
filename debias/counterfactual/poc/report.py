"""JSON writers + survey-row aggregation for the PoC."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from debias.counterfactual.io_utils import thumb_path
from debias.counterfactual.schemas import (
    AttrSurveyRow,
    EditResult,
    EditTask,
    PoCConfig,
    PromptAttrSelection,
    SourceImage,
)
from search.utils.io import save_json


def write_selected_pairs(
    run_id: str,
    topic_id: int,
    step_idx: int,
    cfg: PoCConfig,
    humanness_source: str,
    n_undesirable_attrs: int,
    selections: list[PromptAttrSelection],
    sources_by_key: dict[tuple[str, str], list[SourceImage]],
    out_path: Path,
) -> None:
    """`sources_by_key` is keyed by (prompt_text, attr)."""
    rows = []
    for sel in selections:
        srcs = sources_by_key.get((sel.prompt_text, sel.attr), [])
        rows.append({
            "prompt_text": sel.prompt_text,
            "attr": sel.attr,
            "w_value": sel.w_value,
            "rank_in_prompt": sel.rank_in_prompt,
            "source_image_ids": [s.image_id for s in srcs],
            "source_image_paths": [str(s.image_path) for s in srcs],
        })
    payload = {
        "run_id": run_id,
        "topic_id": topic_id,
        "step_idx": step_idx,
        "tau": cfg.tau,
        "top_n_per_prompt": cfg.top_n_per_prompt,
        "humanness_source": humanness_source,
        "n_undesirable_attrs": n_undesirable_attrs,
        "selections": rows,
    }
    save_json(payload, out_path)


def write_edit_results(
    run_id: str,
    topic_id: int,
    tasks: list[EditTask],
    results: list[EditResult],
    out_path: Path,
) -> None:
    rows = []
    for t, r in zip(tasks, results):
        rows.append({
            "prompt_text": t.selection.prompt_text,
            "attr": t.selection.attr,
            "source_image_id": t.source.image_id,
            "source_image_path": str(t.source.image_path),
            "edited_image_path": str(t.edited_output_path),
            "instruction": t.instruction,
            "success": bool(r.success),
            "edited_attr_detected": r.edited_attr_detected,
            "original_attr_detected": r.original_attr_detected,
            "side_effect_drift": (
                {k: list(v) for k, v in r.side_effect_drift.items()}
                if r.side_effect_drift else None
            ),
            "error": r.error,
            "orig_reward":   r.orig_reward,
            "edited_reward": r.edited_reward,
            "reward_drop":   r.reward_drop,
            "reward_model_name": r.reward_model_name,
        })
    save_json({"run_id": run_id, "topic_id": topic_id, "results": rows}, out_path)


def build_survey_rows(
    results: list[EditResult],
    *,
    n_samples: int = 4,
) -> list[AttrSurveyRow]:
    """Per-attr aggregation: success rate + a few sample pairs for visual inspection."""
    by_attr: dict[str, list[EditResult]] = defaultdict(list)
    for r in results:
        by_attr[r.task.selection.attr].append(r)

    rows: list[AttrSurveyRow] = []
    for attr, rs in by_attr.items():
        n_attempted = len(rs)
        successes = [r for r in rs if r.success]
        failures = [r for r in rs if not r.success]
        n_success = len(successes)
        # Reward stats (only over edits that were actually scored)
        drops = [r.reward_drop for r in rs if r.reward_drop is not None]
        rm_names = {r.reward_model_name for r in rs if r.reward_model_name}
        if drops:
            mean_drop = sum(drops) / len(drops)
            pct_pos = sum(1 for d in drops if d > 0) / len(drops)
        else:
            mean_drop = None
            pct_pos = None
        rows.append(AttrSurveyRow(
            attr=attr,
            n_attempted=n_attempted,
            n_success=n_success,
            success_rate=(n_success / n_attempted) if n_attempted else 0.0,
            sample_success=[
                (Path(r.task.source.image_path), Path(r.task.edited_output_path))
                for r in successes[:n_samples]
            ],
            sample_fail=[
                (Path(r.task.source.image_path), Path(r.task.edited_output_path))
                for r in failures[:n_samples]
            ],
            reward_n_scored=len(drops),
            reward_drop_mean=mean_drop,
            reward_drop_pct_positive=pct_pos,
            reward_model_name=next(iter(rm_names), None) if rm_names else None,
        ))
    rows.sort(key=lambda r: r.success_rate, reverse=True)
    return rows


def write_survey_report(
    run_id: str,
    topic_id: int,
    cfg: PoCConfig,
    rows: list[AttrSurveyRow],
    out_path: Path,
    editable_threshold: float = 0.5,
) -> None:
    per_attr = []
    for row in rows:
        per_attr.append({
            "attr": row.attr,
            "n_attempted": row.n_attempted,
            "n_success": row.n_success,
            "success_rate": row.success_rate,
            "thumbnail_path": str(row.thumbnail_path) if row.thumbnail_path else None,
            "sample_success_paths": [[str(a), str(b)] for a, b in row.sample_success],
            "sample_fail_paths": [[str(a), str(b)] for a, b in row.sample_fail],
            "reward_model_name": row.reward_model_name,
            "reward_n_scored": row.reward_n_scored,
            "reward_drop_mean": row.reward_drop_mean,
            "reward_drop_pct_positive": row.reward_drop_pct_positive,
        })
    n_attempted_total = sum(r.n_attempted for r in rows)
    n_success_total = sum(r.n_success for r in rows)
    mean_success_rate = (n_success_total / n_attempted_total) if n_attempted_total else 0.0
    editable_subset = [r.attr for r in rows if r.success_rate >= editable_threshold]
    # Aggregate reward stats across all rows that were scored
    all_drops_n = sum(r.reward_n_scored for r in rows)
    weighted_mean = None
    if all_drops_n:
        # Weighted mean across attrs (each attr weighted by its reward_n_scored)
        s = sum((r.reward_drop_mean or 0.0) * r.reward_n_scored for r in rows
                if r.reward_drop_mean is not None)
        weighted_mean = s / all_drops_n
    rm_name = next(
        (r.reward_model_name for r in rows if r.reward_model_name), None
    )
    payload = {
        "run_id": run_id,
        "topic_id": topic_id,
        "config": {
            "tau": cfg.tau,
            "top_n_per_prompt": cfg.top_n_per_prompt,
            "n_prompts_per_attr": cfg.n_prompts_per_attr,
            "n_images_per_prompt": cfg.n_images_per_prompt,
            "humanness_recheck": cfg.humanness_recheck,
            "side_effect_check": cfg.side_effect_check,
            "seed": cfg.seed,
        },
        "per_attr": per_attr,
        "global": {
            "n_attrs": len(rows),
            "n_attempted_total": n_attempted_total,
            "n_success_total": n_success_total,
            "mean_success_rate": mean_success_rate,
            "editable_subset": editable_subset,
            "editable_threshold": editable_threshold,
            "reward_model_name": rm_name,
            "reward_n_scored_total": all_drops_n,
            "reward_drop_weighted_mean": weighted_mean,
        },
    }
    save_json(payload, out_path)


def attach_thumbnail_paths(
    rows: list[AttrSurveyRow],
    run_id: str,
    report_root: Path,
) -> None:
    """Populate `row.thumbnail_path` in place using io_utils.thumb_path()."""
    for row in rows:
        row.thumbnail_path = thumb_path(run_id, row.attr, report_root)
