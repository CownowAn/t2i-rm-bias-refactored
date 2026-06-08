"""PoC orchestration: selection → edit → verify → report."""
from __future__ import annotations

from pathlib import Path
from random import Random
from typing import TYPE_CHECKING

from loguru import logger

from debias.counterfactual.edit.editor_runner import EditorRunner
from debias.counterfactual.edit.instruction_builder import build_instruction
from debias.counterfactual.io_utils import edited_image_path, poc_report_dir
from debias.counterfactual.poc.report import (
    attach_thumbnail_paths,
    build_survey_rows,
    write_edit_results,
    write_selected_pairs,
    write_survey_report,
)
from debias.counterfactual.selection.attr_selector import (
    group_by_attr,
    select_per_prompt,
)
from debias.counterfactual.selection.detection_lookup import load_detection_cache
from debias.counterfactual.selection.humanness import resolve_undesirable_set
from debias.counterfactual.selection.per_prompt_w_loader import (
    limit_attrs,
    load_per_prompt_w,
)
from debias.counterfactual.selection.source_image_sampler import sample_source_images
from debias.counterfactual.schemas import (
    EditTask,
    PoCConfig,
    PromptAttrSelection,
    SourceImage,
    SurveyResult,
)
from debias.counterfactual.verify.pair_validator import validate_edits
from debias.counterfactual.verify.thumbnail_grid import build_grid

if TYPE_CHECKING:
    from search.config import DetectorConfig


def _subsample_by_attr(
    selections: list[PromptAttrSelection],
    n_prompts_per_attr: int,
    seed: int,
) -> list[PromptAttrSelection]:
    """Per-attr down-sampling: keep at most `n_prompts_per_attr` prompts per attr."""
    by_attr = group_by_attr(selections)
    out: list[PromptAttrSelection] = []
    for attr, sels in by_attr.items():
        if len(sels) <= n_prompts_per_attr:
            out.extend(sels)
            continue
        rng = Random(seed ^ (hash(attr) & 0xFFFFFFFF))
        out.extend(rng.sample(sels, n_prompts_per_attr))
    logger.info(
        f"  attr-level subsampling: {len(selections)} → {len(out)} "
        f"({len(by_attr)} attrs)"
    )
    return out


async def run_survey(
    *,
    ppw_path: Path,
    ba_expand_path: Path | None,
    detection_cache_path: Path,
    detector_key: str,
    baseline_manifest_path: Path,
    baseline_root: str,
    topic_id: int,
    cfg: PoCConfig,
    editor: EditorRunner,
    detector,                                  # DetectorModel
    humanness_model: str,
    cf_root: Path | None,
    report_root: Path | None,
    reward_model=None,                         # optional RewardModel
    reward_model_name: str | None = None,
    source_consistency_n: int = 0,             # 0 = skip; ≥1 = re-query detector that many times per (image, attr)
    instruction_mode: str = "correct",         # "correct" | "remove"
    limit_n_attrs: int = 0,                    # 0 = no limit; >0 = keep only the first N attrs
) -> SurveyResult:
    """End-to-end PoC. Writes selected_pairs.json + edit_results.json + survey_report.json."""

    # ── 1. per_prompt_W ────────────────────────────────────────────────────
    ppw = load_per_prompt_w(ppw_path)
    logger.info(
        f"per_prompt_W: step={ppw.step_idx} topic={ppw.topic_id} "
        f"K={len(ppw.attrs)} P={len(ppw.per_prompt_W)}"
    )
    assert ppw.topic_id == topic_id, (
        f"topic_id mismatch: ppw.topic_id={ppw.topic_id} vs --topic_id={topic_id}"
    )

    if limit_n_attrs > 0 and limit_n_attrs < len(ppw.attrs):
        K_before = len(ppw.attrs)
        ppw = limit_attrs(ppw, limit_n_attrs)
        logger.info(f"  limit_attrs: {K_before} → {len(ppw.attrs)} (kept first {limit_n_attrs})")

    # ── 2. undesirable set ────────────────────────────────────────────────
    undesirable = await resolve_undesirable_set(
        ppw.attrs, ba_expand_path,
        recheck=cfg.humanness_recheck,
        humanness_model=humanness_model,
    )
    humanness_source = "recheck" if cfg.humanness_recheck else "search"

    # ── 3. per-prompt selection ──────────────────────────────────────────
    selections = select_per_prompt(ppw, undesirable, cfg.tau, cfg.top_n_per_prompt)
    logger.info(
        f"selections (W > tau={cfg.tau}, top-{cfg.top_n_per_prompt} per prompt): "
        f"{len(selections)} (prompt, attr) pairs"
    )

    # ── 4. per-attr subsampling ──────────────────────────────────────────
    selections = _subsample_by_attr(selections, cfg.n_prompts_per_attr, cfg.seed)
    if not selections:
        logger.warning("no selections after subsampling — exiting early.")
        return SurveyResult(selections=[], edit_results=[], rows=[])

    # ── 5. detection cache + baselines ───────────────────────────────────
    detection = load_detection_cache(detection_cache_path, detector_key)
    from search.data.state import TopicState
    from search.data.types import Prompt
    ts = TopicState(
        topic_id=topic_id,
        prompts=[
            Prompt(text=p, topic_id=topic_id)
            for p in {s.prompt_text for s in selections}
        ],
        cluster_summary="",   # not used by the edit path; required by dataclass
    )
    from search.pipeline.baselines import load_baselines_from_manifest
    load_baselines_from_manifest(ts, baseline_manifest_path, baseline_root)

    # ── 6. sample candidate source images per (prompt, attr) ────────────
    sources_by_key: dict[tuple[str, str], list[SourceImage]] = {}
    for sel in selections:
        baselines_for_prompt = ts.baselines.get(sel.prompt_text, [])
        if not baselines_for_prompt:
            logger.warning(
                f"  no baseline images for prompt={sel.prompt_text[:40]!r}"
            )
            continue
        sources = sample_source_images(
            sel, baselines_for_prompt, detection,
            k_img=cfg.n_images_per_prompt, rng_seed=cfg.seed,
        )
        sources_by_key[(sel.prompt_text, sel.attr)] = sources
    n_candidate = sum(len(v) for v in sources_by_key.values())
    logger.info(f"candidate sources: {n_candidate} across "
                f"{len(sources_by_key)} (prompt, attr) pairs")

    # ── 6b. (optional) source-consistency check ──────────────────────────
    if source_consistency_n and source_consistency_n > 0:
        from debias.counterfactual.selection.source_consistency_check import (
            filter_consistent_sources,
        )
        sources_by_key = await filter_consistent_sources(
            sources_by_key, detector, n_repeats=source_consistency_n,
        )
        sources_by_key = {k: v for k, v in sources_by_key.items() if v}

    # ── 7. build EditTasks from surviving sources ────────────────────────
    tasks: list[EditTask] = []
    surviving_sel_keys = set(sources_by_key.keys())
    for sel in selections:
        srcs = sources_by_key.get((sel.prompt_text, sel.attr), [])
        for src in srcs:
            instr = build_instruction(sel.attr, mode=instruction_mode)
            out = edited_image_path(
                sel.topic_id, sel.attr, src.image_id,
                cf_root=cf_root, instruction_mode=instruction_mode,
            )
            tasks.append(EditTask(
                selection=sel, source=src, instruction=instr,
                edited_output_path=out,
            ))
    logger.info(f"built {len(tasks)} EditTasks from surviving sources")

    report_dir = poc_report_dir(cfg.run_id, report_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_selected_pairs(
        run_id=cfg.run_id,
        topic_id=topic_id,
        step_idx=ppw.step_idx,
        cfg=cfg,
        humanness_source=humanness_source,
        n_undesirable_attrs=len(undesirable),
        # Only emit selections that still have at least one surviving source
        selections=[s for s in selections
                    if (s.prompt_text, s.attr) in surviving_sel_keys],
        sources_by_key=sources_by_key,
        out_path=report_dir / "selected_pairs.json",
    )
    if not tasks:
        logger.warning("no EditTasks built — exiting before editor.")
        return SurveyResult(selections=selections, edit_results=[], rows=[])

    # ── 7. edit ────────────────────────────────────────────────────────────
    await editor.edit_many(tasks)

    # ── 8. verify ─────────────────────────────────────────────────────────
    results = await validate_edits(
        tasks, detector, check_original=True, side_effect=cfg.side_effect_check,
    )

    # ── 8b. reward Δ (optional) ────────────────────────────────────────────
    if reward_model is not None and reward_model_name is not None:
        from debias.counterfactual.verify.reward_drop import measure_reward_drop
        results = await measure_reward_drop(results, reward_model, reward_model_name)

    write_edit_results(
        cfg.run_id, topic_id, tasks, results, report_dir / "edit_results.json",
    )

    # ── 9. per-attr aggregation + report ─────────────────────────────────
    rows = build_survey_rows(results)
    if cfg.make_thumbnails:
        attach_thumbnail_paths(rows, cfg.run_id, report_root or report_dir.parent)
        for row in rows:
            pairs = row.sample_success + row.sample_fail
            if pairs and row.thumbnail_path is not None:
                build_grid(
                    pairs, row.thumbnail_path,
                    title=f"{row.attr[:80]}  success={row.n_success}/{row.n_attempted}",
                )

    write_survey_report(
        run_id=cfg.run_id, topic_id=topic_id, cfg=cfg, rows=rows,
        out_path=report_dir / "survey_report.json",
    )
    logger.info(f"PoC report dir → {report_dir}")
    return SurveyResult(selections=selections, edit_results=results, rows=rows)
