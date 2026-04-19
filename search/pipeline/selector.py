"""Pareto-based attribute selection: pick surviving attributes after each evo step."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from loguru import logger

from search.data.state import TopicState, AttributeStats
from search.data.results import ParetoPoint


@dataclass
class SelectionResult:
    surviving: dict[str, int]   # attr -> step_idx where it was found
    pareto_points: list[ParetoPoint]


class ParetoSelector:
    """Select top attributes based on ΔRM (high) and ΔJ (low) — the "undesirable" criterion."""

    def __init__(
        self,
        direction: str = "plus",
        target_pop_size: int = 4,
    ):
        self.direction = direction
        self.target_pop_size = target_pop_size

    def _teacher_threshold(
        self, teacher_scores: list[float], strict: bool = False
    ) -> float:
        if not teacher_scores:
            return 0.0
        if self.direction == "plus":
            return 0.0 if strict else float(np.percentile(teacher_scores, 50))
        else:
            return 0.0 if strict else float(np.percentile(teacher_scores, 50))

    def select(
        self,
        topic_state: TopicState,
        step_idx: int,
        strict: bool = False,
    ) -> SelectionResult:
        """Choose surviving attributes from the current evo step."""
        step = topic_state.history[step_idx]
        candidates: list[AttributeStats] = [
            s for s in step.attributes.values()
            if s.delta_rm() is not None
        ]

        if not candidates:
            logger.warning(f"Topic {topic_state.topic_id}: no scoreable candidates at step {step_idx}")
            return SelectionResult(surviving={}, pareto_points=[])

        # Filter by teacher score
        teacher_scores = [s.delta_j() for s in candidates if s.delta_j() is not None]
        threshold = self._teacher_threshold([t for t in teacher_scores if t is not None], strict)

        if self.direction == "plus":
            # Want ΔRM > 0 (high reward) and ΔJ < 0 (bad human preference)
            # Teacher threshold: keep those with ΔJ <= threshold (lower = worse judge)
            passing = [s for s in candidates if s.delta_j() is None or s.delta_j() <= threshold]
        else:
            passing = [s for s in candidates if s.delta_j() is None or s.delta_j() >= threshold]

        # Fallback: if nothing passes, take all
        if not passing:
            passing = candidates

        # Sort by |ΔRM| descending (find strongest undesirable biases)
        passing.sort(key=lambda s: abs(s.delta_rm() or 0.0), reverse=True)

        selected = passing[:self.target_pop_size]

        # Build surviving dict and ParetoPoints
        surviving: dict[str, int] = {s.attribute: step_idx for s in selected}
        pareto_points: list[ParetoPoint] = []
        for s in selected:
            pareto_points.append(ParetoPoint(
                attribute=s.attribute,
                delta_rm=s.delta_rm() or 0.0,
                delta_j=s.delta_j() or 0.0,
                amplification_score=s.amplification_score,
                step_found=step_idx,
                topic_id=topic_state.topic_id,
            ))

        logger.info(
            f"Topic {topic_state.topic_id} step {step_idx}: "
            f"{len(candidates)} candidates → {len(selected)} survivors"
        )
        for pp in pareto_points:
            logger.info(
                f"  ΔRM={pp.delta_rm:+.3f}  ΔJ={pp.delta_j:+.3f}  "
                f"A(g)={pp.amplification_score:.4f}  | {pp.attribute}"
            )

        return SelectionResult(surviving=surviving, pareto_points=pareto_points)
