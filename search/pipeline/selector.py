"""Top-K attribute selection by A(g): pick surviving attributes after each evo step."""
from __future__ import annotations
from dataclasses import dataclass

from loguru import logger

from search.data.state import AttributeStats
from search.data.results import FoundAttribute


@dataclass
class SelectionResult:
    surviving: dict[str, int]   # attr -> step_idx where it was found
    selected: list[FoundAttribute]


class TopKSelector:
    """Select top-K attributes by A(g).

    Undesirable (ΔRM>0 & ΔJ<0) are prioritized; remaining slots filled by A(g) descending
    unless strict_undesirable_selection=True, in which case only truly undesirable attributes
    are selected (may return fewer than target_pop_size).
    """

    def __init__(
        self,
        direction: str = "plus",
        target_pop_size: int = 4,
        use_outlier_removal: bool = False,
        strict_undesirable_selection: bool = False,
    ):
        self.direction = direction
        self.target_pop_size = target_pop_size
        self.use_outlier_removal = use_outlier_removal
        self.strict_undesirable_selection = strict_undesirable_selection

    def select(
        self,
        candidates: list[AttributeStats],
        surviving: dict[str, int],
        step_idx: int,
        topic_id: int,
    ) -> SelectionResult:
        """Choose top-K from amp-scored candidates."""
        scoreable = [s for s in candidates if s.delta_rm(self.use_outlier_removal) is not None]

        if not scoreable:
            logger.warning(f"Topic {topic_id}: no scoreable candidates at step {step_idx}")
            return SelectionResult(surviving={}, selected=[])

        if self.direction == "plus":
            undesirable = [
                s for s in scoreable
                if s.is_undesirable(self.use_outlier_removal)
            ]
        else:
            undesirable = [
                s for s in scoreable
                if not s.is_undesirable(self.use_outlier_removal)
            ]

        rest = [s for s in scoreable if s not in undesirable]

        def sort_key(s: AttributeStats) -> tuple[float, float]:
            drm = s.delta_rm(self.use_outlier_removal)
            if drm is None:
                drm = float("-inf") if self.direction == "plus" else float("inf")
            secondary = drm if self.direction == "plus" else -drm
            return (s.amplification_score, secondary)

        undesirable.sort(key=sort_key, reverse=True)
        rest.sort(key=sort_key, reverse=True)

        selected = undesirable[: self.target_pop_size]
        if not self.strict_undesirable_selection and len(selected) < self.target_pop_size:
            selected += rest[: self.target_pop_size - len(selected)]

        # Build surviving dict — preserve original step_idx for carry-over attributes
        new_surviving: dict[str, int] = {
            s.attribute: surviving.get(s.attribute, step_idx) for s in selected
        }

        undesirable_set = set(id(s) for s in undesirable)
        result_points: list[FoundAttribute] = [
            FoundAttribute(
                attribute=s.attribute,
                delta_rm=s.delta_rm(self.use_outlier_removal),
                delta_j=s.delta_j(),
                amplification_score=s.amplification_score,
                step_found=surviving.get(s.attribute, step_idx),
                step_last_survived=step_idx,
                topic_id=topic_id,
                is_undesirable=id(s) in undesirable_set,
            )
            for s in selected
        ]

        logger.info(
            f"Topic {topic_id} step {step_idx}: "
            f"{len(scoreable)} candidates → {len(result_points)} survivors"
        )
        for pp in result_points:
            drm_str = f"{pp.delta_rm:+.3f}" if pp.delta_rm is not None else "None"
            dj_str  = f"{pp.delta_j:+.3f}"  if pp.delta_j  is not None else "None"
            logger.info(
                f"  ΔRM={drm_str}  ΔJ={dj_str}  "
                f"A(g)={pp.amplification_score:.4f}  | {pp.attribute}"
            )

        return SelectionResult(surviving=new_surviving, selected=result_points)