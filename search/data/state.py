from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from search.data.types import Prompt, BaselineImage, CounterfactualPair
from search.utils.stats import remove_outliers


@dataclass
class AttributeMeta:
    time_step: int
    parent: str | None
    parent_time_step: int | None
    operation: str              # "initial" | "mutate"
    planner_model: str
    reasoning_effort: str | None
    planner_prompt: str | None = None
    planner_reasoning: str | None = None
    amplification_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class AttributeStats:
    attribute: str
    meta: AttributeMeta
    pairs: dict[str, list[CounterfactualPair]] = field(default_factory=dict)
    # prompt_text -> list of CounterfactualPairs (one per rollout)

    def _all_delta_rm(self) -> list[float]:
        scores = []
        for rollouts in self.pairs.values():
            for p in rollouts:
                if p.delta_rm is not None:
                    scores.append(p.delta_rm)
        return scores

    def _all_delta_j(self) -> list[float]:
        scores = []
        for rollouts in self.pairs.values():
            for p in rollouts:
                if p.delta_j is not None:
                    scores.append(p.delta_j)
        return scores

    def delta_rm(self) -> float | None:
        scores = self._all_delta_rm()
        if not scores:
            return None
        # Only apply IQR outlier removal for continuous RM scores, not for {-1,0,1} judge scores
        all_discrete = all(math.isclose(s, 1, abs_tol=1e-6) or math.isclose(s, 0, abs_tol=1e-6) or math.isclose(s, -1, abs_tol=1e-6) for s in scores)
        if not all_discrete:
            scores = remove_outliers(scores)
        return float(np.mean(scores)) if scores else None

    def delta_j(self) -> float | None:
        scores = self._all_delta_j()
        if not scores:
            return None
        return float(np.mean(scores))

    def is_undesirable(self) -> bool:
        rm, j = self.delta_rm(), self.delta_j()
        return rm is not None and j is not None and rm > 0 and j < 0

    @property
    def amplification_score(self) -> float:
        return self.meta.amplification_score

    def __repr__(self) -> str:
        return (
            f"AttributeStats(\n"
            f"  attribute={self.attribute[:50]!r},\n"
            f"  n_prompts={len(self.pairs)},\n"
            f"  delta_rm={self.delta_rm()},\n"
            f"  delta_j={self.delta_j()},\n"
            f"  amp={self.amplification_score:.4f},\n"
            f")"
        )


@dataclass
class EvoStep:
    step_idx: int
    attributes: dict[str, AttributeStats] = field(default_factory=dict)


@dataclass
class TopicState:
    """State for a single topic (cluster) across all evo steps."""
    topic_id: int
    prompts: list[Prompt]
    cluster_summary: str
    baselines: dict[str, list[BaselineImage]] = field(default_factory=dict)
    # prompt_text -> baselines
    history: list[EvoStep] = field(default_factory=list)
    surviving: dict[str, int] = field(default_factory=dict)
    # attr_text -> step_idx where it was first found

    @property
    def current_step(self) -> EvoStep | None:
        return self.history[-1] if self.history else None

    def train_prompts(self) -> list[str]:
        return [p.text for p in self.prompts]
