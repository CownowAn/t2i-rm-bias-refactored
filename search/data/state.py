from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from search.data.types import Prompt, BaselineImage, CounterfactualPair
from search.utils.stats import remove_outliers as remove_outliers_fn

if TYPE_CHECKING:
    from search.data.baseline_pair_types import BaselinePairStep


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
    amp_mean_p1: float = 0.0          # mean P(g=1) across prompts
    amp_mean_p0: float = 0.0          # mean P(g=0) across prompts
    amp_mean_mu1: float | None = None  # mean E[reward | g=1]; None if never any g=1 images
    amp_mean_mu0: float | None = None  # mean E[reward | g=0]; None if never any g=0 images

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class AttributeStats:
    attribute: str
    meta: AttributeMeta
    pairs: dict[str, list[CounterfactualPair]] = field(default_factory=dict)
    # prompt_text -> list of CounterfactualPairs (one per rollout)
    baseline_detected: dict[str, int] = field(default_factory=dict)
    # image_id -> 0/1 VLM detection result, populated by AmplificationScorer

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

    def delta_rm(self, remove_outliers: bool = False) -> float | None:
        scores = self._all_delta_rm()
        if not scores:
            return None
        if remove_outliers:
            scores = remove_outliers_fn(scores)
        return float(np.mean(scores)) if scores else None

    def delta_j(self) -> float | None:
        scores = self._all_delta_j()
        if not scores:
            return None
        return float(np.mean(scores))

    def is_undesirable(self, use_outlier_removal: bool = False) -> bool:
        rm = self.delta_rm(use_outlier_removal)
        j = self.delta_j()
        return rm is not None and j is not None and rm > 0 and j <= 0

    @property
    def amplification_score(self) -> float:
        return self.meta.amplification_score

    def __repr__(self) -> str:
        return (
            f"AttributeStats(\n"
            f"  attribute={self.attribute!r},\n"
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
    bp_history: list["BaselinePairStep"] = field(default_factory=list)
    # baseline-pairs mode: per-step data (detection, pairs, Lasso results)

    @property
    def current_step(self) -> EvoStep | None:
        return self.history[-1] if self.history else None

    def train_prompts(self) -> list[str]:
        return [p.text for p in self.prompts]

    def get_accumulated_stats(self, attribute: str) -> "AttributeStats | None":
        """Merge pairs from all steps where this attribute was evaluated."""
        merged_pairs: dict[str, list[CounterfactualPair]] = {}
        base_stats: AttributeStats | None = None
        for step in self.history:
            stats = step.attributes.get(attribute)
            if stats is None:
                continue
            if base_stats is None:
                base_stats = stats
            for prompt, pairs in stats.pairs.items():
                merged_pairs.setdefault(prompt, []).extend(pairs)
        if base_stats is None:
            return None
        return AttributeStats(
            attribute=attribute,
            meta=base_stats.meta,
            pairs=merged_pairs,
            baseline_detected=base_stats.baseline_detected,
        )

    def get_latest_stats(self, attribute: str) -> "AttributeStats | None":
        """Return the most recent completed step's AttributeStats for this attribute."""
        for step in reversed(self.history[:-1]):  # exclude the new (current) step
            stats = step.attributes.get(attribute)
            if stats is not None:
                return stats
        return None
