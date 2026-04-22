"""Amplification scorer: A(g) = E_x[p1(x)*p0(x)*(E[r|g=1]-E[r|g=0])]"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from search.data.types import BaselineImage
from search.models.base import DetectorModel

if TYPE_CHECKING:
    from search.data.state import AttributeStats


class AmplificationScorer:
    """Computes the RLHF amplification score A(g) for a set of attributes."""

    def __init__(self, detector: DetectorModel):
        self.detector = detector

    async def compute(
        self,
        attribute: str,
        baselines_by_prompt: dict[str, list[BaselineImage]],
        reward_model_name: str,
        attr_stats: "AttributeStats | None" = None,
    ) -> float:
        """
        A(g) = E_x[ p1(x) * p0(x) * (mu1(x) - mu0(x)) ]

        baselines_by_prompt: prompt_text -> list of BaselineImage (already reward-scored).
        Uses VLM detection to split images into g=1 / g=0 groups per prompt.
        """
        per_prompt_scores: list[float] = []
        per_prompt_p1:  list[float] = []
        per_prompt_p0:  list[float] = []
        per_prompt_mu1: list[float] = []
        per_prompt_mu0: list[float] = []

        for prompt_text, baselines in baselines_by_prompt.items():
            scored = [b for b in baselines if reward_model_name in b.reward_scores]
            if len(scored) < 2:
                continue

            image_paths = [str(b.image_path) for b in scored]
            prompts = [prompt_text] * len(scored)

            detections = await self.detector.detect(image_paths, prompts, attribute)
            if attr_stats is not None:
                for b, d in zip(scored, detections):
                    attr_stats.baseline_detected[b.image_id] = int(d)
            rewards = [b.reward_scores[reward_model_name] for b in scored]

            g1_rewards = [r for r, d in zip(rewards, detections) if d == 1]
            g0_rewards = [r for r, d in zip(rewards, detections) if d == 0]

            if not g1_rewards or not g0_rewards:
                continue

            n = len(scored)
            p1 = len(g1_rewards) / n
            p0 = len(g0_rewards) / n
            mu1 = float(np.mean(g1_rewards))
            mu0 = float(np.mean(g0_rewards))

            per_prompt_scores.append(p1 * p0 * (mu1 - mu0))
            per_prompt_p1.append(p1)
            per_prompt_p0.append(p0)
            per_prompt_mu1.append(mu1)
            per_prompt_mu0.append(mu0)

            logger.debug(
                f"  A(g) prompt '{prompt_text[:40]}': "
                f"p1={p1:.3f} p0={p0:.3f} mu1={mu1:.3f} mu0={mu0:.3f} "
                f"→ {p1 * p0 * (mu1 - mu0):.4f}"
            )

        if not per_prompt_scores:
            return 0.0

        score = float(np.mean(per_prompt_scores))
        mean_p1  = float(np.mean(per_prompt_p1))
        mean_p0  = float(np.mean(per_prompt_p0))
        mean_mu1 = float(np.mean(per_prompt_mu1))
        mean_mu0 = float(np.mean(per_prompt_mu0))

        if attr_stats is not None:
            attr_stats.meta.amp_mean_p1  = mean_p1
            attr_stats.meta.amp_mean_p0  = mean_p0
            attr_stats.meta.amp_mean_mu1 = mean_mu1
            attr_stats.meta.amp_mean_mu0 = mean_mu0

        logger.debug(
            f"A(g) for '{attribute}': {score:.4f} (over {len(per_prompt_scores)} prompts) | "
            f"mean p1={mean_p1:.3f} p0={mean_p0:.3f} mu1={mean_mu1:.3f} mu0={mean_mu0:.3f}"
        )
        return score

    async def compute_batch(
        self,
        attributes: list[str],
        baselines_by_prompt: dict[str, list[BaselineImage]],
        reward_model_name: str,
        attrs_stats_map: "dict[str, AttributeStats] | None" = None,
    ) -> dict[str, float]:
        """Compute A(g) for multiple attributes against the same baseline pool."""
        results: dict[str, float] = {}
        for attr in attributes:
            results[attr] = await self.compute(
                attr, baselines_by_prompt, reward_model_name,
                attr_stats=attrs_stats_map.get(attr) if attrs_stats_map else None,
            )
        return results