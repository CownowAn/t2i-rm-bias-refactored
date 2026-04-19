"""Amplification scorer: A(g) = E_x[p1(x)*p0(x)*(E[r|g=1]-E[r|g=0])]"""
from __future__ import annotations

import numpy as np
from loguru import logger

from search.data.types import BaselineImage
from search.models.base import JudgeModel


class AmplificationScorer:
    """Computes the RLHF amplification score A(g) for a set of attributes."""

    def __init__(self, detector: JudgeModel):
        self.detector = detector

    async def compute(
        self,
        attribute: str,
        baselines_by_prompt: dict[str, list[BaselineImage]],
        reward_model_name: str,
    ) -> float:
        """
        A(g) = E_x[ p1(x) * p0(x) * (mu1(x) - mu0(x)) ]

        baselines_by_prompt: prompt_text -> list of BaselineImage (already reward-scored).
        Uses VLM detection to split images into g=1 / g=0 groups per prompt.
        """
        per_prompt_scores: list[float] = []

        for prompt_text, baselines in baselines_by_prompt.items():
            scored = [b for b in baselines if reward_model_name in b.reward_scores]
            if len(scored) < 2:
                continue

            image_paths = [str(b.image_path) for b in scored]
            prompts = [prompt_text] * len(scored)

            detections = await self.detector.detect(image_paths, prompts, attribute)
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

        if not per_prompt_scores:
            return 0.0

        score = float(np.mean(per_prompt_scores))
        logger.debug(f"A(g) for '{attribute}': {score:.4f} (over {len(per_prompt_scores)} prompts)")
        return score

    async def compute_batch(
        self,
        attributes: list[str],
        baselines_by_prompt: dict[str, list[BaselineImage]],
        reward_model_name: str,
    ) -> dict[str, float]:
        """Compute A(g) for multiple attributes against the same baseline pool."""
        results: dict[str, float] = {}
        for attr in attributes:
            results[attr] = await self.compute(attr, baselines_by_prompt, reward_model_name)
        return results