"""Filters attributes by human-preference undesirability and μ1 > μ0 criterion."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from caller import AutoCaller, ChatHistory, ChatMessage

if TYPE_CHECKING:
    from search.data.types import BaselineImage


_HUMANNESS_PROMPT = """\
Imagine you are a typical person casually browsing AI-generated images online.

Would the following visual attribute make an AI-generated image look worse \
to most viewers?

Attribute: "{attribute}"

Respond with exactly one word: YES or NO."""


class AttributeUndesirabilityFilter:
    """Filters attribute candidates by humanness and μ1 > μ0 reward criterion."""

    def __init__(self, model_name: str, max_tokens: int = 4096, max_parallel: int = 8):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_parallel = max_parallel
        self.caller = AutoCaller(dotenv_path=".env")

    async def filter_by_humanness(self, attr_names: list[str]) -> list[str]:
        """Keep only attributes that LLM judges as undesirable for human viewers."""
        if not attr_names:
            return []

        # One LLM call per attribute in parallel — avoids key/index mismatch entirely
        histories: list[ChatHistory] = [
            ChatHistory(messages=[ChatMessage(
                role="user",
                content=[{"type": "input_text",
                          "text": _HUMANNESS_PROMPT.format(attribute=a)}],
            )])
            for a in attr_names
        ]

        responses = await self.caller.call(
            messages=histories,
            model=self.model_name,
            max_tokens=16,       # YES or NO is all we need
            max_parallel=self.max_parallel,
        )

        passed = []
        for attr, resp in zip(attr_names, responses):
            if resp is None:
                logger.info(f"Humanness filter: no response for '{attr}', keeping")
                passed.append(attr)
                continue
            raw = (resp.first_response or "").strip().upper()
            if raw.startswith("YES"):
                passed.append(attr)
                logger.info(f"Humanness filter: '{attr}' → UNDESIRABLE (YES)")
            else:
                logger.info(f"Humanness filter: '{attr}' → desirable ({raw!r})")

        logger.info(
            f"Humanness filter: {len(attr_names)} → {len(passed)} attrs "
            f"({len(attr_names) - len(passed)} removed as desirable)"
        )
        return passed

    def filter_by_mu(
        self,
        attr_names: list[str],
        detection: dict[str, dict[str, int]],
        amp_baselines: dict[str, list["BaselineImage"]],
        reward_model_name: str,
    ) -> tuple[list[str], dict[str, tuple[float | None, float | None, int, int]]]:
        """Keep only attrs where μ1 > μ0.

        Returns:
            (passed, mu_stats) where mu_stats = {attr: (μ1, μ0, g1_count, g0_count)}.
            μ1/μ0 are None when the corresponding group has no images.
        """
        if not attr_names:
            return [], {}

        image_rewards: dict[str, float] = {}
        for images in amp_baselines.values():
            for img in images:
                if reward_model_name in img.reward_scores:
                    image_rewards[img.image_id] = img.reward_scores[reward_model_name]

        passed: list[str] = []
        mu_stats: dict[str, tuple[float | None, float | None, int, int]] = {}
        for attr in attr_names:
            g1_rewards, g0_rewards = [], []
            for image_id, reward in image_rewards.items():
                if image_id not in detection or attr not in detection[image_id]:
                    continue
                (g1_rewards if detection[image_id][attr] == 1 else g0_rewards).append(reward)

            n1, n0 = len(g1_rewards), len(g0_rewards)
            if not g1_rewards and not g0_rewards:
                logger.info(f"μ filter: '{attr}' skipped — no images detected at all")
                mu_stats[attr] = (None, None, 0, 0)
                continue
            if not g1_rewards:
                logger.info(
                    f"μ filter: '{attr}' skipped — attr never present "
                    f"(g1=0, g0={n0}, p1=0.00)"
                )
                mu_stats[attr] = (None, float(np.mean(g0_rewards)), 0, n0)
                continue
            if not g0_rewards:
                logger.info(
                    f"μ filter: '{attr}' skipped — attr always present "
                    f"(g1={n1}, g0=0, p0=0.00)"
                )
                mu_stats[attr] = (float(np.mean(g1_rewards)), None, n1, 0)
                continue

            n = n1 + n0
            p1, p0 = n1 / n, n0 / n
            mu1 = float(np.mean(g1_rewards))
            mu0 = float(np.mean(g0_rewards))
            mu_stats[attr] = (mu1, mu0, n1, n0)
            if mu1 > mu0:
                passed.append(attr)
                logger.info(
                    f"μ filter: '{attr}' PASS  "
                    f"μ1={mu1:.3f} > μ0={mu0:.3f}  "
                    f"(g1={n1} p1={p1:.2f}, g0={n0} p0={p0:.2f})"
                )
            else:
                logger.info(
                    f"μ filter: '{attr}' FAIL  "
                    f"μ1={mu1:.3f} ≤ μ0={mu0:.3f}  "
                    f"(g1={n1} p1={p1:.2f}, g0={n0} p0={p0:.2f})"
                )

        logger.info(f"μ1>μ0 filter: {len(attr_names)} → {len(passed)} attrs")
        return passed, mu_stats

    async def shutdown(self) -> None:
        await self.caller.shutdown()
