"""Score original and edited images with the reward model and report ΔR.

Goal: confirm that removing the targeted bias attribute actually moves the
reward in the expected direction. We score both images through the SAME RM
instance to avoid version mismatch with manifest-stored scores.

A positive `reward_drop` (= R(orig) − R(edited)) means the edit successfully
lowered the reward, which is exactly the signal we want for BT-loss training.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from debias.counterfactual.schemas import EditResult

if TYPE_CHECKING:
    from search.models.base import RewardModel


async def measure_reward_drop(
    results: list[EditResult],
    reward_model: "RewardModel",
    reward_model_name: str,
) -> list[EditResult]:
    """For each successful edit, score both (orig, edited) and record ΔR.

    Returns a NEW list of EditResult (the dataclass is frozen). Failed tasks
    and tasks whose edited PNG is missing are passed through unchanged.
    """
    if not results:
        return results

    # Eligible: editor said success AND the edited PNG actually exists.
    eligible_idx: list[int] = []
    image_pairs: list[tuple[str, str]] = []           # (orig_path, edited_path)
    prompts: list[str] = []                            # same prompt for both
    for i, r in enumerate(results):
        if not r.success:
            continue
        ep = Path(r.task.edited_output_path)
        if not ep.exists():
            continue
        eligible_idx.append(i)
        image_pairs.append((str(r.task.source.image_path), str(ep)))
        prompts.append(r.task.source.prompt_text)

    if not eligible_idx:
        logger.warning(f"  no successful edits to score with {reward_model_name}")
        return results

    n = len(eligible_idx)
    logger.info(
        f"  scoring {n} (orig, edited) pairs with {reward_model_name} "
        f"({2 * n} rate calls)"
    )

    # Two batched calls — clean and simple. Could be combined but keeps the
    # mapping obvious.
    orig_paths   = [p[0] for p in image_pairs]
    edited_paths = [p[1] for p in image_pairs]
    try:
        orig_ratings   = await reward_model.rate(orig_paths,   prompts)
        edited_ratings = await reward_model.rate(edited_paths, prompts)
    except Exception as e:
        logger.exception(f"  reward scoring failed: {e}")
        return results

    out = list(results)
    drops: list[float] = []
    n_pos = 0
    for k, i in enumerate(eligible_idx):
        orig_score   = _score_value(orig_ratings[k])
        edited_score = _score_value(edited_ratings[k])
        if orig_score is None or edited_score is None:
            continue
        drop = orig_score - edited_score
        drops.append(drop)
        if drop > 0:
            n_pos += 1
        out[i] = replace(
            results[i],
            orig_reward=orig_score,
            edited_reward=edited_score,
            reward_drop=drop,
            reward_model_name=reward_model_name,
        )

    if drops:
        mean = sum(drops) / len(drops)
        logger.info(
            f"  ΔR summary [{reward_model_name}]: "
            f"{n_pos}/{len(drops)} dropped (mean Δ = {mean:+.4f})"
        )
    return out


def _score_value(rating) -> float | None:
    if rating is None:
        return None
    score = getattr(rating, "score", None)
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None