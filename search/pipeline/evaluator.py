"""CounterfactualEvaluator: edit images and compute ΔRM for all attributes in one step."""
from __future__ import annotations
import asyncio
import time
from pathlib import Path
from random import Random

from loguru import logger

from search.data.types import CounterfactualPair, BaselineImage
from search.data.state import TopicState, EvoStep
from search.models.base import RewardModel
from search.models.editor.flux_kontext import FluxKontextApplier
from search.models.editor.instruction_gen import EditInstructionGenerator
from search.utils.async_utils import GpuApplierPool, bounded_gather
from search.utils.io import edited_image_filename


class CounterfactualEvaluator:
    """Edit baseline images for each attribute and compute ΔRM (reward(edited) - reward(baseline))."""

    def __init__(
        self,
        instruction_gen: EditInstructionGenerator,
        gpu_pool: GpuApplierPool,
        reward_model: RewardModel,
        output_dir: Path,
        n_instruction_parallel: int = 128,
    ):
        self.instruction_gen = instruction_gen
        self.gpu_pool = gpu_pool
        self.reward_model = reward_model
        self.output_dir = output_dir
        self.n_instruction_parallel = n_instruction_parallel

    async def _edit_one(
        self,
        attribute: str,
        baseline: BaselineImage,
    ) -> tuple[BaselineImage, str, str] | None:
        """Generate edit instruction + apply via FluxKontext. Returns (baseline, edited_path, instruction)."""
        try:
            instruction = await self.instruction_gen.generate(
                image_path=str(baseline.image_path),
                attribute=attribute,
                prompt_text=baseline.prompt.text,
            )
            out_filename = edited_image_filename(attribute, baseline.prompt.text, baseline.image_path)
            out_path = str(self.output_dir / out_filename)
            edited_path = await self.gpu_pool.apply(
                str(baseline.image_path), instruction, out_path
            )
            return baseline, edited_path, instruction
        except Exception as e:
            logger.error(f"Edit failed for attribute '{attribute}', image '{baseline.image_path}': {e}")
            return None

    async def evaluate_step(
        self,
        topic_state: TopicState,
        step: EvoStep,
        batch_prompts: list[str],
        n_rollouts: int,
        reward_model_name: str,
        rng: Random,
    ) -> None:
        """For every attribute in step, edit baselines and compute ΔRM. Results stored in-place."""
        t0 = time.time()
        attributes = list(step.attributes.keys())
        logger.info(
            f"Topic {topic_state.topic_id} step {step.step_idx}: "
            f"evaluating {len(attributes)} attributes × {len(batch_prompts)} prompts"
        )

        # Build all edit tasks
        edit_tasks = []
        task_meta = []  # (attribute, prompt_text)
        for attr in attributes:
            for prompt_text in batch_prompts:
                baselines = topic_state.baselines.get(prompt_text, [])
                scored = [b for b in baselines if reward_model_name in b.reward_scores]
                if not scored:
                    continue
                chosen = rng.sample(scored, min(n_rollouts, len(scored)))
                for b in chosen:
                    edit_tasks.append(self._edit_one(attr, b))
                    task_meta.append((attr, prompt_text))

        if not edit_tasks:
            logger.warning("No edit tasks generated — check that baselines are scored")
            return

        logger.info(f"  Running {len(edit_tasks)} edit tasks...")
        edit_results = await bounded_gather(
            edit_tasks, max_parallel=self.n_instruction_parallel, desc="Editing images"
        )

        # Batch reward scoring — edited images only; baselines already scored
        valid = [(res, meta) for res, meta in zip(edit_results, task_meta) if res is not None]
        if not valid:
            return

        edited_paths: list[str] = []
        edited_prompts: list[str] = []
        for (baseline, edited_path, _instruction), (attr, prompt_text) in valid:
            edited_paths.append(edited_path)
            edited_prompts.append(prompt_text)

        logger.info(f"  Scoring {len(edited_paths)} edited images with {self.reward_model.model_name}...")
        ratings = await self.reward_model.rate(edited_paths, edited_prompts)

        # Assign ΔRM to CounterfactualPairs in step
        for i, ((baseline, edited_path, instruction), (attr, prompt_text)) in enumerate(valid):
            edited_score = ratings[i].score
            baseline_score = baseline.reward_scores.get(reward_model_name)

            if edited_score is None or baseline_score is None:
                delta_rm = None
            else:
                delta_rm = edited_score - baseline_score

            pair = CounterfactualPair(
                baseline=baseline,
                edited_image_path=Path(edited_path),
                edit_instruction=instruction,
                delta_rm=delta_rm,
            )
            step.attributes[attr].pairs.setdefault(prompt_text, []).append(pair)

        logger.info(
            f"  Evaluation done in {time.time()-t0:.1f}s — "
            f"{sum(len(s.pairs) for s in step.attributes.values())} prompts with pairs"
        )
