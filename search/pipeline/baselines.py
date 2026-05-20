"""Load and score baseline images from a pregenerated manifest."""
from __future__ import annotations
import json
from pathlib import Path

from loguru import logger

from search.data.types import Prompt, BaselineImage
from search.data.state import TopicState
from search.models.base import RewardModel


def load_topic_states(
    prompts_dir: str | Path,
    topic_ids: list[int],
    val_split_size: int = 40,
    random_seed: int = 42,
) -> list[TopicState]:
    """Load cluster JSON files and build TopicState objects (without baselines)."""
    from random import Random
    prompts_dir = Path(prompts_dir)

    states: list[TopicState] = []
    for topic_id in sorted(topic_ids):
        cluster_path = prompts_dir / f"cluster_{topic_id}.json"
        with open(cluster_path) as f:
            data = json.load(f)

        rng = Random(random_seed + topic_id)
        all_prompts: list[str] = data["prompts"]

        if len(all_prompts) < max(1, val_split_size):
            raise ValueError(f"Not enough prompts for topic {topic_id}: {len(all_prompts)}")

        rng.shuffle(all_prompts)
        train_texts = all_prompts[:-val_split_size] if val_split_size > 0 else all_prompts
        val_texts   = all_prompts[-val_split_size:]  if val_split_size > 0 else []

        prompts = [Prompt(text=t, topic_id=topic_id) for t in train_texts]
        states.append(TopicState(
            topic_id=topic_id,
            prompts=prompts,
            cluster_summary=data.get("summary", ""),
        ))
        logger.info(
            f"Topic {topic_id}: {len(train_texts)} train / {len(val_texts)} val prompts\n"
            f"  Summary: {data.get('summary', '')[:80]}"
        )

    return states


def load_baselines_from_manifest(
    topic_state: TopicState,
    manifest_path: str | Path,
    baseline_root: str | Path = "",
) -> None:
    """Populate topic_state.baselines from a pregenerated manifest JSON (in-place)."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    all_prompts = {p.text for p in topic_state.prompts}
    raw: dict = manifest.get("baselines", {})
    _root = Path(baseline_root) if baseline_root else None

    loaded, missing = 0, 0
    for prompt_text in all_prompts:
        entries = raw.get(prompt_text)
        if not entries:
            missing += 1
            continue

        rollouts: list[BaselineImage] = []
        for entry in entries:
            img_path = Path(entry["image_path"])
            if _root is not None and not img_path.is_absolute():
                img_path = _root / img_path
            if not img_path.exists():
                logger.warning(f"Missing baseline image: {img_path}")
                continue
            rollouts.append(BaselineImage(
                image_path=img_path,
                image_id=entry["image_id"],
                prompt=Prompt(text=prompt_text, topic_id=topic_state.topic_id),
                policy_model=entry.get("policy_model", "pregenerated"),
                reward_scores=dict(entry.get("reward_scores", {})),
            ))
        if rollouts:
            topic_state.baselines[prompt_text] = rollouts
            loaded += 1

    if missing:
        logger.warning(f"Topic {topic_state.topic_id}: {missing} prompts not found in manifest")
    logger.info(
        f"Topic {topic_state.topic_id}: loaded baselines for {loaded}/{len(all_prompts)} prompts"
    )


def load_val_topic_state(
    prompts_dir: str | Path,
    topic_id: int,
    val_split_size: int = 40,
    random_seed: int = 42,
) -> TopicState:
    """Load val split for a single topic using the same shuffle as load_topic_states()."""
    from random import Random
    prompts_dir = Path(prompts_dir)

    cluster_path = prompts_dir / f"cluster_{topic_id}.json"
    with open(cluster_path) as f:
        data = json.load(f)

    rng = Random(random_seed + topic_id)
    all_prompts: list[str] = data["prompts"]

    if len(all_prompts) < max(1, val_split_size):
        raise ValueError(f"Not enough prompts for topic {topic_id}: {len(all_prompts)}")

    rng.shuffle(all_prompts)
    val_texts = all_prompts[-val_split_size:] if val_split_size > 0 else []

    prompts = [Prompt(text=t, topic_id=topic_id) for t in val_texts]
    state = TopicState(
        topic_id=topic_id,
        prompts=prompts,
        cluster_summary=data.get("summary", ""),
    )
    logger.info(
        f"Topic {topic_id}: loaded {len(val_texts)} val prompts\n"
        f"  Summary: {data.get('summary', '')[:80]}"
    )
    return state


def all_baselines_have_scores(manifest_path: str | Path, model_name: str) -> bool:
    """Return True iff every image entry in the manifest already has
    reward_scores[model_name]. Lets the engine skip loading the reward model."""
    path = Path(manifest_path)
    if not path.exists():
        return False
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    baselines = data.get("baselines", {})
    if not baselines:
        return False
    for entries in baselines.values():
        for entry in entries:
            scores = entry.get("reward_scores") or {}
            if model_name not in scores:
                return False
    return True


async def score_baselines(
    topic_state: TopicState,
    reward_model: RewardModel,
) -> None:
    """Score any unscored baselines with the reward model (in-place)."""
    model_name = reward_model.model_name
    to_score: list[tuple[str, int, BaselineImage]] = []

    for prompt_text, baselines in topic_state.baselines.items():
        for i, b in enumerate(baselines):
            if model_name not in b.reward_scores:
                to_score.append((prompt_text, i, b))

    if not to_score:
        logger.info(f"Topic {topic_state.topic_id}: all baselines already scored")
        return

    logger.info(f"Topic {topic_state.topic_id}: scoring {len(to_score)} baselines with {model_name}")
    image_paths = [str(item[2].image_path) for item in to_score]
    prompts     = [item[0] for item in to_score]

    results = await reward_model.rate(image_paths, prompts)
    for (prompt_text, idx, _), result in zip(to_score, results):
        if result.score is not None:
            topic_state.baselines[prompt_text][idx].reward_scores[model_name] = result.score
