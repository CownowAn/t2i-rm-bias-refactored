"""Config system: YAML → dataclasses with dot-path CLI overrides."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from search.utils.io import timestamp


# ─── Leaf config dataclasses ──────────────────────────────────────────────────

@dataclass
class RunConfig:
    name: str | None = None
    output_dir: str = "outputs/"
    random_seed: int = 42


@dataclass
class DataConfig:
    baseline_manifest: str = ""
    baseline_root: str = ""  # prefix for relative image_path entries in manifest; empty = CWD
    prompts_dir: str = ""
    topic_ids: list[int] = field(default_factory=lambda: [0])
    val_split_size: int = 40


@dataclass
class RewardModelConfig:
    name: str = "imagereward"
    device: str = "cuda:0"
    hf_cache_dir: str = "/nfs/data/sohyun/models"


@dataclass
class EditorConfig:
    instruction_model: str = "openai/gpt-4o-mini"
    flux_model: str = "black-forest-labs/FLUX.1-Kontext-dev"
    flux_devices: list[str] = field(default_factory=lambda: ["cuda:0"])
    guidance_scale: float = 2.5


@dataclass
class JudgeConfig:
    model: str = "openai/gpt-4o-mini"
    max_tokens: int = 50000
    max_parallel: int = 32


@dataclass
class DetectorConfig:
    model: str = "openai/gpt-4o-mini"
    max_tokens: int = 50000
    max_parallel: int = 32


@dataclass
class PlannerConfig:
    model: str = "openai/gpt-5"
    reasoning: str | None = "high"
    max_tokens: int = 50000
    max_parallel: int = 64


@dataclass
class ClusterConfig:
    model: str = "openai/gpt-5.2"
    max_tokens: int = 50000
    reasoning: str | None = "high"
    max_parallel: int = 64


@dataclass
class ModelsConfig:
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)
    editor: EditorConfig = field(default_factory=EditorConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)


@dataclass
class EvolutionConfig:
    n_steps: int = 5
    initial_pop_size: int = 8
    target_pop_sizes: list[int] = field(default_factory=lambda: [4, 4, 4, 4, 4])
    n_mutations: int = 1
    n_context_imgs: int = 16
    n_attrs_per_prompt: int = 4
    n_per_user_prompt: int = 1
    n_initial_plan_prompts: int | None = None  # None = use all train prompts
    initial_context_sampling: str = "random"  # "random" | "stratified" (top-half + bottom-half by reward)
    use_cluster_summary: bool = True
    direction: str = "plus"
    image_order: str = "descending"   # order images are shown to the planner LLM
    context: str = "ancestry"
    mutation_context_source: str = "origin"  # "origin" | "accumulated" | "latest"
    lasso_min_pairs: int = 5
    cosine_sim_threshold_initial: float = 0.9
    cosine_sim_threshold_evolution: float = 0.9
    replan_if_no_undesirable: bool = False


@dataclass
class EvaluationConfig:
    train_batch_size: list[int] = field(default_factory=lambda: [32, 32, 32, 32, 32])
    n_rollouts_per_prompt: int = 1
    judge_first_n_prompts: int = 32
    judge_first_n_rollouts: int = 1
    amp_n_prompts: int = 32          # prompts to use for A(g) computation
    amp_n_images_per_prompt: int = 64  # baseline images per prompt for A(g)
    use_outlier_removal: bool = False


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "t2i-rm-bias"
    entity: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class LoggingConfig:
    wandb: WandbConfig = field(default_factory=WandbConfig)
    log_images_every_n_steps: int = 1
    console_level: str = "INFO"


# ─── Root config ──────────────────────────────────────────────────────────────

@dataclass
class SearchConfig:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ── Loaders ───────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] | None = None) -> "SearchConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        if overrides:
            raw = _apply_overrides(raw, overrides)

        config = _from_dict(cls, raw)
        if config.run.name is None:
            config.run.name = f"{timestamp()}"
        return config

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self) -> None:
        n = self.evolution.n_steps
        assert len(self.evolution.target_pop_sizes) == n, (
            f"target_pop_sizes length ({len(self.evolution.target_pop_sizes)}) "
            f"must equal n_steps ({n})"
        )
        assert len(self.evaluation.train_batch_size) == n, (
            f"train_batch_size length ({len(self.evaluation.train_batch_size)}) "
            f"must equal n_steps ({n})"
        )
        assert self.evolution.direction in ("plus", "minus"), (
            f"direction must be 'plus' or 'minus', got {self.evolution.direction!r}"
        )
        assert self.evolution.context in ("all", "ancestry", "vanilla", "residual"), (
            f"context must be 'all', 'ancestry', 'vanilla', or 'residual'"
        )
        assert self.evolution.initial_context_sampling in ("random", "stratified"), (
            f"initial_context_sampling must be 'random' or 'stratified', "
            f"got {self.evolution.initial_context_sampling!r}"
        )

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def run_output_dir(self) -> Path:
        return Path(self.run.output_dir) / self.run.name


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """Apply dot-path overrides, e.g. ['evolution.n_steps=3', 'run.name=foo']."""
    raw = copy.deepcopy(raw)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override!r}")
        key, value = override.split("=", 1)
        parts = key.strip().split(".")
        d = raw
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = _cast_value(value.strip())
    return raw


def _cast_value(s: str) -> Any:
    """Try to cast a string CLI value to int, float, bool, null, or list."""
    if s.lower() == "null":
        return None
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    # Simple list: "[1,2,3]"
    if s.startswith("[") and s.endswith("]"):
        items = [_cast_value(x.strip()) for x in s[1:-1].split(",") if x.strip()]
        return items
    return s


_LEAF_TYPES = {
    RunConfig, DataConfig, RewardModelConfig, EditorConfig, JudgeConfig,
    DetectorConfig, PlannerConfig, ClusterConfig, ModelsConfig, EvolutionConfig,
    EvaluationConfig, WandbConfig, LoggingConfig, SearchConfig,
}


def _from_dict(cls, data: dict | None) -> Any:
    """Recursively construct a dataclass from a dict."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        return data

    import dataclasses
    if not dataclasses.is_dataclass(cls):
        return data

    hints = {f.name: f for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for fname, fld in hints.items():
        if fname not in data:
            continue
        ftype = fld.type
        # Resolve string annotations
        if isinstance(ftype, str):
            ftype = eval(ftype)  # noqa: S307 – safe, only our own types
        val = data[fname]
        if dataclasses.is_dataclass(ftype) and isinstance(val, dict):
            kwargs[fname] = _from_dict(ftype, val)
        else:
            kwargs[fname] = val
    return cls(**kwargs)
