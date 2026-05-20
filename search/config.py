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
    name: str = "imagereward"  # "imagereward" | "pickscore" | "hpsv3"
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
    image_detail: str = "auto"  # "auto" | "high" | "low"
    use_batch_api: bool = False  # use OpenAI Batch API (50% discount, async; openai/ models only)


@dataclass
class DetectorConfig:
    model: str = "openai/gpt-4o-mini"
    max_tokens: int = 50000
    max_parallel: int = 32
    image_detail: str = "auto"  # "auto" | "high" | "low"
    use_batch_api: bool = False  # use OpenAI Batch API (50% discount, async; openai/ models only)
    # vLLM serving: set to the base URL of a running vLLM server (e.g. "http://localhost:8000/v1")
    # When set, requests are sent to the local server instead of the cloud API.
    # Model name should be the exact model ID as registered in the vLLM server.
    vllm_base_url: str | None = None
    # General sampling parameters — apply to both cloud and vLLM backends.
    # None = use the model's server-side default.
    temperature: float | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    # Arbitrary extra_body dict forwarded to the API (e.g. vLLM-specific params).
    # Example: {"top_k": 20, "chat_template_kwargs": {"enable_thinking": false}}
    extra_body: dict | None = None
    # Prompt content controls
    use_prompt: bool = True    # include the image generation prompt in the detection query
    use_reasoning: bool = True  # request a reasoning field in the JSON response


@dataclass
class PlannerConfig:
    model: str = "openai/gpt-5"
    reasoning: str | None = "high"
    max_tokens: int = 50000
    max_parallel: int = 64


@dataclass
class AttrFilterConfig:
    """Model config for AttributeUndesirabilityFilter (humanness check)."""
    model: str = "openai/gpt-5"
    max_tokens: int = 16        # only YES/NO needed
    max_parallel: int = 64


@dataclass
class ProposerConfig:
    """Model config for BonResidualProposer (VLM attr mining with images)."""
    model: str = "openai/gpt-5"
    reasoning: str | None = "high"
    max_tokens: int = 50000
    max_parallel: int = 64
    image_detail: str = "auto"  # image detail level shown to the proposer VLM


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
    attr_filter: AttrFilterConfig = field(default_factory=AttrFilterConfig)
    proposer: ProposerConfig = field(default_factory=ProposerConfig)
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
    reg_min_pairs: int = 5
    cosine_sim_threshold_initial: float = 0.9
    cosine_sim_threshold_evolution: float = 0.9
    replan_if_no_undesirable: bool = False
    strict_undesirable_selection: bool = False
    n_prompts_per_plan_call: int = 1  # >1: show multiple prompts per InitialPlanner VLM call


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
class PipelineConfig:
    mode: str = "edit"  # "edit" | "baseline_pairs" | "bon_amplified"


@dataclass
class BonAmplifiedConfig:
    N: int = 16                         # BoN parameter for U^{N-1} quantile
    tau: float = 0.0                    # A_hat pruning threshold (keep attrs with A_hat > tau)
    n_top_residual: int = 2             # P+/P- images extracted per prompt
    n_prompts_vlm: int = 8              # prompts shown to VLM per mining call
    n_proposals: int = 10               # attr proposals per VLM mining call
    n_proposer_calls: int = 1           # number of proposer VLM calls per EXPAND step
    select_all_passing: bool = True     # all A_hat > tau attrs enter acc_pool (no top-K cap)
    # Prevalence filter: reject attrs whose global detection rate p1 falls
    # outside [p1_min, p1_max].  Attrs that are too rare or too ubiquitous
    # have limited discriminative power for OLS (low corr with U^{N-1}).
    use_p1_filter: bool = False
    p1_min: float = 0.1
    p1_max: float = 0.9
    detection_cache_path: str | None = None
    proposer_use_cluster_summary: bool = False  # include cluster summary in proposer prompt
    use_per_prompt_ols: bool = False  # fit a separate OLS W_x per prompt (for residuals)
    # Monotonic pool + partial A_hat admission
    use_monotonic_pool: bool = False  # admit via partial_A_hat instead of A_hat; skip top-K
    tau_partial: float = 0.0          # partial_A_hat threshold for admission
    n_admit_per_step: int = 4         # max admissions per EXPAND step (Top-K')
    # Proposer prompt selection strategy
    prompt_select_strategy: str = "random"     # "random" | "middle_band"
    prompt_select_exclude_pct: float = 0.2     # top/bottom % trim for middle_band


@dataclass
class BaselinePairsConfig:
    n_pairs_per_prompt: int = 20
    use_judge: bool = True
    judge_filter_position: str = "before_regression"  # "before_regression" | "before_residual_select" | "lazy"
    judge_lazy_overshoot: int = 2  # lazy mode: judge n_high_residual_pairs × overshoot top-residual pairs
    # Prompt diversity controls for lazy judge and high-residual pair selection.
    # null = no limit.
    judge_lazy_prompt_cap: int | None = None  # max candidate pairs per prompt for judging
    n_high_residual_per_prompt: int | None = None  # max pairs per prompt in final high-residual selection
    n_high_residual_pairs: int = 5
    n_proposed_per_step: int = 4
    reg_fit_intercept: bool = True
    pair_constructor: str = "stratified"  # "hamming" | "stratified" | "all"
    n_pairs_per_stratum: int = 4          # pairs per (attr_k, prompt) stratum; used when pair_constructor="stratified"
    # Attr selection filters (applied independently before top-K selection)
    use_mu_filter: bool = True   # keep only attrs where μ1 > μ0 (global mean across all images)
    use_amp_filter: bool = False  # keep only attrs where A(g) > 0 (prompt-averaged amplification)
    # If True, all filter-passing attrs enter acc_pool (no top-K cap).
    # Recommended with pair_constructor="all" where N grows with K, keeping N/K large.
    select_all_passing: bool = False
    # Detection cache persistence: path to JSON file for cross-run reuse of VLM detection results.
    # null = disabled (fresh detection every run). Specify a path to save after each EVALUATE and
    # reload on the next run — eliminates detection cost for already-seen (image, attr) pairs.
    detection_cache_path: str | None = None
    # Amplification score formula: "kl_rlhf" = p1·p0·(μ1−μ0) i.e. Cov(g,r) proxy;
    # "bon" = N·p1·p0·(E[U^{N-1}|g=1]−E[U^{N-1}|g=0]) per BoN theorem (amplification.md §4).
    amp_mode: str = "kl_rlhf"
    amp_bon_n: int = 16  # Best-of-N sample size N (only used when amp_mode="bon")
    # Regression model for linear probing (residual computation)
    regression_model: str = "elasticnet"  # "lasso" | "ridge" | "elasticnet"
    # Shared hyperparameters
    elasticnet_l1_ratio: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9, 1.0])
    elasticnet_n_alphas: int = 100   # alpha candidates for Lasso/ElasticNet; logspace count for Ridge
    elasticnet_cv: int = 5


@dataclass
class CallerCacheConfig:
    enabled: bool = True
    base_path: str = ".cache/caller"
    max_entries_in_disk: int | None = 131072

    def build(self) -> "CacheConfig":
        from caller.cache import CacheConfig
        if not self.enabled:
            return CacheConfig(base_path=None)
        return CacheConfig(base_path=self.base_path, max_entries_in_disk=self.max_entries_in_disk)


@dataclass
class SearchConfig:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    baseline_pairs: BaselinePairsConfig = field(default_factory=BaselinePairsConfig)
    bon_amplified: BonAmplifiedConfig = field(default_factory=BonAmplifiedConfig)
    caller_cache: CallerCacheConfig = field(default_factory=CallerCacheConfig)

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
        assert self.pipeline.mode in ("edit", "baseline_pairs", "bon_amplified"), (
            f"pipeline.mode must be 'edit', 'baseline_pairs', or 'bon_amplified', "
            f"got {self.pipeline.mode!r}"
        )
        assert self.baseline_pairs.pair_constructor in ("hamming", "stratified", "all"), (
            f"baseline_pairs.pair_constructor must be 'hamming', 'stratified', or 'all', "
            f"got {self.baseline_pairs.pair_constructor!r}"
        )
        assert self.models.reward_model.name in ("imagereward", "pickscore", "hpsv3"), (
            f"models.reward_model.name must be 'imagereward', 'pickscore', or 'hpsv3', "
            f"got {self.models.reward_model.name!r}"
        )
        assert self.baseline_pairs.regression_model in ("lasso", "ridge", "elasticnet"), (
            f"baseline_pairs.regression_model must be 'lasso', 'ridge', or 'elasticnet', "
            f"got {self.baseline_pairs.regression_model!r}"
        )
        assert self.baseline_pairs.judge_filter_position in ("before_regression", "before_residual_select", "lazy"), (
            f"baseline_pairs.judge_filter_position must be 'before_regression', 'before_residual_select', or 'lazy', "
            f"got {self.baseline_pairs.judge_filter_position!r}"
        )
        assert self.models.judge.image_detail in ("auto", "high", "low"), (
            f"models.judge.image_detail must be 'auto', 'high', or 'low', "
            f"got {self.models.judge.image_detail!r}"
        )
        assert self.models.detector.image_detail in ("auto", "high", "low"), (
            f"models.detector.image_detail must be 'auto', 'high', or 'low', "
            f"got {self.models.detector.image_detail!r}"
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
    DetectorConfig, PlannerConfig, AttrFilterConfig, ProposerConfig, ClusterConfig, ModelsConfig, EvolutionConfig,
    EvaluationConfig, WandbConfig, LoggingConfig, PipelineConfig, BaselinePairsConfig,
    BonAmplifiedConfig, SearchConfig,
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
