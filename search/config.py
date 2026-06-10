"""Config system: YAML → dataclasses with dot-path CLI overrides.

Single supported pipeline: BoN-amplified search. The legacy ``edit`` and
``baseline_pairs`` modes have been removed; their config classes are gone too.
"""
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
    cluster_summary_field: str = "summary"  # cluster JSON key used as cluster_summary ("summary" | "category")


@dataclass
class RewardModelConfig:
    name: str = "imagereward"  # "imagereward" | "pickscore" | "hpsv3"
    device: str = "cuda:0"
    hf_cache_dir: str = "/nfs/data/sohyun/models"


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
    # When True, the model is also asked whether the attribute even applies to the image
    # (e.g. "rocks are too smooth" on a portrait with no rocks → applicable=false → -1).
    use_applicability: bool = False
    # When use_applicability=True, the detector still writes -1 ("not applicable") to
    # the cache (so disk preserves the raw signal), but downstream OLS / amplification
    # code treats -1 as absent (0). No effect if use_applicability=False.
    not_applicable_as_absent: bool = False


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
    n_context_imgs: int = 16
    n_attrs_per_prompt: int = 4
    n_per_user_prompt: int = 1
    n_initial_plan_prompts: int | None = None  # None = use all train prompts
    initial_context_sampling: str = "random"  # "random" | "stratified" (top-half + bottom-half by reward)
    use_cluster_summary: bool = True
    direction: str = "plus"
    image_order: str = "descending"   # order images are shown to the planner LLM
    # How reward scores are normalized when shown to InitialPlanner.
    # Stats are computed over the FULL scored pool per prompt (typically 128 imgs),
    # then applied to the sampled images. Aligns the planner's view with the
    # within-prompt quantile signal used by downstream OLS.
    #   "none"     → raw score (current behavior)
    #   "zscore"   → (r - mean_x) / std_x
    #   "minmax"   → (r - min_x) / (max_x - min_x), bounded [0, 1]
    #   "quantile" → within-prompt rank percentile (matches U used by BoN OLS)
    initial_score_normalization: str = "none"
    n_prompts_per_plan_call: int = 1  # >1: show multiple prompts per InitialPlanner VLM call


@dataclass
class EvaluationConfig:
    amp_n_prompts: int = 32          # prompts to use for A(g) computation
    amp_n_images_per_prompt: int = 64  # baseline images per prompt for A(g)


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


# ─── BoN-amplified search config ──────────────────────────────────────────────

@dataclass
class BonAmplifiedConfig:
    N: int = 16                         # BoN parameter for U^{N-1} quantile
    tau: float = 0.0                    # A_hat pruning threshold (keep attrs with A_hat > tau)
    n_top_residual: int = 2             # P+/P- images extracted per prompt
    # P+/P- selection within the sign-split pools (P+ residual>0, P- residual<0):
    #   "extreme"        — most-positive / most-negative residual (current)
    #   "reward_matched" — greedy match so the two groups have similar reward
    #   "random"         — random n_top from each pool
    pplus_pminus_selection: str = "extreme"
    pplus_pminus_reward_tol: float | None = None   # max |reward diff| for reward_matched (None = no cap)
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
    # step 0 starts from this attribute pool (a JSON file OR a run's planner/ dir);
    # planner + humanness + clustering are skipped (pool treated as already filtered).
    initial_pool_path: str | None = None
    proposer_use_cluster_summary: bool = False  # include cluster summary in proposer prompt
    # Restrict the proposer's avoid list to attrs rejected in the CURRENT run only;
    # rejected attrs loaded from the detection cache (previous runs) are excluded from
    # avoid. Cache save/load itself is unaffected (rejections keep accumulating).
    proposer_avoid_current_run_only: bool = False
    use_per_prompt_ols: bool = False  # fit a separate OLS W_x per prompt (for residuals)
    # Monotonic pool + partial A_hat admission
    use_monotonic_pool: bool = False  # admit via partial_A_hat instead of A_hat; skip top-K
    tau_partial: float = 0.0          # partial_A_hat threshold for admission
    n_admit_per_step: int = 4         # max admissions per EXPAND step (Top-K')
    # Proposer prompt selection strategy
    prompt_select_strategy: str = "random"     # "random" | "middle_band"
    prompt_select_exclude_pct: float = 0.2     # top/bottom % trim for middle_band


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
        assert self.evolution.direction in ("plus", "minus"), (
            f"direction must be 'plus' or 'minus', got {self.evolution.direction!r}"
        )
        assert self.evolution.initial_context_sampling in ("random", "stratified"), (
            f"initial_context_sampling must be 'random' or 'stratified', "
            f"got {self.evolution.initial_context_sampling!r}"
        )
        assert self.evolution.initial_score_normalization in ("none", "zscore", "minmax", "quantile"), (
            f"initial_score_normalization must be one of "
            f"'none', 'zscore', 'minmax', 'quantile', "
            f"got {self.evolution.initial_score_normalization!r}"
        )
        assert self.models.reward_model.name in ("imagereward", "pickscore", "hpsv3"), (
            f"models.reward_model.name must be 'imagereward', 'pickscore', or 'hpsv3', "
            f"got {self.models.reward_model.name!r}"
        )
        assert self.data.cluster_summary_field in ("summary", "category"), (
            f"data.cluster_summary_field must be 'summary' or 'category', "
            f"got {self.data.cluster_summary_field!r}"
        )
        assert self.bon_amplified.pplus_pminus_selection in ("extreme", "reward_matched", "random"), (
            f"bon_amplified.pplus_pminus_selection must be 'extreme', 'reward_matched', or 'random', "
            f"got {self.bon_amplified.pplus_pminus_selection!r}"
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


def _from_dict(cls, data: dict | None) -> Any:
    """Recursively construct a dataclass from a dict.

    Unknown keys in `data` are silently dropped — this lets us load
    config_effective.yaml files from older runs that still mention removed
    fields (e.g. `pipeline`, `baseline_pairs`, `evolution.n_mutations`).
    """
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
