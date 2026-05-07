"""Config system for BoN analysis: YAML → dataclasses with dot-path CLI overrides."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from search.config import RewardModelConfig, DetectorConfig, WandbConfig, LoggingConfig
from search.utils.io import timestamp


# ─── BoN-specific config dataclasses ─────────────────────────────────────────

@dataclass
class BonRunConfig:
    name: str | None = None
    output_dir: str = "outputs/bon/"
    random_seed: int = 42


@dataclass
class BonDataConfig:
    name: str = ""
    baseline_manifest: str = ""
    baseline_root: str = ""
    prompts_dir: str = ""
    topic_ids: list[int] = field(default_factory=lambda: [0])
    val_split_size: int = 40  # must match the search run's val_split_size


@dataclass
class BonSamplingConfig:
    n_values: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    n_trials: int = 100  # Monte Carlo trials per (prompt, N)


@dataclass
class BonAttributeConfig:
    # Option A: load from a search run's results.json
    search_results_path: str = ""
    only_undesirable: bool = True  # only applies when search_results_path is set

    # Option B: explicit attribute list (takes precedence over search_results_path)
    attributes: list[str] = field(default_factory=list)


@dataclass
class BonModelsConfig:
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)


# ─── Root config ──────────────────────────────────────────────────────────────

@dataclass
class BonConfig:
    run: BonRunConfig = field(default_factory=BonRunConfig)
    data: BonDataConfig = field(default_factory=BonDataConfig)
    sampling: BonSamplingConfig = field(default_factory=BonSamplingConfig)
    attributes: BonAttributeConfig = field(default_factory=BonAttributeConfig)
    models: BonModelsConfig = field(default_factory=BonModelsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] | None = None) -> "BonConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        if overrides:
            raw = _apply_overrides(raw, overrides)
        config = _from_dict(cls, raw)
        if config.run.name is None:
            config.run.name = f"bon_{timestamp()}"
        return config

    def validate(self) -> None:
        assert self.data.name, "data.name is required"
        assert self.data.baseline_manifest, "data.baseline_manifest is required"
        assert self.data.prompts_dir, "data.prompts_dir is required"
        assert self.attributes.search_results_path or self.attributes.attributes, (
            "either attributes.search_results_path or attributes.attributes must be set"
        )
        assert self.data.val_split_size > 0, "data.val_split_size must be > 0"
        assert self.sampling.n_values, "sampling.n_values must be non-empty"
        assert all(n >= 1 for n in self.sampling.n_values), "all n_values must be >= 1"
        assert self.sampling.n_trials >= 1, "sampling.n_trials must be >= 1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def run_output_dir(self) -> Path:
        return Path(self.run.output_dir) / self.run.name


# ─── Helpers (mirrors search/config.py) ──────────────────────────────────────

def _apply_overrides(raw: dict, overrides: list[str]) -> dict:
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
    if s.startswith("[") and s.endswith("]"):
        items = [_cast_value(x.strip()) for x in s[1:-1].split(",") if x.strip()]
        return items
    return s


def _from_dict(cls, data: dict | None) -> Any:
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
        if isinstance(ftype, str):
            ftype = eval(ftype)  # noqa: S307 – safe, only our own types
        val = data[fname]
        if dataclasses.is_dataclass(ftype) and isinstance(val, dict):
            kwargs[fname] = _from_dict(ftype, val)
        else:
            kwargs[fname] = val
    return cls(**kwargs)
