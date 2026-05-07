from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class BonResults:
    run_id: str
    search_results_path: str
    topic_id: int
    attributes: list[str]
    n_values: list[int]
    n_trials: int
    n_val_prompts: int
    prevalence: dict[str, list[float]]  # attr → prevalence at each n in n_values
    cost_usd: float
    wall_time_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
