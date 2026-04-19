from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ParetoPoint:
    attribute: str
    delta_rm: float
    delta_j: float
    amplification_score: float
    step_found: int
    topic_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResults:
    run_id: str
    config_snapshot: dict[str, Any]
    pareto_front: list[ParetoPoint]
    n_steps_completed: int
    cost_usd: float
    wall_time_seconds: float

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "run_id": self.run_id,
            "config_snapshot": self.config_snapshot,
            "pareto_front": [p.to_dict() for p in self.pareto_front],
            "n_steps_completed": self.n_steps_completed,
            "cost_usd": self.cost_usd,
            "wall_time_seconds": self.wall_time_seconds,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
