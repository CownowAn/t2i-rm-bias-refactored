from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class FoundAttribute:
    attribute: str
    delta_rm: float | None
    delta_j: float | None
    amplification_score: float
    step_found: int
    step_last_survived: int
    topic_id: int
    is_undesirable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResults:
    run_id: str
    config_snapshot: dict[str, Any]
    top_attributes: list[FoundAttribute]
    n_steps_completed: int
    cost_usd: float
    wall_time_seconds: float

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "run_id": self.run_id,
            "config_snapshot": self.config_snapshot,
            "top_attributes": [p.to_dict() for p in self.top_attributes],
            "n_steps_completed": self.n_steps_completed,
            "cost_usd": self.cost_usd,
            "wall_time_seconds": self.wall_time_seconds,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
