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

    @classmethod
    def load(cls, path: Path | str) -> "SearchResults":
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        top_attributes = [FoundAttribute(**a) for a in data.get("top_attributes", [])]
        return cls(
            run_id=data["run_id"],
            config_snapshot=data.get("config_snapshot", {}),
            top_attributes=top_attributes,
            n_steps_completed=data.get("n_steps_completed", 0),
            cost_usd=data.get("cost_usd", 0.0),
            wall_time_seconds=data.get("wall_time_seconds", 0.0),
        )

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
