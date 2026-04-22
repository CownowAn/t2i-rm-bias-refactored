from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RatingResult:
    score: float | None
    reasoning: str | None = None


@dataclass(frozen=True)
class ComparisonResult:
    winner: str | None          # "A" | "B" | "Tie" | None
    score_diff: float | None    # +1.0 (A wins), -1.0 (B wins), 0.0 (Tie)
    reasoning: str | None = None


class RewardModel(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    async def rate(
        self,
        image_paths: list[str],
        prompts: list[str],
    ) -> list[RatingResult]: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


class JudgeModel(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    async def compare(
        self,
        image_A_paths: list[str],
        image_B_paths: list[str],
        prompts: list[str],
    ) -> list[ComparisonResult]: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


class DetectorModel(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    async def detect(
        self,
        image_paths: list[str],
        prompts: list[str],
        attribute: str,
    ) -> list[int]: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...
