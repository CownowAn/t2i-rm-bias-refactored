from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RatingResult:
    score: float | None
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
