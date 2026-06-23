"""Metric interface + registry + factory.

A metric ``score()``s one (sample, prediction) pair and ``aggregate()``s a list
of per-sample scores into headline numbers for the leaderboard.
"""
from __future__ import annotations

import abc
from typing import Any, Callable

from ..schema import Prediction, Sample, Score

_REGISTRY: dict[str, type["Metric"]] = {}


def register_metric(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        _REGISTRY[name] = cls
        cls.metric_name = name
        return cls
    return deco


def create_metric(name: str, config: dict[str, Any] | None = None) -> "Metric":
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown metric {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](config or {})


def available_metrics() -> list[str]:
    return sorted(_REGISTRY)


class Metric(abc.ABC):
    """Abstract scorer."""

    metric_name: str = "base"
    # set True by metrics that need a judge provider injected (llm_judge)
    needs_judge: bool = False

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.judge = None  # injected by the runner when needs_judge

    @abc.abstractmethod
    async def score(self, sample: Sample, pred: Prediction) -> Score:
        """Score a single prediction."""

    def aggregate(self, scores: list[Score]) -> dict[str, Any]:
        """Default aggregation: mean of ``value`` plus count."""
        vals = [s.value for s in scores if s.value is not None]
        mean = sum(vals) / len(vals) if vals else 0.0
        return {self.metric_name: mean, "n": len(scores)}
