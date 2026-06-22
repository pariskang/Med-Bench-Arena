"""DatasetAdapter interface + registry + factory.

An adapter ``load()``s a source into canonical ``Sample``s and ``parse()``s a
model's raw text into a metric-ready ``Prediction``. Common formats are handled
by config-driven adapters (hf_mcq, local_json) so *adding a similar dataset =
editing config, zero code*.
"""
from __future__ import annotations

import abc
from typing import Any, Callable

from ..schema import Prediction, Sample

_REGISTRY: dict[str, type["DatasetAdapter"]] = {}


def register_dataset(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        _REGISTRY[name] = cls
        cls.adapter_name = name
        return cls
    return deco


def create_dataset(config: dict[str, Any]) -> "DatasetAdapter":
    """Factory: instantiate by ``config['adapter']``."""
    name = config.get("adapter")
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown dataset adapter {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](config)


def available_adapters() -> list[str]:
    return sorted(_REGISTRY)


class DatasetAdapter(abc.ABC):
    """Abstract dataset adapter."""

    adapter_name: str = "base"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.id: str = config.get("id", self.adapter_name)
        self.limit: int | None = config.get("limit")
        # metrics requested for this dataset (names resolved by the runner)
        self.metrics: list[str] = list(config.get("metrics", []))
        # optional per-dataset judge override (model id) for llm_judge
        self.judge: str | None = config.get("judge")

    @abc.abstractmethod
    def load(self) -> list[Sample]:
        """Materialize canonical samples (respecting ``self.limit``)."""

    @abc.abstractmethod
    def parse(self, sample: Sample, text: str) -> Prediction:
        """Turn raw model ``text`` into a ``Prediction`` (extract the answer)."""

    def _truncate(self, items: list[Any]) -> list[Any]:
        return items[: self.limit] if self.limit else items
