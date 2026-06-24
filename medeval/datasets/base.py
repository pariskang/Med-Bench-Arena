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

# Comparability tiers (leaderboard ``split_type``). "official" runs are directly
# comparable to public leaderboards; everything else is internal-only.
SPLIT_TYPES = frozenset({
    "official",      # full official split + the official metric/grader
    "validation",    # a dev/val split (not the held-out test)
    "demo",          # a tiny demo subset shipped in lieu of the full corpus
    "sample",        # a small public sample of an otherwise-gated set
    "gated",         # full set requires manual access; partial here
    "approximated",  # a built-in/approximate grader, not the official one
})


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


def _parse_metric_specs(raw: Any) -> list[tuple[str, dict[str, Any]]]:
    """Normalize a ``metrics:`` list into ``[(name, config), ...]``. Entries may be
    plain strings (``llm_judge``) or dicts carrying that metric's config
    (``{name: llm_judge, per_criterion: true}``)."""
    out: list[tuple[str, dict[str, Any]]] = []
    for m in raw or []:
        if isinstance(m, str):
            out.append((m, {}))
        elif isinstance(m, dict):
            name = m.get("name") or m.get("metric")
            if not name:
                raise ValueError(f"metric spec {m!r} needs a 'name' key")
            out.append((name, {k: v for k, v in m.items()
                               if k not in ("name", "metric")}))
        else:
            raise ValueError(f"invalid metric spec {m!r} (want str or dict)")
    return out


class DatasetAdapter(abc.ABC):
    """Abstract dataset adapter."""

    adapter_name: str = "base"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.id: str = config.get("id", self.adapter_name)
        self.limit: int | None = config.get("limit")
        # metrics requested for this dataset. Each entry may be a plain name
        # ("llm_judge") or a dict carrying that metric's config
        # ({name: llm_judge, per_criterion: true}). ``metric_specs`` keeps
        # (name, config); ``metrics`` keeps just the names (back-compat).
        self.metric_specs: list[tuple[str, dict[str, Any]]] = _parse_metric_specs(
            config.get("metrics", []))
        self.metrics: list[str] = [n for n, _ in self.metric_specs]
        # optional per-dataset judge override (model id) for llm_judge
        self.judge: str | None = config.get("judge")
        # comparability tier — keeps "officially comparable" runs from being mixed
        # with internal ones on the leaderboard. One of:
        #   official | validation | demo | sample | gated | approximated
        # (default "official"; adapters whose comparability depends on runtime —
        # e.g. MedAgentBench's built-in grader — may override below).
        self.split_type: str = config.get("split_type", "official")
        if self.split_type not in SPLIT_TYPES:
            import warnings
            warnings.warn(f"{self.id}: unknown split_type {self.split_type!r}; "
                          f"expected one of {sorted(SPLIT_TYPES)}")
        # reliability counters populated during load() (read by `medeval preflight`):
        # {"seen": int, "kept": int, "dropped": {reason: count}}
        self.load_stats: dict[str, Any] = {}

    @abc.abstractmethod
    def load(self) -> list[Sample]:
        """Materialize canonical samples (respecting ``self.limit``)."""

    @abc.abstractmethod
    def parse(self, sample: Sample, text: str) -> Prediction:
        """Turn raw model ``text`` into a ``Prediction`` (extract the answer)."""

    def _truncate(self, items: list[Any]) -> list[Any]:
        return items[: self.limit] if self.limit else items
