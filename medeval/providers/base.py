"""ModelProvider interface + registry + factory.

A provider turns ``list[Message]`` into a ``Generation``. The runner only ever
talks to this interface, so HF / Poe / LiteLLM (or anything you register) are
interchangeable. ``agenerate_many`` defaults to bounded-concurrency fan-out;
the HF provider overrides it with a true vLLM batch.
"""
from __future__ import annotations

import abc
import asyncio
from typing import Any, Callable

from ..schema import Generation, Message

# name -> provider class
_REGISTRY: dict[str, type["ModelProvider"]] = {}


def register_provider(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        _REGISTRY[name] = cls
        cls.provider_type = name
        return cls
    return deco


def create_provider(config: dict[str, Any]) -> "ModelProvider":
    """Factory: instantiate by ``config['type']`` (hf | poe | litellm | mock | ...)."""
    ptype = config.get("type")
    if ptype not in _REGISTRY:
        raise ValueError(
            f"unknown provider type {ptype!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[ptype](config)


def available_providers() -> list[str]:
    return sorted(_REGISTRY)


class ModelProvider(abc.ABC):
    """Abstract backend. Subclasses implement :meth:`agenerate`."""

    provider_type: str = "base"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.id: str = config.get("id", config.get("type", "model"))
        # Backend-side concurrency ceiling for the default fan-out.
        self.concurrency: int = int(config.get("concurrency", 16))
        self.judge_only: bool = bool(config.get("judge_only", False))
        # Per-model generation overrides: merged on top of the run's eval.gen so a
        # single model can carry its authors' recommended sampling (e.g. a reasoning
        # model's temperature) without changing the run default for every other model.
        self.gen_overrides: dict[str, Any] = dict(config.get("gen", {}))

    def _merge_gen(self, gen: dict[str, Any]) -> dict[str, Any]:
        return {**gen, **self.gen_overrides} if self.gen_overrides else gen

    @abc.abstractmethod
    async def agenerate(self, messages: list[Message], **gen: Any) -> Generation:
        """Generate one completion for a single chat."""

    async def agenerate_many(
        self, batch: list[list[Message]], **gen: Any
    ) -> list[Generation]:
        """Generate for many chats. Default = bounded-concurrency fan-out.

        API backends keep this. The HF backend overrides it to feed the whole
        batch to vLLM at once (the single biggest perf lever vs. APIs).
        """
        gen = self._merge_gen(gen)
        sem = asyncio.Semaphore(self.concurrency)

        async def one(msgs: list[Message]) -> Generation:
            async with sem:
                return await self.agenerate(msgs, **gen)

        return await asyncio.gather(*(one(m) for m in batch))

    async def aclose(self) -> None:
        """Release any held resources (HTTP clients, model handles)."""

    # convenience for sync callers / notebooks
    def generate(self, messages: list[Message], **gen: Any) -> Generation:
        return asyncio.run(self.agenerate(messages, **gen))
