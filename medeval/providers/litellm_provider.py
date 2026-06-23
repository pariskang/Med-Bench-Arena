"""LiteLLM backend (``type: litellm``) — the universal adapter.

Wraps OpenAI / Anthropic / Gemini / DeepSeek / Qwen + any OpenAI-compatible
endpoint behind one call. Built-in retry, cost accounting and (optional) cache.
Recommended for the LLM-judge (DeepSeek-R1).
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

from ..schema import Generation, Message, now
from .base import ModelProvider, register_provider


@register_provider("litellm")
class LiteLLMProvider(ModelProvider):
    """config:
      model:        litellm model string, e.g. ``deepseek/deepseek-reasoner``
      api_key_env:  env var holding the key (read into the right slot)
      api_base:     optional custom endpoint (OpenAI-compatible / vllm serve)
      api_key:      literal key (discouraged; prefer api_key_env)
      max_retries:  default 4
      extra:        dict of extra kwargs forwarded to litellm.acompletion
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.model: str = config["model"]
        self.api_base: str | None = config.get("api_base")
        self.max_retries: int = int(config.get("max_retries", 4))
        self.extra: dict[str, Any] = dict(config.get("extra", {}))
        self._key = self._resolve_key(config)
        self._litellm = None

    @staticmethod
    def _resolve_key(config: dict[str, Any]) -> str | None:
        if config.get("api_key"):
            return config["api_key"]
        env = config.get("api_key_env")
        return os.environ.get(env) if env else None

    def _lib(self):
        if self._litellm is None:
            import litellm  # lazy: only import when actually used
            litellm.drop_params = True       # tolerate provider-specific params
            litellm.suppress_debug_info = True
            self._litellm = litellm
        return self._litellm

    async def agenerate(self, messages: list[Message], **gen: Any) -> Generation:
        litellm = self._lib()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "num_retries": self.max_retries,
            **self.extra,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self._key:
            kwargs["api_key"] = self._key
        for k in ("temperature", "max_tokens", "top_p", "stop", "seed"):
            if k in gen and gen[k] is not None:
                kwargs[k] = gen[k]

        t0 = now()
        try:
            resp = await litellm.acompletion(**kwargs)
        except Exception as e:  # surface as empty generation; runner logs it
            return Generation.empty(self.model, error=f"{type(e).__name__}: {e}")
        dt = now() - t0

        choice = resp.choices[0]
        text = (choice.message.content or "")
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        cost = 0.0
        try:
            cost = float(litellm.completion_cost(completion_response=resp) or 0.0)
        except Exception:
            cost = 0.0
        return Generation(
            text=text,
            model=self.model,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=(pt + ct) or getattr(usage, "total_tokens", 0) or 0,
            cost_usd=cost,
            latency_s=dt,
            finish_reason=getattr(choice, "finish_reason", "") or "",
        )
