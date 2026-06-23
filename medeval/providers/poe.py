"""Poe backend (``type: poe``).

Poe exposes a standard OpenAI-compatible endpoint at ``https://api.poe.com/v1``;
the *bot name* is the model (e.g. ``Claude-Opus-4.7``, ``GPT-5.4``, community or
self-hosted bots). Includes a client-side requests-per-minute throttle (Poe caps
around 500 rpm). Billing is credit-based, so ``cost_usd`` is left at 0 (approx).
"""
from __future__ import annotations

import asyncio
import collections
import os
import time
from typing import Any

from ..schema import Generation, Message, now
from .base import ModelProvider, register_provider

POE_BASE_URL = "https://api.poe.com/v1"


class _RpmLimiter:
    """Sliding-window requests-per-minute limiter (shared across coroutines)."""

    def __init__(self, rpm: int):
        self.rpm = max(1, rpm)
        self._times: collections.deque[float] = collections.deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                nowt = time.monotonic()
                while self._times and nowt - self._times[0] > 60.0:
                    self._times.popleft()
                if len(self._times) < self.rpm:
                    self._times.append(nowt)
                    return
                sleep_for = 60.0 - (nowt - self._times[0]) + 0.01
                await asyncio.sleep(max(0.0, sleep_for))


@register_provider("poe")
class PoeProvider(ModelProvider):
    """config:
      bot_name:    Poe bot used as the model id (e.g. ``Claude-Opus-4.7``)
      api_key_env: env var with the Poe key (from poe.com/api_key); default POE_API_KEY
      rpm:         client-side throttle (default 480, under Poe's ~500 cap)
      api_base:    override (default https://api.poe.com/v1)
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.bot_name: str = config.get("bot_name") or config.get("model")
        if not self.bot_name:
            raise ValueError("poe provider requires 'bot_name'")
        self.api_base: str = config.get("api_base", POE_BASE_URL)
        key_env = config.get("api_key_env", "POE_API_KEY")
        self._key = config.get("api_key") or os.environ.get(key_env)
        self.max_retries: int = int(config.get("max_retries", 4))
        self._limiter = _RpmLimiter(int(config.get("rpm", 480)))
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "poe provider needs the 'openai' package (pip install openai)"
                ) from e
            if not self._key:
                raise RuntimeError(
                    "Poe API key missing; set POE_API_KEY (get it at poe.com/api_key)"
                )
            self._client = AsyncOpenAI(api_key=self._key, base_url=self.api_base)
        return self._client

    async def agenerate(self, messages: list[Message], **gen: Any) -> Generation:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.bot_name,
            "messages": [m.to_openai() for m in messages],
        }
        for k in ("temperature", "max_tokens", "top_p", "stop", "seed"):
            if k in gen and gen[k] is not None:
                kwargs[k] = gen[k]

        last_err = ""
        for attempt in range(self.max_retries + 1):
            await self._limiter.acquire()
            t0 = now()
            try:
                resp = await client.chat.completions.create(**kwargs)
                dt = now() - t0
                choice = resp.choices[0]
                usage = getattr(resp, "usage", None)
                pt = getattr(usage, "prompt_tokens", 0) or 0
                ct = getattr(usage, "completion_tokens", 0) or 0
                return Generation(
                    text=choice.message.content or "",
                    model=self.bot_name,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=(pt + ct),
                    cost_usd=0.0,  # credit-based; not a USD figure
                    latency_s=dt,
                    finish_reason=getattr(choice, "finish_reason", "") or "",
                )
            except Exception as e:  # retry with exponential backoff
                last_err = f"{type(e).__name__}: {e}"
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
        return Generation.empty(self.bot_name, error=last_err)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
