"""Deterministic, offline mock provider.

Needs no network, GPU, or keys — it inspects the prompt and returns a plausible
answer so the *entire* pipeline (generate -> parse -> score -> aggregate, incl.
LLM-judge and agent rollouts) can be smoke-tested end to end. Behaviour is a
pure function of the prompt, so caching and reruns are stable.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from ..schema import Generation, Message
from .base import ModelProvider, register_provider

_LETTER_LINE = re.compile(r"^\s*([A-Z])\s*[.):：、]\s*", re.MULTILINE)
_RUBRIC_ID = re.compile(r"\(id=([^,)]+)")


def _hash_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)


@register_provider("mock")
class MockProvider(ModelProvider):
    """A scripted, deterministic stand-in for a real backend.

    config:
      behavior: auto (default) | echo | "letter:C" | "text:..."
    In ``auto`` it detects judge prompts, MCQ prompts and agent prompts.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.behavior: str = config.get("behavior", "auto")

    async def agenerate(self, messages: list[Message], **gen: Any) -> Generation:
        # image-aware: fold image presence into the deterministic prompt
        prompt = "\n".join(
            m.content + (f" [IMAGE x{len(m.images)}]" if m.images else "")
            for m in messages)
        text = self._respond(prompt)
        ptok = max(1, len(prompt) // 4)
        ctok = max(1, len(text) // 4)
        return Generation(
            text=text,
            model=self.id,
            prompt_tokens=ptok,
            completion_tokens=ctok,
            total_tokens=ptok + ctok,
            cost_usd=0.0,
            latency_s=0.0,
            finish_reason="stop",
        )

    # --- response logic ---------------------------------------------------
    def _respond(self, prompt: str) -> str:
        b = self.behavior
        if b == "echo":
            return prompt.strip()[-512:]
        if b.startswith("letter:"):
            return f"The answer is {b.split(':', 1)[1].strip()}."
        if b.startswith("text:"):
            return b.split(":", 1)[1]

        # auto
        low = prompt.lower()
        if "json" in low and ("rubric" in low or "criterion" in low or "id=" in prompt):
            return self._judge(prompt)
        if "diagnosis" in low and "doctor" in low:
            return self._agent(prompt)
        letters = sorted(set(_LETTER_LINE.findall(prompt)))
        # MediQ interactive: ask one question, then commit to a lettered answer
        if "answer: <letter>" in low and letters:
            if low.count("please tell me") < 1:   # count our own prior questions in the history
                return "Please tell me the main symptoms, history, and any key findings."
            return f"ANSWER: {letters[_hash_int(prompt) % len(letters)]}"
        if letters and ("correct option" in low or "letter" in low or "answer with" in low):
            pick = letters[_hash_int(prompt) % len(letters)]
            return f"The answer is {pick}."
        # open-ended: deterministic short clinical-ish stub
        return (
            "Based on the presentation, the most appropriate response addresses "
            "the key findings and recommends safe, guideline-concordant next steps."
        )

    def _judge(self, prompt: str) -> str:
        # Award every listed criterion (id=...) — exercises the per-criterion path.
        ids = _RUBRIC_ID.findall(prompt)
        scores = {cid: 1.0 for cid in ids}
        payload = {"scores": scores, "overall": 1.0,
                   "explanation": "mock judge: all criteria satisfied"}
        return json.dumps(payload, ensure_ascii=False)

    def _agent(self, prompt: str) -> str:
        # Deterministically commit to a diagnosis drawn from the prompt's hints.
        m = re.search(r"candidate (?:diagnoses|diagnosis)[:\s]+([^\n]+)", prompt, re.I)
        if m:
            opts = [x.strip() for x in re.split(r"[,/;]", m.group(1)) if x.strip()]
            if opts:
                return "DIAGNOSIS: " + opts[_hash_int(prompt) % len(opts)]
        return "DIAGNOSIS: unspecified"
