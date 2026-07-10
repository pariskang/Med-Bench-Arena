"""Canonical schema — the shared intermediate representation (the "地基").

Everything in MedEval speaks this vocabulary. Datasets *produce* ``Sample``s,
providers *produce* ``Generation``s, metrics *consume* ``Sample`` + ``Prediction``
and *produce* ``Score``s. Because the three sides only depend on these types
(and never on each other), any one of them can be swapped or extended freely.
"""
from __future__ import annotations

import base64
import mimetypes
import re
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class TaskType(str, Enum):
    """The task family a sample belongs to. Drives prompt shaping + metric choice."""

    MCQ = "mcq"                    # single/multi choice (MedQA, MMLU, CMB, TCMBench...)
    OPEN_QA = "open_qa"            # free-text answer graded by rubric (HealthBench...)
    SDT = "sdt"                    # syndrome differentiation 辨证 (TCMEval-SDT)
    PRESCRIPTION = "prescription"  # 方剂生成 (MTCMB prescription)
    SAFETY = "safety"              # safety/refusal (CSEDB, MedSafetyBench)
    AGENT = "agent"                # interactive rollout (AgentClinic, MedAgentBench)


def image_to_url(img: str) -> str:
    """Normalize an image reference to a URL the vision APIs accept: pass http(s)
    and data: URIs through; base64-encode a local file path into a data URI."""
    if img.startswith(("http://", "https://", "data:")):
        return img
    mime = mimetypes.guess_type(img)[0] or "image/jpeg"
    with open(img, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def encode_images(val: Any, base: str = "", strip: str = "") -> list[str]:
    """Normalize many image representations to a list of URL/data-URI strings.

    Handles: str (URL/path, ``base`` prepended to relative, ``strip`` prefix removed),
    HF Image dict ``{bytes|path|url}``, raw ``bytes`` (parquet binary), nested lists,
    PIL, and a bare base64 string. ``strip`` removes a leading path prefix such as
    ``"../"`` (e.g. MedBookVQA's ``../figures/x.jpg``) before prepending ``base``.
    """
    if val is None:
        return []
    if isinstance(val, (bytes, bytearray)):
        mime = "image/png" if bytes(val[:8]).startswith(b"\x89PNG") else "image/jpeg"
        return [f"data:{mime};base64,{base64.b64encode(bytes(val)).decode('ascii')}"]
    if isinstance(val, str):
        if not val:
            return []
        if val.startswith(("http://", "https://", "data:")):
            return [val]
        # bare base64 image string (e.g. GMAI-MMBench TSV): long, pure base64
        # charset (paths/text have ./-/_ or spaces and break the match)
        if len(val) > 256 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", val):
            return [f"data:image/jpeg;base64,{val.strip()}"]
        if strip and val.startswith(strip):
            val = val[len(strip):]
        return [base + val]
    if isinstance(val, dict):  # HF Image feature
        if val.get("url"):
            return [val["url"]]
        if val.get("path"):
            p = val["path"]
            if strip and p.startswith(strip):
                p = p[len(strip):]
            return [(base if not p.startswith(("http", "data:")) else "") + p]
        if val.get("bytes"):
            return encode_images(val["bytes"], base, strip)
        return []
    if isinstance(val, (list, tuple)):
        out: list[str] = []
        for v in val:
            out.extend(encode_images(v, base, strip))
        return out
    try:  # PIL.Image
        import io
        from PIL import Image as _PIL
        if isinstance(val, _PIL.Image):
            buf = io.BytesIO()
            val.convert("RGB").save(buf, format="PNG")
            return [f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"]
    except Exception:
        pass
    return []


@dataclass
class Message:
    """One chat turn. ``content`` is text; ``images`` (optional) are http(s)/data
    URIs or local paths attached to this turn (for vision models)."""

    role: str           # "system" | "user" | "assistant"
    content: str
    images: list[str] | None = None

    def to_openai(self) -> dict[str, Any]:
        if not self.images:
            return {"role": self.role, "content": self.content}
        blocks: list[dict[str, Any]] = []
        if self.content:
            blocks.append({"type": "text", "text": self.content})
        for img in self.images:
            blocks.append({"type": "image_url", "image_url": {"url": image_to_url(img)}})
        return {"role": self.role, "content": blocks}


@dataclass
class Sample:
    """One evaluation item, fully decoupled from its source dataset.

    ``reference`` is intentionally polymorphic so a single set of metrics can
    serve every task type:

    * MCQ          -> ``{"index": 1, "letter": "B", "text": "..."}``
    * SDT          -> ``{"syndrome": "...", "rubric": [...]}``
    * SAFETY       -> ``{"label": "unsafe", "rubric": [...]}``
    * PRESCRIPTION -> ``{"reference": "...", "rubric": [...]}``
    """

    id: str
    task_type: TaskType
    messages: list[Message]
    choices: Optional[list[str]] = None      # MCQ option texts, in canonical order
    reference: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    # For agent tasks: an opaque handle the AgentAdapter knows how to reset/step.
    env_spec: Optional[dict[str, Any]] = None

    @property
    def prompt_messages(self) -> list[dict[str, str]]:
        return [m.to_openai() for m in self.messages]


@dataclass
class Generation:
    """A single model generation plus accounting (tokens / cost / latency)."""

    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    finish_reason: str = ""
    raw: Optional[dict[str, Any]] = None     # provider-native payload (debug)

    @classmethod
    def empty(cls, model: str = "", error: str = "") -> "Generation":
        g = cls(text="", model=model)
        if error:
            g.raw = {"error": error}
        return g


@dataclass
class Prediction:
    """What a model predicted for a sample.

    ``parsed`` is the metric-ready interpretation of the generation (e.g. an
    MCQ option index). ``trajectory`` holds the full step list for agent tasks.
    ``generations`` carries every rollout when k>1 (pass^k); ``generation`` is
    the canonical/first one.
    """

    sample_id: str
    generation: Generation
    parsed: Any = None
    trajectory: Optional[list[dict[str, Any]]] = None
    generations: Optional[list[Generation]] = None   # all k rollouts (pass^k)
    rollouts: Optional[list[dict[str, Any]]] = None   # per-rollout agent results
    # Agent support-role (patient/measurement/moderator) cost/tokens, summed
    # across every turn and rollout: {role: {cost_usd, prompt_tokens,
    # completion_tokens}}. ``generation.cost_usd`` on this Prediction is the
    # DOCTOR's cost only — without this, a faithful multi-agent AgentClinic run
    # (3 extra LLMs per turn) would report the same "cost" as the fully
    # scripted, single-LLM approximation.
    support_cost: Optional[dict[str, dict[str, float]]] = None

    @property
    def text(self) -> str:
        return self.generation.text


@dataclass
class Score:
    """One metric's verdict on one sample."""

    metric: str
    value: float
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def now() -> float:
    return time.perf_counter()
