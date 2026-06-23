"""TCMBench adapter (``adapter: tcmbench``).

TCMBench (ywjawmw/TCMBench, arXiv:2406.01126) ships per-task JSON files whose
questions don't fit the generic ``hf_mcq`` mold: each file is a JSON *object*
``{type, prefix_prompt, keywords, example}`` with the questions under
``example``, and **options are embedded inside the question text** (``A．…``)
rather than a separate column. Two record shapes exist:

* flat:        ``{question (options embedded), answer: [letters], ...}``
* shared-stem: ``{share_content, question: [{sub_question, answer: [...]}, ...]}``
  (the B1/A3 类型 files KHC / CVR)

This subclasses :class:`HFMCQAdapter` to reuse its robust answer extraction and
just overrides loading. Demo data lives under ``data_demo/`` in the repo (the
full 5,473-question set is not public).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from ..schema import Message, Sample, TaskType
from .base import register_dataset
from .hf_mcq import HFMCQAdapter, LETTERS
from .agent_env import _download
from .local_json import CACHE_DIR


@register_dataset("tcmbench")
class TCMBenchAdapter(HFMCQAdapter):
    """config:
      source_url:  raw URL of a TCMBench task file, e.g.
        https://raw.githubusercontent.com/ywjawmw/TCMBench/main/data_demo/first_level/FKU.json
      path:        local file path (alternative to source_url)
      answer_format: multi (default; answers are letter lists)
      limit:       cap samples
    """

    def __init__(self, config: dict[str, Any]):
        cfg = dict(config)
        cfg.setdefault("path", "tcmbench")
        cfg.setdefault("field_map", {"question": "question", "options": None, "answer": "answer"})
        cfg.setdefault("answer_format", "multi")
        super().__init__(cfg)
        self.source_url = config.get("source_url")
        self.local_path = config.get("path") if config.get("path") not in (None, "tcmbench") else None

    def load(self) -> list[Sample]:
        from pathlib import Path
        srcs = self.source_url if isinstance(self.source_url, list) else (
            [self.source_url] if self.source_url else [self.local_path])
        samples: list[Sample] = []
        for si, src in enumerate(srcs):
            if str(src).startswith(("http://", "https://")):
                h = hashlib.sha256(str(src).encode()).hexdigest()[:16]
                fp = _download(str(src), CACHE_DIR / f"{self.id}_{h}.json")
            else:
                fp = Path(src)
            obj = json.loads(fp.read_text(encoding="utf-8"))
            examples = obj.get("example", obj if isinstance(obj, list) else [])
            for i, rec in enumerate(examples):
                built = self._build_shared(rec, f"{si}:{i}") if isinstance(rec.get("question"), list) \
                    else [self._build_flat(rec, f"{si}:{i}")]
                for s in built:
                    if s is not None:
                        samples.append(s)
                    if self.limit and len(samples) >= self.limit:
                        return samples
        return samples

    # --- record builders --------------------------------------------------
    def _make_sample(self, sid: str, stem: str, opt_text: str,
                     ans_letters: list[str]) -> Sample | None:
        parsed = self._split_inline(opt_text)
        if len(parsed) < 2:
            return None  # not an MCQ (e.g. fill-in / herb_predict)
        keys = [k for k, _ in parsed]
        choices = [t for _, t in parsed]
        idxs = [keys.index(L.upper()) for L in ans_letters if L.upper() in keys]
        if not idxs:
            return None
        reference = ({"indices": sorted(set(idxs))} if len(idxs) > 1
                     else {"index": idxs[0], "letter": LETTERS[idxs[0]],
                           "text": choices[idxs[0]]})
        prompt = f"{stem.strip()}\n\n{opt_text.strip()}\n\n{self.instruction}"
        return Sample(id=sid, task_type=TaskType.MCQ,
                      messages=[Message("user", prompt)], choices=choices,
                      reference=reference, meta={"source": self.source_url})

    def _build_flat(self, rec: dict[str, Any], i: int) -> Sample | None:
        q = str(rec.get("question", ""))
        ans = rec.get("answer", [])
        ans = ans if isinstance(ans, list) else [ans]
        # the question already embeds the lettered options; split off the stem
        idx = self._first_option_pos(q)
        stem, opts = (q[:idx], q[idx:]) if idx > 0 else (q, q)
        return self._make_sample(f"{self.id}:{rec.get('index', i)}", stem, opts, ans)

    def _build_shared(self, rec: dict[str, Any], i: int) -> list[Sample | None]:
        share = str(rec.get("share_content", "")).strip()
        out: list[Sample | None] = []
        for j, sub in enumerate(rec.get("question", [])):
            subq = str(sub.get("sub_question", ""))
            ans = sub.get("answer", [])
            ans = ans if isinstance(ans, list) else [ans]
            # Split the lettered options off whichever block carries them — the
            # sub-question (A3 类型) or the shared stem (B1 类型) — so stem and the
            # option block are disjoint (no double-render, vignette preserved).
            si = self._first_option_pos(subq)
            if si > 0:                                   # options embedded in sub-question
                stem = f"{share}\n{subq[:si]}".strip() if share else subq[:si].strip()
                opt_text = subq[si:]
            else:
                bi = self._first_option_pos(share)
                if bi > 0:                               # options in the shared block (B1)
                    stem = f"{share[:bi]}\n{subq}".strip()
                    opt_text = share[bi:]
                else:                                    # no parseable options -> skipped downstream
                    stem, opt_text = f"{share}\n{subq}".strip(), subq
            out.append(self._make_sample(f"{self.id}:{rec.get('index', i)}:{j}",
                                        stem, opt_text, ans))
        return out

    @staticmethod
    def _first_option_pos(text: str) -> int:
        import re
        m = re.search(r"(?m)^\s*[A-Za-z][\s．.、:：)]", text)
        return m.start() if m else -1
