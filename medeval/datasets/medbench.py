"""MedBench data adapter (``adapter: medbench``).

Loads MedBench evaluation files (``<DATASET>_test.jsonl`` with records
``{question, passage, options, answer, other:{source,id}}``) so you can run your
model over the platform's data and then export a submission with
``medeval.submit`` / ``python -m medeval export``.

MedBench test answers are **held out** (``answer: null``), so there is nothing to
score locally — the point is to produce predictions for upload. If you point it
at a ``提交结果示例.jsonl`` (answers filled), the gold is kept and you can add
``mcq_accuracy`` to sanity-check.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..schema import Message, Prediction, Sample, TaskType
from .base import DatasetAdapter, register_dataset
from .hf_mcq import HFMCQAdapter, LETTERS
from .agent_env import _download
from .local_json import CACHE_DIR

_OPT_PREFIX = re.compile(r"^\s*([A-Za-z])\s*[:：.、)]\s*(.*)$")


@register_dataset("medbench")
class MedBenchAdapter(DatasetAdapter):
    """config:
      path:        a MedBench *_test.jsonl file OR a directory (loads *_test.jsonl)
      source_url:  alternatively download one test file
      instruction: trailing MCQ instruction
      limit:       cap samples
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.path = config.get("path")
        self.source_url = config.get("source_url")
        self.instruction = config.get(
            "instruction",
            "请直接给出正确答案的选项字母。Answer with the option letter.")
        # held-out answers -> no local metric by default
        self._mcq = HFMCQAdapter({"id": "_p", "path": "x",
                                  "field_map": {"question": "q", "options": "o", "answer": "a"}})

    # --- source files -----------------------------------------------------
    def _files(self) -> list[Path]:
        if self.source_url:
            h = hashlib.sha256(self.source_url.encode()).hexdigest()[:16]
            return [_download(self.source_url, CACHE_DIR / f"{self.id}_{h}.jsonl")]
        p = Path(self.path)
        if p.is_dir():
            return sorted(p.rglob("*_test.jsonl")) or sorted(p.rglob("*.jsonl"))
        return [p]

    def load(self) -> list[Sample]:
        samples: list[Sample] = []
        for fp in self._files():
            for i, line in enumerate(fp.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                s = self._to_sample(json.loads(line), i)
                if s is not None:
                    samples.append(s)
                if self.limit and len(samples) >= self.limit:
                    return samples
        return samples

    def _to_sample(self, rec: dict[str, Any], i: int) -> Sample | None:
        other = rec.get("other") or {}
        source = other.get("source", self.id)
        oid = other.get("id", rec.get("id", i))
        question = str(rec.get("question", ""))
        passage = rec.get("passage")
        head = f"{passage}\n\n" if passage else ""
        opts = rec.get("options")

        if isinstance(opts, list) and opts:  # MCQ
            choices, rendered = [], []
            for opt in opts:
                m = _OPT_PREFIX.match(str(opt))
                choices.append(m.group(2) if m else str(opt))
                rendered.append(str(opt) if m else f"{LETTERS[len(rendered)]}: {opt}")
            user = f"{head}{question}\n" + "\n".join(rendered) + f"\n\n{self.instruction}"
            reference = self._gold(rec, choices)
            return Sample(id=f"{self.id}:{source}:{oid}", task_type=TaskType.MCQ,
                          messages=[Message("user", user)], choices=choices,
                          reference=reference, meta={"source": source, "id": oid})
        # generation / open task
        user = f"{head}{question}".strip()
        ref = {}
        if rec.get("answer"):
            ref = {"reference": str(rec["answer"])}
        return Sample(id=f"{self.id}:{source}:{oid}", task_type=TaskType.OPEN_QA,
                      messages=[Message("user", user)], reference=ref,
                      meta={"source": source, "id": oid})

    @staticmethod
    def _gold(rec: dict[str, Any], choices: list[str]) -> dict[str, Any]:
        ans = rec.get("answer")
        if not ans:
            return {}                      # held out
        letter = str(ans).strip().upper()[:1]
        if letter in LETTERS and LETTERS.index(letter) < len(choices):
            idx = LETTERS.index(letter)
            return {"index": idx, "letter": letter, "text": choices[idx]}
        return {"reference": str(ans)}

    def parse(self, sample: Sample, text: str) -> Prediction:
        if sample.task_type == TaskType.MCQ:
            return self._mcq.parse(sample, text)
        from ..schema import Generation
        return Prediction(sample_id=sample.id, generation=Generation(text=text), parsed=text)
