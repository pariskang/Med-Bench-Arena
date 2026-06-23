"""Numeric-answer matching (``numeric_match``) — for calculation tasks like
MedCalc-Bench. Extracts the model's final number and checks it against the gold
value within a tolerance, or against an explicit ``[lower_limit, upper_limit]``
range when the dataset provides one.
"""
from __future__ import annotations

import re
from typing import Any

from ..schema import Prediction, Sample, Score
from .base import Metric, register_metric

_NUM = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.\d+|-?\.\d+|-?\d+")
# an explicit final answer marker, e.g. "answer is 12.5", "= 12.5", "答案：12.5"
_ANSWER = re.compile(
    r"(?:answer|result|final|总分|得分|答案|结果)\s*(?:is|are|:|：|=|为|是)?\s*([-\d.,]+)",
    re.IGNORECASE)


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", "").rstrip(".%"))
    except (TypeError, ValueError):
        return None


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text or ""):
        v = _to_float(m)
        if v is not None:
            out.append(v)
    return out


def _pred_number(text: str) -> float | None:
    m = _ANSWER.search(text or "")
    if m:
        v = _to_float(m.group(1))
        if v is not None:
            return v
    nums = _numbers(text)
    return nums[-1] if nums else None   # models usually end with the final value


@register_metric("numeric_match")
class NumericMatch(Metric):
    """config: rel_tol (default 0.05), abs_tol (default 0.0)."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.rel_tol = float(self.config.get("rel_tol", 0.05))
        self.abs_tol = float(self.config.get("abs_tol", 0.0))

    def _gold(self, sample: Sample) -> float | tuple[float, float] | None:
        ref = sample.reference
        lo, hi = ref.get("lower_limit"), ref.get("upper_limit")
        if lo is not None and hi is not None:
            flo, fhi = _to_float(str(lo)), _to_float(str(hi))
            if flo is not None and fhi is not None:
                return (flo, fhi)
        raw = ref.get("reference") or ref.get("answer") or ref.get("ground_truth")
        nums = _numbers(str(raw)) if raw is not None else []
        return nums[0] if nums else None

    async def score(self, sample: Sample, pred: Prediction) -> Score:
        gold = self._gold(sample)
        cand = _pred_number(pred.text)
        if gold is None or cand is None:
            correct = False
        elif isinstance(gold, tuple):
            correct = gold[0] <= cand <= gold[1]
        else:
            correct = abs(cand - gold) <= max(self.abs_tol, self.rel_tol * abs(gold))
        return Score(metric="numeric_match", value=1.0 if correct else 0.0,
                     detail={"gold": gold, "pred": cand,
                             "parsed_ok": cand is not None})

    def aggregate(self, scores: list[Score]) -> dict[str, Any]:
        n = len(scores) or 1
        return {"numeric_accuracy": sum(s.value for s in scores) / n,
                "unparsed": sum(1 for s in scores if not s.detail.get("parsed_ok")),
                "n": len(scores)}
