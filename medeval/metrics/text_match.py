"""Text-overlap metrics: token-F1 and ROUGE (ROUGE-1/2/L).

Both compare the model's answer against a reference string
(``sample.reference['reference']``). Tokenization is CJK-aware and
dependency-free: ``auto`` uses character-level tokens for predominantly Chinese
text and word-level for Latin text (use ``jieba`` if installed for word-level
Chinese). These suit open_qa / sdt / prescription where a gold answer exists.
"""
from __future__ import annotations

import re
from typing import Any

from ..schema import Prediction, Sample, Score
from .base import Metric, register_metric

_CJK = r"一-鿿㐀-䶿぀-ヿ가-힯"
_TOKEN_RE = re.compile(rf"[a-zA-Z0-9]+|[{_CJK}]")
_WORD_RE = re.compile(r"[a-z0-9]+")
_CJK_COUNT = re.compile(rf"[{_CJK}]")


def _is_cjk_heavy(text: str) -> bool:
    if not text:
        return False
    cjk = len(_CJK_COUNT.findall(text))
    return cjk >= max(1, len(text.replace(" ", "")) * 0.2)


def tokenize(text: str, mode: str = "auto") -> list[str]:
    text = text or ""
    if mode == "auto":
        mode = "char" if _is_cjk_heavy(text) else "word"
    if mode == "jieba":
        try:
            import jieba
            return [t for t in jieba.lcut(text) if t.strip()]
        except Exception:
            mode = "char"
    if mode == "char":
        # ASCII words kept whole; CJK split per character
        return _TOKEN_RE.findall(text)
    return _WORD_RE.findall(text.lower())


def _overlap(pred: list[str], ref: list[str]) -> int:
    from collections import Counter
    cp, cr = Counter(pred), Counter(ref)
    return sum((cp & cr).values())


def prf(common: int, n_pred: int, n_ref: int) -> tuple[float, float, float]:
    p = common / n_pred if n_pred else 0.0
    r = common / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _reference_text(sample: Sample) -> str:
    ref = sample.reference
    return str(ref.get("reference") or ref.get("answer") or ref.get("syndrome") or "")


class _RefMetric(Metric):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.mode = self.config.get("tokenization", "auto")


@register_metric("f1")
class TokenF1(_RefMetric):
    """Token-level F1 (SQuAD-style multiset overlap) vs. the reference answer."""

    async def score(self, sample: Sample, pred: Prediction) -> Score:
        ref_text = _reference_text(sample)
        pt = tokenize(pred.text, self.mode)
        rt = tokenize(ref_text, self.mode)
        common = _overlap(pt, rt)
        p, r, f = prf(common, len(pt), len(rt))
        return Score(metric="f1", value=f,
                     detail={"precision": p, "recall": r, "f1": f,
                             "has_reference": bool(ref_text)})

    def aggregate(self, scores: list[Score]) -> dict[str, Any]:
        n = len(scores) or 1
        return {"f1": sum(s.value for s in scores) / n,
                "precision": sum(s.detail["precision"] for s in scores) / n,
                "recall": sum(s.detail["recall"] for s in scores) / n,
                "n": len(scores)}


@register_metric("rouge")
class Rouge(_RefMetric):
    """ROUGE-1 / ROUGE-2 / ROUGE-L (F-measures) vs. the reference answer.
    The headline ``value`` is ROUGE-L F1."""

    async def score(self, sample: Sample, pred: Prediction) -> Score:
        ref_text = _reference_text(sample)
        pt = tokenize(pred.text, self.mode)
        rt = tokenize(ref_text, self.mode)
        # ROUGE-1 / ROUGE-2
        r1 = prf(_overlap(pt, rt), len(pt), len(rt))[2]
        p2, r2 = _ngrams(pt, 2), _ngrams(rt, 2)
        rouge2 = prf(_overlap(p2, r2), len(p2), len(r2))[2]
        # ROUGE-L (LCS-based F)
        lcs = _lcs(pt, rt)
        rl = prf(lcs, len(pt), len(rt))[2]
        return Score(metric="rouge", value=rl,
                     detail={"rouge1": r1, "rouge2": rouge2, "rougeL": rl,
                             "has_reference": bool(ref_text)})

    def aggregate(self, scores: list[Score]) -> dict[str, Any]:
        n = len(scores) or 1
        return {"rougeL": sum(s.value for s in scores) / n,
                "rouge1": sum(s.detail["rouge1"] for s in scores) / n,
                "rouge2": sum(s.detail["rouge2"] for s in scores) / n,
                "n": len(scores)}
