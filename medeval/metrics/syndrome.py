"""证型链结构分 — syndrome-differentiation chain score (``syndrome_chain``).

For 辨证 (TCMEval-SDT). Scores the reasoning chain 症状→病机→证型 against the
structured gold, with **partial credit for 同病异治** (multiple acceptable
syndromes → recall-based credit). Components read from ``sample.reference``:

* 证型 syndrome  : ``reference['syndrome']`` (or ``label``) — may be multi
                   (``"热伤阳络;血热妄行"``), split on ``;，、 / 和``.
* 病机 pathogenesis: ``reference['pathogenesis']`` (optional; map it in field_map).
* 辨证 final      : ``reference['reference']`` (the full differentiation text).

Headline ``value`` = weighted chain score over whichever components exist
(default weights syndrome .5 / pathogenesis .3 / final .2, renormalized).
"""
from __future__ import annotations

import re
from typing import Any

from ..schema import Prediction, Sample, Score
from .base import Metric, register_metric
from .text_match import tokenize, _overlap, prf

_SPLIT = re.compile(r"[;；,，、/\s]+|和(?=\S)|与|及|兼|夹")
# a 证型/辨证/诊断 section in the model's free text (for precision estimation)
_SYND_SECTION = re.compile(r"(?:证型|证候|辨证|证属|诊断|中医诊断)\s*[:：为是]?\s*(.+)")


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).strip()


def _syndrome_set(text: str) -> set[str]:
    return {_norm(t) for t in _SPLIT.split(text or "") if _norm(t) and len(_norm(t)) >= 2}


def _text_f1(pred: str, ref: str) -> float:
    if not ref:
        return 0.0
    pt, rt = tokenize(pred), tokenize(ref)
    return prf(_overlap(pt, rt), len(pt), len(rt))[2]


@register_metric("syndrome_chain")
class SyndromeChain(Metric):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        w = self.config.get("weights", {})
        self.w_syndrome = float(w.get("syndrome", 0.5))
        self.w_patho = float(w.get("pathogenesis", 0.3))
        self.w_final = float(w.get("final", 0.2))

    async def score(self, sample: Sample, pred: Prediction) -> Score:
        ref = sample.reference
        gold_synd = _syndrome_set(ref.get("syndrome") or ref.get("label") or "")
        gold_patho = str(ref.get("pathogenesis") or "")
        gold_final = str(ref.get("reference") or "")
        text = pred.text or ""

        detail: dict[str, Any] = {}
        components: list[tuple[float, float]] = []  # (score, weight)

        # 证型: recall over gold (同病异治 partial credit) + best-effort precision
        if gold_synd:
            found = {s for s in gold_synd if s in text}
            recall = len(found) / len(gold_synd)
            m = _SYND_SECTION.search(text)
            pred_synd = _syndrome_set(m.group(1)) if m else set()
            if pred_synd:
                precision = len(pred_synd & gold_synd) / len(pred_synd)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            else:
                precision, f1 = recall, recall  # can't penalize without a clear section
            detail.update({"syndrome_recall": recall, "syndrome_precision": precision,
                           "syndrome_f1": f1, "gold_syndromes": sorted(gold_synd),
                           "matched_syndromes": sorted(found)})
            components.append((f1, self.w_syndrome))

        # 病机 / 辨证: token overlap with the gold text
        if gold_patho:
            pf = _text_f1(text, gold_patho)
            detail["pathogenesis_f1"] = round(pf, 4)
            components.append((pf, self.w_patho))
        if gold_final:
            ff = _text_f1(text, gold_final)
            detail["final_f1"] = round(ff, 4)
            components.append((ff, self.w_final))

        wsum = sum(w for _, w in components)
        if not components:
            # no gold at all (no 证型/病机/辨证 reference): nothing to score against —
            # excluded from the aggregate rather than counted as a hard 0.
            return Score(metric="syndrome_chain", value=None,
                         detail={"skipped": "no_gold"})
        value = sum(s * w for s, w in components) / wsum if wsum else 0.0
        detail["chain_score"] = value
        return Score(metric="syndrome_chain", value=value, detail=detail)

    def aggregate(self, scores: list[Score]) -> dict[str, Any]:
        vals = [s.value for s in scores if s.value is not None]
        out = {"chain_score": sum(vals) / len(vals) if vals else 0.0,
               "n": len(scores), "n_scored": len(vals),
               "skipped_no_gold": len(scores) - len(vals)}
        for key in ("syndrome_f1", "syndrome_recall", "pathogenesis_f1", "final_f1"):
            kvals = [s.detail[key] for s in scores if key in s.detail]
            if kvals:
                out[key] = sum(kvals) / len(kvals)
        return out
