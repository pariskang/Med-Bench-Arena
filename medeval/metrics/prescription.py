"""方剂结构匹配 — prescription / formula structure match (``prescription_match``).

For TCM prescription tasks (MTCMB TCM-FRD / TCM-PR), compare the *structure* of
the model's prescription against the gold one, not surface text:

* herb set overlap (君臣佐使 / 药物组成) → precision / recall / F1  (the headline)
* 方剂 formula-name match
* 治法 treatment-principle token overlap

Gold is read from ``reference_raw`` when present (e.g. the TCM-FRD dict
``{治法, 方剂, 药物组成}`` or a stringified herb list), else parsed from the
reference string. Herb names are normalized (dosages / parentheticals removed).
"""
from __future__ import annotations

import ast
import re
from typing import Any

from ..schema import Prediction, Sample, Score
from .base import Metric, register_metric
from .text_match import tokenize, _overlap, prf

# Separators between herbs. NOTE: 加 / 配 were removed — they split *inside*
# common phrases (加减, 配伍) and inside herb names, manufacturing phantom tokens.
_SEP = re.compile(r"[、,，;；/\|\s]+|和(?=\S)|与|及")
_DOSE = re.compile(r"\d+\.?\d*\s*(?:g|kg|mg|ml|克|毫升|钱|两|分|片|枚|条|根|个|只|对|粒)?", re.I)
_PAREN = re.compile(r"[（(【\[][^）)】\]]*[)）】\]]")
_HERB_SECTION = re.compile(r"(?:药物组成|药物|组成|方药|处方|用药|方剂组成)\s*[:：]?\s*(.+)", re.S)
_CJK = re.compile(r"[一-鿿]")
# Preparation / instruction words that are NOT herbs but ride along in a 药物组成
# string ("…水煎服", "加减", "每日一剂"). Dropped so they don't dilute herb-set F1.
_NON_HERB = {
    "水煎服", "水煎", "煎服", "顿服", "温服", "分服", "冲服", "代茶饮", "加减",
    "随证加减", "上药", "每日一剂", "日一剂", "一剂", "适量", "若", "等",
    "共研细末", "研末", "为末", "若干", "克", "用法", "用量",
}


def _norm_herb(h: str) -> str:
    h = _PAREN.sub("", h)
    h = _DOSE.sub("", h)
    h = re.sub(r"[\s·•・,.。:：;；]", "", h)
    return h.strip()


def _split_herbs(text: str) -> set[str]:
    out = set()
    for tok in _SEP.split(text or ""):
        h = _norm_herb(tok)
        # require ≥2 CJK chars (single chars are almost always fragments) and skip
        # known non-herb preparation words.
        if h and _CJK.search(h) and 2 <= len(h) <= 8 and h not in _NON_HERB:
            out.add(h)
    return out


def _ref_fields(sample: Sample) -> tuple[set[str], str, str]:
    """Return (herbs, formula_name, principle) from the structured/raw reference."""
    raw = sample.reference.get("reference_raw")
    herbs: set[str] = set()
    formula = principle = ""
    if isinstance(raw, dict):
        for k, v in raw.items():
            if any(x in k for x in ("药物", "组成", "方药")):
                herbs |= _split_herbs(v if isinstance(v, str) else " ".join(map(str, v)))
            elif "方" in k:
                formula = str(v)
            elif "治法" in k or "治则" in k:
                principle = str(v)
        if not herbs:  # dict but no obvious herb key -> flatten values
            herbs = _split_herbs(" ".join(str(v) for v in raw.values()))
        return herbs, formula, principle
    if isinstance(raw, list):
        return _split_herbs("、".join(map(str, raw))), "", ""
    # fall back to the stringified reference
    text = str(sample.reference.get("reference", ""))
    if text.strip().startswith("[") :   # stringified python list (TCM-PR)
        try:
            return _split_herbs("、".join(map(str, ast.literal_eval(text)))), "", ""
        except Exception:
            pass
    m = _HERB_SECTION.search(text)
    return _split_herbs(m.group(1) if m else text), "", ""


def _pred_herbs(text: str) -> set[str]:
    m = _HERB_SECTION.search(text or "")
    return _split_herbs(m.group(1) if m else text)


@register_metric("prescription_match")
class PrescriptionMatch(Metric):
    """Structure-aware scoring for 方剂生成. Headline value = herb-set F1."""

    async def score(self, sample: Sample, pred: Prediction) -> Score:
        ref_herbs, formula, principle = _ref_fields(sample)
        pred_herbs = _pred_herbs(pred.text)
        common = len(ref_herbs & pred_herbs)
        p, r, f = prf(common, len(pred_herbs), len(ref_herbs))

        formula_match = 1.0 if formula and formula in (pred.text or "") else 0.0
        principle_f1 = 0.0
        if principle:
            principle_f1 = prf(_overlap(tokenize(pred.text), tokenize(principle)),
                               len(tokenize(pred.text)), len(tokenize(principle)))[2]
        return Score(metric="prescription_match", value=f,
                     detail={"herb_precision": p, "herb_recall": r, "herb_f1": f,
                             "formula_match": formula_match,
                             "principle_f1": round(principle_f1, 4),
                             "n_ref_herbs": len(ref_herbs), "n_pred_herbs": len(pred_herbs),
                             "matched_herbs": sorted(ref_herbs & pred_herbs)})

    def aggregate(self, scores: list[Score]) -> dict[str, Any]:
        n = len(scores) or 1
        return {"herb_f1": sum(s.value for s in scores) / n,
                "herb_precision": sum(s.detail["herb_precision"] for s in scores) / n,
                "herb_recall": sum(s.detail["herb_recall"] for s in scores) / n,
                "formula_match": sum(s.detail["formula_match"] for s in scores) / n,
                "n": len(scores)}
