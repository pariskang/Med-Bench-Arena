"""Structured TCM metrics: 经络腧穴 (meridians/acupoints) and 古籍本体 (classical
literature ontology).

These target gaps the field repeatedly flags — models are weakest on 经络腧穴,
and existing TCM benchmarks tend to flatten the classics into "standard answers".
Both score by extracting **canonical ontology terms** from the prediction and the
gold (a built-in, extensible lexicon) and computing set precision/recall/F1, so
答案被奖励的是"命中正确的经络/腧穴/古籍来源", not surface text overlap.

Gold is read from explicit reference lists when present
(``reference['meridians'|'acupoints'|'classics'|'source']``), otherwise extracted
from the reference text. Extend the lexicons via config
(``extra_meridians`` / ``extra_acupoints`` / ``extra_classics`` lists, or
``lexicon_file`` — a JSON with those keys).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schema import Prediction, Sample, Score
from .base import Metric, register_metric

# --- 经络: 12 正经 + 奇经八脉, canonical -> aliases ---------------------------
_MERIDIANS: dict[str, list[str]] = {
    "手太阴肺经": ["手太阴肺经", "手太阴", "肺经"],
    "手阳明大肠经": ["手阳明大肠经", "手阳明", "大肠经"],
    "足阳明胃经": ["足阳明胃经", "足阳明", "胃经"],
    "足太阴脾经": ["足太阴脾经", "足太阴", "脾经"],
    "手少阴心经": ["手少阴心经", "手少阴", "心经"],
    "手太阳小肠经": ["手太阳小肠经", "手太阳", "小肠经"],
    "足太阳膀胱经": ["足太阳膀胱经", "足太阳", "膀胱经"],
    "足少阴肾经": ["足少阴肾经", "足少阴", "肾经"],
    "手厥阴心包经": ["手厥阴心包经", "手厥阴", "心包经"],
    "手少阳三焦经": ["手少阳三焦经", "手少阳", "三焦经"],
    "足少阳胆经": ["足少阳胆经", "足少阳", "胆经"],
    "足厥阴肝经": ["足厥阴肝经", "足厥阴", "肝经"],
    "督脉": ["督脉"], "任脉": ["任脉"], "冲脉": ["冲脉"], "带脉": ["带脉"],
    "阴维脉": ["阴维脉"], "阳维脉": ["阳维脉"],
    "阴跷脉": ["阴跷脉", "阴蹻脉"], "阳跷脉": ["阳跷脉", "阳蹻脉"],
}

# --- 腧穴: a curated common-point lexicon (extensible) ------------------------
_ACUPOINTS: list[str] = [
    "足三里", "三阴交", "手三里", "上巨虚", "下巨虚", "阳陵泉", "阴陵泉", "足临泣",
    "悬钟", "昆仑", "至阴", "委中", "承山", "承筋", "飞扬", "申脉", "京骨", "束骨",
    "合谷", "曲池", "手五里", "肩髃", "迎香", "二间", "三间", "阳溪", "偏历", "温溜",
    "列缺", "太渊", "鱼际", "少商", "尺泽", "孔最", "中府", "云门", "侠白",
    "内关", "外关", "支沟", "中渚", "液门", "翳风", "耳门", "丝竹空",
    "神门", "通里", "少海", "极泉", "少冲", "少府",
    "后溪", "前谷", "腕骨", "养老", "天宗", "秉风", "肩贞", "听宫",
    "太冲", "行间", "大敦", "曲泉", "蠡沟", "章门", "期门",
    "太溪", "照海", "然谷", "复溜", "涌泉", "太白", "公孙", "商丘", "大都", "隐白",
    "百会", "印堂", "神庭", "水沟", "大椎", "命门", "腰阳关", "至阳", "风府", "哑门",
    "关元", "气海", "中脘", "下脘", "神阙", "膻中", "天突", "中极", "曲骨",
    "天枢", "梁门", "梁丘", "犊鼻", "解溪", "内庭", "厉兑", "丰隆", "条口", "地仓",
    "颊车", "下关", "头维", "四白", "风池", "风市", "环跳", "肩井", "日月",
    "睛明", "攒竹", "承泣", "血海", "箕门", "曲泽", "劳宫", "中冲",
    "肺俞", "心俞", "肝俞", "脾俞", "肾俞", "胃俞", "胆俞", "膈俞", "厥阴俞",
    "大肠俞", "小肠俞", "膀胱俞", "三焦俞", "肾俞", "膏肓", "志室",
]

# --- 古籍本体: classical sources, canonical -> aliases ------------------------
_CLASSICS: dict[str, list[str]] = {
    "黄帝内经": ["黄帝内经", "内经"], "素问": ["素问"], "灵枢": ["灵枢经", "灵枢"],
    "难经": ["黄帝八十一难经", "难经"], "伤寒杂病论": ["伤寒杂病论"],
    "伤寒论": ["伤寒论"], "金匮要略": ["金匮要略", "金匮"],
    "神农本草经": ["神农本草经", "本草经", "本经"], "脉经": ["脉经"],
    "针灸甲乙经": ["针灸甲乙经", "甲乙经"], "诸病源候论": ["诸病源候论"],
    "千金要方": ["备急千金要方", "千金要方", "千金方"], "外台秘要": ["外台秘要", "外台"],
    "温病条辨": ["温病条辨"], "温热论": ["温热论"], "湿热条辨": ["湿热条辨", "湿热病篇"],
    "脾胃论": ["脾胃论"], "内外伤辨惑论": ["内外伤辨惑论"], "景岳全书": ["景岳全书"],
    "医宗金鉴": ["医宗金鉴"], "本草纲目": ["本草纲目"], "濒湖脉学": ["濒湖脉学"],
    "医林改错": ["医林改错"], "血证论": ["血证论"], "丹溪心法": ["丹溪心法"],
    "三因极一病证方论": ["三因极一病证方论", "三因方"], "小儿药证直诀": ["小儿药证直诀"],
    "本草纲目拾遗": ["本草纲目拾遗"], "证治准绳": ["证治准绳"], "医学心悟": ["医学心悟"],
}


def _alias_map(canon: dict[str, list[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for c, aliases in canon.items():
        for a in aliases:
            out[a] = c
    return out


def extract_terms(text: str, alias_to_canon: dict[str, str]) -> set[str]:
    """Longest-match extraction: blank each hit so sub-aliases don't double-count
    (so '黄帝内经' is not also counted as '内经', '伤寒杂病论' not as '伤寒论')."""
    work = text or ""
    found: set[str] = set()
    for alias in sorted(alias_to_canon, key=len, reverse=True):
        if alias and alias in work:
            found.add(alias_to_canon[alias])
            work = work.replace(alias, "　")
    return found


def _prf(gold: set[str], pred: set[str]) -> tuple[float, float, float]:
    if not gold and not pred:
        return 0.0, 0.0, 0.0
    common = len(gold & pred)
    p = common / len(pred) if pred else 0.0
    r = common / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _ref_text(sample: Sample) -> str:
    ref = sample.reference
    return str(ref.get("reference") or ref.get("text") or ref.get("answer") or "")


class _LexiconMetric(Metric):
    """Shared base: build alias maps, allow per-config lexicon extension."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.extra = self._load_extra()

    def _load_extra(self) -> dict[str, list[str]]:
        extra: dict[str, list[str]] = {"meridians": [], "acupoints": [], "classics": []}
        lf = self.config.get("lexicon_file")
        if lf and Path(lf).exists():
            data = json.loads(Path(lf).read_text(encoding="utf-8"))
            for k in extra:
                extra[k] += list(data.get(k, []))
        for k in extra:
            extra[k] += list(self.config.get(f"extra_{k}", []))
        return extra


@register_metric("meridian_acupoint")
class MeridianAcupoint(_LexiconMetric):
    """经络腧穴: meridian-set F1 and acupoint-set F1 vs. the gold (headline = mean)."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.w_mer = float(self.config.get("weight_meridian", 0.5))
        self.w_acu = float(self.config.get("weight_acupoint", 0.5))
        mer = dict(_MERIDIANS)
        for m in self.extra["meridians"]:
            mer.setdefault(m, [m])
        self._mer_map = _alias_map(mer)
        acu = list(_ACUPOINTS) + self.extra["acupoints"]
        self._acu_map = {a: a for a in acu}

    def _gold(self, sample: Sample, key: str, amap: dict[str, str]) -> set[str]:
        explicit = sample.reference.get(key)
        if explicit:
            if isinstance(explicit, str):       # free text / stringified list -> extract
                return extract_terms(explicit, amap)
            return {amap.get(str(x), str(x)) for x in explicit}
        return extract_terms(_ref_text(sample), amap)

    async def score(self, sample: Sample, pred: Prediction) -> Score:
        gold_mer = self._gold(sample, "meridians", self._mer_map)
        gold_acu = self._gold(sample, "acupoints", self._acu_map)
        pred_mer = extract_terms(pred.text, self._mer_map)
        pred_acu = extract_terms(pred.text, self._acu_map)

        mp, mr, mf = _prf(gold_mer, pred_mer)
        ap, ar, af = _prf(gold_acu, pred_acu)
        comps, weights = [], []
        if gold_mer:
            comps.append((mf, self.w_mer)); weights.append(self.w_mer)
        if gold_acu:
            comps.append((af, self.w_acu)); weights.append(self.w_acu)
        wsum = sum(weights)
        value = sum(s * w for s, w in comps) / wsum if wsum else 0.0
        return Score(metric="meridian_acupoint", value=value, detail={
            "meridian_f1": mf, "meridian_p": mp, "meridian_r": mr,
            "acupoint_f1": af, "acupoint_p": ap, "acupoint_r": ar,
            "gold_meridians": sorted(gold_mer), "matched_meridians": sorted(gold_mer & pred_mer),
            "gold_acupoints": sorted(gold_acu), "matched_acupoints": sorted(gold_acu & pred_acu)})

    def aggregate(self, scores: list[Score]) -> dict[str, Any]:
        n = len(scores) or 1
        return {"meridian_acupoint_f1": sum(s.value for s in scores) / n,
                "meridian_f1": sum(s.detail["meridian_f1"] for s in scores) / n,
                "acupoint_f1": sum(s.detail["acupoint_f1"] for s in scores) / n,
                "n": len(scores)}


@register_metric("classics_ontology")
class ClassicsOntology(_LexiconMetric):
    """古籍本体: did the answer ground itself in the correct classical source(s)?
    Set-F1 over canonical classics, plus source_recall (all gold sources cited)."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        cl = dict(_CLASSICS)
        for c in self.extra["classics"]:
            cl.setdefault(c, [c])
        self._map = _alias_map(cl)

    def _gold(self, sample: Sample) -> set[str]:
        ref = sample.reference
        explicit = ref.get("classics") or ref.get("source") or ref.get("ontology")
        if explicit:
            if isinstance(explicit, dict):
                explicit = explicit.get("source") or explicit.get("classics") or []
            if isinstance(explicit, str):       # free text / stringified -> extract
                return extract_terms(explicit, self._map)
            return {self._map.get(str(x), str(x)) for x in explicit}
        return extract_terms(_ref_text(sample), self._map)

    async def score(self, sample: Sample, pred: Prediction) -> Score:
        gold = self._gold(sample)
        got = extract_terms(pred.text, self._map)
        p, r, f = _prf(gold, got)
        source_correct = 1.0 if gold and gold <= got else 0.0
        return Score(metric="classics_ontology", value=f, detail={
            "source_precision": p, "source_recall": r, "source_f1": f,
            "all_sources_cited": source_correct,
            "gold_sources": sorted(gold), "cited_sources": sorted(got),
            "matched": sorted(gold & got)})

    def aggregate(self, scores: list[Score]) -> dict[str, Any]:
        n = len(scores) or 1
        return {"source_f1": sum(s.value for s in scores) / n,
                "source_recall": sum(s.detail["source_recall"] for s in scores) / n,
                "all_sources_cited": sum(s.detail["all_sources_cited"] for s in scores) / n,
                "n": len(scores)}
