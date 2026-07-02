"""Unit tests for the F1 / ROUGE / 方剂结构匹配 metrics (offline)."""
from __future__ import annotations

import asyncio

from medeval import Generation, Message, Prediction, Sample, Score, TaskType, create_metric
from medeval.datasets.base import _parse_metric_specs
from medeval.providers.mock import MockProvider


def _sample(reference, raw=None, task=TaskType.OPEN_QA):
    ref = {"reference": reference}
    if raw is not None:
        ref["reference_raw"] = raw
    return Sample(id="s", task_type=task, messages=[Message("user", "q")], reference=ref)


def _pred(text):
    return Prediction(sample_id="s", generation=Generation(text=text), parsed=text)


def _score(metric, sample, pred):
    return asyncio.run(create_metric(metric).score(sample, pred))


def test_f1_identical_and_partial():
    s = _sample("the patient has hypertension and diabetes")
    assert _score("f1", s, _pred("the patient has hypertension and diabetes")).value == 1.0
    partial = _score("f1", s, _pred("the patient has hypertension")).value
    assert 0.0 < partial < 1.0


def test_f1_chinese_char_level():
    s = _sample("高血压会增加心脏病和卒中风险")
    v = _score("f1", s, _pred("高血压会增加心脏病、卒中和肾病风险")).value
    assert 0.5 < v < 1.0   # strong char overlap, not identical


def test_rouge_l_orders_and_caps():
    s = _sample("a b c d e f")
    sc_full = _score("rouge", s, _pred("a b c d e f"))
    assert sc_full.value == 1.0 and sc_full.detail["rouge2"] == 1.0
    sc_part = _score("rouge", s, _pred("a c e"))
    assert 0.0 < sc_part.value < 1.0


def test_prescription_match_dict_reference():
    raw = {"治法": "辛温解表", "方剂": "荆防败毒散",
           "药物组成": "荆芥、防风、茯苓、羌活、独活、柴胡、甘草"}
    s = _sample("方剂: 荆防败毒散", raw=raw, task=TaskType.PRESCRIPTION)
    # exact herbs (with a dosage that must be stripped) + the formula name
    sc = _score("prescription_match", s, _pred("用荆防败毒散。药物组成：荆芥10g、防风、茯苓、羌活、独活、柴胡、甘草"))
    assert sc.value == 1.0                       # all 7 herbs recovered
    assert sc.detail["formula_match"] == 1.0
    assert "荆芥" in sc.detail["matched_herbs"]
    # missing herbs -> recall drops
    sc2 = _score("prescription_match", s, _pred("药物组成：荆芥、防风"))
    assert sc2.detail["herb_recall"] < 1.0 and sc2.detail["herb_precision"] == 1.0


def test_prescription_match_stringified_list_reference():
    # TCM-PR style: reference is a stringified python list of herbs
    s = _sample("['玄参', '麦冬', '生地']", task=TaskType.PRESCRIPTION)
    sc = _score("prescription_match", s, _pred("处方：玄参、麦冬、生地、甘草"))
    assert sc.detail["herb_recall"] == 1.0       # all 3 reference herbs present
    assert sc.detail["n_ref_herbs"] == 3


def test_numeric_match_tolerance_and_range():
    # ±5% relative tolerance (default)
    s = _sample("100")
    assert _score("numeric_match", s, _pred("The result is 103.")).value == 1.0
    assert _score("numeric_match", s, _pred("The result is 130.")).value == 0.0
    # explicit [lower, upper] range (MedCalc-Bench)
    s2 = Sample(id="r", task_type=TaskType.OPEN_QA, messages=[Message("u", "q")],
                reference={"lower_limit": "23.97", "upper_limit": "26.50"})
    assert _score("numeric_match", s2, _pred("...creatinine clearance is 25.2 mL/min.")).value == 1.0
    assert _score("numeric_match", s2, _pred("answer = 30")).value == 0.0
    # commas + "answer is" marker
    assert _score("numeric_match", _sample("1500"), _pred("Total is 1,500 mL")).value == 1.0


def test_bleu_orders_and_smoothing():
    s = _sample("a b c d e f g h")
    full = _score("bleu", s, _pred("a b c d e f g h"))
    assert full.value > 0.99 and full.detail["bleu1"] > 0.99
    # partial overlap, no 4-gram match -> smoothed (small but > 0)
    part = _score("bleu", s, _pred("a c e g"))
    assert 0.0 < part.value < full.value


def test_syndrome_chain_tongbing_yizhi_partial_credit():
    ref = {"syndrome": "热伤阳络;血热妄行",
           "pathogenesis": "热伤肺络，血热妄行",
           "reference": "辨证：热伤阳络，血热妄行"}
    s = Sample(id="c", task_type=TaskType.SDT, messages=[Message("user", "q")], reference=ref)
    one = _score("syndrome_chain", s, _pred("证型：血热妄行。病机为热伤肺络，血热妄行。"))
    both = _score("syndrome_chain", s, _pred("辨证：热伤阳络，血热妄行。病机：热伤肺络血热妄行。"))
    assert one.detail["syndrome_recall"] == 0.5       # 1 of 2 acceptable syndromes
    assert both.detail["syndrome_recall"] == 1.0
    assert both.value > one.value                      # more chain coverage scores higher
    # no gold at all -> excluded (None), never a hard 0 against the model
    empty = _score("syndrome_chain", Sample(id="e", task_type=TaskType.SDT,
                   messages=[Message("user", "q")], reference={}), _pred("anything"))
    assert empty.value is None and empty.detail["skipped"] == "no_gold"
    agg = create_metric("syndrome_chain").aggregate([both, empty])
    assert agg["chain_score"] == both.value            # skipped sample excluded from mean
    assert agg["n"] == 2 and agg["n_scored"] == 1 and agg["skipped_no_gold"] == 1


def test_meridian_acupoint_extraction_and_aliases():
    s = _sample("本病循足阳明胃经，主穴取足三里、合谷。")
    sc = _score("meridian_acupoint", s, _pred("取足阳明胃经的足三里，配三阴交。"))
    assert sc.detail["meridian_f1"] == 1.0                  # 胃经 alias matched
    assert sc.detail["matched_acupoints"] == ["足三里"]     # 合谷 missed, 三阴交 extra
    assert 0.0 < sc.detail["acupoint_f1"] < 1.0
    # explicit gold lists + alias normalization (胃经 -> 足阳明胃经)
    s2 = Sample(id="x", task_type=TaskType.OPEN_QA, messages=[Message("user", "q")],
                reference={"meridians": ["胃经"], "acupoints": ["足三里"]})
    assert _score("meridian_acupoint", s2, _pred("取足三里，属胃经。")).value == 1.0


def test_classics_ontology_longest_match_and_recall():
    s = _sample("此方出自《伤寒论》，理论本于《黄帝内经》。")
    sc = _score("classics_ontology", s, _pred("该方见于《伤寒论》，并参《金匮要略》。"))
    assert set(sc.detail["gold_sources"]) == {"伤寒论", "黄帝内经"}   # 内经 not double-counted
    assert sc.detail["source_f1"] == 0.5 and sc.detail["all_sources_cited"] == 0.0
    # longest-match: 伤寒杂病论 must NOT also register as 伤寒论
    s2 = _sample("《伤寒杂病论》")
    sc2 = _score("classics_ontology", s2, _pred("《伤寒杂病论》为张仲景所著。"))
    assert sc2.detail["gold_sources"] == ["伤寒杂病论"] and sc2.value == 1.0


def test_metric_specs_parsing():
    # plain names, dict form, and a mix all normalize to (name, config)
    assert _parse_metric_specs(["mcq_accuracy"]) == [("mcq_accuracy", {})]
    assert _parse_metric_specs([{"name": "llm_judge", "per_criterion": True}]) == \
        [("llm_judge", {"per_criterion": True})]
    mixed = _parse_metric_specs(["bleu", {"name": "llm_judge", "per_criterion": True}])
    assert mixed == [("bleu", {}), ("llm_judge", {"per_criterion": True})]
    try:
        _parse_metric_specs([{"per_criterion": True}])   # missing name
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_healthbench_per_criterion_signed_points():
    # Faithful HealthBench: one judge call per item, score = Σ(signed met pts) /
    # Σ(positive pts). The mock judge marks every criterion "met", so a +10 and a
    # -5 item give achieved = 10 + (-5) = 5 over possible = 10 -> 0.5.
    rubric = [{"id": "good", "points": 10, "criterion": "covers the key point"},
              {"id": "bad", "points": -5, "criterion": "contains a dangerous error"}]
    s = Sample(id="s", task_type=TaskType.OPEN_QA,
               messages=[Message("user", "What should I do?")],
               reference={"rubric": rubric})
    m = create_metric("llm_judge", {"per_criterion": True})
    assert m.per_criterion is True
    m.judge = MockProvider({"id": "mock-judge", "behavior": "auto"})
    sc = asyncio.run(m.score(s, _pred("A safe, complete answer.")))
    assert sc.detail["style"] == "healthbench"
    assert sc.detail["achieved"] == 5.0 and sc.detail["possible"] == 10.0
    assert abs(sc.value - 0.5) < 1e-9
    # aggregate clips the dataset mean into [0, 1]
    assert 0.0 <= m.aggregate([sc])["judge_score"] <= 1.0


class _StubJudge:
    """Judge stub that replays a fixed sequence of raw responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.id = "stub-judge"
        self.calls = 0

    async def agenerate(self, messages, **gen):
        text = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return Generation(text=text)


def _judged(judge_responses, reference=None, config=None):
    m = create_metric("llm_judge", config or {})
    m.judge = _StubJudge(judge_responses)
    s = Sample(id="s", task_type=TaskType.OPEN_QA,
               messages=[Message("user", "q")], reference=reference or {})
    return m, asyncio.run(m.score(s, _pred("an answer"))), m.judge


def test_llm_judge_failure_excluded_not_zero():
    # unparseable judge output (even after the retry) -> value None + judge_failed,
    # and the aggregate excludes it instead of folding a 0 into the mean.
    m, bad, judge = _judged(["I cannot grade this.", "still not json"])
    assert bad.value is None and bad.detail["judge_failed"] is True
    assert judge.calls == 2                            # one retry before giving up
    good = Score(metric="llm_judge", value=0.8, detail={})
    agg = m.aggregate([good, bad])
    assert agg["judge_score"] == 0.8                   # failure did NOT drag the mean
    assert agg["n"] == 2 and agg["n_scored"] == 1 and agg["judge_failures"] == 1


def test_llm_judge_retry_recovers():
    # a transiently-garbled first response is retried and scored normally
    ok_json = '{"scores": {"accuracy": 1.0, "completeness": 1.0, "safety": 1.0}}'
    m, sc, judge = _judged(["garbage", ok_json])
    assert judge.calls == 2 and sc.value == 1.0 and not sc.detail.get("judge_failed")


def test_llm_judge_partial_key_match_excludes_unmatched():
    # judge covers 1 of 3 default criteria and adds an extra key: the matched
    # criterion scores; the 2 ungraded ones are excluded (not coerced to 0).
    m, sc, _ = _judged(['{"scores": {"accuracy": 1.0, "overall": 0.9}}'])
    assert sc.value == 1.0
    assert sc.detail["keys_matched"] == 1
    assert sorted(sc.detail["unmatched_criteria"]) == ["completeness", "safety"]


def test_llm_judge_normalized_key_match():
    # keys matched case/whitespace-insensitively (" Accuracy " -> accuracy)
    m, sc, _ = _judged(['{"scores": {" Accuracy ": 1.0, "COMPLETENESS": 0.5, '
                        '"Safety": 1.0}}'])
    assert sc.detail["keys_matched"] == 3
    assert abs(sc.value - (2.5 / 3)) < 1e-9


def test_llm_judge_total_key_mismatch_is_failure_not_zero():
    # keys match nothing and the count differs (positional fallback impossible):
    # the sample is judge-failed/excluded, not silently 0.
    m, sc, _ = _judged(['{"scores": {"x": 1.0, "y": 1.0, "z": 1.0, "w": 1.0}}'])
    assert sc.value is None and sc.detail["judge_failed"] is True


def test_llm_judge_healthbench_failed_criterion_excluded():
    # per-criterion path: criterion 1 grades met, criterion 2 is unparseable
    # (twice) -> excluded from achieved AND possible, so value = 1/1, not 1/2.
    rubric = [{"id": "a", "points": 1, "criterion": "covers A"},
              {"id": "b", "points": 1, "criterion": "covers B"}]
    m, sc, judge = _judged(
        ['{"explanation": "ok", "criteria_met": true}', "not json", "not json"],
        reference={"rubric": rubric}, config={"per_criterion": True})
    assert judge.calls == 3
    assert sc.value == 1.0 and sc.detail["failed_criteria"] == ["b"]
    # all criteria failing -> the whole sample is judge-failed (excluded)
    m2, sc2, _ = _judged(["no", "no", "no", "no"],
                         reference={"rubric": rubric}, config={"per_criterion": True})
    assert sc2.value is None and sc2.detail["judge_failed"] is True


def test_no_gold_excluded_across_structured_metrics():
    empty = Sample(id="e", task_type=TaskType.OPEN_QA,
                   messages=[Message("user", "q")], reference={})
    for name in ("meridian_acupoint", "tongue_pulse", "classics_ontology",
                 "prescription_match"):
        sc = _score(name, empty, _pred("足三里 舌红苔黄 脉弦 《伤寒论》 荆芥防风"))
        assert sc.value is None and sc.detail["skipped"] == "no_gold", name
        m = create_metric(name)
        real = asyncio.run(m.score(_sample("取足三里。舌红苔黄，脉弦。出自《伤寒论》。"
                                           "药物组成：荆芥、防风"), _pred("取足三里，舌红苔黄，脉弦，"
                                           "见《伤寒论》。药物组成：荆芥、防风")))
        agg = m.aggregate([real, sc])
        assert agg["n"] == 2 and agg["n_scored"] == 1 and agg["skipped_no_gold"] == 1, name
        head = next(iter(agg.values()))
        assert head == real.value, name                 # mean over scored only


def test_no_gold_component_does_not_dilute_sub_f1():
    # sample 1 has only meridian gold; its (empty) acupoint side must not drag
    # acupoint_f1 down in the aggregate.
    m = create_metric("meridian_acupoint")
    s1 = _sample("循胃经。")
    sc1 = asyncio.run(m.score(s1, _pred("足阳明胃经")))
    s2 = _sample("主穴取足三里。")
    sc2 = asyncio.run(m.score(s2, _pred("取足三里")))
    agg = m.aggregate([sc1, sc2])
    assert agg["meridian_f1"] == 1.0 and agg["acupoint_f1"] == 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("OK: all metric tests passed")
