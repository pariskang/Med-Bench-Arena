"""Unit tests for the F1 / ROUGE / 方剂结构匹配 metrics (offline)."""
from __future__ import annotations

import asyncio

from medeval import Generation, Message, Prediction, Sample, TaskType, create_metric


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
    # no gold at all -> 0
    empty = _score("syndrome_chain", Sample(id="e", task_type=TaskType.SDT,
                   messages=[Message("user", "q")], reference={}), _pred("anything"))
    assert empty.value == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("OK: all metric tests passed")
