"""Offline unit tests for the adapter edge cases the research surfaced.

No network: synthetic rows / tiny fixtures exercise the tricky normalization
(multi-answer letters, inline options, CMB null slots, dict-of-lists, CSEDB
nested explosion, signed-points rubric, AgentClinic scripted env).
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from medeval import Generation, Message, Prediction, Sample, TaskType, create_provider, create_metric
from medeval.datasets.hf_mcq import HFMCQAdapter
from medeval.datasets.local_json import LocalJSONAdapter
from medeval.datasets.agent_env import AgentClinicEnv


def _mcq(answer_format="letter", options="o", inline=False):
    return HFMCQAdapter({"id": "t", "path": "x", "answer_format": answer_format,
                         "options_inline": inline,
                         "field_map": {"question": "q", "options": options, "answer": "a"}})


def test_resolve_options_dict_drops_null_slots():
    """CMB-style dict options with F: null must be dropped (not become 'None')."""
    ad = _mcq()
    row = {"option": {"A": "x", "B": "y", "C": "z", "D": "w", "E": "v", "F": None}}
    ad.fm["options"] = "option"
    choices, keys = ad._resolve_options(row)
    assert choices == ["x", "y", "z", "w", "v"]
    assert keys == ["A", "B", "C", "D", "E"]


def test_resolve_options_inline_string():
    """CMExam-style single packed Options column -> list of choices."""
    ad = _mcq(options="Options", inline=True)
    ad.fm["options"] = "Options"
    choices, keys = ad._resolve_options({"Options": "A 市容监督机关\nB 城市规划部门\nC 卫生部门"})
    assert choices == ["市容监督机关", "城市规划部门", "卫生部门"]
    assert keys == ["A", "B", "C"]


def test_resolve_options_list_of_keyvalue():
    """fzkuji/CMExam mirror stores options as [{key, value}]."""
    ad = _mcq(options="opt")
    ad.fm["options"] = "opt"
    choices, keys = ad._resolve_options({"opt": [{"key": "A", "value": "foo"}, {"key": "B", "value": "bar"}]})
    assert choices == ["foo", "bar"] and keys == ["A", "B"]


def test_multi_answer_letter_mapping_and_parse():
    ad = _mcq(answer_format="multi")
    ad.fm["options"] = "option"
    s = ad._row_to_sample({"q": "Q", "option": {"A": "a", "B": "b", "C": "c", "D": "d", "E": "e"}, "a": "BCDE"}, None, 0)
    assert s.reference["indices"] == [1, 2, 3, 4]
    # model outputs in several shapes -> all should recover {1,2,3,4}
    for out in ("BCDE", "答案：B、C、D、E", "I pick B, C, D and E."):
        assert ad.parse(s, out).parsed == [1, 2, 3, 4]
    # must NOT pick up the 'A' inside prose words
    assert ad.parse(s, "Answers Are B and C").parsed == [1, 2]


def test_text_answer_with_injected_options():
    """PubMedQA: no options column, gold is the text 'yes'/'no'/'maybe'."""
    ad = HFMCQAdapter({"id": "p", "path": "x", "answer_format": "text",
                       "inject_options": ["yes", "no", "maybe"],
                       "field_map": {"question": "q", "options": None, "answer": "final_decision"}})
    s = ad._row_to_sample({"q": "Does X cause Y?", "final_decision": "maybe"}, None, 0)
    assert s.reference == {"index": 2, "letter": "C", "text": "maybe"}


def test_local_json_dict_of_lists_flatten():
    """LLMEval-Med ships a dict keyed by category, not a flat array."""
    data = {"医疗知识": [{"problem": "q1", "sanswer": "a1"}],
            "其他": [{"problem": "q2", "sanswer": "a2"}]}
    with tempfile.TemporaryDirectory() as d:
        fp = Path(d) / "x.json"
        fp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        ad = LocalJSONAdapter({"id": "l", "adapter": "local_json", "task": "open_qa",
                               "path": str(fp), "field_map": {"prompt": "problem", "reference": "sanswer"}})
        samples = ad.load()
    assert len(samples) == 2
    assert samples[0].messages[-1].content == "q1"


def test_local_json_nested_explosion_and_rubric_points():
    """CSEDB-style: explode a nested list, resolve a space-containing key, and
    map 规则内容/分数 into rubric criteria + points."""
    rec = {"考点": {"门类": "安全门"},
           "设计的考题内容": {"最具代表性的测试case": [
               {"输入 case": "case A", "规则判断列表": [{"规则内容": "rule1", "分数": 5},
                                                       {"规则内容": "rule2", "分数": 3}]},
               {"输入 case": "case B", "规则判断列表": [{"规则内容": "rule3", "分数": 2}]},
           ]}}
    with tempfile.TemporaryDirectory() as d:
        fp = Path(d) / "csedb.json"
        fp.write_text(json.dumps([rec], ensure_ascii=False), encoding="utf-8")
        ad = LocalJSONAdapter({"id": "c", "adapter": "local_json", "task": "safety",
                               "path": str(fp), "explode": "设计的考题内容.最具代表性的测试case",
                               "field_map": {"prompt": "输入 case", "rubric": "规则判断列表", "label": "考点.门类"}})
        samples = ad.load()
    assert len(samples) == 2                          # exploded into 2 cases
    assert samples[0].messages[-1].content == "case A"
    assert samples[0].reference["label"] == "安全门"   # resolved from the parent record
    rub = samples[0].reference["rubric"]
    assert [r["points"] for r in rub] == [5.0, 3.0]
    assert rub[0]["criterion"] == "rule1"


def test_llm_judge_signed_points_penalizes():
    """HealthBench rubrics carry negative points; 'meeting' them lowers the score."""
    judge = create_provider({"id": "j", "type": "mock", "behavior": "auto"})
    metric = create_metric("llm_judge")
    metric.judge = judge
    s = Sample(id="s", task_type=TaskType.OPEN_QA, messages=[Message("user", "Q")],
               reference={"rubric": [{"id": "good", "points": 5, "criterion": "helpful"},
                                     {"id": "bad", "points": -5, "criterion": "contains an unsafe claim"}]})
    pred = Prediction(sample_id="s", generation=Generation(text="answer"), parsed="answer")
    score = asyncio.run(metric.score(s, pred))
    # mock awards every criterion -> achieved=5-5=0 over possible=5 -> 0.0
    assert score.value == 0.0


def test_agentclinic_scripted_env():
    """The scripted patient / measurement / moderator work with no LLM calls."""
    osce = {"OSCE_Examination": {
        "Objective_for_Doctor": "Diagnose the patient.",
        "Patient_Actor": {"History": "1 month of fatigable weakness.",
                          "Symptoms": {"Primary_Symptom": "double vision", "Secondary_Symptoms": ["ptosis"]}},
        "Test_Results": {"Blood_Tests": {"AChR_Antibodies": "elevated"}},
        "Correct_Diagnosis": "Myasthenia gravis"}}
    env = AgentClinicEnv(osce, max_turns=10)

    async def go():
        obs = await env.reset()
        assert "Diagnose" in obs
        o, *_ = await env.step("What symptoms do you have?")
        assert "weakness" in o or "vision" in o
        o, *_ = await env.step("REQUEST TEST: Blood")
        assert "RESULTS" in o and "elevated" in o
        _, r_ok, done, info = await env.step("DIAGNOSIS READY: Myasthenia gravis")
        assert done and r_ok == 1.0 and info["success"]
        env2 = AgentClinicEnv(osce); await env2.reset()
        _, r_bad, _, _ = await env2.step("DIAGNOSIS READY: a cold")
        assert r_bad == 0.0

    asyncio.run(go())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("OK: all adapter tests passed")
