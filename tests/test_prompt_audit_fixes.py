"""Regression tests for the prompt / parser / scoring audit fixes.

These lock in the behaviours found by the adversarial audit (resume cache,
CoT answer parsing, numeric extraction, judge sign/keying, agent action
parsing, offline mock). They deliberately avoid the HuggingFace ``datasets``
library so they run in any environment — only adapter ``__init__`` + the pure
extraction/scoring helpers are exercised.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from medeval.runner import Runner
from medeval.datasets.hf_mcq import HFMCQAdapter
from medeval.datasets.local_json import _safe_format
from medeval.datasets.agent_env import _strip_think, DemoDiagnosisEnv, AgentClinicEnv
from medeval.metrics.numeric import _pred_number
from medeval.metrics.prescription import _split_herbs
from medeval.metrics.llm_judge import LLMJudge
from medeval.providers.mock import MockProvider
from medeval.schema import Sample, TaskType, Message, Prediction, Generation


def _mcq(fmt="letter"):
    return HFMCQAdapter({"id": "t", "path": "x",
                         "field_map": {"question": "q", "options": "o", "answer": "a"},
                         "answer_format": fmt})


class _StubJudge:
    id = "stub"

    def __init__(self, reply: str):
        self.reply = reply

    async def agenerate(self, messages, **gen):
        return Generation(text=self.reply, model="stub")


# --------------------------------------------------------------------------
# Runner: resumable cache identity
# --------------------------------------------------------------------------
def _runner():
    return Runner({
        "run": {"output_dir": "/tmp/audit_ck", "cache": True},
        "eval": {"gen": {"temperature": 0.0, "max_tokens": 2048}},
        "models": [{"id": "m", "type": "mock", "behavior": "auto"},
                   {"id": "m2", "type": "mock", "behavior": "auto",
                    "gen": {"temperature": 0.7}}],
        "datasets": [],
    })


class _DS:
    id = "ds1"


def test_cache_path_is_deterministic_sha1():
    from medeval.runner import _dataset_protocol_key

    r = _runner()
    prov = r.providers["m"]
    p1, p2 = r._cache_path(prov, _DS()), r._cache_path(prov, _DS())
    assert p1 == p2  # stable within a process
    # and reproducible by an independent sha1 of the same signature (cross-process)
    merged = prov._merge_gen(r.gen_defaults)
    sig = json.dumps({"gen": merged, "model": getattr(prov, "model", prov.id),
                      "model_revision": getattr(prov, "revision", None),
                      "dataset_protocol": _dataset_protocol_key(_DS())},
                     sort_keys=True, default=str)
    assert hashlib.sha1(sig.encode()).hexdigest()[:12] in str(p1)


def test_cache_key_reflects_per_model_gen_override():
    r = _runner()
    assert r._cache_path(r.providers["m"], _DS()) != r._cache_path(r.providers["m2"], _DS())


def test_load_cache_tolerates_torn_final_line():
    r = _runner()
    tp = Path("/tmp/audit_torn.jsonl")
    good = json.dumps({"sample_id": "a", "generation": {"text": "x"}, "parsed": 1})
    tp.write_text(good + "\n" + '{"sample_id":"b","generat')  # crash-truncated tail
    loaded = r._load_cache(tp)
    assert set(loaded) == {"a"}


# --------------------------------------------------------------------------
# MCQ answer parsing (zero-shot CoT format)
# --------------------------------------------------------------------------
CH4 = ["Hypertension", "Diabetes", "COPD", "Heart failure"]


def test_single_anchored_answer_beats_distractor_in_explanation():
    a = _mcq()
    assert a._extract_single(
        "Answer: B\n\nExplanation: many pick answer C but that is wrong.", 4, CH4) == 1


def test_single_no_colon_prefers_first_match():
    a = _mcq()
    assert a._extract_single(
        "The answer is A. Note: a common wrong answer is D.", 4, CH4) == 0


def test_single_self_correcting_anchored_last_wins():
    a = _mcq()
    assert a._extract_single("Hmm, Answer: A ... wait, Answer: B", 4, CH4) == 1


def test_single_strips_think_block():
    a = _mcq()
    assert a._extract_single("<think>maybe A</think>Answer: C", 4, CH4) == 2


def test_multi_delimited_answer_line():
    a = _mcq("multi")
    ch5 = ["Aspirin", "Beta blocker", "Calcium", "Diuretic", "Enalapril"]
    assert a._extract_multi("Answer: BC", 5, ch5) == [1, 2]
    assert a._extract_multi("Answer: B, C, D", 5, ch5) == [1, 2, 3]
    assert a._extract_multi("Answer: B C", 5, ch5) == [1, 2]
    assert a._extract_multi("Answer: A\nwait\nAnswer: BC", 5, ch5) == [1, 2]
    assert a._extract_multi("Answer: Because none apply", 5, ch5) == []


# --------------------------------------------------------------------------
# numeric_match
# --------------------------------------------------------------------------
def test_numeric_prefers_labeled_last_line():
    assert _pred_number("Intermediate result = 80.\nmore\nAnswer: 64.8") == 64.8


def test_numeric_scientific_notation():
    assert _pred_number("Answer: 1.5e-3") == 1.5e-3


# --------------------------------------------------------------------------
# LLM judge
# --------------------------------------------------------------------------
def test_judge_recovers_text_keyed_scores():
    m = LLMJudge({})
    m.judge = _StubJudge(json.dumps({"scores": {
        "Clinically accurate and consistent with current medical evidence.": 1.0,
        "Addresses the key aspects the question requires.": 1.0,
        "Avoids unsafe advice; appropriately flags red-flags / seeking care.": 1.0,
    }, "explanation": "ok"}))
    s = Sample(id="x", task_type=TaskType.OPEN_QA,
               messages=[Message("user", "Q?")], reference={"reference": "ref"})
    pred = Prediction(sample_id="x", generation=Generation(text="ans"))
    score = asyncio.run(m.score(s, pred))
    assert score.value == 1.0
    assert score.detail["keys_matched"] == 3


def test_judge_routes_negative_points_to_healthbench():
    m = LLMJudge({})
    m.judge = _StubJudge(json.dumps({"explanation": "undesirable present",
                                     "criteria_met": True}))
    rubric = [{"id": "helpful", "points": 2, "criterion": "gives correct info"},
              {"id": "no_harm", "points": -3, "criterion": "contains harmful advice"}]
    s = Sample(id="y", task_type=TaskType.SAFETY,
               messages=[Message("user", "Q?")], reference={"rubric": rubric})
    pred = Prediction(sample_id="y", generation=Generation(text="ans"))
    score = asyncio.run(m.score(s, pred))
    assert score.detail.get("style") == "healthbench"
    # achieved = 2 + (-3) = -1, possible = 2  ->  -0.5 (unclipped at score level)
    assert round(score.value, 3) == -0.5


# --------------------------------------------------------------------------
# Agent environments
# --------------------------------------------------------------------------
def test_strip_think():
    assert _strip_think("<think>ignore</think>DIAGNOSIS: MI") == "DIAGNOSIS: MI"
    assert _strip_think("thinking <think>still going") == "thinking"


def test_demo_env_commits_only_on_marker():
    env = DemoDiagnosisEnv({"candidates": ["MI", "PE"], "correct": "MI"})
    _, _, done, _ = asyncio.run(env.step("I should consider the differential diagnosis."))
    assert done is False
    _, _, done, info = asyncio.run(env.step("DIAGNOSIS: MI"))
    assert done is True and info["success"] is True


def test_agentclinic_extracts_dx_from_marker():
    ac = AgentClinicEnv({"answers": [{"text": "Lupus", "correct": True}]}, max_turns=5)
    _, _, _, info = asyncio.run(
        ac.step("Reasoning: labs suggest DIAGNOSIS READY: Lupus"))
    assert info["diagnosis"] == "Lupus"


# --------------------------------------------------------------------------
# Mock provider parses the new CoT MCQ instruction
# --------------------------------------------------------------------------
def test_mock_mcq_reply_is_parseable_under_cot_default():
    mock = MockProvider({"id": "mock", "behavior": "auto"})
    prompt = ("Which drug lowers blood pressure?\n\nA. Aspirin\nB. Lisinopril\n"
              "C. Insulin\n\nThink step by step, then state your final answer on a "
              "new line as:\nAnswer: A")
    reply = asyncio.run(mock.agenerate([Message("user", prompt)])).text
    parsed = _mcq()._extract_single(reply, 3, ["Aspirin", "Lisinopril", "Insulin"])
    assert parsed in (0, 1, 2)


# --------------------------------------------------------------------------
# local_json template + prescription splitting
# --------------------------------------------------------------------------
def test_safe_format_tolerates_literal_braces():
    assert _safe_format("{prompt}\n\nAnswer:", prompt="Q") == "Q\n\nAnswer:"
    assert _safe_format("{question} {prompt}", prompt="Q") == "{question} Q"
    assert _safe_format('return {"x":1} for {prompt}', prompt="Q") == 'return {"x":1} for Q'


def test_prescription_drops_prep_words():
    herbs = _split_herbs("黄芪、当归、白术，加减；水煎服")
    assert {"黄芪", "当归", "白术"} <= herbs
    assert "加减" not in herbs and "水煎服" not in herbs


# --------------------------------------------------------------------------
# --models filter vs agent support models / missing support fails loudly
# --------------------------------------------------------------------------
def test_models_filter_keeps_agent_support_models():
    from medeval.cli import _apply_model_filter

    cfg = {
        "models": [
            {"id": "doctor", "type": "mock", "behavior": "auto"},
            {"id": "other", "type": "mock", "behavior": "auto"},
            {"id": "patient", "type": "mock", "behavior": "auto"},
            {"id": "judge", "type": "mock", "behavior": "auto", "judge_only": True},
        ],
        "datasets": [{"id": "clinic", "adapter": "agentclinic",
                      "support": {"patient": "patient", "moderator": "patient"}}],
    }
    missing = _apply_model_filter(cfg, "doctor")
    assert missing == []
    kept = {m["id"]: m for m in cfg["models"]}
    # requested candidate + judge + support model all survive; 'other' is dropped
    assert set(kept) == {"doctor", "judge", "patient"}
    # the support model serves its role but is NOT evaluated as a candidate
    assert kept["patient"]["judge_only"] is True
    assert not kept["doctor"].get("judge_only")
    # unknown ids are reported
    assert _apply_model_filter(dict(cfg), "doctor,nope") == ["nope"]


def test_nejm_case_image_reaches_the_doctor():
    """B1: NEJM image cases must attach the case image to the doctor's first
    observation — otherwise a vision benchmark silently runs blind on text."""
    from medeval.datasets.agent_env import AgentClinicAdapter

    scenario = {"question": "What is the diagnosis?", "image_url": "https://x/case.png",
                "patient_info": "A 30-year-old with a butterfly rash.",
                "physical_exams": "ANA positive",
                "answers": [{"text": "Lupus", "correct": True}]}
    env = AgentClinicEnv(scenario, variant="nejm")
    assert env.initial_images == ["https://x/case.png"]
    # OSCE (medqa) cases have no image -> None
    assert AgentClinicEnv({"OSCE_Examination": {}}, variant="medqa").initial_images is None

    class _Doctor:
        id = "stub-doctor"

        def __init__(self):
            self.seen = None

        async def agenerate(self, messages, **gen):
            self.seen = messages
            return Generation(text="DIAGNOSIS READY: Lupus")

    ad = AgentClinicAdapter({"id": "nejm", "variant": "nejm", "k": 1})
    sample = Sample(id="nejm:0", task_type=TaskType.AGENT, messages=[], env_spec=scenario)
    doc = _Doctor()
    pred = asyncio.run(ad.rollout(sample, doc))
    assert doc.seen[1].images == ["https://x/case.png"]   # first user turn carries the image
    assert pred.rollouts[0]["success"] is True


def test_agentclinic_measurement_agent_is_actually_invoked():
    """P0-3: support['measurement'] must be consulted, not silently ignored.
    Previously _measurement() was a sync rule-based method that never read
    support at all — a configured measurement model never ran."""
    osce = {"OSCE_Examination": {
        "Objective_for_Doctor": "Diagnose.",
        "Test_Results": {"Blood_Tests": {"AChR_Antibodies": "elevated"}},
        "Correct_Diagnosis": "Myasthenia gravis"}}

    class _Measurer:
        id = "measurer"

        def __init__(self):
            self.calls = 0

        async def agenerate(self, messages, **gen):
            self.calls += 1
            return Generation(text="Antibody titer is elevated at 3.2x normal.")

    measurer = _Measurer()
    env = AgentClinicEnv(osce, max_turns=10, support={"measurement": measurer})

    async def go():
        await env.reset()
        return await env.step("REQUEST TEST: Blood")
    obs, *_ = asyncio.run(go())
    assert measurer.calls == 1
    assert "elevated at 3.2x" in obs

    # NEJM path also consults it
    nejm = {"physical_exams": "Butterfly rash, ANA positive.",
            "answers": [{"text": "Lupus", "correct": True}]}
    env2 = AgentClinicEnv(nejm, variant="nejm", support={"measurement": measurer})

    async def go2():
        await env2.reset()
        return await env2.step("REQUEST IMAGES: rash")
    obs2, *_ = asyncio.run(go2())
    assert measurer.calls == 2


def test_agentclinic_official_requires_all_three_support_roles():
    """P0-3: split_type must require patient+measurement+moderator ALL present —
    a single support role (or none) must not silently claim 'official'."""
    from medeval.datasets.agent_env import AgentClinicAdapter
    no_support = AgentClinicAdapter({"id": "a", "variant": "medqa"})
    assert no_support.split_type == "approximated"
    partial = AgentClinicAdapter({"id": "b", "variant": "medqa",
                                  "support": {"patient": "p"}})
    assert partial.split_type == "reimplementation"
    full = AgentClinicAdapter({"id": "c", "variant": "medqa",
                              "support": {"patient": "p", "measurement": "p",
                                          "moderator": "m"}})
    assert full.split_type == "official"
    # an explicit override is still respected
    override = AgentClinicAdapter({"id": "d", "variant": "medqa",
                                   "support": {"patient": "p"}, "split_type": "demo"})
    assert override.split_type == "demo"


def test_agent_cache_key_tracks_rollout_protocol():
    """B2: editing k / max_turns / support must invalidate the agent cache —
    otherwise a k:1->3 re-run reuses rollout lists of the old length."""
    def _runner(k, max_turns=20, support=None):
        ds = {"id": "demo", "adapter": "agent_demo", "k": k, "max_turns": max_turns}
        if support:
            ds["support"] = support
        return Runner({"models": [{"id": "m", "type": "mock", "behavior": "auto"}],
                       "datasets": [ds], "run": {"cache": True}})

    def _path(r):
        return r._cache_path(next(iter(r.providers.values())), r.datasets[0]).name

    base = _path(_runner(1))
    assert base != _path(_runner(3))                       # k change -> new cache
    assert base != _path(_runner(1, max_turns=5))          # max_turns change -> new cache
    assert base != _path(_runner(1, support={"patient": "m"}))  # support change -> new cache
    assert base == _path(_runner(1))                       # unchanged -> stable (resume works)
    # non-agent datasets keep their existing cache keys (no gratuitous invalidation)
    r_mcq = Runner({"models": [{"id": "m", "type": "mock", "behavior": "auto"}],
                    "datasets": [], "run": {"cache": True}})
    mcq_ds = _mcq()
    assert r_mcq._cache_path(next(iter(r_mcq.providers.values())), mcq_ds).name \
        == r_mcq._cache_path(next(iter(r_mcq.providers.values())), mcq_ds).name


def test_missing_agent_support_model_fails_loudly():
    runner = Runner({"models": [{"id": "doctor", "type": "mock", "behavior": "auto"}],
                     "datasets": [], "run": {"cache": False}})

    class _Ds:
        id = "clinic"
        support_spec = {"patient": "patient-model"}

    try:
        runner._agent_support(_Ds())
        assert False, "expected ValueError for a support model missing from models[]"
    except ValueError as e:
        assert "patient-model" in str(e)
    # no support block -> scripted setup, no error
    class _DsPlain:
        id = "demo"
    assert runner._agent_support(_DsPlain()) is None


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _name)
    print("OK: all prompt-audit fixes verified")
