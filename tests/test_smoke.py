"""End-to-end offline smoke test (no network / keys / GPU).

Runs the mock provider through local_json (open_qa / safety / sdt) and the agent
demo (pass^k), and checks the schema/registry/parse/judge/aggregate plumbing.
Run with:  python -m pytest tests/test_smoke.py  (or)  python tests/test_smoke.py
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import yaml

import medeval
from medeval import Message, Sample, TaskType, create_metric, create_provider
from medeval.datasets.hf_mcq import HFMCQAdapter


ROOT = Path(__file__).resolve().parents[1]


def test_registries_populated():
    assert {"hf", "poe", "litellm", "mock"} <= set(medeval.available_providers())
    assert {"hf_mcq", "local_json", "agent_demo", "agentclinic"} <= set(medeval.available_adapters())
    assert {"mcq_accuracy", "pass_k", "llm_judge"} <= set(medeval.available_metrics())


def test_mcq_answer_parsing():
    """The robust two-level extractor handles letters, 'answer is', and text."""
    ad = HFMCQAdapter({"id": "t", "path": "x", "field_map": {"question": "q", "options": "o", "answer": "a"}})
    s = Sample(id="s1", task_type=TaskType.MCQ,
               messages=[Message("user", "q")],
               choices=["Hyperplasia", "Hypertrophy", "Atrophy", "Dysplasia"],
               reference={"index": 2})
    assert ad.parse(s, "The answer is C.").parsed == 2
    assert ad.parse(s, "(C)").parsed == 2
    assert ad.parse(s, "I believe the correct option is Atrophy").parsed == 2
    assert ad.parse(s, "Clearly D, no wait, the answer is C").parsed == 2


def test_mock_provider_judge_and_mcq():
    prov = create_provider({"id": "m", "type": "mock", "behavior": "auto"})

    async def go():
        mcq = await prov.agenerate([Message("user",
            "Q?\nA. one\nB. two\nAnswer with the letter of the correct option.")])
        assert any(L in mcq.text for L in ("A", "B"))
        judge = await prov.agenerate([Message("user",
            "Grade this. Return JSON. - (id=foo, points=1) crit\n- (id=bar, points=2) crit2")])
        assert '"scores"' in judge.text and "foo" in judge.text

    asyncio.run(go())


def test_llm_judge_default_rubric_runs():
    """llm_judge with the mock judge yields a normalized [0,1] score."""
    judge = create_provider({"id": "j", "type": "mock", "behavior": "auto"})
    metric = create_metric("llm_judge")
    metric.judge = judge
    s = Sample(id="s", task_type=TaskType.SAFETY,
               messages=[Message("user", "Is this safe?")],
               reference={"reference": "A safe answer."})
    from medeval import Prediction, Generation
    pred = Prediction(sample_id="s", generation=Generation(text="A careful, safe response."), parsed="...")
    score = asyncio.run(metric.score(s, pred))
    assert 0.0 <= score.value <= 1.0
    assert score.value == 1.0  # mock judge awards all default-rubric criteria


def test_end_to_end_smoke():
    cfg = yaml.safe_load((ROOT / "configs/example_smoke.yaml").read_text())
    cfg["run"]["output_dir"] = str(ROOT / "results/smoke_test")
    rows = medeval.run_config(cfg)
    # 1 testable model × 4 datasets (judge_only model excluded)
    assert len(rows) == 4
    by_ds = {r["dataset"]: r for r in rows}
    assert by_ds["demo_open_qa"]["metrics"]["llm_judge"]["judge_score"] == 1.0
    assert by_ds["demo_agent"]["n"] == 9
    pk = by_ds["demo_agent"]["metrics"]["pass_k"]
    assert "pass^k" in pk and "pass@1" in pk
    # leaderboard artifacts exist
    assert (ROOT / "results/smoke_test/leaderboard.md").exists()
    assert (ROOT / "results/smoke_test/leaderboard.json").exists()


if __name__ == "__main__":
    test_registries_populated()
    test_mcq_answer_parsing()
    test_mock_provider_judge_and_mcq()
    test_llm_judge_default_rubric_runs()
    test_end_to_end_smoke()
    print("OK: all smoke tests passed")
