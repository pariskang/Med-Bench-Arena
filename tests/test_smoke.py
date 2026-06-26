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


def test_catalog_configs_valid():
    """Every dataset in the shipped catalogs parses, has a unique id, references a
    registered adapter, and carries a field_map / source — guards config typos
    (incl. the new ethics-safety + cn_tcm entries) without any network."""
    adapters = set(medeval.available_adapters())
    for name in ("catalog_ethics_safety.yaml", "catalog_cn_tcm.yaml",
                 "catalog_mcq.yaml", "example_open_safety.yaml", "example_tcm.yaml",
                 "catalog_med_models.yaml"):
        cfg = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
        ids = [d["id"] for d in cfg["datasets"]]
        assert len(ids) == len(set(ids)), f"{name}: duplicate dataset id"
        for d in cfg["datasets"]:
            assert d["adapter"] in adapters, f"{name}/{d['id']}: unknown adapter {d['adapter']!r}"
            if d["adapter"] in ("hf_mcq", "local_json"):   # config-driven adapters need a field_map
                assert "field_map" in d, f"{name}/{d['id']}: missing field_map"
            assert any(k in d for k in ("path", "data_files", "source_url", "hf")), \
                f"{name}/{d['id']}: no data source (path/data_files/source_url/hf)"


def test_med_models_catalog_wellformed():
    """The medical-model catalog's hf entries have a model id and unique ids; the
    LoRA entry (ZhongJing-2) points its `model` at a base + `lora` at the adapter."""
    cfg = yaml.safe_load((ROOT / "configs/catalog_med_models.yaml").read_text(encoding="utf-8"))
    ids = [m["id"] for m in cfg["models"]]
    assert len(ids) == len(set(ids)), "duplicate model id in catalog_med_models"
    hf = [m for m in cfg["models"] if m.get("type") == "hf"]
    assert len(hf) >= 17, "expected the full medical-model roster"
    for m in hf:
        assert m.get("model"), f"{m['id']}: hf entry needs a `model` repo id"
        assert m.get("dtype") in ("bfloat16", "float16", "auto"), f"{m['id']}: bad dtype"
    zj = next(m for m in hf if m["id"] == "zhongjing-2-1_8b")
    assert zj["lora"] == "CMLM/ZhongJing-2-1_8b" and "Qwen" in zj["model"]


def test_hf_per_model_gen_and_system_prompt():
    """A model entry can carry `gen` overrides + a `system_prompt`; the Dao entry
    uses the transformers backend with eager attention. No model is loaded."""
    from medeval.providers.hf import HFProvider
    cfg = yaml.safe_load((ROOT / "configs/catalog_med_models.yaml").read_text(encoding="utf-8"))
    dao = next(m for m in cfg["models"] if m["id"] == "dao1-30b-a3b")
    assert dao["backend"] == "transformers" and dao["attn_implementation"] == "eager"
    p = HFProvider(dao)                       # lazy: no engine/torch needed
    merged = p._merge_gen({"temperature": 0.0, "max_tokens": 2048})
    assert merged["temperature"] == 0.7 and merged["repetition_penalty"] == 1.1
    chat = p._to_chat([Message("user", "问题")])
    assert chat[0]["role"] == "system" and "小道" in chat[0]["content"]
    # a plain entry (no gen/system_prompt) is unchanged
    plain = HFProvider(next(m for m in cfg["models"] if m["id"] == "biancang"))
    assert plain._merge_gen({"temperature": 0.0}) == {"temperature": 0.0}
    assert plain._to_chat([Message("user", "x")])[0]["role"] == "user"


def test_cli_models_filter():
    """`medeval run --models <id>` keeps only that model + judge_only models."""
    from medeval.cli import main
    out = ROOT / "results/cli_filter_test"
    rc = main(["run", str(ROOT / "configs/example_smoke.yaml"),
               "--models", "mock-model", "--output", str(out), "--no-cache"])
    assert rc == 0
    rows = yaml.safe_load((out / "leaderboard.json").read_text(encoding="utf-8"))
    assert {r["model"] for r in rows} == {"mock-model"}   # judge ran but isn't ranked
    # an unknown id is a hard error (exit 2), not a silent empty run
    assert main(["run", str(ROOT / "configs/example_smoke.yaml"), "--models", "nope"]) == 2


def test_continue_on_error_skips_bad_dataset():
    """run.continue_on_error keeps the sweep alive when one dataset fails to load,
    still scoring the good ones and writing a leaderboard (the Colab full-sweep)."""
    out = ROOT / "results/coe_test"
    cfg = {
        "run": {"output_dir": str(out), "cache": False, "continue_on_error": True},
        "eval": {"gen": {"temperature": 0.0, "max_tokens": 64}, "judge_model": "mock-judge"},
        "models": [{"id": "mock-model", "type": "mock", "behavior": "auto"},
                   {"id": "mock-judge", "type": "mock", "behavior": "auto", "judge_only": True}],
        "datasets": [
            {"id": "good_qa", "adapter": "local_json", "task": "open_qa",
             "path": "data/samples/demo_open_qa.jsonl",
             "field_map": {"prompt": "question", "rubric": "rubric", "reference": "reference"},
             "metrics": ["llm_judge"]},
            {"id": "bad_ds", "adapter": "local_json", "task": "open_qa",
             "path": "/no/such/file.jsonl",
             "field_map": {"prompt": "q", "reference": "r"}, "metrics": ["llm_judge"]},
        ],
    }
    rows = medeval.run_config(cfg)
    assert [r["dataset"] for r in rows] == ["good_qa"]
    assert (out / "leaderboard.md").exists()


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
    test_catalog_configs_valid()
    test_med_models_catalog_wellformed()
    test_hf_per_model_gen_and_system_prompt()
    test_cli_models_filter()
    test_continue_on_error_skips_bad_dataset()
    test_end_to_end_smoke()
    print("OK: all smoke tests passed")
