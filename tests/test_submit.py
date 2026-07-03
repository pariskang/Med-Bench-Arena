"""MedBench data adapter + OpenCompass/MedBench submission export (offline)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import medeval
from medeval.submit import to_opencompass, to_medbench, answer_for_submission


# a tiny MedBench-format test set (answers held out, like the real platform)
RECORDS = [
    {"passage": None, "question": "门静脉高压最常见的病因是？",
     "options": ["A: 肝硬化", "B: 肝囊肿", "C: 胆结石", "D: 胃炎", "E: 阑尾炎"],
     "answer": None, "other": {"source": "Med-Exam", "id": 1}},
    {"passage": None, "question": "急性心梗最典型的症状是？",
     "options": ["A: 头痛", "B: 胸痛", "C: 腹泻", "D: 皮疹", "E: 耳鸣"],
     "answer": None, "other": {"source": "Med-Exam", "id": 2}},
    {"passage": None, "question": "请简述高血压的生活方式干预。",
     "options": None, "answer": None, "other": {"source": "Med-Exam", "id": 3}},
]


def _write_tree(root: Path) -> Path:
    d = root / "MedBench" / "Med-Exam"
    d.mkdir(parents=True)
    fp = d / "Med-Exam_test.jsonl"
    fp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in RECORDS), encoding="utf-8")
    return root / "MedBench"


def _run(tree: Path, results: Path):
    cfg = {
        "run": {"output_dir": str(results), "cache": False},
        "eval": {"gen": {"temperature": 0.0, "max_tokens": 64}},
        "models": [{"id": "mock-model", "type": "mock", "behavior": "auto"}],
        "datasets": [{"id": "Med-Exam", "adapter": "medbench",
                      "path": str(tree / "Med-Exam" / "Med-Exam_test.jsonl"), "metrics": []}],
    }
    medeval.run_config(cfg)


def test_medbench_adapter_loads_mcq_and_open():
    from medeval.datasets.medbench import MedBenchAdapter
    with tempfile.TemporaryDirectory() as d:
        tree = _write_tree(Path(d))
        ad = MedBenchAdapter({"id": "Med-Exam", "adapter": "medbench",
                              "path": str(tree / "Med-Exam" / "Med-Exam_test.jsonl")})
        samples = ad.load()
    assert len(samples) == 3
    mcq = samples[0]
    assert mcq.task_type.value == "mcq" and mcq.choices[0] == "肝硬化"  # "A: " stripped
    assert mcq.reference == {}                       # held-out gold
    assert samples[2].task_type.value == "open_qa"   # no options -> generation


def test_export_opencompass_and_medbench():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        tree = _write_tree(d)
        results = d / "results"
        _run(tree, results)

        # OpenCompass: dict keyed by str(idx) with origin_prompt/prediction
        oc = to_opencompass(results, d / "oc")
        ocf = oc / "predictions" / "mock-model" / "Med-Exam.json"
        assert ocf.exists()
        oc_data = json.loads(ocf.read_text())
        assert set(oc_data) == {"0", "1", "2"}
        assert "origin_prompt" in oc_data["0"] and "prediction" in oc_data["0"]
        assert oc_data["0"]["prediction"] in list("ABCDE")   # MCQ -> a letter

        # MedBench fill-in: preserve original records, fill `answer` by other.id
        mb = to_medbench(results, d / "mb", test_dir=tree)
        out = mb / "Med-Exam" / "Med-Exam.jsonl"
        assert out.exists()
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        assert len(rows) == 3
        assert rows[0]["other"] == {"source": "Med-Exam", "id": 1}   # untouched
        assert rows[0]["question"] == RECORDS[0]["question"]          # untouched
        assert rows[0]["answer"] in list("ABCDE")                     # filled
        assert isinstance(rows[2]["answer"], str) and rows[2]["answer"]  # generation text

        # MedBench synthesize mode (no test_dir): build records from details
        mb2 = to_medbench(results, d / "mb2")
        syn = json.loads((mb2 / "mock-model" / "Med-Exam.jsonl").read_text().splitlines()[0])
        assert syn["options"][0].startswith("A:") and "answer" in syn


def test_answer_for_submission_letter_and_text():
    assert answer_for_submission({"task": "mcq", "parsed": 4}) == "E"
    assert answer_for_submission({"task": "mcq", "parsed": [1, 3]}) == "BD"
    assert answer_for_submission({"task": "open_qa", "prediction": "hello"}) == "hello"


def test_load_details_merges_shard_files():
    """A sharded run writes detail__m__ds__shard{i}of{N}.jsonl; export must merge
    them into one (model, dataset) group, not fragment by shard."""
    from medeval.submit import load_details
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        def _w(name, rows):
            (d / name).write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        _w("detail__m__exam__shard0of2.jsonl",
           [{"sample_id": "exam:0", "task": "mcq", "parsed": 0},
            {"sample_id": "exam:2", "task": "mcq", "parsed": 1}])
        _w("detail__m__exam__shard1of2.jsonl",
           [{"sample_id": "exam:1", "task": "mcq", "parsed": 2},
            {"sample_id": "exam:2", "task": "mcq", "parsed": 1}])  # dup id across shards
        groups = load_details(d)
        assert set(groups) == {("m", "exam")}          # one group, suffix stripped
        rows = groups[("m", "exam")]
        assert {r["sample_id"] for r in rows} == {"exam:0", "exam:1", "exam:2"}  # deduped


if __name__ == "__main__":
    test_medbench_adapter_loads_mcq_and_open()
    test_export_opencompass_and_medbench()
    test_answer_for_submission_letter_and_text()
    test_load_details_merges_shard_files()
    print("OK: submission export tests passed")
