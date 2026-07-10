"""Pairwise model comparison (medeval compare): paired bootstrap CI + McNemar."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from medeval.compare import compare, render_compare


def _write_detail(root: Path, model: str, dataset: str, rows: list[dict]) -> None:
    fp = root / f"detail__{model}__{dataset}.jsonl"
    fp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def _mcq_rows(values: list[int]) -> list[dict]:
    return [{"sample_id": f"ds:{i}", "task": "mcq",
             "scores": {"mcq_accuracy": {"value": float(v), "detail": {}}}}
            for i, v in enumerate(values)]


def test_compare_identical_models_zero_diff():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        rows = _mcq_rows([1, 0, 1, 1, 0, 1, 1, 0, 1, 1])
        _write_detail(root, "m1", "ds", rows)
        _write_detail(root, "m2", "ds", rows)
        r = compare(root, "ds", "m1", "m2", metric="mcq_accuracy")
        assert r["n"] == 10
        assert r["diff"] == 0.0
        assert r["diff_ci95"] == [0.0, 0.0]
        assert "mcnemar" in r
        assert r["mcnemar"]["method"] == "degenerate"


def test_compare_clear_winner_has_positive_diff():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_detail(root, "good", "ds", _mcq_rows([1] * 20))
        _write_detail(root, "bad", "ds", _mcq_rows([0] * 20))
        r = compare(root, "ds", "good", "bad", metric="mcq_accuracy")
        assert r["diff"] == 1.0
        lo, hi = r["diff_ci95"]
        assert lo > 0  # CI excludes 0 -> significant


def test_compare_aligns_by_sample_id_not_position():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        rows_a = [{"sample_id": "ds:0", "scores": {"mcq_accuracy": {"value": 1.0}}},
                  {"sample_id": "ds:1", "scores": {"mcq_accuracy": {"value": 0.0}}},
                  {"sample_id": "ds:2", "scores": {"mcq_accuracy": {"value": 1.0}}}]
        # model b is missing sample 1 (e.g. it failed to load) but has an extra id
        rows_b = [{"sample_id": "ds:0", "scores": {"mcq_accuracy": {"value": 1.0}}},
                  {"sample_id": "ds:2", "scores": {"mcq_accuracy": {"value": 0.0}}},
                  {"sample_id": "ds:3", "scores": {"mcq_accuracy": {"value": 1.0}}}]
        _write_detail(root, "a", "ds", rows_a)
        _write_detail(root, "b", "ds", rows_b)
        r = compare(root, "ds", "a", "b", metric="mcq_accuracy")
        assert r["n"] == 2  # only ds:0 and ds:2 are shared


def test_compare_auto_picks_metric():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        rows = _mcq_rows([1, 1, 0, 1])
        _write_detail(root, "m1", "ds", rows)
        _write_detail(root, "m2", "ds", rows)
        r = compare(root, "ds", "m1", "m2")  # no metric passed
        assert r["metric"] == "mcq_accuracy"


def test_compare_missing_model_raises():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_detail(root, "m1", "ds", _mcq_rows([1, 0]))
        try:
            compare(root, "ds", "m1", "ghost", metric="mcq_accuracy")
            assert False, "expected ValueError"
        except ValueError as e:
            assert "ghost" in str(e)


def test_render_compare_flags_significance():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_detail(root, "good", "ds", _mcq_rows([1] * 20))
        _write_detail(root, "bad", "ds", _mcq_rows([0] * 20))
        r = compare(root, "ds", "good", "bad", metric="mcq_accuracy")
        text = render_compare(r)
        assert "good" in text and "bad" in text
        assert "significant" in text


if __name__ == "__main__":
    test_compare_identical_models_zero_diff()
    test_compare_clear_winner_has_positive_diff()
    test_compare_aligns_by_sample_id_not_position()
    test_compare_auto_picks_metric()
    test_compare_missing_model_raises()
    test_render_compare_flags_significance()
    print("OK: compare CLI tests passed")
