"""Preflight profiling — offline, against a tiny local fixture (no network).

Verifies the reliability stats `medeval preflight` reports: sample count vs. raw
rows seen, the option-count distribution, the answer-parse success rate (kept /
seen, with a deliberately-unparseable row dropped), and the example rows.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from medeval.preflight import preflight, format_reports


def _fixture() -> str:
    # 4 rows, dict options + letter answer; row 2 has an out-of-range answer ("Z")
    # so it must be dropped as answer_unparsed (3 kept of 4 seen -> 0.75).
    rows = [
        {"id": "q1", "question": "Most specific marker of MI?",
         "options": {"A": "Troponin", "B": "CK-MB", "C": "AST", "D": "LDH"}, "answer_idx": "A"},
        {"id": "q2", "question": "Unparseable gold here",
         "options": {"A": "x", "B": "y", "C": "z", "D": "w"}, "answer_idx": "Z"},
        {"id": "q3", "question": "First-line for anaphylaxis?",
         "options": {"A": "Diphenhydramine", "B": "Epinephrine", "C": "Steroids", "D": "Albuterol"},
         "answer_idx": "B"},
        {"id": "q4", "question": "Vitamin deficiency in scurvy?",
         "options": {"A": "B12", "B": "D", "C": "C", "D": "K"}, "answer_idx": "C"},
    ]
    fp = Path(tempfile.mkdtemp()) / "mini_mcq.json"
    fp.write_text(json.dumps(rows), encoding="utf-8")
    return str(fp)


def _config(path: str) -> dict:
    return {"datasets": [{
        "id": "mini_mcq", "adapter": "hf_mcq", "format": "json",
        "data_files": path, "split": "train",
        "field_map": {"question": "question", "options": "options", "answer": "answer_idx"},
        "answer_format": "letter", "metrics": ["mcq_accuracy"],
    }]}


def test_preflight_counts_dist_and_parse_rate():
    reports = preflight(_config(_fixture()), n_examples=3)
    assert len(reports) == 1
    r = reports[0]
    assert "error" not in r, r
    assert r["id"] == "mini_mcq"
    assert r["rows_seen"] == 4 and r["n_samples"] == 3        # one row dropped
    assert r["option_dist"] == {4: 3}                          # all 4-option
    assert abs(r["answer_parse_rate"] - 0.75) < 1e-9           # 3 / 4
    assert r["dropped"] == {"answer_unparsed": 1}
    # examples: first 3 KEPT samples (the unparseable q2 is skipped)
    assert len(r["examples"]) == 3
    ex_ids = [e["id"].split(":")[-1] for e in r["examples"]]
    assert ex_ids == ["q1", "q3", "q4"]
    first = r["examples"][0]
    assert first["n_options"] == 4 and first["gold"] == "A"
    assert "Troponin" in first["choices"]


def test_preflight_limit_and_dataset_filter():
    cfg = _config(_fixture())
    # filter to a non-existent id -> empty
    assert preflight(cfg, dataset_ids=["nope"]) == []
    # limit caps the rows seen
    r = preflight(cfg, dataset_ids=["mini_mcq"], limit=2)[0]
    assert r["n_samples"] <= 2


def test_format_reports_smoke():
    text = format_reports(preflight(_config(_fixture())))
    assert "mini_mcq" in text
    assert "answer parse" in text
    assert "PREFLIGHT:" in text


if __name__ == "__main__":
    test_preflight_counts_dist_and_parse_rate()
    test_preflight_limit_and_dataset_filter()
    test_format_reports_smoke()
    print("OK: preflight tests passed")
