"""Preflight profiling — offline, against a tiny local fixture (no network).

Verifies the reliability stats `medeval preflight` reports: sample count vs. raw
rows seen, the option-count distribution, the answer-parse success rate (kept /
seen, with a deliberately-unparseable row dropped), and the example rows.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from medeval.preflight import find_near_duplicates, preflight, format_reports


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


# --------------------------------------------------------------------------
# Near-duplicate / contamination scan (MinHash + LSH)
# --------------------------------------------------------------------------
def test_find_near_duplicates_flags_paraphrase_level_repeat():
    items = [
        ("q1", "A 45 year old man presents with crushing chest pain radiating to the left arm."),
        ("q2", "A 45 year old man presents with crushing chest pain radiating to the left arm!"),
        ("q3", "What is the boiling point of water at sea level in degrees Celsius?"),
    ]
    pairs = find_near_duplicates(items, threshold=0.8)
    ids = {(p["a"], p["b"]) for p in pairs}
    assert ("q1", "q2") in ids
    assert not any("q3" in (p["a"], p["b"]) for p in pairs)


def test_find_near_duplicates_none_below_threshold():
    items = [("q1", "Completely unrelated question about diabetes management."),
             ("q2", "A totally different question regarding fracture healing times.")]
    assert find_near_duplicates(items, threshold=0.8) == []


def test_find_near_duplicates_ignores_empty_text():
    items = [("q1", ""), ("q2", "   "), ("q3", "Some real question text here for shingling.")]
    assert find_near_duplicates(items, threshold=0.8) == []


def test_find_near_duplicates_exact_duplicate_is_similarity_one():
    items = [("q1", "The exact same question text, word for word, repeated."),
             ("q2", "The exact same question text, word for word, repeated.")]
    pairs = find_near_duplicates(items, threshold=0.8)
    assert len(pairs) == 1
    assert pairs[0]["similarity"] == 1.0


def test_preflight_report_includes_near_duplicate_fields():
    r = preflight(_config(_fixture()), n_examples=3)[0]
    assert "near_duplicate_count" in r
    assert "near_duplicates" in r
    assert r["near_duplicate_count"] == 0   # the 4-row fixture has no duplicates


def test_preflight_dup_threshold_none_skips_scan():
    r = preflight(_config(_fixture()), n_examples=3, dup_threshold=None)[0]
    assert r["near_duplicate_count"] == 0
    assert r["near_duplicates"] == []


if __name__ == "__main__":
    test_preflight_counts_dist_and_parse_rate()
    test_preflight_limit_and_dataset_filter()
    test_format_reports_smoke()
    test_find_near_duplicates_flags_paraphrase_level_repeat()
    test_find_near_duplicates_none_below_threshold()
    test_find_near_duplicates_ignores_empty_text()
    test_find_near_duplicates_exact_duplicate_is_similarity_one()
    test_preflight_report_includes_near_duplicate_fields()
    test_preflight_dup_threshold_none_skips_scan()
    print("OK: preflight tests passed")
