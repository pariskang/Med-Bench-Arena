"""Judge-calibration harness: metric math, verdict logic, the frozen-set result,
the deterministic sampler, and the leaderboard auxiliary-metric policy.

Plain-assert style (no pytest) to match the repo's `python tests/test_x.py` runner.
"""
import json
import math
import tempfile
from pathlib import Path

from medeval.calibrate import (
    _pairwise_table, balanced_f1, cohen_kappa, raw_agreement,
    CalItem, rater_pairs, physician_pairs, evaluate, load_label_preds,
    load_calibration_set, build_meta_set,
)

ROOT = Path(__file__).resolve().parent.parent
ITEMS = ROOT / "data/calibration/healthbench_meta_items.jsonl"
GOLD = ROOT / "data/calibration/healthbench_meta_gold.jsonl"
LABELS = ROOT / "data/calibration/healthbench_meta_strongmodel_labels.jsonl"


def _approx(a, b, tol=1e-5):
    assert abs(a - b) < tol, f"{a} != {b} (tol {tol})"


# --- metric math (hand-computed reference) ---------------------------------
def test_metric_math_hand_example():
    pairs = [(True, True)] * 3 + [(True, False)] + [(False, False)] * 2
    t = _pairwise_table(pairs)
    assert t == {"tt": 3, "tf": 1, "ft": 0, "ff": 2, "n": 6}
    _approx(raw_agreement(t), 5 / 6)
    _approx(cohen_kappa(t), 2 / 3)
    m = balanced_f1(t)
    _approx(m["f1_pos"], 0.8571428)
    _approx(m["f1_neg"], 0.8)
    _approx(m["f1_balanced"], (0.8571428 + 0.8) / 2)


def test_perfect_and_random_kappa():
    t = _pairwise_table([(True, True)] * 5 + [(False, False)] * 5)
    _approx(cohen_kappa(t), 1.0)
    _approx(balanced_f1(t)["f1_balanced"], 1.0)
    t2 = _pairwise_table([(True, False)] * 4 + [(False, True)] * 4)
    assert cohen_kappa(t2) < 0


def test_balanced_f1_single_class_is_nan_safe():
    t = _pairwise_table([(True, True)] * 4)
    m = balanced_f1(t)
    _approx(m["f1_pos"], 1.0)
    assert math.isnan(m["f1_neg"])
    _approx(m["f1_balanced"], 1.0)  # averages over present classes only


# --- pairing helpers --------------------------------------------------------
def test_rater_and_physician_pairs():
    items = [
        CalItem("a", [], "", "", "c1", physician_labels=[True, True]),
        CalItem("b", [], "", "", "c1", physician_labels=[True, False]),
    ]
    preds = {"a": True, "b": False}
    assert sorted(rater_pairs(items, preds)) == sorted(
        [(True, True), (True, True), (False, True), (False, False)])
    pp = physician_pairs(items)
    assert pp.count((True, False)) == 1 and pp.count((False, True)) == 1


# --- verdict logic ----------------------------------------------------------
def _items(labels_list):
    return [CalItem(f"i{k}", [], "", "", "cat", physician_labels=list(lb))
            for k, lb in enumerate(labels_list)]


def test_verdict_structure_and_consistency():
    items = _items([[True, True]] * 60 + [[False, False]] * 30 + [[True, False]] * 30)
    preds = {it.item_id: (sum(it.physician_labels) >= 1) for it in items}
    v = evaluate(items, preds, "rater")["verdict"]
    assert {"calibrated", "physician_equivalent", "substantial_absolute"} <= set(v)
    assert v["calibrated"] == (v["physician_equivalent"] and v["substantial_absolute"])


def test_verdict_needs_min_items():
    items = _items([[True, True], [False, False]] * 10)  # 20 items < MIN_ITEMS
    preds = {it.item_id: it.physician_labels[0] for it in items}
    v = evaluate(items, preds, "rater")["verdict"]
    assert v["physician_equivalent"] is False
    assert any("reviewed" in r for r in v["reasons"])


# --- the real frozen result (regression-locks the headline numbers) ---------
def test_frozen_strongmodel_result():
    if not (ITEMS.exists() and GOLD.exists() and LABELS.exists()):
        print("skip test_frozen_strongmodel_result (no frozen set)"); return
    items = load_calibration_set(ITEMS, GOLD)
    assert len(items) >= 100
    preds = load_label_preds(LABELS)
    assert len(preds) == len(items)
    rep = evaluate(items, preds, "strong-model judge")
    r, p = rep["rater_vs_physician"], rep["physician_ceiling"]
    assert abs(r["f1_balanced"] - p["f1_balanced"]) < 0.05      # tracks human ceiling
    assert rep["verdict"]["physician_equivalent"] is True
    assert rep["verdict"]["calibrated"] is False               # moderate -> auxiliary
    assert 0.60 < r["f1_balanced"] < 0.80


# --- deterministic sampler --------------------------------------------------
def _synthetic_meta(path: Path, n=60):
    rows = []
    for i in range(n):
        rows.append({
            "completion_id": f"c{i}", "prompt_id": f"p{i%10}",
            "prompt": [{"role": "user", "content": f"question {i}"}],
            "rubric": f"rubric text {i}", "completion": f"answer {i}",
            "category": f"cluster:cat{i % 5}",
            "binary_labels": [(i % 3) == 0, (i % 4) == 0],
            "anonymized_physician_ids": [f"phys{i%2}", f"phys{(i+1)%2}"],
        })
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_build_meta_set_deterministic():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src = d / "meta.jsonl"; _synthetic_meta(src, 60)
        n1 = build_meta_set(str(src), d / "i1.jsonl", d / "g1.jsonl", n=20, cap_per_cat=6)
        n2 = build_meta_set(str(src), d / "i2.jsonl", d / "g2.jsonl", n=20, cap_per_cat=6)
        assert n1 == n2 and n1 <= 20
        assert (d / "i1.jsonl").read_text() == (d / "i2.jsonl").read_text()
        items = [json.loads(l) for l in (d / "i1.jsonl").read_text().splitlines()]
        gold = [json.loads(l) for l in (d / "g1.jsonl").read_text().splitlines()]
        assert all("binary_labels" not in r for r in items)   # blind
        assert all("binary_labels" in g for g in gold)
        assert {r["item_id"] for r in items} == {g["item_id"] for g in gold}


# --- leaderboard auxiliary-metric policy ------------------------------------
def test_leaderboard_demotes_uncalibrated_judge():
    from medeval.runner import write_leaderboard
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rows = [
            {"model": "m", "dataset": "medqa", "n": 10, "split_type": "official",
             "metrics": {"mcq_accuracy": {"accuracy": 0.9, "n": 10}}, "model_cost_usd": 0.0},
            {"model": "m", "dataset": "healthbench", "n": 10, "split_type": "official",
             "metrics": {"llm_judge": {"judge_score": 0.7, "n": 10}}, "model_cost_usd": 0.0},
        ]
        write_leaderboard(rows, d)
        md = (d / "leaderboard.md").read_text()
        assert "Auxiliary (open-ended" in md
        assert md.index("Official (comparable)") < md.index("Auxiliary (open-ended")
        assert "healthbench" in md.split("Auxiliary (open-ended")[1]
        assert "medqa" in md.split("Auxiliary (open-ended")[0]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("OK: all calibration tests passed")
