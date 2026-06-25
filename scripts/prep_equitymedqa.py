#!/usr/bin/env python3
"""Preprocess EquityMedQA into a scorable questions JSONL for Med-Bench-Arena.

EquityMedQA (`katielink/EquityMedQA`, CC-BY-4.0; Pfohl et al. 2024,
"A toolbox for surfacing health equity harms and biases in large language models",
arXiv:2403.12025) is a collection of **adversarial questions** designed to surface
health-equity harms / bias in LLM answers. It ships **no reference answers and no
gold labels** — answers are meant to be rated against a *bias rubric* — and the
CSVs are **header-less single-column** lists (the CC-* subsets carry two
counterfactual questions per line). Loading them naively (`pandas.read_csv` /
`load_dataset`) silently eats the first question as a column name.

This step fixes both quirks and emits one clean record per question:

    {"id": "<subset>:<i>[:a|:b]", "subset": "<subset>", "question": "...",
     "pair_id": "<subset>:<i>"?}          # pair_id only for the CC counterfactual pairs

Then score it with the shared bias rubric wired in
`configs/catalog_ethics_safety.yaml` (`local_json` + `llm_judge`, the 6 EquityMedQA
independent-assessment dimensions). The questions remain CC-BY-4.0 © Google LLC.

Usage:
    python scripts/prep_equitymedqa.py                       # -> data/equitymedqa.jsonl (full, ~6k Q)
    python scripts/prep_equitymedqa.py --out data/eq.jsonl --limit 50
    python scripts/prep_equitymedqa.py --demo data/samples/equitymedqa_demo.jsonl  # also (re)write the demo
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

# pin to an immutable commit so the question set can't change underneath you
REVISION = "ae9bb54e1e70b9894896c1b88f2c145869f29993"
BASE = f"https://huggingface.co/datasets/katielink/EquityMedQA/resolve/{REVISION}/"

# file -> (subset name, n question columns). CC-* carry a counterfactual PAIR per line.
SUBSETS: list[tuple[str, str, int]] = [
    ("equitymedqa_omaq.csv", "omaq", 1),
    ("equitymedqa_ehai.csv", "ehai", 1),
    ("equitymedqa_fbrt_manual.csv", "fbrt_manual", 1),
    ("equitymedqa_fbrt_llm.csv", "fbrt_llm", 1),
    ("equitymedqa_fbrt_llm_661_sampled.csv", "fbrt_llm_661_sampled", 1),
    ("equitymedqa_trinds.csv", "trinds", 1),
    ("equitymedqa_cc_manual.csv", "cc_manual", 2),       # counterfactual pairs
    ("equitymedqa_cc_llm.csv", "cc_llm", 2),             # counterfactual pairs
    ("other_datasets_mixed_mmqa_omaq.csv", "mixed_mmqa_omaq", 1),
    ("other_datasets_multimedqa.csv", "multimedqa", 1),
    ("other_datasets_omiye_et_al.csv", "omiye_et_al", 1),
]


def _fetch(url: str, retries: int = 5) -> str | None:
    """Robust-ish text fetch (retry on truncation / transient errors). Prefer the
    package streamer if importable; fall back to urllib."""
    try:
        from medeval.assets import _download_stream  # robust .part + Range resume
        import hashlib
        cache = Path("data/cache"); cache.mkdir(parents=True, exist_ok=True)
        dest = cache / f"eqmedqa_{hashlib.sha256(url.encode()).hexdigest()[:12]}.csv"
        return _download_stream(url, dest).read_text(encoding="utf-8")
    except Exception:
        pass
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "medeval/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read().decode("utf-8")
        except Exception as e:  # noqa: BLE001
            last = e
    print(f"  ! failed {url}: {last}", file=sys.stderr)
    return None


def _rows(text: str) -> list[list[str]]:
    # header-less: csv.reader keeps EVERY line (the first question is NOT a header)
    return [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]


def build(out: Path, limit: int | None = None) -> int:
    records: list[dict] = []
    for fname, subset, ncols in SUBSETS:
        text = _fetch(BASE + fname)
        if text is None:
            continue
        rows = _rows(text)
        n_before = len(records)
        for i, row in enumerate(rows):
            cells = [c.strip() for c in row if c.strip()]
            if not cells:
                continue
            if ncols == 2 and len(cells) >= 2:           # CC counterfactual pair
                pair = f"{subset}:{i}"
                for tag, q in zip(("a", "b"), cells[:2]):
                    records.append({"id": f"{pair}:{tag}", "subset": subset,
                                    "question": q, "pair_id": pair})
            else:
                records.append({"id": f"{subset}:{i}", "subset": subset, "question": cells[0]})
            if limit and len(records) >= limit:
                break
        print(f"  {subset:22s} {len(records) - n_before:5d} questions")
        if limit and len(records) >= limit:
            break
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[prep] wrote {len(records)} questions -> {out}")
    return len(records)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/equitymedqa.jsonl", help="output JSONL path")
    ap.add_argument("--limit", type=int, default=None, help="cap total questions")
    ap.add_argument("--demo", default=None,
                    help="also write a small per-subset demo sample to this path")
    args = ap.parse_args()
    build(Path(args.out), args.limit)
    if args.demo:                                        # one question per subset (a quick demo)
        full = [json.loads(l) for l in Path(args.out).read_text(encoding="utf-8").splitlines()]
        seen, demo = set(), []
        for r in full:
            if r["subset"] not in seen:
                seen.add(r["subset"]); demo.append(r)
        Path(args.demo).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.demo).open("w", encoding="utf-8") as f:
            for r in demo:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[prep] wrote {len(demo)} demo questions -> {args.demo}")


if __name__ == "__main__":
    main()
