"""Export MedEval results into leaderboard-submission formats.

Two targets, both confirmed against upstream source:

* **OpenCompass** — ``predictions/<model>/<dataset>.json`` = a dict keyed by
  ``str(idx)`` with ``{origin_prompt, prediction, gold?}`` (gold only when known).
  This is exactly what ``GenInferencerOutputHandler.save_results`` writes.

* **MedBench** (medbench.opencompass.org.cn) — one JSONL per evaluation set, each
  record ``{question, passage, options, answer, other:{source,id}}`` with the model
  output written into ``answer`` (a letter for MCQ, free text for generation). The
  faithful path preserves the platform's original test records and only fills
  ``answer`` (matched by ``other.id``); pass ``test_dir`` for that.

The exporter reads the runner's per-sample ``detail__<model>__<dataset>.jsonl``
files, so it is fully decoupled from how the run was produced.
"""
from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any

LETTERS = string.ascii_uppercase

_SHARD_RE = re.compile(r"__shard\d+of\d+$")   # matches distributed.py's shard suffix


# --- reading the runner's detail output ---------------------------------------
def load_details(results_dir: str | Path) -> dict[tuple[str, str], list[dict]]:
    """Return {(model, dataset): [detail rows]} from detail__*.jsonl files.

    Distributed runs write one detail file per shard (``…__shard{i}of{N}.jsonl``).
    Strip that suffix and merge the shards (dedup by sample_id) so a sharded run
    exports as one submission per (model, dataset) — not one broken fragment per
    shard, and not a dataset id polluted with a ``__shardIofN`` tail."""
    results_dir = Path(results_dir)
    groups: dict[tuple[str, str], dict[str, dict]] = {}
    for fp in sorted(results_dir.glob("detail__*.jsonl")):
        stem = _SHARD_RE.sub("", fp.stem[len("detail__"):])
        model, _, dataset = stem.partition("__")
        seen = groups.setdefault((model, dataset), {})
        for i, l in enumerate(fp.read_text(encoding="utf-8").splitlines()):
            if l.strip():
                r = json.loads(l)
                seen[str(r.get("sample_id", f"{fp.stem}:{i}"))] = r
    return {k: list(v.values()) for k, v in groups.items()}


def _orig_id(sample_id: str) -> str:
    return sample_id.split(":")[-1]


def _letters(parsed: Any) -> str:
    """Map an MCQ parsed index / index-list to letter(s)."""
    if isinstance(parsed, list):
        return "".join(LETTERS[i] for i in parsed if isinstance(i, int) and 0 <= i < 26)
    if isinstance(parsed, int) and 0 <= parsed < 26:
        return LETTERS[parsed]
    return ""


def answer_for_submission(row: dict) -> Any:
    """The value to put in MedBench's ``answer`` / OpenCompass ``prediction``."""
    if row.get("task") == "mcq":
        return _letters(row.get("parsed"))
    return row.get("prediction", "")


def _gold(row: dict) -> Any:
    ref = row.get("reference") or {}
    if row.get("task") == "mcq":
        return ref.get("letter") or _letters(ref.get("index", ref.get("indices")))
    return ref.get("reference") or ""


# --- OpenCompass --------------------------------------------------------------
def to_opencompass(results_dir: str | Path, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    for (model, dataset), rows in load_details(results_dir).items():
        d: dict[str, dict] = {}
        for i, r in enumerate(rows):
            entry = {"origin_prompt": r.get("prompt", ""),
                     "prediction": str(answer_for_submission(r))}
            gold = _gold(r)
            if gold:
                entry["gold"] = gold
            d[str(i)] = entry
        dest = out / "predictions" / model / f"{dataset}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(d, ensure_ascii=False, indent=4), encoding="utf-8")
    return out


# --- MedBench -----------------------------------------------------------------
def _options_block(choices: list[str] | None) -> list[str] | None:
    if not choices:
        return None
    return [f"{LETTERS[i]}: {c}" for i, c in enumerate(choices)]


def _synthesize_medbench(rows: list[dict], dataset: str) -> list[dict]:
    out = []
    for r in rows:
        out.append({
            "question": r.get("prompt", ""),
            "passage": None,
            "options": _options_block(r.get("choices")),
            "answer": answer_for_submission(r),
            "other": {"source": dataset, "id": _orig_id(r.get("sample_id", ""))},
        })
    return out


def _fill_medbench(test_file: Path, answers: dict[str, Any]) -> list[dict]:
    """Preserve the platform's original records; only fill ``answer`` by id."""
    filled = []
    for line in test_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        oid = str(((rec.get("other") or {}).get("id", rec.get("id", ""))))
        if oid in answers:
            rec["answer"] = answers[oid]
        filled.append(rec)
    return filled


def to_medbench(results_dir: str | Path, out_dir: str | Path,
                test_dir: str | Path | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups = load_details(results_dir)

    if test_dir:  # faithful fill-in mode: preserve original test records
        test_dir = Path(test_dir)
        # id -> answer, pooled across datasets (ids are unique within a set)
        by_dataset: dict[str, dict[str, Any]] = {}
        for (model, dataset), rows in groups.items():
            by_dataset.setdefault(dataset, {})
            for r in rows:
                by_dataset[dataset][_orig_id(r.get("sample_id", ""))] = answer_for_submission(r)
        for test_file in sorted(test_dir.rglob("*_test.jsonl")):
            name = test_file.stem.replace("_test", "")
            answers = by_dataset.get(name) or {k: v for d in by_dataset.values() for k, v in d.items()}
            filled = _fill_medbench(test_file, answers)
            dest = out / test_file.parent.name / f"{name}.jsonl"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "w", encoding="utf-8") as f:
                for rec in filled:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return out

    # synthesize mode: build MedBench-shaped records from our detail rows
    for (model, dataset), rows in groups.items():
        dest = out / model / f"{dataset}.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            for rec in _synthesize_medbench(rows, dataset):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out


def export(results_dir: str | Path, fmt: str, out_dir: str | Path,
           test_dir: str | Path | None = None) -> Path:
    if fmt == "opencompass":
        return to_opencompass(results_dir, out_dir)
    if fmt == "medbench":
        return to_medbench(results_dir, out_dir, test_dir)
    raise ValueError(f"unknown export format {fmt!r}; use opencompass|medbench")
