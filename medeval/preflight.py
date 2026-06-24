"""Dataset preflight — reliability check for MCQ benchmarks, **no model required**.

For each dataset in a run spec it loads the data through the *real* adapter and
reports, without calling any LLM:

  * **样本数**            number of usable samples (and raw rows seen)
  * **选项数分布**        distribution of #options per sample (e.g. {4: 1273})
  * **答案解析成功率**     fraction of raw rows whose gold answer resolved to a valid
                          choice (kept / seen); the rest are listed by drop-reason
  * **前 3 条样例**       the first few rendered samples (question · choices · gold)

This catches the failure modes that silently corrupt an MCQ eval — a mis-mapped
``field_map``, an answer column in an unexpected encoding, options that don't
parse — *before* you spend tokens. Run it after pinning revisions to lock in a
known-good profile:

    python -m medeval preflight configs/catalog_mcq.yaml
"""
from __future__ import annotations

from typing import Any

from .datasets.base import create_dataset
from .schema import TaskType


def _example(sample) -> dict[str, Any]:
    ref = sample.reference or {}
    if "indices" in ref:
        gold = "".join(chr(65 + i) for i in ref["indices"])
    else:
        gold = ref.get("letter") or ref.get("index")
    user = next((m.content for m in sample.messages if m.role == "user"), "")
    return {
        "id": sample.id,
        "n_options": len(sample.choices or []),
        "gold": gold,
        "choices": list(sample.choices or []),
        "question": (user[:240] + ("…" if len(user) > 240 else "")),
    }


def _profile(adapter, samples: list, n_examples: int) -> dict[str, Any]:
    from collections import Counter
    n = len(samples)
    opt_dist = dict(sorted(Counter(len(s.choices or []) for s in samples).items()))

    stats = getattr(adapter, "load_stats", {}) or {}
    seen = stats.get("seen")
    kept = stats.get("kept", n)
    dropped = dict(stats.get("dropped", {}))
    if seen:                                  # adapter tracked raw rows -> true rate
        parse_rate = kept / seen
    else:                                     # fallback: validity among loaded samples
        valid = 0
        for s in samples:
            ref, nopt = s.reference or {}, len(s.choices or [])
            if "indices" in ref:
                valid += bool(ref["indices"]) and all(0 <= i < nopt for i in ref["indices"])
            else:
                idx = ref.get("index")
                valid += isinstance(idx, int) and 0 <= idx < nopt
        parse_rate = (valid / n) if n else 0.0

    mcq = sum(1 for s in samples if s.task_type == TaskType.MCQ)
    return {
        "id": adapter.id,
        "adapter": getattr(adapter, "adapter_name", "?"),
        "revision": getattr(adapter, "revision", None),
        "n_samples": n,
        "rows_seen": seen,
        "mcq_samples": mcq,
        "option_dist": opt_dist,
        "answer_parse_rate": round(parse_rate, 4),
        "dropped": dropped,
        "examples": [_example(s) for s in samples[:n_examples]],
    }


def preflight(config: dict[str, Any], dataset_ids: list[str] | None = None,
              limit: int | None = None, n_examples: int = 3) -> list[dict[str, Any]]:
    """Profile every dataset in ``config`` (or only ``dataset_ids``). ``limit``
    caps rows per dataset (``None`` = full load — needed for a true sample count)."""
    reports: list[dict[str, Any]] = []
    for dcfg in config.get("datasets", []):
        did = dcfg.get("id")
        if dataset_ids and did not in dataset_ids:
            continue
        cfg = dict(dcfg)
        if limit is not None:
            cfg["limit"] = limit
        else:
            cfg.pop("limit", None)            # full load by default
        try:
            adapter = create_dataset(cfg)
            samples = adapter.load()
        except Exception as e:                # report, don't abort the whole sweep
            reports.append({"id": did, "error": f"{type(e).__name__}: {e}"})
            continue
        reports.append(_profile(adapter, samples, n_examples))
    return reports


def _bar(rate: float, width: int = 20) -> str:
    fill = int(round(rate * width))
    return "█" * fill + "░" * (width - fill)


def format_reports(reports: list[dict[str, Any]]) -> str:
    """Render the preflight reports as a human-readable terminal report."""
    out: list[str] = []
    ok = True
    for r in reports:
        out.append("─" * 72)
        if "error" in r:
            ok = False
            out.append(f"✗ {r['id']}\n    ERROR: {r['error']}")
            continue
        rate = r["answer_parse_rate"]
        flag = "✓" if rate >= 0.999 else ("⚠" if rate >= 0.95 else "✗")
        if rate < 0.999:
            ok = False
        seen = r["rows_seen"]
        seen_s = f" of {seen} rows" if seen else ""
        out.append(f"{flag} {r['id']}   [{r['adapter']}]")
        if r.get("revision"):
            out.append(f"    revision : {r['revision']}")
        out.append(f"    样本数 samples        : {r['n_samples']}{seen_s}"
                   + (f"  ({r['mcq_samples']} MCQ)" if r['mcq_samples'] != r['n_samples'] else ""))
        out.append(f"    选项数 option dist     : {r['option_dist']}")
        out.append(f"    解析率 answer parse    : {rate*100:5.1f}%  {_bar(rate)}")
        if r["dropped"]:
            out.append(f"    dropped rows          : {r['dropped']}")
        for j, ex in enumerate(r["examples"], 1):
            q = ex["question"].replace("\n", " ")
            out.append(f"    例{j} [{ex['n_options']} opts, gold={ex['gold']}] {ex['id']}")
            out.append(f"        {q}")
    out.append("─" * 72)
    out.append("PREFLIGHT: " + ("ALL CLEAR ✓" if ok else "ISSUES FOUND — review ⚠/✗ above"))
    return "\n".join(out)
