"""Dataset preflight — reliability check for MCQ benchmarks, **no model required**.

For each dataset in a run spec it loads the data through the *real* adapter and
reports, without calling any LLM:

  * **样本数**            number of usable samples (and raw rows seen)
  * **选项数分布**        distribution of #options per sample (e.g. {4: 1273})
  * **答案解析成功率**     fraction of raw rows whose gold answer resolved to a valid
                          choice (kept / seen); the rest are listed by drop-reason
  * **前 3 条样例**       the first few rendered samples (question · choices · gold)
  * **近重复/污染扫描**    within-dataset near-duplicate question pairs (MinHash/LSH),
                          a cheap signal for contamination or accidental duplication

This catches the failure modes that silently corrupt an MCQ eval — a mis-mapped
``field_map``, an answer column in an unexpected encoding, options that don't
parse — *before* you spend tokens. Run it after pinning revisions to lock in a
known-good profile:

    python -m medeval preflight configs/catalog_mcq.yaml
"""
from __future__ import annotations

import re
import zlib
from typing import Any

from .datasets.base import create_dataset
from .schema import TaskType

DEFAULT_DUP_THRESHOLD = 0.85
_SHINGLE_K = 5
_NUM_HASHES = 32
_BANDS = 8   # NUM_HASHES must be divisible by BANDS
_ROWS_PER_BAND = _NUM_HASHES // _BANDS


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _shingles(text: str, k: int = _SHINGLE_K) -> frozenset[str]:
    """k-word shingles of ``text`` — the near-duplicate unit. Short texts (fewer
    than k words) shingle as their whole normalized form."""
    words = _normalize_text(text).split()
    if len(words) < k:
        return frozenset([" ".join(words)]) if words else frozenset()
    return frozenset(" ".join(words[i:i + k]) for i in range(len(words) - k + 1))


def _hash_i(x: str, i: int) -> int:
    """Deterministic, seeded hash — NOT the builtin ``hash()``, which is salted
    per-process (PYTHONHASHSEED) and would make MinHash signatures irreproducible
    across runs/processes. ``zlib.crc32`` is plenty for a similarity signal."""
    return zlib.crc32(f"{i}:{x}".encode("utf-8"))


def _minhash_signature(shingles: frozenset[str], num_hashes: int = _NUM_HASHES
                       ) -> tuple[int, ...]:
    if not shingles:
        return tuple([2**32 - 1] * num_hashes)
    return tuple(min(_hash_i(s, i) for s in shingles) for i in range(num_hashes))


def _estimate_jaccard(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return matches / len(sig_a)


def find_near_duplicates(
    items: list[tuple[str, str]],
    threshold: float = DEFAULT_DUP_THRESHOLD,
    num_hashes: int = _NUM_HASHES,
    bands: int = _BANDS,
) -> list[dict[str, Any]]:
    """Near-duplicate question pairs within one dataset via MinHash + LSH banding.

    ``items`` is ``[(sample_id, text), ...]``. Comparing every pair directly is
    O(n^2) — fine for small demo sets, prohibitive for an 11k-row benchmark — so
    candidate pairs are first narrowed via LSH: split each MinHash signature into
    ``bands`` bands, bucket samples that share an identical band, and only run
    the (still-approximate) Jaccard estimate on same-bucket candidates. Only
    approximately-duplicate content (paraphrase-level, not just exact text
    matches) is caught, which is the more useful contamination signal.
    """
    if bands and num_hashes % bands != 0:
        bands = 1  # degrade to a single band (full-signature match) rather than crash
    rows_per_band = num_hashes // bands if bands else num_hashes

    sigs: dict[str, tuple[int, ...]] = {}
    for sid, text in items:
        if not text or not text.strip():
            continue
        sigs[sid] = _minhash_signature(_shingles(text), num_hashes)

    buckets: dict[tuple[int, Any], list[str]] = {}
    for sid, sig in sigs.items():
        for b in range(bands):
            band_key = sig[b * rows_per_band:(b + 1) * rows_per_band]
            buckets.setdefault((b, band_key), []).append(sid)

    candidates: set[tuple[str, str]] = set()
    for bucket_ids in buckets.values():
        if len(bucket_ids) < 2:
            continue
        for i in range(len(bucket_ids)):
            for j in range(i + 1, len(bucket_ids)):
                a, b_id = sorted((bucket_ids[i], bucket_ids[j]))
                candidates.add((a, b_id))

    pairs = []
    for a, b_id in candidates:
        sim = _estimate_jaccard(sigs[a], sigs[b_id])
        if sim >= threshold:
            pairs.append({"a": a, "b": b_id, "similarity": round(sim, 4)})
    pairs.sort(key=lambda p: -p["similarity"])
    return pairs


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


def _profile(adapter, samples: list, n_examples: int,
            dup_threshold: float | None = DEFAULT_DUP_THRESHOLD,
            max_dup_pairs_reported: int = 20) -> dict[str, Any]:
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

    dup_pairs: list[dict[str, Any]] = []
    if dup_threshold is not None:
        items = [(s.id, next((m.content for m in s.messages if m.role == "user"), ""))
                 for s in samples]
        dup_pairs = find_near_duplicates(items, threshold=dup_threshold)

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
        "near_duplicate_count": len(dup_pairs),
        "near_duplicates": dup_pairs[:max_dup_pairs_reported],
    }


def preflight(config: dict[str, Any], dataset_ids: list[str] | None = None,
              limit: int | None = None, n_examples: int = 3,
              dup_threshold: float | None = DEFAULT_DUP_THRESHOLD) -> list[dict[str, Any]]:
    """Profile every dataset in ``config`` (or only ``dataset_ids``). ``limit``
    caps rows per dataset (``None`` = full load — needed for a true sample count).
    ``dup_threshold`` sets the MinHash near-duplicate similarity cutoff
    (``None`` skips the scan — it's O(dataset size) extra work per dataset)."""
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
        reports.append(_profile(adapter, samples, n_examples, dup_threshold=dup_threshold))
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
        if r.get("near_duplicate_count"):
            out.append(f"    ⚠ 近重复/污染 near-dup : {r['near_duplicate_count']} pair(s) "
                       f"≥ similarity threshold — possible contamination or accidental "
                       f"duplication, review below")
            for p in r["near_duplicates"][:5]:
                out.append(f"        {p['a']}  ≈  {p['b']}   (sim={p['similarity']})")
            if r["near_duplicate_count"] > 5:
                out.append(f"        … {r['near_duplicate_count'] - 5} more "
                           f"(see the JSON report with --output)")
        for j, ex in enumerate(r["examples"], 1):
            q = ex["question"].replace("\n", " ")
            out.append(f"    例{j} [{ex['n_options']} opts, gold={ex['gold']}] {ex['id']}")
            out.append(f"        {q}")
    out.append("─" * 72)
    out.append("PREFLIGHT: " + ("ALL CLEAR ✓" if ok else "ISSUES FOUND — review ⚠/✗ above"))
    return "\n".join(out)
