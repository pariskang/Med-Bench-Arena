"""Pairwise model comparison with statistical rigor (``medeval compare``).

A leaderboard rank ordering from two point estimates invites over-reading a
gap that's within noise. This does the paired comparison properly: aligns
both models' per-sample scores by ``sample_id`` (so a model that skipped or
gained a sample from a resumed run doesn't silently misalign), then reports
a paired-bootstrap CI for the mean difference plus (when the metric is
binary, e.g. MCQ accuracy) an exact/continuity-corrected McNemar test.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .stats import mcnemar_test, paired_bootstrap_diff_ci
from .submit import load_details

# same priority order as runner.headline() — picks the metric a leaderboard
# would actually rank on when the caller doesn't name one explicitly.
_HEADLINE_KEYS = ("accuracy", "judge_score", "pass^k", "chain_score", "herb_f1")


def _values(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    """{sample_id: score value} for one metric, skipping rows that lack it."""
    out: dict[str, float] = {}
    for r in rows:
        sc = (r.get("scores") or {}).get(metric)
        if sc is not None and sc.get("value") is not None:
            out[r["sample_id"]] = sc["value"]
    return out


def _auto_metric(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    available = set()
    for r in rows:
        available.update((r.get("scores") or {}).keys())
    for k in _HEADLINE_KEYS:
        if k in available:
            return k
    return next(iter(sorted(available)), None)


def compare(results_dir: str | Path, dataset: str, model_a: str, model_b: str,
           metric: str | None = None) -> dict[str, Any]:
    """Compare two models on one dataset from an existing results directory.

    Returns a dict with the paired sample count, both means, the paired-
    bootstrap 95% CI for ``mean(a) - mean(b)``, and (metric permitting) a
    McNemar test. Raises ``ValueError`` if either (model, dataset) has no
    detail file, or the two share no sample ids (nothing to pair on).
    """
    groups = load_details(results_dir)
    rows_a = groups.get((model_a, dataset))
    rows_b = groups.get((model_b, dataset))
    if rows_a is None:
        raise ValueError(f"no detail file for model={model_a!r} dataset={dataset!r} in {results_dir}")
    if rows_b is None:
        raise ValueError(f"no detail file for model={model_b!r} dataset={dataset!r} in {results_dir}")

    metric = metric or _auto_metric(rows_a) or _auto_metric(rows_b)
    if metric is None:
        raise ValueError(f"no scored metric found for dataset={dataset!r}")

    va = _values(rows_a, metric)
    vb = _values(rows_b, metric)
    shared = sorted(set(va) & set(vb))
    if not shared:
        raise ValueError(
            f"model={model_a!r} and model={model_b!r} share no sample_ids "
            f"with a scored {metric!r} on dataset={dataset!r} — nothing to pair")

    a = [va[sid] for sid in shared]
    b = [vb[sid] for sid in shared]
    n = len(shared)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    lo, hi = paired_bootstrap_diff_ci(a, b)

    result: dict[str, Any] = {
        "dataset": dataset, "metric": metric, "n": n,
        "model_a": model_a, "model_b": model_b,
        "mean_a": round(mean_a, 6), "mean_b": round(mean_b, 6),
        "diff": round(mean_a - mean_b, 6),
        "diff_ci95": [round(lo, 6), round(hi, 6)],
    }
    # McNemar only makes sense for a binary (0/1) outcome — e.g. MCQ accuracy,
    # not a continuous judge score. Detect rather than require the caller to know.
    if all(v in (0, 1, 0.0, 1.0) for v in a + b):
        result["mcnemar"] = mcnemar_test([int(v) for v in a], [int(v) for v in b])
    return result


def render_compare(r: dict[str, Any]) -> str:
    lo, hi = r["diff_ci95"]
    lines = [
        f"# {r['model_a']} vs {r['model_b']}  ·  {r['dataset']}  ({r['metric']}, n={r['n']})",
        "",
        f"- **{r['model_a']}**: {r['mean_a']:.4f}",
        f"- **{r['model_b']}**: {r['mean_b']:.4f}",
        f"- **diff ({r['model_a']} - {r['model_b']})**: {r['diff']:+.4f}  "
        f"(95% CI [{lo:+.4f}, {hi:+.4f}])",
    ]
    if lo <= 0 <= hi:
        lines.append("  - ≈ not statistically significant (CI includes 0)")
    else:
        lines.append("  - significant at the 95% CI level")
    if "mcnemar" in r:
        m = r["mcnemar"]
        lines += [
            "",
            f"McNemar test ({m['method']}): n01={m['n01']}, n10={m['n10']}, "
            f"p={m['p_value']:.4f}"
            + ("  - significant at p<0.05" if m["p_value"] < 0.05 else "  - not significant"),
        ]
    return "\n".join(lines)
