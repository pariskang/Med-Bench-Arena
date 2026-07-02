"""Deterministic metrics: MCQ accuracy and agent pass^k."""
from __future__ import annotations

from typing import Any

from ..schema import Prediction, Sample, Score
from .base import Metric, register_metric


@register_metric("mcq_accuracy")
class MCQAccuracy(Metric):
    """Exact-match accuracy over parsed option index/indices vs. the gold.

    Single-answer: ``reference['index']`` (int) vs ``pred.parsed`` (int).
    Multi-answer:  ``reference['indices']`` (set) vs ``pred.parsed`` (set) — all
    selected options must match exactly.
    """

    async def score(self, sample: Sample, pred: Prediction) -> Score:
        ref = sample.reference
        parsed = pred.parsed
        if "indices" in ref:  # multi-answer
            gold = set(ref["indices"])
            got = set(parsed) if isinstance(parsed, (list, set, tuple)) else (
                {parsed} if parsed is not None else set()
            )
            correct = got == gold
        else:
            correct = parsed is not None and parsed == ref.get("index")
        return Score(
            metric="mcq_accuracy",
            value=1.0 if correct else 0.0,
            detail={"gold": ref.get("indices", ref.get("index")),
                    "pred": parsed, "parsed_ok": parsed is not None},
        )

    def aggregate(self, scores: list[Score]) -> dict[str, Any]:
        n = len(scores)
        acc = sum(s.value for s in scores) / n if n else 0.0
        unparsed = sum(1 for s in scores if not s.detail.get("parsed_ok"))
        return {"accuracy": acc, "n": n, "unparsed": unparsed}


@register_metric("pass_k")
class PassK(Metric):
    """Agent reliability: pass^k (all k rollouts succeed) plus pass@1.

    Expects ``pred.rollouts`` = list of per-rollout dicts each carrying a
    boolean ``success``. A single high-scoring trajectory over-states
    reliability, so the headline is pass^k.
    """

    async def score(self, sample: Sample, pred: Prediction) -> Score:
        rollouts = pred.rollouts or []
        # rollouts the grader flagged as unverifiable offline (e.g. MedAgentBench
        # query tasks without shipped gold, conditional no-ops) are not failures —
        # they simply cannot be graded. A sample where EVERY rollout is ungradable
        # is excluded from the aggregate; mixed samples score over the gradable ones.
        gradable = [r for r in rollouts
                    if not (r.get("info") or {}).get("ungradable")]
        if rollouts and not gradable:
            return Score(metric="pass_k", value=None,
                         detail={"skipped": "ungradable", "k": len(rollouts)})
        successes = [bool(r.get("success")) for r in gradable]
        k = len(successes)
        pass_pow_k = 1.0 if k > 0 and all(successes) else 0.0
        pass_at_1 = 1.0 if successes and successes[0] else 0.0
        frac = (sum(successes) / k) if k else 0.0
        turns = [r.get("turns", 0) for r in rollouts]
        timed_out = any(r.get("info", {}).get("timeout") for r in rollouts)
        return Score(
            metric="pass_k",
            value=pass_pow_k,
            detail={"k": k, "pass^k": pass_pow_k, "pass@1": pass_at_1,
                    "success_fraction": frac, "successes": successes,
                    "ungradable_rollouts": len(rollouts) - len(gradable),
                    "mean_turns": (sum(turns) / len(turns)) if turns else 0.0,
                    "timeout": timed_out},
        )

    def aggregate(self, scores: list[Score]) -> dict[str, Any]:
        scored = [s for s in scores if s.value is not None]
        n, m = len(scores), len(scored)
        if not m:
            return {"pass^k": 0.0, "pass@1": 0.0, "n": n, "n_scored": 0,
                    "ungradable": n}
        return {
            "pass^k": sum(s.detail["pass^k"] for s in scored) / m,
            "pass@1": sum(s.detail["pass@1"] for s in scored) / m,
            "mean_success_fraction": sum(s.detail["success_fraction"] for s in scored) / m,
            "avg_turns": sum(s.detail.get("mean_turns", 0.0) for s in scored) / m,
            "timeout_rate": sum(1 for s in scored if s.detail.get("timeout")) / m,
            "n": n, "n_scored": m, "ungradable": n - m,
        }
