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
        successes = [bool(r.get("success")) for r in rollouts]
        k = len(successes)
        pass_pow_k = 1.0 if k > 0 and all(successes) else 0.0
        pass_at_1 = 1.0 if successes and successes[0] else 0.0
        frac = (sum(successes) / k) if k else 0.0
        return Score(
            metric="pass_k",
            value=pass_pow_k,
            detail={"k": k, "pass^k": pass_pow_k, "pass@1": pass_at_1,
                    "success_fraction": frac, "successes": successes},
        )

    def aggregate(self, scores: list[Score]) -> dict[str, Any]:
        n = len(scores)
        if not n:
            return {"pass^k": 0.0, "pass@1": 0.0, "n": 0}
        return {
            "pass^k": sum(s.detail["pass^k"] for s in scores) / n,
            "pass@1": sum(s.detail["pass@1"] for s in scores) / n,
            "mean_success_fraction": sum(s.detail["success_fraction"] for s in scores) / n,
            "n": n,
        }
