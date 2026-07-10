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

    ``k`` is not actually fixed across samples once ungradable rollouts are
    dropped: a sample where all 3 configured rollouts graded and a sample
    where only 1 of 3 did are NOT comparable "pass^3" numbers — averaging them
    together silently changes what's being measured per sample. So the headline
    ``pass^k`` in ``aggregate()`` is computed ONLY over samples where every
    configured rollout was gradable (``strict_eligible``); samples with a
    partial gradable count still contribute pass@1 / success_fraction (which
    only need the first rollout) and are visible in ``k_effective_distribution``
    rather than silently blended into a mislabeled "pass^k".
    """

    async def score(self, sample: Sample, pred: Prediction) -> Score:
        rollouts = pred.rollouts or []
        k_configured = len(rollouts)
        # rollouts the grader flagged as unverifiable offline (e.g. MedAgentBench
        # query tasks without shipped gold, conditional no-ops) are not failures —
        # they simply cannot be graded. A sample where EVERY rollout is ungradable
        # is excluded from the aggregate; mixed samples score over the gradable ones.
        gradable = [r for r in rollouts
                    if not (r.get("info") or {}).get("ungradable")]
        if rollouts and not gradable:
            return Score(metric="pass_k", value=None,
                         detail={"skipped": "ungradable", "k": k_configured})
        successes = [bool(r.get("success")) for r in gradable]
        k_eff = len(successes)
        strict_eligible = k_eff == k_configured   # ALL configured rollouts graded
        pass_pow_k = 1.0 if k_eff > 0 and all(successes) else 0.0
        pass_at_1 = 1.0 if successes and successes[0] else 0.0
        frac = (sum(successes) / k_eff) if k_eff else 0.0
        turns = [r.get("turns", 0) for r in rollouts]
        timed_out = any(r.get("info", {}).get("timeout") for r in rollouts)

        # Optional, env-specific signals (only MediQ populates these today) —
        # generic so AgentClinic/MedAgentBench rollouts are unaffected.
        infos = [r.get("info") or {} for r in rollouts]
        n_abstained = sum(1 for i in infos if i.get("abstained"))
        questions = [i["questions"] for i in infos if i.get("questions") is not None]
        conf_pairs = [(i["confidence"], 1.0 if r.get("success") else 0.0)
                      for r, i in zip(rollouts, infos)
                      if i.get("confidence") is not None and not i.get("abstained")]

        return Score(
            metric="pass_k",
            value=pass_pow_k,
            detail={"k_configured": k_configured, "k_effective": k_eff,
                    "strict_eligible": strict_eligible,
                    "pass^k": pass_pow_k, "pass@1": pass_at_1,
                    "success_fraction": frac, "successes": successes,
                    "ungradable_rollouts": k_configured - k_eff,
                    "mean_turns": (sum(turns) / len(turns)) if turns else 0.0,
                    "timeout": timed_out,
                    "n_abstained": n_abstained, "questions": questions,
                    "confidence_pairs": conf_pairs},
        )

    def aggregate(self, scores: list[Score]) -> dict[str, Any]:
        scored = [s for s in scores if s.value is not None]
        n, m = len(scores), len(scored)
        if not m:
            return {"pass^k": 0.0, "pass@1": 0.0, "n": n, "n_scored": 0,
                    "ungradable": n}
        strict = [s for s in scored if s.detail.get("strict_eligible")]
        k_dist: dict[str, int] = {}
        for s in scored:
            key = str(s.detail.get("k_effective", 0))
            k_dist[key] = k_dist.get(key, 0) + 1

        out: dict[str, Any] = {
            # headline: strict-k only — samples with a partial gradable count
            # never silently blend a "pass^2" into a "pass^3" average.
            "pass^k": (sum(s.detail["pass^k"] for s in strict) / len(strict)
                      if strict else 0.0),
            "pass@1": sum(s.detail["pass@1"] for s in scored) / m,
            "mean_success_fraction": sum(s.detail["success_fraction"] for s in scored) / m,
            "avg_turns": sum(s.detail.get("mean_turns", 0.0) for s in scored) / m,
            "timeout_rate": sum(1 for s in scored if s.detail.get("timeout")) / m,
            "n": n, "n_scored": m, "ungradable": n - m,
            "strict_k_eligible": len(strict), "strict_k_rate": len(strict) / m,
            "k_effective_distribution": k_dist,
        }

        n_abstained = sum(s.detail.get("n_abstained", 0) for s in scored)
        n_rollouts = sum(s.detail.get("k_configured", 0) for s in scored)
        if n_abstained:  # only surfaced when the env supports abstention (MediQ)
            out["abstain_rate"] = n_abstained / n_rollouts if n_rollouts else 0.0

        all_questions = [q for s in scored for q in s.detail.get("questions", [])]
        if all_questions:
            out["avg_questions"] = sum(all_questions) / len(all_questions)

        conf_pairs = [p for s in scored for p in s.detail.get("confidence_pairs", [])]
        if conf_pairs:
            out["mean_confidence"] = sum(c for c, _ in conf_pairs) / len(conf_pairs)
            # Brier score: mean squared error between stated confidence and
            # actual correctness — lower is better-calibrated (0 = perfect).
            out["confidence_brier_score"] = (
                sum((c - correct) ** 2 for c, correct in conf_pairs) / len(conf_pairs))
        return out
