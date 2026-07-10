"""Orchestrator: schedule, cache, resume, score, rank.

Wires the three decoupled layers together for the full ``N datasets × M models ×
K metrics`` grid. Generations are cached on disk keyed by a CONTENT-ADDRESSED
signature (sample content + dataset protocol config + adapter version + model/
generation config) so reruns are cheap and resumable, but stale generations are
never silently reused after the underlying data or protocol changes. Every run
also writes a ``run_manifest.json`` recording exactly what produced it.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema import Generation, Prediction, Sample
from .providers.base import create_provider, ModelProvider
from .datasets.base import create_dataset
from .datasets.agent_env import AgentAdapter
from .metrics.base import create_metric
from .eligibility import CONTENT_ADAPTERS, enforce_official_eligibility, has_pin_evidence

# Import implementation modules so their @register_* decorators run.
from .providers import mock, litellm_provider, poe, hf  # noqa: F401
from .datasets import hf_mcq, local_json, agent_env, tcmbench, medbench  # noqa: F401
from .metrics import (  # noqa: F401
    mcq, llm_judge, text_match, prescription, syndrome, tcm_struct, numeric)

# Bump manually whenever adapter parsing / prompt-rendering / rollout logic
# changes in a way that should invalidate previously-cached generations (a
# format-string tweak, a new default instruction, a changed action parser).
# Deliberately NOT the git commit: folding the commit hash into the cache key
# would invalidate every cache on every unrelated code change, defeating
# resumability during normal development. This is a precise, hand-controlled
# knob instead — the git commit is still recorded in ``run_manifest.json`` for
# full provenance, just not baked into the cache key.
ADAPTER_PROTOCOL_VERSION = 1

# Dataset-config keys that affect SCORING only, never what the model-under-test
# is asked to generate — excluded from the cache signature so editing a rubric
# or switching judges doesn't force an expensive, pointless regeneration.
# ``limit``/``id`` are orchestration knobs (how many samples, filename), not
# protocol — also excluded so `--limit 5` then `--limit 0` share one cache.
_CACHE_IRRELEVANT_CONFIG_KEYS = frozenset({"limit", "id", "metrics", "judge", "split_type"})

_SECRET_KEY_PATTERN = re.compile(r"(api[_-]?key|secret|token|password)$", re.IGNORECASE)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s))


def _sample_content_hash(s: Sample) -> str:
    """Hash of everything about a sample that affects what the model is asked
    and what it will be scored against: rendered messages (already baked-in
    prompt template / system prompt / instruction), choices, reference, and
    env_spec (agent tasks). If upstream data drifts under an unpinned source —
    same ``sample.id``, different content — this hash changes, so the cache
    MISSES and the sample is regenerated instead of silently scoring a new
    reference against a stale cached output."""
    payload = {
        "messages": [{"role": m.role, "content": m.content, "images": m.images}
                     for m in s.messages],
        "choices": s.choices,
        "reference": s.reference,
        "env_spec": s.env_spec,
    }
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _dataset_protocol_key(ds: Any) -> dict[str, Any]:
    """The dataset-level part of the cache signature: raw config (minus the
    scoring-only/orchestration keys above) + adapter class + protocol version.
    Changing a pinned revision, a field_map, a prompt template, an instruction,
    or upgrading the adapter's parsing logic all change this — and therefore
    the cache filename — so a stale generation is never picked up by accident."""
    cfg = {k: v for k, v in (getattr(ds, "config", None) or {}).items()
           if k not in _CACHE_IRRELEVANT_CONFIG_KEYS}
    return {
        "config": cfg,
        "adapter_class": f"{type(ds).__module__}.{type(ds).__qualname__}",
        "adapter_protocol_version": ADAPTER_PROTOCOL_VERSION,
    }


def _protocol_hash(ds: Any) -> str:
    sig = json.dumps(_dataset_protocol_key(ds), sort_keys=True, default=str)
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:12]


def _git_commit() -> str | None:
    """Best-effort short commit hash for the run manifest (never for the cache
    key — see ADAPTER_PROTOCOL_VERSION). None outside a git checkout."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _redact_secrets(obj: Any) -> Any:
    """Recursively blank literal-secret config values (e.g. a discouraged
    ``api_key: sk-...`` inline in YAML) before writing the run manifest. Env-var
    NAMES (``api_key_env: OPENAI_API_KEY``) are not secrets and are kept."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if (isinstance(k, str) and _SECRET_KEY_PATTERN.search(k)
                    and not k.lower().endswith("_env")):
                out[k] = "***REDACTED***" if v else v
            else:
                out[k] = _redact_secrets(v)
        return out
    if isinstance(obj, list):
        return [_redact_secrets(v) for v in obj]
    return obj


def _gen_to_dict(g: Generation) -> dict[str, Any]:
    return {"text": g.text, "model": g.model, "prompt_tokens": g.prompt_tokens,
            "completion_tokens": g.completion_tokens, "total_tokens": g.total_tokens,
            "cost_usd": g.cost_usd, "latency_s": g.latency_s,
            "finish_reason": g.finish_reason}


def _gen_from_dict(d: dict[str, Any]) -> Generation:
    return Generation(text=d.get("text", ""), model=d.get("model", ""),
                      prompt_tokens=d.get("prompt_tokens", 0),
                      completion_tokens=d.get("completion_tokens", 0),
                      total_tokens=d.get("total_tokens", 0),
                      cost_usd=d.get("cost_usd", 0.0), latency_s=d.get("latency_s", 0.0),
                      finish_reason=d.get("finish_reason", ""))


def _make_pbar(total: int, desc: str, initial: int = 0):
    """tqdm progress bar with a plain-print fallback when tqdm is absent."""
    try:
        from tqdm.auto import tqdm as _tqdm
        return _tqdm(total=total, initial=initial, desc=desc,
                     unit="sample", dynamic_ncols=True, leave=True)
    except ImportError:
        class _FallbackBar:
            def __init__(self) -> None:
                self.n = initial
                self._last_report = initial

            def update(self, n: int = 1) -> None:
                self.n += n
                if not total:
                    return
                pct = self.n * 100 // total
                if (self.n - self._last_report >= max(1, total // 20)
                        or self.n >= total):
                    self._last_report = self.n
                    print(f"  [{desc}] {self.n}/{total} ({pct}%)", flush=True)

            def close(self) -> None:
                pass
        return _FallbackBar()


def headline(agg: dict[str, Any]) -> tuple[str, float]:
    """Pick the salient (name, value) from a metric aggregation for ranking."""
    for metric, d in agg.items():
        for key in ("accuracy", "judge_score", "pass^k", "chain_score", "herb_f1"):
            if key in d:
                return key, d[key]
    for metric, d in agg.items():
        for k, v in d.items():
            if isinstance(v, (int, float)) and k != "n":
                return f"{metric}.{k}", v
    return "score", 0.0


_INTERNAL_NOTE = (
    "> ⚠️ **Not comparable to official leaderboards.** These runs use a "
    "validation/dev split, a tiny demo subset, a small public sample, a gated "
    "partial set, a built-in (approximate) grader, an unpinned/unverified data "
    "source, or a from-scratch protocol reimplementation not confirmed to match "
    "the original paper — see each section's `split_type`. Reported for internal "
    "tracking only; do **not** publish them as official scores. `unverified` is "
    "the DEFAULT tier: a dataset must actively earn `official` (a pinned "
    "commit/sha for static benchmarks, or full protocol-fidelity — e.g. all "
    "AgentClinic support roles present — for agent benchmarks); it is never "
    "granted just because a config says so.")

_AUXILIARY_NOTE = (
    "> ⚠️ **Auxiliary metric — open-ended, LLM-judge scored.** These scores come "
    "from an LLM-as-judge whose agreement with human experts has not (yet) cleared "
    "calibration **for this exact judge model + grading prompt**, so they are "
    "reported as a secondary signal and never as a headline rank. A calibration "
    "only lifts a row out of this section when it was measured against the SAME "
    "judge model+revision AND the SAME grading prompt (today: the HealthBench "
    "per-criterion style) that produced the row — calibrating one judge on "
    "HealthBench's English criteria never grants headline status to a different "
    "judge, or to TCM/ethics/safety rows graded by the default rubric prompt. "
    "Calibrate with `medeval calibrate` (see `data/calibration/`); a judge becomes "
    "headline-eligible only when it matches the physician ceiling **and** reaches "
    "substantial absolute agreement (κ ≥ 0.40). On HealthBench's physician "
    "meta-eval a strong-model judge is *physician-equivalent* yet only *moderate* "
    "in absolute terms (κ≈0.39 vs a 0.44 human ceiling) — rubric grading is "
    "intrinsically subjective — so open-ended scores stay auxiliary by default.")


def _load_calibration_report(output_dir: Path) -> dict[str, Any] | None:
    """The calibration report dict (run-local, else the canonical
    ``data/calibration/``), or None if absent/unparseable."""
    for cand in (output_dir / "calibration_report.json",
                 Path("data/calibration/calibration_report.json")):
        try:
            return json.loads(cand.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def _judge_calibrated(output_dir: Path) -> bool:
    """True iff *some* calibration report marking *a* judge headline-eligible is
    present — a coarse existence check for the run manifest only. The
    leaderboard's actual headline-eligibility decision is per-row and much
    stricter: see ``_row_judge_calibrated``."""
    report = _load_calibration_report(output_dir)
    return bool(report and report.get("verdict", {}).get("calibrated") is True)


def _row_judge_calibrated(row: dict[str, Any], report: dict[str, Any] | None) -> bool:
    """Whether THIS row's specific judge usage is covered by a headline-eligible
    calibration — not merely whether *some* judge, on *some* task, was once
    measured. A calibration report is only a green light for a row when:

      1. it is itself headline-eligible (``verdict.calibrated``);
      2. it records a concrete ``judge_model`` it was measured against — a
         ``--labels``/human-rater calibration never binds to one (we cannot
         mechanically verify an arbitrary configured judge is "equivalent" to
         whoever produced those labels), so it is reported for research
         purposes only and never auto-applied to a live run;
      3. that judge_model + judge_revision matches the judge THIS row actually
         used — swapping the judge must not silently inherit someone else's
         calibration;
      4. the grading prompt/protocol matches too — today calibration only
         exercises the HealthBench-style per-criterion grader
         (``prompt_style: healthbench_per_criterion``); a dataset graded by the
         default single-call rubric prompt was never measured and stays
         auxiliary regardless of what else was calibrated.

    This is what stops "a model calibrated on HealthBench English criteria"
    from silently granting headline status to TCM syndrome-differentiation,
    ethics, or safety scores graded by a different judge and a different
    prompt — each task family needs its OWN calibration evidence.
    """
    sig = row.get("judge_signature")
    if not sig or not report:
        return False
    if not report.get("verdict", {}).get("calibrated"):
        return False
    rsig = report.get("signature") or {}
    if not rsig.get("judge_model"):
        return False
    if rsig.get("judge_model") != sig.get("judge_model"):
        return False
    if rsig.get("judge_revision") != sig.get("judge_revision"):
        return False
    if rsig.get("prompt_style") != sig.get("prompt_style"):
        return False
    return True


def _is_judge_headline(row: dict[str, Any]) -> bool:
    """Whether this row is ranked by an LLM-judge open-ended score."""
    return headline(row["metrics"])[0] == "judge_score"


# A run whose judge failed to grade more than this fraction of samples (empty/
# garbled JSON, refusals, API errors — see llm_judge.py's judge_failures) is
# not a trustworthy headline number even if the judge MODEL itself is
# calibrated in general: something went wrong on THIS run specifically, and
# high-failure samples are plausibly the hardest/most-ambiguous ones (a
# non-random subset), not a harmless random sample of missingness.
JUDGE_FAILURE_HEADLINE_THRESHOLD = 0.02


def _judge_failure_rate(row: dict[str, Any]) -> float:
    m = (row.get("metrics") or {}).get("llm_judge")
    if not m or not m.get("n"):
        return 0.0
    return m.get("judge_failures", 0) / m["n"]


def _leaderboard_section(subset: list[dict[str, Any]]) -> list[str]:
    """Per-dataset ranked tables for one comparability tier."""
    by_ds: dict[str, list] = {}
    for r in subset:
        by_ds.setdefault(r["dataset"], []).append(r)
    out: list[str] = []
    for dsid, rs in by_ds.items():
        st = rs[0].get("split_type", "unverified")
        tag = "" if st == "official" else f"  ·  `split_type: {st}`"
        out += [f"### {dsid}{tag}", "",
                "| Model | Score | Metric | n | Cost (USD) | Notes |",
                "|---|---|---|---|---|---|"]
        for r in sorted(rs, key=lambda r: headline(r["metrics"])[1], reverse=True):
            key, val = headline(r["metrics"])
            rate = _judge_failure_rate(r)
            note = (f"⚠️ {rate:.1%} judge-ungraded — exceeds {JUDGE_FAILURE_HEADLINE_THRESHOLD:.0%}, "
                    "manual review recommended" if rate > JUDGE_FAILURE_HEADLINE_THRESHOLD else "")
            out.append(f"| {r['model']} | {val:.4f} | {key} | {r['n']} | "
                       f"{r.get('model_cost_usd', 0.0):.4f} | {note} |")
        out.append("")
    return out


def write_leaderboard(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Write leaderboard.json + leaderboard.md, keeping **officially-comparable**
    runs (``split_type == official``) in a separate section from internal ones."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "leaderboard.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # Open-ended (LLM-judge) scores are auxiliary until THIS row's specific judge
    # model + grading prompt clears calibration (not just "some judge, once") —
    # AND stay auxiliary regardless of calibration if THIS run's judge failed to
    # grade too many samples (a calibrated judge model doesn't make a flaky RUN
    # trustworthy; high-failure samples are plausibly the hardest ones, not a
    # random, harmless subset).
    report = _load_calibration_report(output_dir)
    auxiliary = [r for r in rows if _is_judge_headline(r) and (
        not _row_judge_calibrated(r, report)
        or _judge_failure_rate(r) > JUDGE_FAILURE_HEADLINE_THRESHOLD)]
    ranked = [r for r in rows if r not in auxiliary]
    official = [r for r in ranked if r.get("split_type", "unverified") == "official"]
    internal = [r for r in ranked if r.get("split_type", "unverified") != "official"]
    lines = ["# MedEval Leaderboard", ""]
    if official:
        lines += ["## ✅ Official (comparable)", ""]
        lines += _leaderboard_section(official)
    if internal:
        lines += ["## ⚠️ Internal / non-comparable", "", _INTERNAL_NOTE, ""]
        lines += _leaderboard_section(internal)
    if auxiliary:
        lines += ["## 🧪 Auxiliary (open-ended · LLM-judge, uncalibrated)", "",
                  _AUXILIARY_NOTE, ""]
        lines += _leaderboard_section(auxiliary)
    (output_dir / "leaderboard.md").write_text("\n".join(lines), encoding="utf-8")


class Runner:
    def __init__(self, config: dict[str, Any]):
        self.cfg = config
        run = config.get("run", {})
        self.output_dir = Path(run.get("output_dir", "./results"))
        self.concurrency = int(run.get("concurrency", 16))
        self.cache = bool(run.get("cache", True))
        # distributed sharding: this worker handles samples[shard_index::num_shards]
        self.num_shards = max(1, int(run.get("num_shards", 1) or 1))
        self.shard_index = int(run.get("shard_index", 0) or 0)
        self.sharded = self.num_shards > 1
        # When sweeping many datasets in one process (one model load), keep going if
        # a single dataset fails to load or eval (gated, offline, needs a server).
        self.continue_on_error = bool(run.get("continue_on_error", False))
        self.shard_suffix = f"__shard{self.shard_index}of{self.num_shards}" if self.sharded else ""
        self.eval_cfg = config.get("eval", {})
        self.gen_defaults: dict[str, Any] = dict(
            self.eval_cfg.get("gen", {"temperature": 0.0, "max_tokens": 1024}))
        self.default_judge = self.eval_cfg.get("judge_model")

        self.providers: dict[str, ModelProvider] = {}
        for m in config.get("models", []):
            prov = create_provider(m)
            prov.concurrency = self.concurrency
            self.providers[prov.id] = prov

        self.datasets = [create_dataset(d) for d in config.get("datasets", [])]
        # Apply the official-tier eligibility gate to EVERY dataset, regardless
        # of how split_type was set (explicit YAML, or a subclass default) — a
        # config can declare split_type: official, but for the static content
        # adapters that claim is only honored with mechanical pin evidence.
        for ds in self.datasets:
            enforce_official_eligibility(ds)
        self.cache_dir = self.output_dir / "cache"

    # --- judge / agent-support resolution --------------------------------
    def _judge_for(self, ds) -> ModelProvider | None:
        jid = ds.judge or self.default_judge
        if jid is None:
            return None
        if jid not in self.providers:
            raise ValueError(
                f"dataset {ds.id!r} needs judge {jid!r}, which is not in models[]")
        return self.providers[jid]

    def _agent_support(self, ds) -> dict[str, ModelProvider] | None:
        spec = getattr(ds, "support_spec", {}) or {}
        missing = sorted({mid for mid in spec.values() if mid not in self.providers})
        if missing:
            # Falling back to the scripted patient/moderator here would silently
            # swap the evaluation protocol (and its split_type claim). Fail loudly,
            # like _judge_for does for a missing judge.
            raise ValueError(
                f"dataset {ds.id!r} declares agent support model(s) {missing} that "
                "are not in models[]. Add them to the config (they are kept "
                "automatically under --models), or remove the support: block to "
                "use the offline scripted setup.")
        support = {role: self.providers[mid] for role, mid in spec.items()}
        return support or None

    # --- caching ----------------------------------------------------------
    def _support_signature(self, spec: dict[str, str]) -> dict[str, Any]:
        """Identity (+ pin, if any) of each agent support-role provider — a
        support model swap (or its own revision bump) changes the patient/
        measurement/moderator's behavior and must invalidate the doctor's
        cached rollouts too, since the trajectory depends on both."""
        out = {}
        for role, mid in spec.items():
            p = self.providers.get(mid)
            out[role] = {"id": mid, "model": getattr(p, "model", mid),
                        "revision": getattr(p, "revision", None)} if p else {"id": mid}
        return out

    def _cache_path(self, prov: ModelProvider, ds) -> Path:
        """Content-addressed generation cache path.

        Two independent layers of protection against silently scoring a stale
        generation against new/changed data:
          1. FILENAME signature (this method): model + effective gen params +
             the full dataset protocol config (revision, field_map, templates,
             adapter class + version, agent k/max_turns/support model identities)
             — any of these changing routes to a DIFFERENT cache file entirely.
          2. PER-SAMPLE content hash (``_sample_content_hash``, checked in
             ``_predict``): even with an IDENTICAL config, if an unpinned
             source's actual fetched content drifted for a given sample_id
             between runs, that sample's hash changes and it is treated as a
             cache miss — never scored with a stale output against a new
             reference.
        """
        merged = (prov._merge_gen(self.gen_defaults)
                  if hasattr(prov, "_merge_gen") else self.gen_defaults)
        key: dict[str, Any] = {
            "gen": merged,
            "model": getattr(prov, "model", prov.id),
            "model_revision": getattr(prov, "revision", None),
            "dataset_protocol": _dataset_protocol_key(ds),
        }
        if isinstance(ds, AgentAdapter):
            # Agent rollouts also depend on the rollout protocol: editing k /
            # max_turns / support in the config MUST invalidate the cache, or a
            # reliability re-run (k: 1 -> 3) silently reuses rollout lists of the
            # old length and reports the old k's pass^k.
            spec = getattr(ds, "support_spec", {}) or {}
            key["agent"] = {"k": getattr(ds, "k", 1),
                            "max_turns": getattr(ds, "max_turns", None),
                            "support": self._support_signature(spec)}
        sig = json.dumps(key, sort_keys=True, default=str)
        # hashlib, NOT the builtin hash(): hash() of a str is salted per process
        # via PYTHONHASHSEED, so it returns a different value every launch — the
        # cache filename would change on each run and resume would never find the
        # prior cache (the whole checkpoint-resume feature would be silently dead).
        h = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:12]
        return self.cache_dir / f"{_safe(ds.id)}__{_safe(prov.id)}__{h}{self.shard_suffix}.jsonl"

    def _load_cache(self, path: Path) -> dict[str, dict[str, Any]]:
        """Raw on-disk records keyed by sample_id (NOT yet ``Prediction``s — the
        caller must additionally check ``content_hash`` per sample before
        trusting a hit; see ``_predict``)."""
        out: dict[str, dict[str, Any]] = {}
        if not (self.cache and path.exists()):
            return out
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                # A crash mid-write (or a non-atomic Drive/FUSE flush) can leave a
                # truncated final JSONL line — skip it rather than poisoning the
                # whole resumable cache.
                continue
            out[d["sample_id"]] = d
        return out

    @staticmethod
    def _prediction_from_record(d: dict[str, Any]) -> Prediction:
        return Prediction(
            sample_id=d["sample_id"], generation=_gen_from_dict(d["generation"]),
            parsed=d.get("parsed"), rollouts=d.get("rollouts"),
            trajectory=d.get("trajectory"))

    def _append_cache(self, path: Path, pred: Prediction, content_hash: str) -> None:
        if not self.cache:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"sample_id": pred.sample_id, "content_hash": content_hash,
               "generation": _gen_to_dict(pred.generation),
               "parsed": pred.parsed if not isinstance(pred.parsed, set) else list(pred.parsed),
               "rollouts": pred.rollouts, "trajectory": pred.trajectory}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _append_cache_batch(self, path: Path, preds: list[Prediction],
                           content_hashes: list[str]) -> None:
        """Write a batch of predictions in one file open — Drive-friendly."""
        if not self.cache or not preds:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for pred, chash in zip(preds, content_hashes):
                rec = {
                    "sample_id": pred.sample_id,
                    "content_hash": chash,
                    "generation": _gen_to_dict(pred.generation),
                    "parsed": (pred.parsed if not isinstance(pred.parsed, set)
                               else list(pred.parsed)),
                    "rollouts": pred.rollouts,
                    "trajectory": pred.trajectory,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- prediction -------------------------------------------------------
    async def _predict(self, prov: ModelProvider, ds, samples: list[Sample]
                       ) -> dict[str, Prediction]:
        is_agent = isinstance(ds, AgentAdapter)
        cpath = self._cache_path(prov, ds)
        raw_cache = self._load_cache(cpath)

        # A cache hit requires BOTH the sample_id AND its current content hash
        # to match the stored record — an unpinned source that silently drifted
        # (same id, different question/reference) is treated as a miss, not a
        # stale reuse. Records from before this check existed (no content_hash
        # field) also miss, so upgrading medeval pays a one-time cache-bust
        # rather than keep trusting unverifiable old entries.
        preds: dict[str, Prediction] = {}
        todo: list[Sample] = []
        for s in samples:
            rec = raw_cache.get(s.id)
            if rec is not None and rec.get("content_hash") == _sample_content_hash(s):
                preds[s.id] = self._prediction_from_record(rec)
            else:
                todo.append(s)

        n_cached, n_total = len(preds), len(samples)
        if not todo:
            return preds

        if n_cached:
            print(f"[medeval]   resuming: {n_cached}/{n_total} cached, "
                  f"{len(todo)} remaining", flush=True)

        pbar = _make_pbar(n_total, ds.id, initial=n_cached)

        if is_agent:
            support = self._agent_support(ds)
            # Some agent adapters cap concurrency below the run default for
            # correctness (MedAgentBench: concurrent episodes racing against
            # one shared, unreset live FHIR server is a correctness risk, not
            # just contention) — consult the dataset if it declares one.
            eff_concurrency = getattr(ds, "effective_concurrency", lambda r: r)(self.concurrency)
            sem = asyncio.Semaphore(max(1, eff_concurrency))

            async def roll(s: Sample) -> None:
                async with sem:
                    pred = await ds.rollout(s, prov, gen=self.gen_defaults, support=support)
                preds[s.id] = pred
                self._append_cache(cpath, pred, _sample_content_hash(s))
                pbar.update(1)

            await asyncio.gather(*(roll(s) for s in todo))
        else:
            chunk_size = max(1, int(
                self.cfg.get("run", {}).get("checkpoint_every", 64)))
            for i in range(0, len(todo), chunk_size):
                chunk = todo[i: i + chunk_size]
                gens = await prov.agenerate_many(
                    [s.messages for s in chunk], **self.gen_defaults)
                batch_preds: list[Prediction] = []
                for s, g in zip(chunk, gens):
                    pred = ds.parse(s, g.text)
                    pred.generation = g
                    preds[s.id] = pred
                    batch_preds.append(pred)
                self._append_cache_batch(
                    cpath, batch_preds, [_sample_content_hash(s) for s in chunk])
                pbar.update(len(batch_preds))

        pbar.close()
        return preds

    # --- scoring ----------------------------------------------------------
    async def _score_all(self, metric, samples, preds) -> list:
        sem = asyncio.Semaphore(self.concurrency)
        pbar = (_make_pbar(len(samples), f"scoring [{metric.metric_name}]")
                if metric.needs_judge and len(samples) > 10 else None)

        async def one(s: Sample):
            async with sem:
                result = await metric.score(s, preds[s.id])
            if pbar:
                pbar.update(1)
            return result

        results = await asyncio.gather(*(one(s) for s in samples))
        if pbar:
            pbar.close()
        return results

    async def _eval(self, prov: ModelProvider, ds, samples) -> dict[str, Any]:
        preds = await self._predict(prov, ds, samples)
        metrics = []
        judge_sig: dict[str, Any] | None = None
        for name, mcfg in ds.metric_specs:
            m = create_metric(name, mcfg)
            if m.needs_judge:
                judge = self._judge_for(ds)
                if judge is None:
                    raise ValueError(
                        f"{ds.id}: metric {name} needs a judge; set eval.judge_model "
                        "or dataset.judge")
                m.judge = judge
                if name == "llm_judge":
                    # Recorded so a calibration report can be checked against THE
                    # SAME judge model+revision and grading prompt this row
                    # actually used — see _row_judge_calibrated.
                    judge_sig = {
                        "judge_id": judge.id,
                        "judge_model": getattr(judge, "model", judge.id),
                        "judge_revision": getattr(judge, "revision", None),
                        "prompt_style": ("healthbench_per_criterion"
                                        if getattr(m, "per_criterion", False)
                                        else "default_rubric"),
                    }
            metrics.append(m)

        agg: dict[str, Any] = {}
        scores_by_metric: dict[str, list] = {}
        for m in metrics:
            scores = await self._score_all(m, samples, preds)
            scores_by_metric[m.metric_name] = scores
            agg[m.metric_name] = m.aggregate(scores)

        model_cost = sum(preds[s.id].generation.cost_usd for s in samples)
        self._write_detail(prov, ds, samples, preds, scores_by_metric)
        row: dict[str, Any] = {"model": prov.id, "dataset": ds.id, "n": len(samples),
                               "split_type": getattr(ds, "split_type", "unverified"),
                               "metrics": agg, "model_cost_usd": round(model_cost, 6)}
        if judge_sig is not None:
            row["judge_signature"] = judge_sig
        # Agent support-role (patient/measurement/moderator) cost, summed across
        # every sample — without this a faithful multi-agent AgentClinic run
        # (3 extra LLMs/turn) is invisible next to model_cost_usd (the doctor
        # only), making single-agent and multi-agent setups look equally cheap.
        role_cost: dict[str, float] = {}
        for s in samples:
            for role, d in (preds[s.id].support_cost or {}).items():
                role_cost[role] = role_cost.get(role, 0.0) + d.get("cost_usd", 0.0)
        if role_cost:
            row["role_cost_usd"] = {"doctor": round(model_cost, 6),
                                    **{k: round(v, 6) for k, v in role_cost.items()}}
        return row

    # --- output -----------------------------------------------------------
    def _write_detail(self, prov, ds, samples, preds, scores_by_metric) -> None:
        path = self.output_dir / f"detail__{_safe(prov.id)}__{_safe(ds.id)}{self.shard_suffix}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for i, s in enumerate(samples):
                p = preds[s.id]
                prompt = s.messages[-1].content if s.messages else ""
                row = {
                    "sample_id": s.id, "task": s.task_type.value,
                    "split_type": getattr(ds, "split_type", "unverified"),
                    "prompt": prompt,
                    "choices": s.choices,
                    "prediction": p.generation.text[:2000],
                    "parsed": p.parsed if not isinstance(p.parsed, set) else list(p.parsed),
                    "reference": s.reference,
                    "cost_usd": p.generation.cost_usd,
                    "prompt_tokens": p.generation.prompt_tokens,
                    "completion_tokens": p.generation.completion_tokens,
                    # surface length-capped generations: a CoT trace that runs past
                    # max_tokens never emits its "Answer:" line and scores 0 — this
                    # flag lets you spot a truncation-driven accuracy drop in review.
                    "truncated": p.generation.finish_reason == "length",
                    "scores": {name: {"value": sc[i].value, "detail": sc[i].detail}
                               for name, sc in scores_by_metric.items()},
                }
                if p.rollouts is not None:
                    row["rollouts"] = p.rollouts
                if p.support_cost is not None:
                    row["support_cost"] = p.support_cost
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_leaderboard(self, rows: list[dict[str, Any]]) -> None:
        if self.sharded:  # partial result; the global leaderboard comes from `merge`
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / f"leaderboard{self.shard_suffix}.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            return
        write_leaderboard(rows, self.output_dir)

    def _write_run_manifest(self, leaderboard: list[dict[str, Any]]) -> None:
        """Record exactly what produced this run — the code version, every
        dataset's resolved comparability tier + pin evidence + load reliability,
        every model's resolved identity, and judge-calibration status — so a
        result can be traced back to a precise, reproducible configuration
        rather than "whatever the YAML said this week"."""
        datasets_info = []
        for ds in self.datasets:
            adapter_name = getattr(ds, "adapter_name", type(ds).__name__)
            datasets_info.append({
                "id": ds.id,
                "adapter": adapter_name,
                "split_type": getattr(ds, "split_type", "unverified"),
                "pin_evidence": (has_pin_evidence(ds)
                                if adapter_name in CONTENT_ADAPTERS else None),
                "protocol_hash": _protocol_hash(ds),
                "load_stats": getattr(ds, "load_stats", {}),
            })
        models_info = [
            {"id": p.id, "type": p.provider_type, "model": getattr(p, "model", None),
             "revision": getattr(p, "revision", None), "judge_only": p.judge_only}
            for p in self.providers.values()
        ]
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "adapter_protocol_version": ADAPTER_PROTOCOL_VERSION,
            "config": _redact_secrets(self.cfg),
            "datasets": datasets_info,
            "models": models_info,
            "judge_calibrated": _judge_calibrated(self.output_dir),
            "rows": len(leaderboard),
            "shard": (f"{self.shard_index}/{self.num_shards}" if self.sharded else None),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / f"run_manifest{self.shard_suffix}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # --- entry points -----------------------------------------------------
    async def arun(self) -> list[dict[str, Any]]:
        leaderboard: list[dict[str, Any]] = []
        failures: list[str] = []
        for ds in self.datasets:
            try:
                samples = ds.load()
            except Exception as e:  # dataset failed to load (gated / offline / moved)
                if not self.continue_on_error:
                    raise
                msg = f"{ds.id}: load failed — {type(e).__name__}: {e}"
                print(f"[medeval] SKIP {msg}")
                failures.append(msg)
                continue
            if self.sharded:
                samples = samples[self.shard_index::self.num_shards]  # strided shard
            if not samples:
                print(f"[medeval] WARNING: dataset {ds.id!r} produced 0 samples"
                      + (f" for shard {self.shard_index}/{self.num_shards}" if self.sharded else ""))
            for prov in self.providers.values():
                if prov.judge_only:
                    continue
                tag = f" [shard {self.shard_index}/{self.num_shards}]" if self.sharded else ""
                print(f"[medeval] {prov.id} × {ds.id}  ({len(samples)} samples){tag}")
                try:
                    row = await self._eval(prov, ds, samples)
                except Exception as e:  # eval/score failed for this (model, dataset)
                    if not self.continue_on_error:
                        raise
                    msg = f"{prov.id} × {ds.id}: eval failed — {type(e).__name__}: {e}"
                    print(f"[medeval] SKIP {msg}")
                    failures.append(msg)
                    continue
                leaderboard.append(row)
                key, val = headline(row["metrics"])
                print(f"           -> {key}={val:.4f}")
                self._write_leaderboard(leaderboard)  # incremental flush after each (model, dataset)
        if failures:
            print(f"[medeval] {len(failures)} dataset(s) skipped:")
            for m in failures:
                print(f"           - {m}")
        for prov in self.providers.values():
            await prov.aclose()
        self._write_run_manifest(leaderboard)
        if self.sharded:
            print(f"[medeval] shard {self.shard_index}/{self.num_shards} done; "
                  f"run `medeval merge {self.output_dir}` to build the leaderboard")
        else:
            print(f"[medeval] wrote {self.output_dir/'leaderboard.md'}")
        return leaderboard

    def run(self) -> list[dict[str, Any]]:
        return asyncio.run(self.arun())


def run_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a :class:`Runner` from a config dict and execute it."""
    return Runner(config).run()
