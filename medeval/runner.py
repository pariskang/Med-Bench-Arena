"""Orchestrator: schedule, cache, resume, score, rank.

Wires the three decoupled layers together for the full ``N datasets × M models ×
K metrics`` grid. Generations are cached on disk keyed by
``hash(model + gen-params + sample)`` so reruns are cheap and resumable.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schema import Generation, Prediction, Sample
from .providers.base import create_provider, ModelProvider
from .datasets.base import create_dataset
from .datasets.agent_env import AgentAdapter
from .metrics.base import create_metric

# Import implementation modules so their @register_* decorators run.
from .providers import mock, litellm_provider, poe, hf  # noqa: F401
from .datasets import hf_mcq, local_json, agent_env, tcmbench, medbench  # noqa: F401
from .metrics import (  # noqa: F401
    mcq, llm_judge, text_match, prescription, syndrome, tcm_struct, numeric)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s))


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
    "partial set, or a built-in (approximate) grader — see each section's "
    "`split_type`. Reported for internal tracking only; do **not** publish them "
    "as official scores.")

_AUXILIARY_NOTE = (
    "> ⚠️ **Auxiliary metric — open-ended, LLM-judge scored.** These scores come "
    "from an LLM-as-judge whose agreement with human experts has not (yet) cleared "
    "calibration, so they are reported as a secondary signal and never as a headline "
    "rank. Calibrate a judge with `medeval calibrate` (see `data/calibration/`); a "
    "judge becomes headline-eligible only when it matches the physician ceiling **and** "
    "reaches substantial absolute agreement (κ ≥ 0.40). On HealthBench's physician "
    "meta-eval a strong-model judge is *physician-equivalent* yet only *moderate* in "
    "absolute terms (κ≈0.39 vs a 0.44 human ceiling) — rubric grading is intrinsically "
    "subjective — so open-ended scores stay auxiliary by default.")


def _judge_calibrated(output_dir: Path) -> bool:
    """True iff a calibration report marking the judge headline-eligible is present.

    Looks in the run's own dir then the canonical ``data/calibration/``. Absent or
    failing report → open-ended judge scores are treated as auxiliary (the safe default).
    """
    for cand in (output_dir / "calibration_report.json",
                 Path("data/calibration/calibration_report.json")):
        try:
            data = json.loads(cand.read_text(encoding="utf-8"))
            if data.get("verdict", {}).get("calibrated") is True:
                return True
        except Exception:
            continue
    return False


def _is_judge_headline(row: dict[str, Any]) -> bool:
    """Whether this row is ranked by an LLM-judge open-ended score."""
    return headline(row["metrics"])[0] == "judge_score"


def _leaderboard_section(subset: list[dict[str, Any]]) -> list[str]:
    """Per-dataset ranked tables for one comparability tier."""
    by_ds: dict[str, list] = {}
    for r in subset:
        by_ds.setdefault(r["dataset"], []).append(r)
    out: list[str] = []
    for dsid, rs in by_ds.items():
        st = rs[0].get("split_type", "official")
        tag = "" if st == "official" else f"  ·  `split_type: {st}`"
        out += [f"### {dsid}{tag}", "",
                "| Model | Score | Metric | n | Cost (USD) |",
                "|---|---|---|---|---|"]
        for r in sorted(rs, key=lambda r: headline(r["metrics"])[1], reverse=True):
            key, val = headline(r["metrics"])
            out.append(f"| {r['model']} | {val:.4f} | {key} | {r['n']} | "
                       f"{r.get('model_cost_usd', 0.0):.4f} |")
        out.append("")
    return out


def write_leaderboard(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Write leaderboard.json + leaderboard.md, keeping **officially-comparable**
    runs (``split_type == official``) in a separate section from internal ones."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "leaderboard.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # Open-ended (LLM-judge) scores are auxiliary until the judge clears calibration.
    judge_ok = _judge_calibrated(output_dir)
    auxiliary = [r for r in rows if _is_judge_headline(r) and not judge_ok]
    ranked = [r for r in rows if r not in auxiliary]
    official = [r for r in ranked if r.get("split_type", "official") == "official"]
    internal = [r for r in ranked if r.get("split_type", "official") != "official"]
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
        support = {role: self.providers[mid] for role, mid in spec.items()
                   if mid in self.providers}
        return support or None

    # --- caching ----------------------------------------------------------
    def _cache_path(self, prov: ModelProvider, ds) -> Path:
        sig = json.dumps({"gen": self.gen_defaults, "model": getattr(prov, "model", prov.id)},
                         sort_keys=True)
        h = abs(hash(sig)) % (10 ** 10)
        return self.cache_dir / f"{_safe(ds.id)}__{_safe(prov.id)}__{h}{self.shard_suffix}.jsonl"

    def _load_cache(self, path: Path) -> dict[str, Prediction]:
        out: dict[str, Prediction] = {}
        if not (self.cache and path.exists()):
            return out
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            out[d["sample_id"]] = Prediction(
                sample_id=d["sample_id"], generation=_gen_from_dict(d["generation"]),
                parsed=d.get("parsed"), rollouts=d.get("rollouts"),
                trajectory=d.get("trajectory"))
        return out

    def _append_cache(self, path: Path, pred: Prediction) -> None:
        if not self.cache:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"sample_id": pred.sample_id, "generation": _gen_to_dict(pred.generation),
               "parsed": pred.parsed if not isinstance(pred.parsed, set) else list(pred.parsed),
               "rollouts": pred.rollouts, "trajectory": pred.trajectory}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _append_cache_batch(self, path: Path, preds: list[Prediction]) -> None:
        """Write a batch of predictions in one file open — Drive-friendly."""
        if not self.cache or not preds:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for pred in preds:
                rec = {
                    "sample_id": pred.sample_id,
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
        preds = self._load_cache(cpath)
        todo = [s for s in samples if s.id not in preds]

        n_cached, n_total = len(preds), len(samples)
        if not todo:
            return preds

        if n_cached:
            print(f"[medeval]   resuming: {n_cached}/{n_total} cached, "
                  f"{len(todo)} remaining", flush=True)

        pbar = _make_pbar(n_total, ds.id, initial=n_cached)

        if is_agent:
            support = self._agent_support(ds)
            sem = asyncio.Semaphore(self.concurrency)

            async def roll(s: Sample) -> None:
                async with sem:
                    pred = await ds.rollout(s, prov, gen=self.gen_defaults, support=support)
                preds[pred.sample_id] = pred
                self._append_cache(cpath, pred)
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
                self._append_cache_batch(cpath, batch_preds)
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
        for name, mcfg in ds.metric_specs:
            m = create_metric(name, mcfg)
            if m.needs_judge:
                judge = self._judge_for(ds)
                if judge is None:
                    raise ValueError(
                        f"{ds.id}: metric {name} needs a judge; set eval.judge_model "
                        "or dataset.judge")
                m.judge = judge
            metrics.append(m)

        agg: dict[str, Any] = {}
        scores_by_metric: dict[str, list] = {}
        for m in metrics:
            scores = await self._score_all(m, samples, preds)
            scores_by_metric[m.metric_name] = scores
            agg[m.metric_name] = m.aggregate(scores)

        model_cost = sum(preds[s.id].generation.cost_usd for s in samples)
        self._write_detail(prov, ds, samples, preds, scores_by_metric)
        return {"model": prov.id, "dataset": ds.id, "n": len(samples),
                "split_type": getattr(ds, "split_type", "official"),
                "metrics": agg, "model_cost_usd": round(model_cost, 6)}

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
                    "split_type": getattr(ds, "split_type", "official"),
                    "prompt": prompt,
                    "choices": s.choices,
                    "prediction": p.generation.text[:2000],
                    "parsed": p.parsed if not isinstance(p.parsed, set) else list(p.parsed),
                    "reference": s.reference,
                    "cost_usd": p.generation.cost_usd,
                    "prompt_tokens": p.generation.prompt_tokens,
                    "completion_tokens": p.generation.completion_tokens,
                    "scores": {name: {"value": sc[i].value, "detail": sc[i].detail}
                               for name, sc in scores_by_metric.items()},
                }
                if p.rollouts is not None:
                    row["rollouts"] = p.rollouts
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_leaderboard(self, rows: list[dict[str, Any]]) -> None:
        if self.sharded:  # partial result; the global leaderboard comes from `merge`
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / f"leaderboard{self.shard_suffix}.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            return
        write_leaderboard(rows, self.output_dir)

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
