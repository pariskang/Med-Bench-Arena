"""Distributed scheduling: shard → run → merge.

The grid is embarrassingly parallel, so MedEval distributes by **strided
sharding**: each worker runs ``samples[i::N]`` and writes its own shard-scoped
detail + cache files (``..._shard{i}of{N}.jsonl``), so workers never collide and
each is independently resumable. There is no central server — workers can be
separate processes on one box or separate machines over a shared filesystem.

* ``run_pool`` launches N local workers (optionally one per GPU) and merges.
* multi-machine: run ``medeval run cfg --shard i --num-shards N`` on each, then
  ``medeval merge <shared_dir>`` once all are done.
* ``merge_results`` re-aggregates the per-sample scores from every shard into the
  final leaderboard (correct even when shards have unequal sizes — it aggregates
  raw scores, not means-of-means).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from .schema import Score
from .metrics.base import create_metric
from .runner import write_leaderboard

_SHARD_RE = re.compile(r"__shard\d+of\d+$")


def merge_results(results_dir: str | Path, output_dir: str | Path | None = None
                  ) -> list[dict[str, Any]]:
    """Combine all detail__*.jsonl shards in ``results_dir`` into a leaderboard."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir

    # group shard files by (model, dataset)
    groups: dict[tuple[str, str], list[Path]] = {}
    for fp in sorted(results_dir.glob("detail__*.jsonl")):
        stem = _SHARD_RE.sub("", fp.stem[len("detail__"):])
        model, _, dataset = stem.partition("__")
        groups.setdefault((model, dataset), []).append(fp)

    import json
    rows: list[dict[str, Any]] = []
    for (model, dataset), files in groups.items():
        seen: dict[str, dict] = {}
        for fp in files:
            for line in fp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    seen[r["sample_id"]] = r          # dedupe across shards
        detail = list(seen.values())

        metric_names: set[str] = set()
        for r in detail:
            metric_names |= set((r.get("scores") or {}).keys())
        agg: dict[str, Any] = {}
        for name in sorted(metric_names):
            scores = [Score(metric=name, value=r["scores"][name]["value"],
                            detail=r["scores"][name].get("detail", {}))
                      for r in detail if name in (r.get("scores") or {})]
            try:
                agg[name] = create_metric(name).aggregate(scores)
            except Exception as e:  # unknown/un-loaded metric -> mean fallback
                vals = [s.value for s in scores]
                agg[name] = {name: sum(vals) / len(vals) if vals else 0.0, "n": len(scores),
                             "warning": str(e)}
        cost = sum(float(r.get("cost_usd", 0.0)) for r in detail)
        rows.append({"model": model, "dataset": dataset, "n": len(detail),
                     "metrics": agg, "model_cost_usd": round(cost, 6),
                     "shards": len(files)})

    write_leaderboard(rows, output_dir)
    print(f"[medeval] merged {sum(r['shards'] for r in rows)} shard files -> "
          f"{output_dir/'leaderboard.md'}")
    return rows


def _resolve_output_dir(config_path: str | Path, override: str | None) -> Path:
    if override:
        return Path(override)
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return Path((cfg.get("run") or {}).get("output_dir", "./results"))


def run_pool(config_path: str | Path, num_shards: int, gpus: str | None = None,
             output_dir: str | None = None, extra_args: list[str] | None = None
             ) -> list[dict[str, Any]]:
    """Launch ``num_shards`` local worker processes, then merge their results.

    ``gpus``: comma-separated device ids (e.g. ``"0,1,2,3"``) assigned round-robin
    via ``CUDA_VISIBLE_DEVICES`` — for data-parallel HF/vLLM inference.
    """
    out = _resolve_output_dir(config_path, output_dir)
    gpu_list = [g.strip() for g in gpus.split(",")] if gpus else []
    extra = extra_args or []

    procs: list[subprocess.Popen] = []
    for i in range(num_shards):
        cmd = [sys.executable, "-m", "medeval", "run", str(config_path),
               "--shard", str(i), "--num-shards", str(num_shards),
               "--output", str(out), *extra]
        env = dict(os.environ)
        if gpu_list:
            env["CUDA_VISIBLE_DEVICES"] = gpu_list[i % len(gpu_list)]
        print(f"[medeval] launching shard {i}/{num_shards}"
              + (f" on GPU {env['CUDA_VISIBLE_DEVICES']}" if gpu_list else ""))
        procs.append(subprocess.Popen(cmd, env=env))

    failures = [i for i, p in enumerate(procs) if p.wait() != 0]
    if failures:
        raise RuntimeError(f"shards {failures} failed; not merging")
    return merge_results(out)
