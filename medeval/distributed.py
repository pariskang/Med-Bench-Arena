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
                vals = [s.value for s in scores if s.value is not None]
                agg[name] = {name: sum(vals) / len(vals) if vals else 0.0, "n": len(scores),
                             "warning": str(e)}
        cost = sum(float(r.get("cost_usd", 0.0)) for r in detail)
        split_type = next((r.get("split_type") for r in detail if r.get("split_type")),
                          "unverified")
        rows.append({"model": model, "dataset": dataset, "n": len(detail),
                     "split_type": split_type,
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


def _load_config(config: str | Path | dict) -> dict:
    if isinstance(config, dict):
        return config
    return yaml.safe_load(Path(config).read_text(encoding="utf-8"))


def run_ray(config: str | Path | dict, num_shards: int, num_gpus: float = 0,
            address: str | None = None, output_dir: str | None = None,
            limit: int | None = None, cache: bool = True) -> list[dict[str, Any]]:
    """Distribute shards as Ray tasks (one per shard), then merge.

    Each task runs ``Runner`` with its shard params; assumes a shared filesystem
    for ``output_dir`` (true on one node or a cluster with shared storage).
    ``num_gpus`` reserves GPUs per shard for HF/vLLM. Connects to an existing
    cluster via ``address`` / ``RAY_ADDRESS``, else starts a local Ray.
    """
    import copy
    import ray

    cfg = copy.deepcopy(_load_config(config))
    cfg.setdefault("run", {})
    if output_dir:
        cfg["run"]["output_dir"] = output_dir
    if limit is not None:
        for d in cfg.get("datasets", []):
            d["limit"] = limit
    cfg["run"]["cache"] = cache
    out = Path(cfg["run"].get("output_dir", "./results"))

    repo_root = str(Path(__file__).resolve().parents[1])
    pythonpath = os.pathsep.join(filter(None, [os.environ.get("PYTHONPATH", ""), repo_root]))
    ray.init(address=address or os.environ.get("RAY_ADDRESS"),
             ignore_reinit_error=True, configure_logging=False,
             runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})

    @ray.remote(num_gpus=num_gpus)
    def _shard_task(c: dict, i: int, n: int) -> dict:
        from medeval.runner import Runner
        cc = copy.deepcopy(c)
        cc["run"]["num_shards"] = n
        cc["run"]["shard_index"] = i
        return {"shard": i, "rows": len(Runner(cc).run())}

    try:
        refs = [_shard_task.remote(cfg, i, num_shards) for i in range(num_shards)]
        ray.get(refs)
    finally:
        ray.shutdown()
    return merge_results(out)


_SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=medeval
#SBATCH --array=0-{last}{throttle}
#SBATCH --output={out}/slurm/%A_%a.out
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time}
{extra_directives}
set -euo pipefail
{setup}
python -m medeval run {config} \\
    --shard ${{SLURM_ARRAY_TASK_ID}} --num-shards {n} --output {out}{run_args}
"""

_MERGE_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=medeval-merge
#SBATCH --output={out}/slurm/merge_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
{merge_directives}
set -euo pipefail
{setup}
python -m medeval merge {out}
"""


def submit_slurm(config: str | Path, num_shards: int, output_dir: str | None = None,
                 partition: str | None = None, gpus_per_task: int = 0,
                 cpus_per_task: int = 4, mem: str | None = None, time: str = "04:00:00",
                 account: str | None = None, setup: str = "", max_parallel: int | None = None,
                 limit: int | None = None, no_cache: bool = False, submit: bool = True
                 ) -> dict[str, Any]:
    """Generate a Slurm **job array** (one task per shard) + a dependent merge job.

    Writes the sbatch scripts under ``<output_dir>/slurm/`` and, if ``submit`` and
    ``sbatch`` is available, submits the array and a ``afterok`` merge job. With no
    cluster it just writes the scripts (so you can inspect/submit them yourself).
    ``setup`` is shell run before the command (e.g. ``module load`` / ``conda activate``).
    """
    out = Path(output_dir or (_load_config(config).get("run") or {}).get("output_dir", "./results"))
    (out / "slurm").mkdir(parents=True, exist_ok=True)

    directives = []
    if partition:
        directives.append(f"#SBATCH --partition={partition}")
    if gpus_per_task:
        directives.append(f"#SBATCH --gres=gpu:{gpus_per_task}")
    if mem:
        directives.append(f"#SBATCH --mem={mem}")
    if account:
        directives.append(f"#SBATCH --account={account}")
    run_args = ""
    if limit is not None:
        run_args += f" --limit {limit}"
    if no_cache:
        run_args += " --no-cache"

    array_sh = _SBATCH_TEMPLATE.format(
        last=num_shards - 1, throttle=f"%{max_parallel}" if max_parallel else "",
        out=out, cpus=cpus_per_task, time=time,
        extra_directives="\n".join(directives), setup=setup,
        config=config, n=num_shards, run_args=run_args)
    merge_sh = _MERGE_TEMPLATE.format(
        out=out, setup=setup,
        merge_directives=f"#SBATCH --partition={partition}" if partition else "")

    array_path = out / "slurm" / "medeval_array.sbatch"
    merge_path = out / "slurm" / "medeval_merge.sbatch"
    array_path.write_text(array_sh, encoding="utf-8")
    merge_path.write_text(merge_sh, encoding="utf-8")

    import shutil
    if submit and shutil.which("sbatch"):
        jid = subprocess.run(["sbatch", "--parsable", str(array_path)],
                             capture_output=True, text=True, check=True).stdout.strip()
        merge = subprocess.run(["sbatch", "--parsable", f"--dependency=afterok:{jid}",
                                str(merge_path)], capture_output=True, text=True, check=True).stdout.strip()
        print(f"[medeval] submitted array job {jid} + merge job {merge} (afterok)")
        return {"array_job": jid, "merge_job": merge,
                "array_script": str(array_path), "merge_script": str(merge_path)}

    print(f"[medeval] wrote {array_path} and {merge_path}\n"
          f"          submit with: sbatch {array_path} && "
          f"sbatch --dependency=afterok:<jobid> {merge_path}")
    return {"array_script": str(array_path), "merge_script": str(merge_path), "submitted": False}
