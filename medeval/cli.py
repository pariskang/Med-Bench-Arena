"""Command-line interface: ``python -m medeval run config.yaml``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from .runner import run_config
from .providers.base import available_providers
from .datasets.base import available_adapters
from .metrics.base import available_metrics


def _load_config(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    if args.limit is not None:
        for d in cfg.get("datasets", []):
            d["limit"] = args.limit
    if args.output:
        cfg.setdefault("run", {})["output_dir"] = args.output
    if args.no_cache:
        cfg.setdefault("run", {})["cache"] = False
    if args.num_shards and args.num_shards > 1:
        cfg.setdefault("run", {})["num_shards"] = args.num_shards
        cfg.setdefault("run", {})["shard_index"] = args.shard
    rows = run_config(cfg)
    print(f"[medeval] {len(rows)} (model × dataset) results")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    from .distributed import merge_results
    rows = merge_results(args.results_dir, args.output)
    print(f"[medeval] merged {len(rows)} (model × dataset) results")
    return 0


def cmd_pool(args: argparse.Namespace) -> int:
    if args.backend == "ray":
        from .distributed import run_ray
        run_ray(args.config, args.num_shards, num_gpus=args.ray_num_gpus,
                address=args.ray_address, output_dir=args.output,
                limit=args.limit, cache=not args.no_cache)
        return 0
    from .distributed import run_pool
    extra = []
    if args.limit is not None:
        extra += ["--limit", str(args.limit)]
    if args.no_cache:
        extra += ["--no-cache"]
    run_pool(args.config, args.num_shards, gpus=args.gpus,
             output_dir=args.output, extra_args=extra)
    return 0


def cmd_slurm(args: argparse.Namespace) -> int:
    from .distributed import submit_slurm
    submit_slurm(args.config, args.num_shards, output_dir=args.output,
                 partition=args.partition, gpus_per_task=args.gpus_per_task,
                 cpus_per_task=args.cpus, mem=args.mem, time=args.time,
                 account=args.account, setup=args.setup or "",
                 max_parallel=args.max_parallel, limit=args.limit,
                 no_cache=args.no_cache, submit=not args.no_submit)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    print("providers:", ", ".join(available_providers()))
    print("adapters :", ", ".join(available_adapters()))
    print("metrics  :", ", ".join(available_metrics()))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .submit import export
    out = export(args.results_dir, args.format, args.output,
                 test_dir=args.medbench_test_dir)
    print(f"[medeval] wrote {args.format} submission to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="medeval", description="MedEval runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run an evaluation from a YAML config")
    p_run.add_argument("config", help="path to the YAML run spec")
    p_run.add_argument("--limit", type=int, default=None,
                       help="cap samples per dataset (overrides config)")
    p_run.add_argument("--output", default=None, help="override run.output_dir")
    p_run.add_argument("--no-cache", action="store_true", help="disable generation cache")
    p_run.add_argument("--num-shards", type=int, default=1,
                       help="total shards for distributed runs")
    p_run.add_argument("--shard", type=int, default=0, help="this worker's shard index")
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list", help="list registered providers / adapters / metrics")
    p_list.set_defaults(func=cmd_list)

    p_exp = sub.add_parser("export", help="export results as an OpenCompass / MedBench submission")
    p_exp.add_argument("results_dir", help="a run's output dir (with detail__*.jsonl)")
    p_exp.add_argument("--format", required=True, choices=["opencompass", "medbench"])
    p_exp.add_argument("--output", "--out", required=True, dest="output")
    p_exp.add_argument("--medbench-test-dir", default=None,
                       help="MedBench data tree; fills answers into original *_test.jsonl")
    p_exp.set_defaults(func=cmd_export)

    p_mrg = sub.add_parser("merge", help="merge sharded results into one leaderboard")
    p_mrg.add_argument("results_dir", help="dir containing detail__*.jsonl shard files")
    p_mrg.add_argument("--output", default=None, help="where to write the leaderboard")
    p_mrg.set_defaults(func=cmd_merge)

    p_pool = sub.add_parser("pool", help="run N shards in parallel (local procs or Ray), then merge")
    p_pool.add_argument("config", help="path to the YAML run spec")
    p_pool.add_argument("--num-shards", type=int, required=True)
    p_pool.add_argument("--backend", choices=["local", "ray"], default="local")
    p_pool.add_argument("--gpus", default=None,
                        help="local: comma-separated CUDA device ids (round-robin per shard)")
    p_pool.add_argument("--ray-num-gpus", type=float, default=0,
                        help="ray: GPUs reserved per shard task")
    p_pool.add_argument("--ray-address", default=None, help="ray: cluster address (or RAY_ADDRESS)")
    p_pool.add_argument("--output", default=None, help="override run.output_dir")
    p_pool.add_argument("--limit", type=int, default=None, help="cap samples per dataset")
    p_pool.add_argument("--no-cache", action="store_true")
    p_pool.set_defaults(func=cmd_pool)

    p_slurm = sub.add_parser("slurm", help="generate + submit a Slurm job array (one task per shard) + merge")
    p_slurm.add_argument("config", help="path to the YAML run spec")
    p_slurm.add_argument("--num-shards", type=int, required=True)
    p_slurm.add_argument("--output", default=None, help="override run.output_dir (shared FS)")
    p_slurm.add_argument("--partition", default=None)
    p_slurm.add_argument("--gpus-per-task", type=int, default=0)
    p_slurm.add_argument("--cpus", type=int, default=4)
    p_slurm.add_argument("--mem", default=None)
    p_slurm.add_argument("--time", default="04:00:00")
    p_slurm.add_argument("--account", default=None)
    p_slurm.add_argument("--max-parallel", type=int, default=None,
                         help="cap concurrent array tasks (sbatch %% throttle)")
    p_slurm.add_argument("--setup", default=None,
                         help="shell to run before the command (module load / conda activate)")
    p_slurm.add_argument("--limit", type=int, default=None)
    p_slurm.add_argument("--no-cache", action="store_true")
    p_slurm.add_argument("--no-submit", action="store_true", help="write scripts but don't sbatch")
    p_slurm.set_defaults(func=cmd_slurm)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
