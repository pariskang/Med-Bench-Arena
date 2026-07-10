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


def _apply_model_filter(cfg: dict[str, Any], models_arg: str) -> list[str]:
    """Filter ``cfg['models']`` down to the --models selection.

    Judges (``judge_only``) are always kept. Models referenced by a dataset's
    agent ``support:`` block (patient / measurement / moderator) are kept too —
    dropping them would silently swap the faithful multi-agent setup for the
    scripted fallback, changing the evaluation protocol. A support model that
    was not itself requested is demoted to ``judge_only`` so it serves its role
    without also being evaluated as a candidate. Returns the requested ids that
    matched nothing (an error for the caller to report)."""
    wanted = {m.strip() for m in models_arg.split(",") if m.strip()}
    support_ids = {mid for d in cfg.get("datasets", [])
                   for mid in (d.get("support") or {}).values()}
    kept: list[dict[str, Any]] = []
    for m in cfg.get("models", []):
        mid = m.get("id")
        if mid in wanted or m.get("judge_only"):
            kept.append(m)
        elif mid in support_ids:
            kept.append({**m, "judge_only": True})   # role provider, not a candidate
    missing = wanted - {m.get("id") for m in kept}
    cfg["models"] = kept
    return sorted(missing)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    if args.models:
        missing = _apply_model_filter(cfg, args.models)
        if missing:
            print(f"[medeval] --models: no such model id(s): {missing}")
            return 2
    if args.limit is not None:
        for d in cfg.get("datasets", []):
            d["limit"] = args.limit
    if args.output:
        cfg.setdefault("run", {})["output_dir"] = args.output
    if args.no_cache:
        cfg.setdefault("run", {})["cache"] = False
    if args.shard and args.num_shards <= 1:
        print("[medeval] --shard requires --num-shards > 1")
        return 2
    if args.num_shards and args.num_shards > 1:
        if not 0 <= args.shard < args.num_shards:
            print(f"[medeval] --shard must be in [0, {args.num_shards}); got {args.shard}")
            return 2
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


def cmd_fetch(args: argparse.Namespace) -> int:
    from .assets import ensure_extracted
    dest = ensure_extracted(args.url, args.output)
    print(f"[medeval] assets ready at {dest}")
    return 0


def cmd_kg(args: argparse.Namespace) -> int:
    from .kg.tcm_classics import build_classics_kg, export_kg
    kg = build_classics_kg()
    if args.stats:
        import json
        print(json.dumps(kg.stats(), ensure_ascii=False, indent=2))
    fmts = None if args.format == "all" else [args.format]
    paths = export_kg(kg, args.output, formats=fmts)
    print(f"[medeval] wrote KG ({kg.stats()['nodes']} nodes, {kg.stats()['edges']} edges): "
          + ", ".join(str(p) for p in paths))
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


def cmd_preflight(args: argparse.Namespace) -> int:
    from .preflight import preflight, format_reports
    cfg = _load_config(args.config)
    ids = args.datasets.split(",") if args.datasets else None
    reports = preflight(cfg, dataset_ids=ids, limit=args.limit, n_examples=args.examples)
    print(format_reports(reports))
    if args.output:
        import json
        Path(args.output).write_text(
            json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[medeval] wrote preflight report -> {args.output}")
    # exit non-zero if any dataset errored or parsed < 100% (CI-friendly)
    bad = any("error" in r or r.get("answer_parse_rate", 1.0) < 0.999 for r in reports)
    return 1 if (bad and args.strict) else 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Measure judge↔human agreement on the frozen, physician-labeled set."""
    import asyncio
    import json
    from . import calibrate as cal

    if args.rebuild_from:
        n = cal.build_meta_set(args.rebuild_from, args.items, args.gold, n=args.n)
        print(f"[medeval] wrote {n} calibration items -> {args.items} (+ gold {args.gold})")
        return 0

    items = cal.load_calibration_set(args.items, args.gold)
    if not items:
        print(f"[medeval] no calibration items found at {args.items}; "
              "run with --rebuild-from <meta_eval.jsonl URL/path> first")
        return 2

    base_sig = {
        "prompt_style": "healthbench_per_criterion",   # the only style calibrated today
        "prompt_hash": cal.healthbench_prompt_hash(),
        "calibration_set_hash": cal.calibration_set_hash(items),
        "language": "en",
        "task_family": "healthbench_meta",
    }
    if args.labels:
        preds = cal.load_label_preds(args.labels)
        name = args.rater_name
        # A labels file (strong-model / human pass) has no concrete, verifiable
        # judge_model tied to it — leave it unset so this report is NEVER
        # auto-applied to a live run's differently-configured judge (see
        # Runner._row_judge_calibrated). It still renders as evidence that
        # physician-equivalence is ACHIEVABLE; making it binding requires a
        # live --config --judge run against the same concrete model.
        signature = {**base_sig, "judge_model": None, "judge_revision": None,
                    "rater_name": name}
    elif args.config and args.judge:
        if not any(it.completion for it in items):
            print(f"[medeval] live-judge mode needs the prose items file ({args.items}); "
                  "regenerate it with: medeval calibrate --rebuild-from <meta_eval.jsonl URL/path>")
            return 2
        from .providers.base import create_provider
        cfg = _load_config(args.config)
        spec = next((m for m in cfg.get("models", []) if m.get("id") == args.judge), None)
        if spec is None:
            print(f"[medeval] judge {args.judge!r} not found in {args.config}")
            return 2
        judge = create_provider(spec)
        preds = asyncio.run(cal.llm_judge_preds(items, judge))
        name = args.judge
        signature = {**base_sig, "judge_model": getattr(judge, "model", judge.id),
                    "judge_revision": getattr(judge, "revision", None)}
    else:
        print("[medeval] provide --labels <reviewer.jsonl> or (--config <run.yaml> --judge <model_id>)")
        return 2

    report = cal.evaluate(items, preds, name, signature=signature)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = cal.render_report(report, dataset=args.dataset_name)
    (out / "calibration_report.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[medeval] wrote {out/'calibration_report.md'} (+ .json)")
    if args.strict and not report["verdict"]["calibrated"]:
        return 1
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


def cmd_compare(args: argparse.Namespace) -> int:
    from .compare import compare, render_compare
    r = compare(args.results_dir, args.dataset, args.model_a, args.model_b,
               metric=args.metric)
    text = render_compare(r)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"[medeval] wrote {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="medeval", description="MedEval runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run an evaluation from a YAML config")
    p_run.add_argument("config", help="path to the YAML run spec")
    p_run.add_argument("--limit", type=int, default=None,
                       help="cap samples per dataset (overrides config)")
    p_run.add_argument("--models", default=None,
                       help="comma-separated model ids to keep (judges and agent "
                            "support: models always kept); select one HF model per "
                            "run since vLLM holds it in GPU memory")
    p_run.add_argument("--output", default=None, help="override run.output_dir")
    p_run.add_argument("--no-cache", action="store_true", help="disable generation cache")
    p_run.add_argument("--num-shards", type=int, default=1,
                       help="total shards for distributed runs")
    p_run.add_argument("--shard", type=int, default=0, help="this worker's shard index")
    p_run.set_defaults(func=cmd_run)

    p_pre = sub.add_parser("preflight",
                           help="profile datasets (count · option dist · answer-parse rate · examples) — no model")
    p_pre.add_argument("config", help="path to the YAML run spec")
    p_pre.add_argument("--datasets", default=None,
                       help="comma-separated dataset ids to check (default: all in the config)")
    p_pre.add_argument("--limit", type=int, default=None,
                       help="cap rows per dataset (default: full load, for a true count)")
    p_pre.add_argument("--examples", type=int, default=3, help="sample rows to show per dataset")
    p_pre.add_argument("--output", "--out", dest="output", default=None,
                       help="also write the full report as JSON")
    p_pre.add_argument("--strict", action="store_true",
                       help="exit non-zero if any dataset errors or parses < 100%%")
    p_pre.set_defaults(func=cmd_preflight)

    p_cal = sub.add_parser("calibrate",
                           help="measure judge↔human agreement on open-ended scoring (physician-labeled set)")
    p_cal.add_argument("--items", default="data/calibration/healthbench_meta_items.jsonl",
                       help="blind reviewer file (conversation/completion/rubric, no labels)")
    p_cal.add_argument("--gold", default="data/calibration/healthbench_meta_gold.jsonl",
                       help="held-out physician-label file")
    p_cal.add_argument("--labels", default=None,
                       help="JSONL of {item_id, criteria_met}: a strong-model / human reviewer pass")
    p_cal.add_argument("--rater-name", default="strong-model judge",
                       help="label for the --labels rater in the report")
    p_cal.add_argument("--config", default=None,
                       help="run YAML providing a judge model (live llm_judge rater)")
    p_cal.add_argument("--judge", default=None, help="judge model id in --config to run live")
    p_cal.add_argument("--dataset-name", default="HealthBench meta-eval",
                       help="dataset label shown in the report")
    p_cal.add_argument("--output", "--out", dest="output", default="data/calibration",
                       help="dir for calibration_report.md / .json")
    p_cal.add_argument("--rebuild-from", default=None,
                       help="regenerate the frozen set from a meta-eval JSONL (URL/path), then exit")
    p_cal.add_argument("--n", type=int, default=120, help="rebuild: target sample size (≥100)")
    p_cal.add_argument("--strict", action="store_true",
                       help="exit non-zero if the judge is NOT calibrated")
    p_cal.set_defaults(func=cmd_calibrate)

    p_list = sub.add_parser("list", help="list registered providers / adapters / metrics")
    p_list.set_defaults(func=cmd_list)

    p_exp = sub.add_parser("export", help="export results as an OpenCompass / MedBench submission")
    p_exp.add_argument("results_dir", help="a run's output dir (with detail__*.jsonl)")
    p_exp.add_argument("--format", required=True, choices=["opencompass", "medbench"])
    p_exp.add_argument("--output", "--out", required=True, dest="output")
    p_exp.add_argument("--medbench-test-dir", default=None,
                       help="MedBench data tree; fills answers into original *_test.jsonl")
    p_exp.set_defaults(func=cmd_export)

    p_cmp = sub.add_parser("compare",
                           help="paired statistical comparison of two models on one dataset "
                                "(bootstrap CI + McNemar for binary metrics)")
    p_cmp.add_argument("results_dir", help="a run's output dir (with detail__*.jsonl)")
    p_cmp.add_argument("--dataset", required=True, help="dataset id to compare on")
    p_cmp.add_argument("--model-a", required=True)
    p_cmp.add_argument("--model-b", required=True)
    p_cmp.add_argument("--metric", default=None,
                       help="metric to compare (default: auto-pick the headline metric)")
    p_cmp.add_argument("--output", default=None, help="also write the report to this path")
    p_cmp.set_defaults(func=cmd_compare)

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

    p_fetch = sub.add_parser("fetch", help="download + unzip a dataset asset (e.g. images.zip)")
    p_fetch.add_argument("url", help="archive URL (or local path)")
    p_fetch.add_argument("--output", "--out", dest="output", required=True, help="extract-to dir")
    p_fetch.set_defaults(func=cmd_fetch)

    p_kg = sub.add_parser("kg", help="build + export the TCM classics knowledge graph")
    p_kg.add_argument("--output", "--out", dest="output", default="data/kg")
    p_kg.add_argument("--format", choices=["json", "turtle", "graphml", "all"], default="all")
    p_kg.add_argument("--stats", action="store_true", help="print node/edge counts")
    p_kg.set_defaults(func=cmd_kg)

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
