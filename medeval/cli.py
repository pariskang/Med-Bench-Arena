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
    rows = run_config(cfg)
    print(f"[medeval] {len(rows)} (model × dataset) results")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    print("providers:", ", ".join(available_providers()))
    print("adapters :", ", ".join(available_adapters()))
    print("metrics  :", ", ".join(available_metrics()))
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
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list", help="list registered providers / adapters / metrics")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
