"""Distributed scheduling: strided sharding + merge (offline, mock backend).

Verifies a sharded run produces shard-scoped files and that merging them gives
exactly the same leaderboard as a single full run.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import medeval
from medeval.distributed import merge_results

ROOT = Path(__file__).resolve().parents[1]


def _cfg(out: Path, num_shards=1, shard=0):
    run = {"output_dir": str(out), "cache": False}
    if num_shards > 1:
        run["num_shards"] = num_shards
        run["shard_index"] = shard
    return {
        "run": run,
        "eval": {"gen": {"temperature": 0.0, "max_tokens": 64}},
        "models": [{"id": "mock-model", "type": "mock", "behavior": "auto"}],
        "datasets": [{"id": "demo_agent", "adapter": "agent_demo", "k": 1, "max_turns": 4,
                      "metrics": ["pass_k"]}],
    }


def _passk(leaderboard_json: Path) -> tuple[float, int]:
    rows = json.loads(leaderboard_json.read_text())
    r = [x for x in rows if x["dataset"] == "demo_agent"][0]
    return r["metrics"]["pass_k"]["pass^k"], r["n"]


def test_shard_then_merge_equals_full_run():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # full (unsharded) reference run
        medeval.run_config(_cfg(d / "full"))
        full_pk, full_n = _passk(d / "full" / "leaderboard.json")
        assert full_n == 9

        # two strided shards into a shared dir
        shard_dir = d / "sharded"
        medeval.run_config(_cfg(shard_dir, num_shards=2, shard=0))
        medeval.run_config(_cfg(shard_dir, num_shards=2, shard=1))
        # each shard wrote its own detail file; no global leaderboard yet
        assert (shard_dir / "detail__mock-model__demo_agent__shard0of2.jsonl").exists()
        assert (shard_dir / "detail__mock-model__demo_agent__shard1of2.jsonl").exists()
        assert not (shard_dir / "leaderboard.md").exists()

        # shards partition the 9 samples (5 + 4), disjointly
        n0 = len((shard_dir / "detail__mock-model__demo_agent__shard0of2.jsonl")
                 .read_text().splitlines())
        n1 = len((shard_dir / "detail__mock-model__demo_agent__shard1of2.jsonl")
                 .read_text().splitlines())
        assert {n0, n1} == {5, 4}

        # merge -> identical headline to the full run
        rows = merge_results(shard_dir)
        merged_pk, merged_n = _passk(shard_dir / "leaderboard.json")
        assert merged_n == 9 and abs(merged_pk - full_pk) < 1e-9
        assert (shard_dir / "leaderboard.md").exists()
        assert [r for r in rows if r["dataset"] == "demo_agent"][0]["shards"] == 2


def test_pool_cli_launches_and_merges():
    """`medeval pool` spawns workers as subprocesses and merges them."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cfg = d / "pool.yaml"
        cfg.write_text(textwrap.dedent(f"""
            run: {{output_dir: {d/'out'}, cache: false}}
            eval: {{gen: {{temperature: 0.0, max_tokens: 64}}}}
            models: [{{id: mock-model, type: mock, behavior: auto}}]
            datasets:
              - {{id: demo_agent, adapter: agent_demo, k: 1, max_turns: 4, metrics: [pass_k]}}
        """), encoding="utf-8")
        env = {"PYTHONPATH": str(ROOT)}
        import os
        env = {**os.environ, **env}
        r = subprocess.run([sys.executable, "-m", "medeval", "pool", str(cfg),
                            "--num-shards", "3"], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        lb = d / "out" / "leaderboard.json"
        assert lb.exists()
        rows = json.loads(lb.read_text())
        assert [x for x in rows if x["dataset"] == "demo_agent"][0]["n"] == 9


def test_split_type_separates_official_from_internal():
    """leaderboard.md keeps `official` runs apart from validation/demo/approximated."""
    from medeval.runner import write_leaderboard
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rows = [
            {"model": "m", "dataset": "medqa", "n": 10, "split_type": "official",
             "metrics": {"mcq_accuracy": {"accuracy": 0.8, "n": 10}}, "model_cost_usd": 0.0},
            {"model": "m", "dataset": "tcmbench_demo", "n": 20, "split_type": "demo",
             "metrics": {"mcq_accuracy": {"accuracy": 0.5, "n": 20}}, "model_cost_usd": 0.0},
            {"model": "m", "dataset": "medagentbench", "n": 5, "split_type": "approximated",
             "metrics": {"pass_k": {"pass^k": 0.4}}, "model_cost_usd": 0.0},
        ]
        write_leaderboard(rows, d)
        md = (d / "leaderboard.md").read_text()
        assert "## ✅ Official (comparable)" in md
        assert "## ⚠️ Internal / non-comparable" in md
        official = md.split("Internal")[0]
        assert "medqa" in official and "tcmbench_demo" not in official  # demo not in official
        assert "`split_type: demo`" in md and "`split_type: approximated`" in md
        by = {r["dataset"]: r["split_type"] for r in json.loads((d / "leaderboard.json").read_text())}
        assert by["medagentbench"] == "approximated"


def test_merge_recovers_split_type():
    """A sharded run records split_type in detail; merge surfaces it on the row."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for shard in (0, 1):
            cfg = _cfg(d / "out", num_shards=2, shard=shard)
            cfg["datasets"][0]["split_type"] = "demo"
            medeval.run_config(cfg)
        rows = merge_results(d / "out")
        row = [r for r in rows if r["dataset"] == "demo_agent"][0]
        assert row["split_type"] == "demo"


if __name__ == "__main__":
    test_shard_then_merge_equals_full_run()
    test_pool_cli_launches_and_merges()
    test_split_type_separates_official_from_internal()
    test_merge_recovers_split_type()
    print("OK: distributed scheduling tests passed")
