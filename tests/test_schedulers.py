"""Ray and Slurm scheduler backends.

Ray is exercised for real with a local cluster (skipped if ray isn't installed).
Slurm is exercised by generating the sbatch job-array + dependent merge scripts
(no cluster needed) and checking their contents.
"""
from __future__ import annotations

import json
import tempfile
import textwrap
from pathlib import Path

import medeval


def _write_cfg(d: Path) -> Path:
    cfg = d / "cfg.yaml"
    cfg.write_text(textwrap.dedent(f"""
        run: {{output_dir: {d/'out'}, cache: false}}
        eval: {{gen: {{temperature: 0.0, max_tokens: 64}}}}
        models: [{{id: mock-model, type: mock, behavior: auto}}]
        datasets:
          - {{id: demo_agent, adapter: agent_demo, k: 1, max_turns: 4, metrics: [pass_k]}}
    """), encoding="utf-8")
    return cfg


def test_slurm_generates_array_and_merge_scripts():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cfg = _write_cfg(d)
        info = medeval.submit_slurm(cfg, num_shards=8, gpus_per_task=1, partition="gpu",
                                    max_parallel=4, setup="conda activate medeval",
                                    submit=False)
        array = Path(info["array_script"]).read_text()
        merge = Path(info["merge_script"]).read_text()
        assert info.get("submitted") is False
        assert "#SBATCH --array=0-7%4" in array          # 8 shards, throttled to 4
        assert "#SBATCH --gres=gpu:1" in array and "#SBATCH --partition=gpu" in array
        assert "--shard ${SLURM_ARRAY_TASK_ID} --num-shards 8" in array
        assert "conda activate medeval" in array
        assert "python -m medeval merge" in merge


def test_ray_backend_runs_and_merges():
    try:
        import ray  # noqa: F401
    except Exception:
        print("ray not installed; skipping ray test")
        return
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cfg = _write_cfg(d)
        rows = medeval.run_ray(cfg, num_shards=3, num_gpus=0)
        r = [x for x in rows if x["dataset"] == "demo_agent"][0]
        assert r["n"] == 9 and r["shards"] == 3          # 9 samples across 3 ray tasks
        # merged headline equals a single full run
        full = medeval.run_config({
            "run": {"output_dir": str(d / "full"), "cache": False},
            "eval": {"gen": {"temperature": 0.0, "max_tokens": 64}},
            "models": [{"id": "mock-model", "type": "mock", "behavior": "auto"}],
            "datasets": [{"id": "demo_agent", "adapter": "agent_demo", "k": 1,
                          "max_turns": 4, "metrics": ["pass_k"]}]})
        full_pk = [x for x in full if x["dataset"] == "demo_agent"][0]["metrics"]["pass_k"]["pass^k"]
        assert abs(r["metrics"]["pass_k"]["pass^k"] - full_pk) < 1e-9


if __name__ == "__main__":
    test_slurm_generates_array_and_merge_scripts()
    test_ray_backend_runs_and_merges()
    print("OK: scheduler backend tests passed")
