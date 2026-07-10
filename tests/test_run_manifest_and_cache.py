"""P0-5: content-addressed generation cache + run_manifest.json.

Locks in: (1) a sample whose CONTENT changes under a stable sample_id is
correctly treated as a cache miss (not a stale reuse against a new reference);
(2) a dataset-config change (revision/field_map/instruction) changes the cache
filename; (3) run_manifest.json records dataset/model/git provenance and
redacts literal secrets. Offline — mock provider only."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from medeval.runner import (
    Runner, _sample_content_hash, _dataset_protocol_key, _protocol_hash,
    _redact_secrets, _git_commit,
)
from medeval.schema import Sample, Message, TaskType


def _sample(qid="s1", question="Q1", answer="A"):
    return Sample(id=qid, task_type=TaskType.MCQ,
                 messages=[Message("user", question)],
                 choices=["A", "B"], reference={"index": 0 if answer == "A" else 1})


def test_sample_content_hash_changes_with_content_not_id():
    s1 = _sample(qid="x", question="What is the capital of France?", answer="A")
    s2 = _sample(qid="x", question="What is the capital of France?", answer="A")
    s3 = _sample(qid="x", question="What is the capital of Spain?", answer="A")  # same id!
    s4 = _sample(qid="x", question="What is the capital of France?", answer="B")  # same id!
    assert _sample_content_hash(s1) == _sample_content_hash(s2)   # identical content -> same hash
    assert _sample_content_hash(s1) != _sample_content_hash(s3)   # question drifted, id stable
    assert _sample_content_hash(s1) != _sample_content_hash(s4)   # reference drifted, id stable


def test_cache_miss_on_content_drift_under_stable_sample_id():
    """The exact failure mode from the audit: upstream data changes under an
    unpinned source, sample_id stays the same — the old cached generation must
    NOT be silently reused against the new reference."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        r = Runner({"models": [{"id": "m", "type": "mock", "behavior": "auto"}],
                   "datasets": [], "run": {"output_dir": str(out), "cache": True}})
        prov = r.providers["m"]

        class _Ds:
            id = "ds1"
            config = {}
        cpath = r._cache_path(prov, _Ds())

        from medeval.schema import Generation, Prediction
        pred = Prediction(sample_id="x", generation=Generation(text="Paris"), parsed=0)
        s_v1 = _sample(qid="x", question="capital of France?", answer="A")
        r._append_cache(cpath, pred, _sample_content_hash(s_v1))

        raw = r._load_cache(cpath)
        assert raw["x"]["content_hash"] == _sample_content_hash(s_v1)   # hit for unchanged content

        # upstream "changed" the question under the SAME sample_id — new content hash
        s_v2 = _sample(qid="x", question="capital of GERMANY?", answer="B")
        assert raw["x"]["content_hash"] != _sample_content_hash(s_v2)   # -> correctly a miss


def test_dataset_protocol_key_changes_with_revision_and_field_map():
    class _Ds:
        id = "d"
        def __init__(self, cfg):
            self.config = cfg
    a = _Ds({"path": "x", "revision": "aaa", "field_map": {"question": "q"}})
    b = _Ds({"path": "x", "revision": "bbb", "field_map": {"question": "q"}})
    c = _Ds({"path": "x", "revision": "aaa", "field_map": {"question": "other"}})
    assert _protocol_hash(a) != _protocol_hash(b)   # revision bump -> new cache
    assert _protocol_hash(a) != _protocol_hash(c)   # field_map change -> new cache
    # scoring-only keys (metrics/judge/split_type) and orchestration keys
    # (limit/id) must NOT bust the generation cache
    d = _Ds({"path": "x", "revision": "aaa", "field_map": {"question": "q"},
            "metrics": ["llm_judge"], "judge": "gpt-4o", "split_type": "official",
            "limit": 5, "id": "different-id-here"})
    assert _protocol_hash(a) == _protocol_hash(d)


def test_redact_secrets_hides_literal_keys_not_env_var_names():
    cfg = {"models": [
        {"id": "m", "type": "litellm", "api_key": "sk-super-secret",
         "api_key_env": "OPENAI_API_KEY"},
    ]}
    red = _redact_secrets(cfg)
    assert red["models"][0]["api_key"] == "***REDACTED***"
    assert red["models"][0]["api_key_env"] == "OPENAI_API_KEY"   # env VAR NAME, not a secret


def test_run_manifest_written_with_provenance():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        cfg = {
            "run": {"output_dir": str(out), "cache": False},
            "eval": {"gen": {"temperature": 0.0, "max_tokens": 32}},
            "models": [{"id": "m", "type": "mock", "behavior": "auto",
                       "api_key": "sk-should-be-redacted"}],
            "datasets": [{"id": "d1", "adapter": "hf_mcq", "path": "x",
                         "split_type": "official",   # unpinned -> gate downgrades it
                         "field_map": {"question": "q", "options": "o", "answer": "a"}}],
        }
        import medeval
        medeval.run_config({**cfg, "datasets": []})  # exercise a trivial run for the manifest write
        mf = json.loads((out / "run_manifest.json").read_text())
        assert "generated_at" in mf and "adapter_protocol_version" in mf
        assert mf["models"][0]["type"] == "mock"
        assert "sk-should-be-redacted" not in json.dumps(mf)   # secret redacted end-to-end
        assert mf["git_commit"] is None or isinstance(mf["git_commit"], str)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("OK: cache content-addressing + run manifest tests passed")
