"""MediQ interactive AgentAdapter + the auto download/unzip asset helper (offline)."""
from __future__ import annotations

import asyncio
import json
import tempfile
import zipfile
from pathlib import Path

from medeval import Sample, TaskType, create_metric, create_provider
from medeval.datasets.agent_env import MediQEnv, MediQAdapter
from medeval.assets import ensure_extracted, ensure_image_base
from medeval.datasets.local_json import LocalJSONAdapter

SCENARIO = {
    "question": "Which antibiotic blocks cell wall synthesis?",
    "options": {"A": "Gentamicin", "B": "Ciprofloxacin", "C": "Ceftriaxone", "D": "Trimethoprim"},
    "answer_idx": "C",
    "context": ["A 21-year-old male with fever and dysuria."],
    "facts": ["Joint fluid culture shows gram-negative diplococci.",
              "The bacteria does not ferment maltose."],
}


def test_mediq_env_reveal_commit_timeout():
    async def go():
        env = MediQEnv(SCENARIO, max_questions=4)
        await env.reset()
        obs, _, done, _ = await env.step("What does the joint fluid culture show?")
        assert "diplococci" in obs and not done            # most relevant fact revealed
        _, r, done, info = await env.step("ANSWER: C")
        assert done and r == 1.0 and info["success"] and info["questions"] == 1
        # wrong commit
        env2 = MediQEnv(SCENARIO); await env2.reset()
        _, r2, _, _ = await env2.step("ANSWER: A")
        assert r2 == 0.0
        # timeout: keep asking past the budget -> forced answer, flagged
        env3 = MediQEnv(SCENARIO, max_questions=2); await env3.reset()
        await env3.step("tell me more")
        _, r3, done3, info3 = await env3.step("and more please")
        assert done3 and info3.get("timeout") and r3 == 0.0

    asyncio.run(go())


def test_mediq_rollout_with_mock_and_passk_metrics():
    ad = MediQAdapter({"id": "mediq", "adapter": "mediq", "max_questions": 6, "k": 1})
    prov = create_provider({"id": "m", "type": "mock", "behavior": "auto"})
    s = Sample(id="mediq:0", task_type=TaskType.AGENT, messages=[], env_spec=SCENARIO)
    pred = asyncio.run(ad.rollout(s, prov, k=1, gen={}))
    assert pred.rollouts[0]["info"].get("answered")       # mock asks then commits
    sc = asyncio.run(create_metric("pass_k").score(s, pred))
    agg = create_metric("pass_k").aggregate([sc])
    assert "avg_turns" in agg and "timeout_rate" in agg and agg["avg_turns"] > 0


def test_asset_ensure_extracted_and_adapter_autofetch():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        zp = d / "imgs.zip"
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr("imgs/0001.jpg", b"\xff\xd8\xff\xe0fake")
        base = ensure_image_base(str(zp), str(d / "ex"))
        assert base.endswith("/") and (Path(base) / "imgs" / "0001.jpg").exists()
        # idempotent (marker present)
        assert (Path(base) / ".medeval_extracted").exists()
        ensure_extracted(str(zp), str(d / "ex"))           # no error, no-op

        # adapter auto-fetches the zip and resolves the relative image path
        rec = {"q": "classify", "a": "A", "img": "imgs/0001.jpg"}
        fp = d / "vqa.jsonl"
        fp.write_text(json.dumps(rec), encoding="utf-8")
        ad = LocalJSONAdapter({"id": "v", "adapter": "local_json", "task": "open_qa",
                               "path": str(fp), "image_zip": str(zp), "image_base": str(d / "ex2"),
                               "field_map": {"prompt": "q", "reference": "a", "image": "img"}})
        s = ad.load()[0]
        assert s.messages[-1].images == [str(d / "ex2" / "imgs" / "0001.jpg")]


if __name__ == "__main__":
    test_mediq_env_reveal_commit_timeout()
    test_mediq_rollout_with_mock_and_passk_metrics()
    test_asset_ensure_extracted_and_adapter_autofetch()
    print("OK: MediQ + asset helper tests passed")
