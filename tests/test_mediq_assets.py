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


def test_mediq_abstain_and_confidence():
    async def go():
        env = MediQEnv(SCENARIO, max_questions=4)
        await env.reset()
        _, r, done, info = await env.step("ABSTAIN — not enough information yet")
        assert done and r == 0.0
        assert info["abstained"] is True and info["success"] is False
        # commit WITH a stated confidence -> parsed and carried in info
        env2 = MediQEnv(SCENARIO); await env2.reset()
        _, r2, done2, info2 = await env2.step("ANSWER: C (confidence: 80%)")
        assert done2 and r2 == 1.0 and abs(info2["confidence"] - 0.8) < 1e-9
        # a mid-conversation question mentioning "abstain" hypothetically must
        # NOT be treated as a commit (only literal ABSTAIN with no other intent)
        env3 = MediQEnv(SCENARIO, max_questions=4); await env3.reset()
        obs3, _, done3, _ = await env3.step("tell me about the joint fluid culture")
        assert not done3
    asyncio.run(go())


def test_mediq_split_type_never_auto_official():
    no_support = MediQAdapter({"id": "m", "adapter": "mediq"})
    assert no_support.split_type == "approximated"
    with_support = MediQAdapter({"id": "m2", "adapter": "mediq", "support": {"patient": "p"}})
    assert with_support.split_type == "reimplementation"
    # explicit override still honored (operator's call, not the code's)
    forced = MediQAdapter({"id": "m3", "adapter": "mediq", "split_type": "official"})
    assert forced.split_type == "official"


def test_passk_abstain_rate_and_confidence_brier_in_aggregate():
    from medeval.metrics.mcq import PassK
    from medeval.schema import Prediction, Generation

    def _pred(sid, rollouts):
        return Prediction(sample_id=sid, generation=Generation(text=""), rollouts=rollouts)

    m = PassK()
    s1 = Sample(id="s1", task_type=TaskType.AGENT, messages=[])
    s2 = Sample(id="s2", task_type=TaskType.AGENT, messages=[])
    # sample 1: one rollout, abstained
    sc1 = asyncio.run(m.score(s1, _pred("s1", [
        {"success": False, "turns": 2, "info": {"abstained": True, "questions": 2}}])))
    # sample 2: one rollout, answered with confidence 0.9 and correct
    sc2 = asyncio.run(m.score(s2, _pred("s2", [
        {"success": True, "turns": 3, "info": {"confidence": 0.9, "questions": 3}}])))
    agg = m.aggregate([sc1, sc2])
    assert agg["abstain_rate"] == 0.5           # 1 of 2 rollouts abstained
    assert agg["avg_questions"] == 2.5          # (2+3)/2
    assert agg["mean_confidence"] == 0.9
    assert abs(agg["confidence_brier_score"] - (0.9 - 1.0) ** 2) < 1e-9
    # AgentClinic/MedAgentBench-style rollouts (no abstain/confidence/questions)
    # must NOT get these keys polluting their leaderboard row
    plain = asyncio.run(m.score(s1, _pred("s1", [{"success": True, "turns": 1, "info": {}}])))
    agg2 = m.aggregate([plain])
    assert "abstain_rate" not in agg2 and "mean_confidence" not in agg2


def test_passk_strict_k_stratification():
    from medeval.metrics.mcq import PassK
    from medeval.schema import Prediction, Generation

    def _pred(sid, rollouts):
        return Prediction(sample_id=sid, generation=Generation(text=""), rollouts=rollouts)

    m = PassK()
    s_full = Sample(id="full", task_type=TaskType.AGENT, messages=[])
    s_partial = Sample(id="partial", task_type=TaskType.AGENT, messages=[])
    # k=3 configured, all 3 gradable and all succeed -> strict-eligible pass^3 = 1.0
    sc_full = asyncio.run(m.score(s_full, _pred("full", [
        {"success": True, "turns": 1, "info": {}} for _ in range(3)])))
    assert sc_full.detail["strict_eligible"] is True and sc_full.value == 1.0
    # k=3 configured, only 1 of 3 gradable (2 ungradable) — NOT strict-eligible,
    # even though the single gradable rollout succeeded
    sc_partial = asyncio.run(m.score(s_partial, _pred("partial", [
        {"success": True, "turns": 1, "info": {}},
        {"success": False, "turns": 1, "info": {"ungradable": True}},
        {"success": False, "turns": 1, "info": {"ungradable": True}}])))
    assert sc_partial.detail["strict_eligible"] is False
    assert sc_partial.detail["k_effective"] == 1 and sc_partial.detail["k_configured"] == 3

    agg = m.aggregate([sc_full, sc_partial])
    # headline pass^k is computed ONLY over the strict-eligible sample
    assert agg["pass^k"] == 1.0 and agg["strict_k_eligible"] == 1 and agg["strict_k_rate"] == 0.5
    # but pass@1 still reflects both samples (only needs the first rollout)
    assert agg["pass@1"] == 1.0
    assert agg["k_effective_distribution"] == {"3": 1, "1": 1}

    # edge case: NO sample is strict-eligible -> pass^k reports 0.0, not a crash
    agg2 = m.aggregate([sc_partial])
    assert agg2["pass^k"] == 0.0 and agg2["strict_k_eligible"] == 0


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
    test_mediq_abstain_and_confidence()
    test_mediq_split_type_never_auto_official()
    test_passk_abstain_rate_and_confidence_brier_in_aggregate()
    test_passk_strict_k_stratification()
    test_mediq_rollout_with_mock_and_passk_metrics()
    test_asset_ensure_extracted_and_adapter_autofetch()
    print("OK: MediQ + asset helper tests passed")
