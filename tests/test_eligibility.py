"""Official-tier eligibility gate: split_type defaults to 'unverified'; a config
can only KEEP 'official' for the static content adapters when there's mechanical
pin evidence (a revision or a commit-hash URL). No test/GPU/network needed."""
from __future__ import annotations

import warnings

from medeval.datasets.hf_mcq import HFMCQAdapter
from medeval.datasets.local_json import LocalJSONAdapter
from medeval.eligibility import has_pin_evidence, enforce_official_eligibility
from medeval.runner import Runner


def _hf(**extra):
    cfg = {"id": "t", "path": "x", "field_map": {"question": "q", "options": "o", "answer": "a"}}
    cfg.update(extra)
    return HFMCQAdapter(cfg)


def test_default_split_type_is_unverified():
    ds = _hf()
    assert ds.split_type == "unverified"


def test_has_pin_evidence_revision():
    assert has_pin_evidence(_hf(revision="abc1234")) is True
    assert has_pin_evidence(_hf()) is False


def test_has_pin_evidence_commit_hash_url():
    pinned = _hf(data_files="https://huggingface.co/datasets/x/y/resolve/"
                            "935fbc09edf1303d89872b21265ff597f426ac0d/z.json")
    unpinned = _hf(data_files="https://huggingface.co/datasets/x/y/resolve/main/z.json")
    assert has_pin_evidence(pinned) is True
    assert has_pin_evidence(unpinned) is False
    gh_pinned = LocalJSONAdapter({
        "id": "t", "source_url": "https://raw.githubusercontent.com/org/repo/"
                                  "6c8ece46097dae736c6805dd3b831e1a38c08971/data.json"})
    gh_unpinned = LocalJSONAdapter({
        "id": "t", "source_url": "https://raw.githubusercontent.com/org/repo/main/data.json"})
    assert has_pin_evidence(gh_pinned) is True
    assert has_pin_evidence(gh_unpinned) is False


def test_enforce_official_eligibility_downgrades_unpinned_claim():
    ds = _hf(split_type="official")
    assert ds.split_type == "official"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        enforce_official_eligibility(ds)
        assert any("official" in str(x.message) for x in w)
    assert ds.split_type == "unverified"


def test_enforce_official_eligibility_keeps_pinned_claim():
    ds = _hf(split_type="official", revision="abc1234")
    enforce_official_eligibility(ds)
    assert ds.split_type == "official"


def test_enforce_official_eligibility_leaves_non_content_adapters_alone():
    # a stub mimicking an agent-family adapter: not in CONTENT_ADAPTERS, so the
    # generic pin gate must not touch it (agent adapters have their own gates).
    class _Stub:
        id = "agentclinic_x"
        adapter_name = "agentclinic"
        split_type = "official"
    s = _Stub()
    enforce_official_eligibility(s)
    assert s.split_type == "official"   # untouched


def test_enforce_official_eligibility_leaves_non_official_alone():
    ds = _hf(split_type="demo")
    enforce_official_eligibility(ds)
    assert ds.split_type == "demo"   # not an "official" claim -> nothing to check


def test_runner_applies_gate_to_every_dataset():
    """Integration: a config that declares split_type: official on an unpinned
    dataset must NOT survive as official once the Runner has been constructed —
    this is the exact MedMCQA-shaped bug the gate exists to catch."""
    cfg = {
        "models": [{"id": "m", "type": "mock", "behavior": "auto"}],
        "datasets": [
            {"id": "unpinned_but_claims_official", "adapter": "hf_mcq",
             "path": "x", "field_map": {"question": "q", "options": "o", "answer": "a"},
             "split_type": "official"},
            {"id": "pinned_official", "adapter": "hf_mcq", "path": "x",
             "revision": "deadbeef", "field_map": {"question": "q", "options": "o", "answer": "a"},
             "split_type": "official"},
        ],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = Runner(cfg)
    by_id = {ds.id: ds for ds in r.datasets}
    assert by_id["unpinned_but_claims_official"].split_type == "unverified"
    assert by_id["pinned_official"].split_type == "official"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("OK: eligibility gate tests passed")
