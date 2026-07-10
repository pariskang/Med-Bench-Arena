"""Official-tier eligibility gate.

``split_type: official`` is a claim that a run is directly comparable to public
leaderboards. That claim is too easy to make by accident: the historical default
was ``official`` itself, so a config author who simply forgot to set
``split_type`` (or copy-pasted a validation-split config) silently produced a
row that *looked* comparable but wasn't (see e.g. MedMCQA — used the validation
split because the test labels are withheld, but shipped with no ``split_type``
and so defaulted straight to ``official``).

The policy this module enforces:

    A dataset defaults to ``unverified``. It may claim ``official`` in config,
    but that claim is only HONORED for the static-benchmark adapter families
    (hf_mcq / local_json / tcmbench / medbench) when there is mechanical
    evidence the source is pinned to an immutable commit — a ``revision:``
    or a commit-hash embedded in a data URL (``resolve/<sha>/`` /
    ``raw.githubusercontent.com/.../<sha>/``). Absent that evidence the claim
    is downgraded (loudly) to ``unverified`` — a config can never *declare*
    its way into the official tier.

Agent-family adapters (AgentClinic / MedAgentBench / MediQ) have their own,
protocol-specific eligibility logic (support-role completeness, refsol
presence, patient-simulator fidelity) implemented where they compute
``self.split_type`` — this module does not override those, only the static
content adapters where "official" is purely a data-provenance question.
"""
from __future__ import annotations

import re
import warnings
from typing import Any

# resolve/<sha> (HF) or raw.githubusercontent.com/<org>/<repo>/<sha>/ (GitHub) —
# a short hex ref (>=7 chars) is accepted since GitHub allows abbreviated shas.
_PIN_PATTERN = re.compile(
    r"/(?:resolve|blob)/([0-9a-f]{7,40})(?:/|$)"
    r"|raw\.githubusercontent\.com/[^/]+/[^/]+/([0-9a-f]{7,40})/",
    re.IGNORECASE,
)

# adapter families whose "official" claim is purely a data-provenance question
# (as opposed to agentclinic/medagentbench/mediq, which gate official-ness via
# protocol fidelity in their own __init__ and are left alone here).
CONTENT_ADAPTERS = frozenset({"hf_mcq", "local_json", "tcmbench", "medbench"})


def _urls_of(ds: Any) -> list[str]:
    """Every URL/path this adapter might source data from."""
    urls: list[str] = []
    for attr in ("source_url", "data_files"):
        v = getattr(ds, attr, None)
        if isinstance(v, str):
            urls.append(v)
        elif isinstance(v, (list, tuple)):
            urls.extend(str(u) for u in v)
    aj = getattr(ds, "answer_join", None)
    if isinstance(aj, dict) and aj.get("data_files"):
        v = aj["data_files"]
        urls.extend(v if isinstance(v, (list, tuple)) else [v])
    hf = getattr(ds, "hf", None)
    if isinstance(hf, dict) and hf.get("revision"):
        urls.append(f"resolve/{hf['revision']}/")  # synthesize a matchable fragment
    return urls


def has_pin_evidence(ds: Any) -> bool:
    """True iff this adapter's source is locked to an immutable commit."""
    if getattr(ds, "revision", None):
        return True
    return any(_PIN_PATTERN.search(u) for u in _urls_of(ds))


def enforce_official_eligibility(ds: Any) -> None:
    """Downgrade ``ds.split_type`` from ``official`` to ``unverified`` in place
    when the mechanical pin-evidence check fails. Call once per dataset right
    after construction (``Runner.__init__`` does this for every dataset in the
    config) so the check applies uniformly regardless of how ``official`` was
    reached — an explicit YAML ``split_type: official``, or a subclass default.
    """
    if getattr(ds, "split_type", None) != "official":
        return
    if getattr(ds, "adapter_name", None) not in CONTENT_ADAPTERS:
        return
    if has_pin_evidence(ds):
        return
    warnings.warn(
        f"{ds.id}: split_type 'official' requires a pinned revision or a "
        "commit-hash URL (resolve/<sha> / raw.githubusercontent.com/.../<sha>/) "
        "— no pin evidence found, so this run is NOT guaranteed reproducible. "
        "Downgrading to 'unverified'. Pin a revision/sha in the config to "
        "restore 'official' (see DATASETS.md).",
        stacklevel=2,
    )
    ds.split_type = "unverified"
