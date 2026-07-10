"""Interactive agent benchmarks (``adapter: agent_demo | agentclinic | medagentbench``).

Static QA is one-shot; agent benchmarks are an *environment loop*: the model acts
as a doctor agent that repeatedly observes -> acts (ask / order test / diagnose)
until ``done``. Every environment implements the same tiny interface, and the same
``ModelProvider`` drives the doctor policy — so HF / Poe / LiteLLM all work
unchanged. Reliability is scored with **pass^k** (k independent rollouts).

Environments
------------
* ``DemoDiagnosisEnv``  — tiny, fully offline; used for smoke tests.
* ``AgentClinicEnv``    — wraps AgentClinic (Schmidgall et al., MIT). Scenario
  JSONL ships in their repo; patient/measurement/moderator can run *scripted*
  (rule-based, no extra LLM calls) so the whole thing runs offline, or as LLMs.
* ``MedAgentBenchEnv``  — scaffold for the FHIR virtual-EHR benchmark. Requires a
  running HAPI-FHIR server + the gated ``refsol.py`` grader (see notes); not part
  of the offline smoke test.
"""
from __future__ import annotations

import abc
import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from ..schema import Generation, Message, Prediction, Sample, TaskType
from .base import DatasetAdapter, register_dataset
from .local_json import CACHE_DIR

DOCTOR_SYSTEM = (
    "You are an expert physician working through a clinical case step by step. "
    "Use the available actions to gather information, then commit to a final answer. "
    "Be concise."
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _strip_think(s: str) -> str:
    """Remove <think>…</think> reasoning traces (and a dangling unterminated
    <think> tail) so agent action parsing sees the model's actual command, not an
    intermediate thought. Mirrors the MCQ path (hf_mcq.py) for reasoning models
    (DeepSeek-R1 / HuatuoGPT-o1 / ClinicalGPT-R1 / Baichuan-M2)."""
    s = re.sub(r"<think>.*?</think>", "", s or "", flags=re.DOTALL)
    s = re.sub(r"<think>.*$", "", s, flags=re.DOTALL)
    return s.strip()


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "medeval/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            f.write(r.read())
    return dest


# ===========================================================================
# Environment interface
# ===========================================================================
class AgentEnvironment(abc.ABC):
    """Minimal Gym-like environment for a doctor agent."""

    # Optional image URLs shown to the doctor with the FIRST observation (e.g. the
    # NEJM case image). The rollout loop attaches them to the initial user turn —
    # without this a vision benchmark would silently run blind on text alone.
    initial_images: Optional[list[str]] = None

    @abc.abstractmethod
    async def reset(self) -> str:
        """Return the initial observation."""

    @abc.abstractmethod
    async def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        """Consume the doctor's action; return (observation, reward, done, info)."""

    def doctor_instructions(self) -> str:
        """Action-space help appended to the doctor system prompt."""
        return ""


# ===========================================================================
# Base adapter: rollout loop + pass^k plumbing
# ===========================================================================
class AgentAdapter(DatasetAdapter):
    """Base class for agent benchmarks. Subclasses implement ``load`` + ``make_env``."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.max_turns = int(config.get("max_turns", 20))
        self.k = int(config.get("k", config.get("pass_k", 1)))
        # role -> model-id map (patient/measurement/moderator), resolved by the
        # runner; also part of the generation-cache key (protocol-relevant).
        self.support_spec: dict[str, str] = config.get("support", {}) or {}
        if not self.metric_specs:
            self.metric_specs = [("pass_k", {})]
            self.metrics = ["pass_k"]

    @abc.abstractmethod
    def make_env(self, sample: Sample, support: Optional[dict] = None) -> AgentEnvironment:
        ...

    def doctor_system_prompt(self, env: AgentEnvironment) -> str:
        instr = env.doctor_instructions()
        return DOCTOR_SYSTEM + ("\n\n" + instr if instr else "")

    async def rollout(self, sample: Sample, provider, k: int | None = None,
                      gen: dict | None = None, support: dict | None = None) -> Prediction:
        gen = dict(gen or {})
        # apply the model's own gen: overrides (the runner's agenerate_many path
        # merges them in the provider; the agent loop calls agenerate directly,
        # so merge here or per-model sampling would be silently ignored)
        if hasattr(provider, "_merge_gen"):
            gen = provider._merge_gen(gen)
        k = k or self.k
        if k > 1 and not float(gen.get("temperature") or 0.0) > 0.0 \
                and not getattr(self, "_warned_deterministic", False):
            self._warned_deterministic = True
            print(f"[medeval] WARNING {self.id}: k={k} rollouts at temperature 0 — "
                  "all rollouts are identical, so pass^k degenerates to pass@1 at "
                  f"{k}x the cost. Set a non-zero temperature (e.g. eval.gen "
                  "temperature: 0.7) for a meaningful reliability estimate.")
        rollouts: list[dict[str, Any]] = []
        first_gen: Generation | None = None
        first_traj: list[dict[str, Any]] | None = None
        # accumulate the doctor model's cost/tokens across EVERY turn and rollout,
        # not just the first rollout's last turn — else the leaderboard Cost column
        # for agent datasets reflects one generation per episode (a large undercount).
        tot_cost = tot_pt = tot_ct = 0.0

        for ki in range(k):
            env = self.make_env(sample, support)
            obs = await env.reset()
            imgs = getattr(env, "initial_images", None) or None
            messages = [Message("system", self.doctor_system_prompt(env)),
                        Message("user", obs, images=imgs)]
            traj: list[dict[str, Any]] = [{"role": "env", "content": obs,
                                           **({"images": imgs} if imgs else {})}]
            reward, done, info = 0.0, False, {}
            last_gen: Generation | None = None
            turn = 0
            for turn in range(self.max_turns):
                g = await provider.agenerate(messages, **gen)
                last_gen = g
                tot_cost += g.cost_usd or 0.0
                tot_pt += g.prompt_tokens or 0
                tot_ct += g.completion_tokens or 0
                action = _strip_think(g.text)   # reasoning models: parse the command, not the <think> trace
                messages.append(Message("assistant", action))
                traj.append({"role": "doctor", "content": action})
                obs, reward, done, info = await env.step(action)
                traj.append({"role": "env", "content": obs, "reward": reward})
                if done:
                    break
                messages.append(Message("user", obs))
            success = bool(info.get("success", reward >= 1.0))
            rollouts.append({"success": success, "reward": reward,
                             "turns": turn + 1, "info": info})
            if ki == 0:
                first_gen = last_gen or Generation.empty(getattr(provider, "id", ""))
                first_traj = traj

        gen_out = first_gen or Generation.empty()
        # report the episode-total accounting (all turns × all rollouts) so the
        # runner's per-sample cost sum is faithful; text stays the first rollout's.
        gen_out.cost_usd = round(tot_cost, 8)
        gen_out.prompt_tokens = int(tot_pt)
        gen_out.completion_tokens = int(tot_ct)
        gen_out.total_tokens = int(tot_pt + tot_ct)
        return Prediction(
            sample_id=sample.id,
            generation=gen_out,
            trajectory=first_traj, rollouts=rollouts,
        )

    def parse(self, sample: Sample, text: str) -> Prediction:  # not used for agents
        return Prediction(sample_id=sample.id, generation=Generation(text=text))


# ===========================================================================
# 1) Offline demo environment (smoke tests)
# ===========================================================================
# Correct answers are spread across positions (not always first) so a random
# baseline naturally scores ~1/3 rather than 0 or 1.
_DEMO_SCENARIOS = [
    {"id": "d1", "vignette": "A 35-year-old woman has 1 month of double vision and fatigable weakness worse in the evening.",
     "candidates": ["Multiple sclerosis", "Stroke", "Myasthenia gravis"], "correct": "Myasthenia gravis"},
    {"id": "d2", "vignette": "A 60-year-old man with crushing substernal chest pain radiating to the left arm and ST elevation.",
     "candidates": ["Pericarditis", "Costochondritis", "Myocardial infarction"], "correct": "Myocardial infarction"},
    {"id": "d3", "vignette": "A child with fever, sore throat, and gray pseudomembrane on the tonsils.",
     "candidates": ["Strep throat", "Diphtheria", "Mononucleosis"], "correct": "Diphtheria"},
    {"id": "d4", "vignette": "A young adult with polyuria, polydipsia, weight loss, and ketones in urine.",
     "candidates": ["Diabetes insipidus", "Type 1 diabetes", "Hyperthyroidism"], "correct": "Type 1 diabetes"},
    {"id": "d5", "vignette": "A 24-year-old returns from the tropics with cyclical high fevers, chills, and sweats.",
     "candidates": ["Malaria", "Influenza", "Typhoid fever"], "correct": "Malaria"},
    {"id": "d6", "vignette": "A 50-year-old smoker has a chronic cough, weight loss, and hemoptysis with a lung mass.",
     "candidates": ["Pneumonia", "Tuberculosis", "Lung cancer"], "correct": "Lung cancer"},
    {"id": "d7", "vignette": "A 30-year-old has a butterfly facial rash, joint pain, and positive ANA.",
     "candidates": ["Systemic lupus erythematosus", "Rosacea", "Rheumatoid arthritis"], "correct": "Systemic lupus erythematosus"},
    {"id": "d8", "vignette": "A 5-year-old has barking cough, inspiratory stridor, and a steeple sign on neck X-ray.",
     "candidates": ["Epiglottitis", "Croup", "Asthma"], "correct": "Croup"},
    {"id": "d9", "vignette": "An older adult has resting tremor, bradykinesia, and cogwheel rigidity.",
     "candidates": ["Essential tremor", "Parkinson disease", "Huntington disease"], "correct": "Parkinson disease"},
]


class DemoDiagnosisEnv(AgentEnvironment):
    def __init__(self, scenario: dict[str, Any]):
        self.s = scenario
        self.turns = 0

    async def reset(self) -> str:
        cands = ", ".join(self.s["candidates"])
        return (f"{self.s['vignette']}\nCandidate diagnoses: {cands}.\n"
                "Ask one question, or commit with 'DIAGNOSIS: <one diagnosis>'.")

    async def step(self, action: str):
        self.turns += 1
        # Commit only on an explicit "DIAGNOSIS: <x>" marker (last occurrence), not
        # on any mention of the word "diagnosis" — a model reasoning about *whether*
        # to diagnose would otherwise trigger a premature, mis-parsed commit.
        ms = list(re.finditer(r"(?i)DIAGNOSIS\s*[:：]\s*([^\n\r]+)", action))
        if ms:
            said = ms[-1].group(1).strip()
            success = _norm(self.s["correct"]) in _norm(said) or _norm(said) in _norm(self.s["correct"])
            return ("", 1.0 if success else 0.0, True,
                    {"success": success, "said": said, "correct": self.s["correct"]})
        return ("The patient confirms the described history and symptoms.", 0.0, False, {})

    def doctor_instructions(self) -> str:
        return ("You are a doctor. Gather info if needed, then output your final "
                "answer as 'DIAGNOSIS: <one of the candidate diagnoses>'.")


@register_dataset("agent_demo")
class DemoAgentAdapter(AgentAdapter):
    """Tiny offline diagnosis environment for pass^k smoke testing."""

    def load(self) -> list[Sample]:
        out = []
        for sc in self._truncate(_DEMO_SCENARIOS):
            out.append(Sample(id=f"{self.id}:{sc['id']}", task_type=TaskType.AGENT,
                              messages=[], env_spec=sc))
        return out

    def make_env(self, sample: Sample, support=None) -> AgentEnvironment:
        return DemoDiagnosisEnv(sample.env_spec)


# ===========================================================================
# 2) AgentClinic wrapper (offline-capable)
# ===========================================================================
class AgentClinicEnv(AgentEnvironment):
    """One AgentClinic episode — MedQA (OSCE) **or** NEJM (image-case) schema.
    Patient/measurement/moderator run rule-based by default (offline); pass LLM
    providers in ``support`` to use the real LLM-backed agents (the original setup)."""

    def __init__(self, scenario: dict[str, Any], max_turns: int = 20,
                 support: Optional[dict] = None, variant: str = "medqa"):
        self.max_turns = max_turns
        self.support = support or {}
        self.turns_left = max_turns
        self.nejm = variant == "nejm" or ("answers" in scenario and "OSCE_Examination" not in scenario)
        if self.nejm:
            self.osce = {}
            self.nejm_rec = scenario
            self.objective = scenario.get("question", "Determine the most likely diagnosis.")
            self.correct = next((a.get("text", "") for a in scenario.get("answers", [])
                                 if a.get("correct")), "")
            # NEJM cases are image cases: surface the case image to the doctor
            # (attached to the first observation by the rollout loop).
            img = scenario.get("image_url") or scenario.get("image")
            self.initial_images = [str(img)] if img else None
        else:
            self.osce = scenario.get("OSCE_Examination", scenario)
            self.objective = self.osce.get("Objective_for_Doctor", "Assess and diagnose the patient.")
            self.correct = self.osce.get("Correct_Diagnosis", "")

    def doctor_instructions(self) -> str:
        return (
            "You are a doctor evaluating a patient. Each turn you may:\n"
            "  - ask the patient a question (free text), or\n"
            "  - request a test with 'REQUEST TEST: <name>', or\n"
            "  - give your final answer with 'DIAGNOSIS READY: <diagnosis>'.\n"
            f"You have at most {self.max_turns} interactions. Reason briefly, then act."
        )

    async def reset(self) -> str:
        return f"Objective: {self.objective}\nThe patient is in front of you. Begin."

    async def step(self, action: str):
        self.turns_left -= 1
        up = action.upper()
        # Commit ONLY on a real "DIAGNOSIS READY: <dx>" marker (last occurrence),
        # not on any mention of the phrase — otherwise "I'm not DIAGNOSIS READY
        # yet" ends the episode and hands garbage to the moderator (mirrors the
        # demo env's explicit-marker guard).
        commits = list(re.finditer(r"(?i)DIAGNOSIS\s+READY\s*[:：]\s*([^\n\r]+)", action))
        if commits:
            dx = commits[-1].group(1).strip()
            success = await self._moderate(dx, self.correct)
            return ("", 1.0 if success else 0.0, True,
                    {"success": success, "diagnosis": dx, "correct": self.correct})
        done = self.turns_left <= 0
        if "REQUEST TEST" in up or "REQUEST IMAGES" in up:
            obs = await self._measurement(action)
        else:
            obs = await self._patient(action)
        # tell the doctor the budget so it can commit before running out of turns
        if not done:
            obs += (f"\n[Turns remaining: {self.turns_left}. On your last turn give "
                    "'DIAGNOSIS READY: <diagnosis>'.]")
        return (obs, 0.0, done, {"success": False} if done else {})

    # --- patient / measurement / moderator -------------------------------
    async def _patient(self, action: str) -> str:
        prov = self.support.get("patient")
        if self.nejm:  # flat NEJM record: patient_info string
            info = str(self.nejm_rec.get("patient_info", ""))
            if prov is not None:
                g = await prov.agenerate(
                    [Message("system", "You are the patient. Answer in 1-3 sentences without "
                             "naming your diagnosis. Your background: " + info),
                     Message("user", action)], temperature=0.7, max_tokens=128)
                return g.text
            # scripted fallback: redact the gold diagnosis so the raw record dump
            # can't leak the answer (which would inflate the offline pass^k).
            return self._redact(info)[:400] or "I'm not feeling well, doctor."
        actor = self.osce.get("Patient_Actor", {})
        if prov is not None:
            sys = ("You are a patient. Reveal only symptoms you have in 1-3 sentences; "
                   "never state your diagnosis. Your profile: " + json.dumps(actor, ensure_ascii=False))
            g = await prov.agenerate([Message("system", sys), Message("user", action)],
                                     temperature=0.7, max_tokens=128)
            return g.text
        # scripted: surface symptoms / history deterministically
        sym = actor.get("Symptoms", {})
        bits = [actor.get("History", "")]
        if isinstance(sym, dict):
            bits.append(str(sym.get("Primary_Symptom", "")))
            sec = sym.get("Secondary_Symptoms", [])
            bits.append(", ".join(sec) if isinstance(sec, list) else str(sec))
        text = " ".join(b for b in bits if b)
        return text[:400] or "I just don't feel well, doctor."

    async def _measurement(self, action: str) -> str:
        """The measurement/imaging agent. AgentClinic's original design gives this
        its own LLM (distinct from the patient) that reveals test/imaging results
        on request — it was previously wired to read ``support['measurement']``
        in the config but the method was never async and never actually consulted
        it, so a configured measurement model silently went unused and every run
        fell through to the scripted branch. Fixed: call it when provided."""
        prov = self.support.get("measurement")
        if self.nejm:  # flat NEJM record: physical_exams string
            pe = str(self.nejm_rec.get("physical_exams", ""))
            if prov is not None:
                g = await prov.agenerate(
                    [Message("system", "You are the clinical measurement/imaging system. Given "
                             "the patient's actual exam/imaging findings below, answer the "
                             "doctor's specific request with ONLY the relevant result(s); if the "
                             "requested test was not performed, say so explicitly — never invent "
                             "a normal result. Findings: " + pe),
                     Message("user", action)], temperature=0.0, max_tokens=128)
                return f"RESULTS: {g.text}"
            return f"RESULTS: {pe}" if pe else "RESULTS: NORMAL READINGS"
        if prov is not None:
            sys = ("You are the clinical measurement/imaging system. Given the patient's actual "
                   "test results and physical-exam findings below, answer the doctor's specific "
                   "request with ONLY the relevant result(s); if the requested test/exam was not "
                   "performed, say so explicitly — never invent a normal result. Test results: " +
                   json.dumps(self.osce.get("Test_Results", {}), ensure_ascii=False) +
                   " Physical exam: " +
                   json.dumps(self.osce.get("Physical_Examination_Findings", {}), ensure_ascii=False))
            g = await prov.agenerate([Message("system", sys), Message("user", action)],
                                     temperature=0.0, max_tokens=128)
            return f"RESULTS: {g.text}"
        results = self.osce.get("Test_Results", {})
        req = action.split(":", 1)[-1].strip().lower() if ":" in action else ""
        flat = self._flatten(results)
        for key, val in flat.items():
            if req and (req in key.lower() or key.lower() in req):
                return f"RESULTS: {key}: {val}"
        # also try physical exam
        flatpe = self._flatten(self.osce.get("Physical_Examination_Findings", {}))
        for key, val in flatpe.items():
            if req and (req in key.lower() or key.lower() in req):
                return f"RESULTS: {key}: {val}"
        # No matching test on record. Report it as *unavailable*, NOT "normal": a
        # false-normal for an actually-abnormal test can steer the doctor to the
        # wrong diagnosis and cause a false failure.
        return ("RESULTS: this test was not ordered for / is not on record for this "
                "patient (no result available).")

    def _redact(self, text: str) -> str:
        c = (self.correct or "").strip()
        if c and len(c) >= 3:
            text = re.sub(re.escape(c), "[redacted]", text, flags=re.IGNORECASE)
        return text

    async def _moderate(self, dx: str, correct: str) -> bool:
        prov = self.support.get("moderator")
        if prov is not None:
            q = (f"Correct diagnosis: {correct}\nDoctor's diagnosis: {dx}\n"
                 "Do they refer to the same condition? Answer yes or no.")
            g = await prov.agenerate([Message("user", q)], temperature=0.0, max_tokens=8)
            tl = g.text.lower()
            # word-boundary match, not `"yes" in tl` (which fires on "eyes"); and
            # veto on an explicit negative so "No, different condition" scores False.
            yes = bool(re.search(r"\b(yes|same|identical|correct|equivalent)\b", tl))
            no = bool(re.search(r"\b(no|not|different|distinct)\b", tl))
            return yes and not no
        return _norm(correct) in _norm(dx) or _norm(dx) in _norm(correct)

    @staticmethod
    def _flatten(d: Any, prefix: str = "") -> dict[str, str]:
        out: dict[str, str] = {}
        if isinstance(d, dict):
            for k, v in d.items():
                out.update(AgentClinicEnv._flatten(v, f"{prefix}{k} " if prefix else f"{k} "))
        else:
            out[prefix.strip()] = str(d)
        return out


@register_dataset("agentclinic")
class AgentClinicAdapter(AgentAdapter):
    """config:
      source_url: AgentClinic scenario JSONL
        (e.g. https://raw.githubusercontent.com/SamuelSchmidgall/AgentClinic/main/agentclinic_medqa.jsonl)
      variant:    medqa (default) | nejm
      max_turns:  default 20
      support:    {patient: <model_id>, measurement: <model_id>, moderator: <model_id>}
                  optional; resolved by the runner. Omit for fully-scripted (offline).
    """

    DEFAULT_URL = ("https://raw.githubusercontent.com/SamuelSchmidgall/"
                   "AgentClinic/main/agentclinic_medqa.jsonl")

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.source_url = config.get("source_url", self.DEFAULT_URL)
        self.variant = config.get("variant", "medqa")
        # Faithful AgentClinic runs ALL THREE roles (patient / measurement /
        # moderator) as their own LLM — that is the original multi-agent setup
        # the paper evaluates. A config with NO support: is fully scripted
        # (approximated). A config with SOME but not all three roles is a
        # partial reimplementation: better than scripted, but not proven to
        # match the paper's exact multi-agent behavior, so it must not silently
        # claim "official" either — that was the bug (`support_spec` truthy at
        # all, e.g. only `patient` set, used to be enough to claim official).
        if "split_type" not in config:
            required_roles = {"patient", "measurement", "moderator"}
            if required_roles.issubset(self.support_spec):
                self.split_type = "official"
            elif self.support_spec:
                self.split_type = "reimplementation"
            else:
                self.split_type = "approximated"

    def load(self) -> list[Sample]:
        h = hashlib.sha256(self.source_url.encode()).hexdigest()[:16]
        fp = _download(self.source_url, CACHE_DIR / f"{self.id}_{h}.jsonl")
        lines = [json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]
        out = []
        for i, rec in enumerate(self._truncate(lines)):
            out.append(Sample(id=f"{self.id}:{i}", task_type=TaskType.AGENT,
                              messages=[], env_spec=rec,
                              meta={"variant": self.variant}))
        return out

    def make_env(self, sample: Sample, support=None) -> AgentEnvironment:
        return AgentClinicEnv(sample.env_spec, max_turns=self.max_turns,
                              support=support, variant=self.variant)


# ===========================================================================
# 3) MedAgentBench scaffold (requires FHIR server + gated grader)
# ===========================================================================
def fhir_request(url: str, method: str = "GET", body: dict | None = None,
                 timeout: int = 30) -> tuple[int, Any]:
    """Real HTTP call to a FHIR server (stdlib only). Returns (status, data)."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"User-Agent": "medeval/1.0", "Accept": "application/fhir+json, application/json"}
    if data is not None:
        headers["Content-Type"] = "application/fhir+json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "ignore")
            status = getattr(r, "status", r.getcode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "ignore")
        status = e.code
    except Exception as e:
        return 0, f"(FHIR server unreachable: {e})"
    try:
        return status, json.loads(raw)
    except Exception:
        return status, raw


class MedAgentBenchEnv(AgentEnvironment):
    """One FHIR virtual-EHR episode. Faithful to the official harness: the doctor
    emits exactly one of ``GET <url>`` / ``POST <url>\\n<json>`` / ``FINISH([...])``;
    GET appends ``&_format=json``; POST writes to the live FHIR server. Scoring is
    done by the official ``refsol`` (if provided) or the built-in grader."""

    def __init__(self, task: dict[str, Any], fhir_base: str, max_rounds: int = 8,
                 funcs: list | None = None, grader=None, refsol=None):
        self.task = task
        self.fhir_base = fhir_base.rstrip("/")
        self.rounds_left = max_rounds
        self.funcs = funcs or []
        self.grader = grader            # optional callable(task, answer, base, posts)
        self.refsol = refsol            # optional official refsol module
        self.posts: list[dict] = []
        self.history: list[dict] = []

    def doctor_instructions(self) -> str:
        if self.funcs:
            cat = "\n".join(f"  - {f.get('name')}" for f in self.funcs)
        else:
            cat = ("  - GET/POST FHIR resources "
                   "(Patient, Observation, Condition, MedicationRequest, ServiceRequest, Procedure)")
        return (
            "You are an expert using FHIR REST APIs to assist a clinician. "
            f"FHIR base URL: {self.fhir_base}\n"
            "Available endpoints:\n" + cat + "\n\n"
            "Each turn output EXACTLY ONE of these, with no other text:\n"
            "  GET <url>?param=value\n"
            "  POST <url>\n<JSON body>\n"
            "  FINISH([<answer1>, <answer2>, ...])"
        )

    async def reset(self) -> str:
        q = f"{self.task.get('instruction', '')}\n{self.task.get('context', '')}".strip()
        self.history.append({"role": "user", "content": q})
        return q

    @staticmethod
    def _extract_action(a: str) -> str:
        """Reduce the model output to its command, tolerating a reasoning preamble.

        The official harness expects a bare ``GET/POST/FINISH``, but reasoning
        models routinely narrate before acting (and not always inside <think>).
        Rather than reject those as "Invalid action" (an unfair 0), take the LAST
        line/segment that begins with a command verb. A fenced ```code``` block or
        a "GET ..." mid-paragraph is recovered too. Returns "" if none is found."""
        a = _strip_think(a).strip()
        a = re.sub(r"```[a-zA-Z]*\n?|```", "", a)   # drop markdown code fences
        # FINISH may span lines? no — it's single-line; GET single-line; POST is
        # "POST <path>\n<json...>". Find the last command anchor and keep the tail.
        anchors = list(re.finditer(r"(?im)^\s*(GET|POST|FINISH)\b", a))
        if not anchors:
            anchors = list(re.finditer(r"(?i)\b(GET|POST|FINISH)\s*[\(/A-Za-z]", a))
        if not anchors:
            return ""
        return a[anchors[-1].start():].strip()

    async def step(self, action: str):
        self.rounds_left -= 1
        a = self._extract_action(action or "")
        self.history.append({"role": "assistant", "content": a})
        done = self.rounds_left <= 0
        if a.startswith("FINISH("):
            inner = a[len("FINISH("):]
            inner = inner[:-1] if inner.endswith(")") else inner
            return self._finish(inner)
        if a.startswith("GET"):
            obs = self._do_get(a[3:].strip())
        elif a.startswith("POST"):
            obs = self._do_post(a[4:].strip())
        else:
            obs = "Invalid action. Use 'GET <url>', 'POST <url>\\n<json>', or 'FINISH([...])'."
        self.history.append({"role": "user", "content": obs})
        return (obs, 0.0, done, {"success": False} if done else {})

    def _do_get(self, q: str) -> str:
        q = q.replace(" ", "%20")  # tolerate raw spaces in model-emitted URLs
        url = f"{self.fhir_base}/{q}" + ("&" if "?" in q else "?") + "_format=json"
        _, data = fhir_request(url, "GET")
        body = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
        if len(body) > 6000:
            # flag the cut so the model can page (add &_count=/&_offset=) instead of
            # silently reasoning over a Bundle whose tail (the value it needs) is gone.
            body = (body[:6000] + f"\n…[TRUNCATED: {len(body)} chars total; refine the "
                    "query or page with &_count=/&_offset= to see the rest]")
        return f"Here is the response from the GET request:\n{body}"

    def _do_post(self, rest: str) -> str:
        head, _, body_text = rest.partition("\n")
        try:
            body = json.loads(body_text)
        except Exception:
            self.posts.append({"path": head.strip(), "status": 0, "body": body_text, "ok": False})
            return "Invalid POST request"
        status, resp = fhir_request(f"{self.fhir_base}/{head.strip().lstrip('/')}", "POST", body)
        self.posts.append({"path": head.strip(), "status": status, "body": body, "response": resp})
        if 200 <= status < 300:
            return "POST request accepted and executed successfully. The resource has been created."
        return f"POST request failed (status {status})."

    def _finish(self, inner: str):
        from .medagentbench_grader import parse_finish, builtin_grade, grade_with_refsol
        answer = parse_finish(inner)
        if self.refsol is not None:
            success, detail = grade_with_refsol(self.refsol, self.task, answer, self.fhir_base)
        elif self.grader is not None:
            success, detail = self.grader(self.task, answer, self.fhir_base, self.posts)
        else:
            success, detail = builtin_grade(self.task, answer, self.fhir_base, self.posts)
        detail.update({"success": success, "answer": answer, "n_posts": len(self.posts)})
        return ("", 1.0 if success else 0.0, True, detail)


@register_dataset("medagentbench")
class MedAgentBenchAdapter(AgentAdapter):
    """config:
      source_url:  test_data_v2.json (Stanford MedAgentBench); 300 tasks
      funcs_url:   funcs_v1.json (FHIR tool catalog) injected into the prompt
      fhir_base:   running HAPI-FHIR base URL (default http://localhost:8080/fhir)
      max_turns:   rounds (paper: 8; repo config: 5)
      refsol_path: path to the official (gated) refsol.py grader; if omitted the
                   built-in grader is used (query tasks exact; action tasks approx)
      episode_reset: {url: <reset-endpoint>, method: POST|GET, body: {...}} — called
                   before EVERY episode (each sample, each rollout) if your FHIR
                   server exposes a snapshot/rollback or reseed endpoint.
      allow_concurrent_episodes: bool — opt in to running episodes concurrently
                   against the shared server without a reset hook, at your own risk
                   (default false: concurrency is forced to 1 for this dataset).

    Needs a running FHIR server (Docker image jyxsu6/medagentbench on :8080) to
    actually exercise GET/POST. See the README for the docker command.

    ISOLATION RISK: every episode (every sample, and every k>1 rollout of the
    same sample) shares ONE live FHIR server with no snapshot/rollback between
    them by default — a POST from one episode is visible to the next episode's
    GETs. The official Stanford harness runs per-task workers against isolated
    state, not a bare shared endpoint reused across the whole run; this adapter
    cannot reproduce that isolation on its own. Without `episode_reset`,
    concurrency is forced to 1 (reduces, does not eliminate, cross-episode
    interference) and split_type can reach at most `reimplementation` — never
    `official`, even with the gated `refsol.py` grader, since a correct grader
    scoring state corrupted by a prior episode is still an unreliable number.
    """

    DEFAULT_URL = ("https://raw.githubusercontent.com/stanfordmlgroup/"
                   "MedAgentBench/main/data/medagentbench/test_data_v2.json")
    FUNCS_URL = ("https://raw.githubusercontent.com/stanfordmlgroup/"
                 "MedAgentBench/main/data/medagentbench/funcs_v1.json")

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.source_url = config.get("source_url", self.DEFAULT_URL)
        self.funcs_url = config.get("funcs_url", self.FUNCS_URL)
        self.fhir_base = config.get("fhir_base", "http://localhost:8080/fhir")
        self.max_turns = int(config.get("max_turns", 8))
        self.refsol_path = config.get("refsol_path")
        self.episode_reset = config.get("episode_reset")
        self.allow_concurrent_episodes = bool(config.get("allow_concurrent_episodes", False))
        self._funcs: list | None = None
        self._refsol = None
        if self.refsol_path:
            from .medagentbench_grader import load_refsol
            self._refsol = load_refsol(self.refsol_path)
        if not self.episode_reset:
            import warnings
            warnings.warn(
                f"{self.id}: no episode_reset configured — every sample (and every "
                "k>1 rollout of the same sample) shares ONE live FHIR server with no "
                "snapshot/rollback between episodes. A write from one episode can leak "
                "into a later episode's reads, corrupting results — the official "
                "Stanford harness isolates state per task worker; this shared-endpoint "
                "setup cannot. Provide `episode_reset: {url: <reset-endpoint>}` if your "
                "server exposes one. Concurrency is forced to 1 for this dataset and "
                "split_type is capped below 'official' until you do.", stacklevel=2)
        # the built-in grader is an approximation; the gated refsol is leaderboard-
        # comparable ONLY together with proven episode isolation — a correct grader
        # scoring state corrupted by a prior unreset episode is still unreliable.
        if "split_type" not in config:
            if self.refsol_path and self.episode_reset:
                self.split_type = "official"
            elif self.refsol_path or self.episode_reset:
                self.split_type = "reimplementation"
            else:
                self.split_type = "approximated"

    def effective_concurrency(self, requested: int) -> int:
        """Cap the rollout semaphore for THIS dataset (Runner._predict consults
        this) — without a reset hook, concurrent episodes racing against the
        same FHIR patient state is a correctness risk, not just a slowdown."""
        if self.episode_reset or self.allow_concurrent_episodes or requested <= 1:
            return requested
        import warnings
        warnings.warn(
            f"{self.id}: forcing concurrency 1 (requested {requested}) — concurrent "
            "MedAgentBench episodes with no episode_reset would race against the same "
            "live FHIR patient state. Set allow_concurrent_episodes: true to override "
            "at your own risk, or episode_reset to run safely in parallel.", stacklevel=2)
        return 1

    def _call_episode_reset(self) -> None:
        spec = self.episode_reset
        if not spec:
            return
        try:
            fhir_request(spec["url"], spec.get("method", "POST"), spec.get("body"))
        except Exception as e:
            import warnings
            warnings.warn(f"{self.id}: episode_reset call failed ({type(e).__name__}: {e}) — "
                          "proceeding WITHOUT a guaranteed-clean FHIR state for this episode.",
                          stacklevel=2)

    def _load_funcs(self) -> list:
        if self._funcs is None:
            try:
                h = hashlib.sha256(self.funcs_url.encode()).hexdigest()[:16]
                fp = _download(self.funcs_url, CACHE_DIR / f"{self.id}_funcs_{h}.json")
                self._funcs = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                self._funcs = []
        return self._funcs

    def load(self) -> list[Sample]:
        h = hashlib.sha256(self.source_url.encode()).hexdigest()[:16]
        fp = _download(self.source_url, CACHE_DIR / f"{self.id}_{h}.json")
        data = json.loads(fp.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("data", list(data.values()))
        self._load_funcs()
        out = []
        for i, rec in enumerate(self._truncate(data)):
            out.append(Sample(id=f"{self.id}:{rec.get('id', i)}",
                              task_type=TaskType.AGENT, messages=[], env_spec=rec))
        return out

    def make_env(self, sample: Sample, support=None) -> AgentEnvironment:
        # Called once per episode (once per sample, and once per k>1 rollout) —
        # the natural single injection point for a pre-episode state reset.
        self._call_episode_reset()
        grader = (support or {}).get("grader")
        return MedAgentBenchEnv(sample.env_spec, self.fhir_base, max_rounds=self.max_turns,
                                funcs=self._funcs or [], grader=grader, refsol=self._refsol)


# ===========================================================================
# 4) MediQ — interactive information-seeking clinical reasoning
# ===========================================================================
def _mediq_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+|[一-鿿]", (text or "").lower()))


class MediQEnv(AgentEnvironment):
    """One MediQ episode (Li et al. 2024). The doctor starts information-starved and
    each turn either ASKS the patient a question (an atomic fact is revealed, the most
    relevant one) or COMMITS with 'ANSWER: <letter>'. Faithful to the proactive
    question-asking protocol; scored by final-answer accuracy + #questions + timeout."""

    def __init__(self, scenario: dict[str, Any], max_questions: int = 15,
                 support: Optional[dict] = None):
        self.s = scenario
        self.options = scenario.get("options", {}) or {}
        self.gold = str(scenario.get("answer_idx", "")).strip().upper()[:1]
        facts = scenario.get("facts") or scenario.get("atomic_facts") or []
        self.unrevealed = [str(f) for f in facts]
        ctx = scenario.get("context") or []
        self.initial = (ctx[0] if isinstance(ctx, list) and ctx else
                        (str(ctx) if ctx else "A patient presents for evaluation."))
        self.max_questions = max_questions
        self.support = support or {}
        self.questions = 0

    def doctor_instructions(self) -> str:
        opts = "\n".join(f"{k}. {v}" for k, v in sorted(self.options.items()))
        return (
            "You are a physician with limited initial information. Each turn, either ASK the "
            "patient ONE question to gather more information, or COMMIT to a final choice "
            "with 'ANSWER: <letter>', or — if you judge the available information genuinely "
            "insufficient to decide reliably — 'ABSTAIN' instead of guessing. Only ask what "
            "you need. Optionally state your confidence after committing, e.g. "
            "'ANSWER: B (confidence: 80%)'.\n"
            f"Question: {self.s.get('question', '')}\n{opts}\n"
            f"You may ask at most {self.max_questions} questions before you must answer."
        )

    async def reset(self) -> str:
        return f"Initial information: {self.initial}\nAsk a question, or commit with 'ANSWER: <letter>'."

    async def step(self, action: str):
        conf = self._confidence(action)
        if self._is_abstain(action):
            # A deliberate non-answer, distinct from a wrong guess: the doctor
            # judged the evidence insufficient rather than committing blind.
            # Scored as a non-pass (reward 0) but flagged `abstained` so the
            # aggregate reports an abstention rate instead of folding it into
            # "wrong answer" — MediQ's construct is proactive info-seeking
            # *and* knowing when NOT to answer, not forced-choice accuracy alone.
            return ("", 0.0, True,
                    {"success": False, "abstained": True, "questions": self.questions,
                     **({"confidence": conf} if conf is not None else {})})
        letter = self._commit_letter(action)
        if letter:
            ok = letter == self.gold
            return ("", 1.0 if ok else 0.0, True,
                    {"success": ok, "answered": True, "questions": self.questions,
                     "choice": letter, **({"confidence": conf} if conf is not None else {})})
        self.questions += 1
        if self.questions >= self.max_questions:  # forced final answer
            guess = self._any_letter(action)
            ok = bool(guess) and guess == self.gold
            return ("", 1.0 if ok else 0.0, True,
                    {"success": ok, "timeout": True, "questions": self.questions})
        obs = await self._patient(action)
        return (obs, 0.0, False, {})

    async def _patient(self, question: str) -> str:
        prov = self.support.get("patient")
        if prov is not None and self.unrevealed:
            sys = ("You are the patient. Reveal ONLY the single most relevant fact that answers "
                   "the doctor's question, drawn from these facts: " + " | ".join(self.unrevealed) +
                   ". If none apply, say you don't have that information.")
            g = await prov.agenerate([Message("system", sys), Message("user", question)],
                                     temperature=0.0, max_tokens=80)
            return g.text
        return self._reveal(question)

    def _reveal(self, question: str) -> str:
        if not self.unrevealed:
            return "I don't have any further information about that."
        q = _mediq_words(question)
        best, best_score = self.unrevealed[0], 0
        for f in self.unrevealed:
            sc = len(q & _mediq_words(f))
            if sc > best_score:
                best, best_score = f, sc
        self.unrevealed.remove(best)
        return best

    @staticmethod
    def _commit_letter(action: str) -> str | None:
        m = re.search(r"(?:ANSWER|FINAL ANSWER|DIAGNOSIS|答案)\s*[:：]?\s*\(?([A-E])\)?",
                      action or "", re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m2 = re.search(r"\b(?:the answer is|answer is)\s*\(?([A-E])\)?", action or "", re.IGNORECASE)
        return m2.group(1).upper() if m2 else None

    @staticmethod
    def _any_letter(action: str) -> str | None:
        m = re.search(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", action or "")
        return m.group(1).upper() if m else None

    @staticmethod
    def _is_abstain(action: str) -> bool:
        return bool(re.search(r"\bABSTAIN\b|不(?:能|予)?回答|无法确定作答", action or "", re.IGNORECASE))

    @staticmethod
    def _confidence(action: str) -> float | None:
        """Optional self-reported confidence following a commit, e.g.
        'ANSWER: B (confidence: 80%)' or 'Confidence: 0.8' -> 0.8. Returns None
        if the doctor didn't state one (confidence reporting is optional, so
        calibration metrics only cover the subset that do)."""
        m = re.search(r"confidence\s*[:：]?\s*(\d+(?:\.\d+)?)\s*%", action or "", re.IGNORECASE)
        if m:
            return max(0.0, min(1.0, float(m.group(1)) / 100.0))
        m = re.search(r"confidence\s*[:：]?\s*(0?\.\d+|1(?:\.0+)?)", action or "", re.IGNORECASE)
        return max(0.0, min(1.0, float(m.group(1)))) if m else None


@register_dataset("mediq")
class MediQAdapter(AgentAdapter):
    """config:
      source_url:  MediQ jsonl (default all_dev_good.jsonl)
      max_questions: question budget per episode (default 15)
      support:     {patient: <model_id>}  optional LLM patient; else scripted (offline)
    Scored with pass_k (pass@1 = accuracy; aggregate also reports avg_turns /
    avg_questions / abstain_rate / timeout_rate, and mean_confidence /
    confidence_brier_score when the doctor states a confidence).

    split_type is capped at "reimplementation", NEVER auto-"official": neither
    the rule-based patient (nearest-atomic-fact-by-word-overlap) nor an LLM
    patient here replicate MediQ's original Patient System / adaptive Expert
    System prompts verbatim, so "official" cannot be earned by config alone —
    an explicit override is honored (and up to the operator to justify), but
    the code will never grant it automatically.
    """

    DEFAULT_URL = "https://raw.githubusercontent.com/stellalisy/MediQ/main/data/all_dev_good.jsonl"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.source_url = config.get("source_url", self.DEFAULT_URL)
        self.max_questions = int(config.get("max_questions", config.get("max_turns", 15)))
        self.max_turns = self.max_questions + 1
        if "split_type" not in config:
            # rule-based token-overlap patient = approximated; an LLM patient is a
            # from-scratch reimplementation of the protocol (not verified to match
            # the paper's exact prompts) — either way, never "official" by default.
            self.split_type = "reimplementation" if self.support_spec else "approximated"

    def load(self) -> list[Sample]:
        h = hashlib.sha256(self.source_url.encode()).hexdigest()[:16]
        fp = _download(self.source_url, CACHE_DIR / f"{self.id}_{h}.jsonl")
        out = []
        for i, line in enumerate(fp.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            rec = json.loads(line)
            out.append(Sample(id=f"{self.id}:{rec.get('id', i)}", task_type=TaskType.AGENT,
                              messages=[], env_spec=rec))
            if self.limit and len(out) >= self.limit:
                break
        return out

    def make_env(self, sample: Sample, support=None) -> AgentEnvironment:
        return MediQEnv(sample.env_spec, max_questions=self.max_questions, support=support)
