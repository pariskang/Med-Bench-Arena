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
        if not self.metrics:
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
        k = k or self.k
        rollouts: list[dict[str, Any]] = []
        first_gen: Generation | None = None
        first_traj: list[dict[str, Any]] | None = None

        for ki in range(k):
            env = self.make_env(sample, support)
            obs = await env.reset()
            messages = [Message("system", self.doctor_system_prompt(env)),
                        Message("user", obs)]
            traj: list[dict[str, Any]] = [{"role": "env", "content": obs}]
            reward, done, info = 0.0, False, {}
            last_gen: Generation | None = None
            turn = 0
            for turn in range(self.max_turns):
                g = await provider.agenerate(messages, **gen)
                last_gen = g
                action = g.text
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

        return Prediction(
            sample_id=sample.id,
            generation=first_gen or Generation.empty(),
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
        if "diagnosis" in action.lower():
            said = action.split(":", 1)[-1].strip()
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
    """One AgentClinic episode (MedQA variant). Patient/measurement/moderator run
    rule-based by default (offline); pass LLM providers in ``support`` to use the
    real LLM-backed agents."""

    def __init__(self, scenario: dict[str, Any], max_turns: int = 20,
                 support: Optional[dict] = None):
        self.osce = scenario.get("OSCE_Examination", scenario)
        self.max_turns = max_turns
        self.support = support or {}
        self.turns_left = max_turns

    def doctor_instructions(self) -> str:
        return (
            "You are a doctor evaluating a patient. Each turn you may:\n"
            "  - ask the patient a question (free text), or\n"
            "  - request a test with 'REQUEST TEST: <name>', or\n"
            "  - give your final answer with 'DIAGNOSIS READY: <diagnosis>'.\n"
            f"You have at most {self.max_turns} interactions. Reason briefly, then act."
        )

    async def reset(self) -> str:
        obj = self.osce.get("Objective_for_Doctor", "Assess and diagnose the patient.")
        return f"Objective: {obj}\nThe patient is in front of you. Begin."

    async def step(self, action: str):
        self.turns_left -= 1
        up = action.upper()
        if "DIAGNOSIS READY" in up:
            dx = action.split(":", 1)[-1].strip() if ":" in action else action
            correct = self.osce.get("Correct_Diagnosis", "")
            success = await self._moderate(dx, correct)
            return ("", 1.0 if success else 0.0, True,
                    {"success": success, "diagnosis": dx, "correct": correct})
        done = self.turns_left <= 0
        if "REQUEST TEST" in up or "REQUEST IMAGES" in up:
            obs = self._measurement(action)
        else:
            obs = await self._patient(action)
        return (obs, 0.0, done, {"success": False} if done else {})

    # --- patient / measurement / moderator -------------------------------
    async def _patient(self, action: str) -> str:
        prov = self.support.get("patient")
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

    def _measurement(self, action: str) -> str:
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
        return "RESULTS: NORMAL READINGS"

    async def _moderate(self, dx: str, correct: str) -> bool:
        prov = self.support.get("moderator")
        if prov is not None:
            q = (f"Correct diagnosis: {correct}\nDoctor's diagnosis: {dx}\n"
                 "Do they refer to the same condition? Answer yes or no.")
            g = await prov.agenerate([Message("user", q)], temperature=0.0, max_tokens=8)
            return "yes" in g.text.lower()
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
        self.support_spec = config.get("support", {})

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
        return AgentClinicEnv(sample.env_spec, max_turns=self.max_turns, support=support)


# ===========================================================================
# 3) MedAgentBench scaffold (requires FHIR server + gated grader)
# ===========================================================================
class MedAgentBenchEnv(AgentEnvironment):
    """FHIR virtual-EHR episode. Actions: 'GET <url>', 'POST <url>\\n<json>',
    'FINISH([...])'. Requires a running HAPI-FHIR server (Docker) at ``fhir_base``
    and the official ``refsol.py`` grader to score; without them, GET/POST are
    inert and reward is 0 (loop still exercised)."""

    def __init__(self, task: dict[str, Any], fhir_base: str, max_rounds: int = 8,
                 grader=None):
        self.task = task
        self.fhir_base = fhir_base.rstrip("/")
        self.rounds_left = max_rounds
        self.grader = grader
        self.posts: list[dict] = []

    def doctor_instructions(self) -> str:
        return (
            "You are using FHIR REST calls to complete the task. Each turn output "
            "exactly ONE of:\n"
            "  GET <url>?param=value\n  POST <url>\\n<json-body>\n  FINISH([<answers>])\n"
            "Output no other text."
        )

    async def reset(self) -> str:
        ctx = self.task.get("context", "")
        return f"{self.task.get('instruction','')}\n{ctx}\nFHIR base: {self.fhir_base}".strip()

    async def step(self, action: str):
        self.rounds_left -= 1
        a = action.strip()
        done = self.rounds_left <= 0
        if a.startswith("FINISH("):
            result = a[len("FINISH("):-1] if a.endswith(")") else a[len("FINISH("):]
            success = self._grade(result)
            return ("", 1.0 if success else 0.0, True,
                    {"success": success, "result": result})
        if a.startswith("GET"):
            return (self._fhir_get(a[3:].strip()), 0.0, done, {})
        if a.startswith("POST"):
            return (self._fhir_post(a[4:].strip()), 0.0, done, {})
        return ("Invalid action. Use GET / POST / FINISH(...).", 0.0, done, {})

    def _grade(self, result: str) -> bool:
        if self.grader is None:
            return False  # refsol.py not provided; cannot score
        try:
            return bool(self.grader(self.task, result, self.fhir_base))
        except Exception:
            return False

    def _fhir_get(self, q: str) -> str:
        try:
            url = f"{self.fhir_base}/{q}" + ("&" if "?" in q else "?") + "_format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "medeval/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read(8000).decode("utf-8", "ignore")
        except Exception as e:
            return f"(FHIR server unreachable: {e})"

    def _fhir_post(self, body: str) -> str:
        self.posts.append({"body": body})
        return "POST recorded."


@register_dataset("medagentbench")
class MedAgentBenchAdapter(AgentAdapter):
    """config:
      source_url:  test_data_v2.json (Stanford MedAgentBench)
      fhir_base:   running HAPI-FHIR base URL (default http://localhost:8080/fhir)
      max_turns:   rounds (paper: 8; repo config: 5)
    Requires Docker FHIR server + gated refsol.py grader to actually score.
    """

    DEFAULT_URL = ("https://raw.githubusercontent.com/stanfordmlgroup/"
                   "MedAgentBench/main/data/medagentbench/test_data_v2.json")

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.source_url = config.get("source_url", self.DEFAULT_URL)
        self.fhir_base = config.get("fhir_base", "http://localhost:8080/fhir")
        self.max_turns = int(config.get("max_turns", 8))

    def load(self) -> list[Sample]:
        h = hashlib.sha256(self.source_url.encode()).hexdigest()[:16]
        fp = _download(self.source_url, CACHE_DIR / f"{self.id}_{h}.json")
        data = json.loads(fp.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("data", list(data.values()))
        out = []
        for i, rec in enumerate(self._truncate(data)):
            out.append(Sample(id=f"{self.id}:{rec.get('id', i)}",
                              task_type=TaskType.AGENT, messages=[], env_spec=rec))
        return out

    def make_env(self, sample: Sample, support=None) -> AgentEnvironment:
        grader = (support or {}).get("grader")
        return MedAgentBenchEnv(sample.env_spec, self.fhir_base,
                                max_rounds=self.max_turns, grader=grader)
