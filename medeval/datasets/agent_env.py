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
        if "DIAGNOSIS READY" in up:
            dx = action.split(":", 1)[-1].strip() if ":" in action else action
            success = await self._moderate(dx, self.correct)
            return ("", 1.0 if success else 0.0, True,
                    {"success": success, "diagnosis": dx, "correct": self.correct})
        done = self.turns_left <= 0
        if "REQUEST TEST" in up or "REQUEST IMAGES" in up:
            obs = self._measurement(action)
        else:
            obs = await self._patient(action)
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
            return info[:400] or "I'm not feeling well, doctor."
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
        if self.nejm:  # flat NEJM record: physical_exams string
            pe = str(self.nejm_rec.get("physical_exams", ""))
            return f"RESULTS: {pe}" if pe else "RESULTS: NORMAL READINGS"
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
        # scripted (rule-based) patient/moderator is an approximation; the faithful
        # multi-agent setup needs LLM `support:` (unless split_type is pinned)
        if "split_type" not in config:
            self.split_type = "official" if self.support_spec else "approximated"

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

    async def step(self, action: str):
        self.rounds_left -= 1
        a = (action or "").strip()
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
        return f"Here is the response from the GET request:\n{body[:6000]}"

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

    Needs a running FHIR server (Docker image jyxsu6/medagentbench on :8080) to
    actually exercise GET/POST. See the README for the docker command.
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
        self._funcs: list | None = None
        self._refsol = None
        if self.refsol_path:
            from .medagentbench_grader import load_refsol
            self._refsol = load_refsol(self.refsol_path)
        # the built-in grader is an approximation; only the official gated refsol
        # is leaderboard-comparable (unless the user pins split_type explicitly)
        if "split_type" not in config:
            self.split_type = "official" if self.refsol_path else "approximated"

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
            "with 'ANSWER: <letter>'. Only ask what you need.\n"
            f"Question: {self.s.get('question', '')}\n{opts}\n"
            f"You may ask at most {self.max_questions} questions before you must answer."
        )

    async def reset(self) -> str:
        return f"Initial information: {self.initial}\nAsk a question, or commit with 'ANSWER: <letter>'."

    async def step(self, action: str):
        letter = self._commit_letter(action)
        if letter:
            ok = letter == self.gold
            return ("", 1.0 if ok else 0.0, True,
                    {"success": ok, "answered": True, "questions": self.questions, "choice": letter})
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


@register_dataset("mediq")
class MediQAdapter(AgentAdapter):
    """config:
      source_url:  MediQ jsonl (default all_dev_good.jsonl)
      max_questions: question budget per episode (default 15)
      support:     {patient: <model_id>}  optional LLM patient; else scripted (offline)
    Scored with pass_k (pass@1 = accuracy; aggregate also reports avg_turns / timeout_rate)."""

    DEFAULT_URL = "https://raw.githubusercontent.com/stellalisy/MediQ/main/data/all_dev_good.jsonl"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.source_url = config.get("source_url", self.DEFAULT_URL)
        self.max_questions = int(config.get("max_questions", config.get("max_turns", 15)))
        self.max_turns = self.max_questions + 1
        self.support_spec = config.get("support", {})

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
