"""MedAgentBench grading.

Two paths, matching the official harness contract
``eval(case_data, results, fhir_api_base)`` where
``task_id = case_data['id'].split('_')[0]`` and
``grader_func = getattr(refsol, task_id)``:

1. **Official refsol** (gated — downloaded from the project's Stanford-Box link,
   deliberately withheld to stop benchmark gaming). Point ``refsol_path`` at it
   and we call ``getattr(refsol, task_id)(task, results, fhir_base)`` exactly as
   the harness does. This is the *faithful* path; use it for real numbers.
2. **Built-in grader** (no gated file). Reconstructed from the public task
   instructions + the published grading methodology ("for query tasks, compare
   the agent answer to the reference and verify no POST was issued; for action
   tasks, rule-based checks on the POST payload"). It dispatches per task id and
   validates the *payload* (resourceType + code system/code + ``subject`` →
   ``Patient/{eval_MRN}`` + the task-specific value/note), not merely that the MRN
   appears somewhere. It is intentionally conservative — it never reports a false
   success — so anything that needs a live-FHIR gold (the query *values* of
   tasks 2/4/6/7 and the conditional *no-op* branch of tasks 5/9/10) is reported
   as unverified rather than passed. Supply ``refsol_path`` to grade those.

Task families (300 tasks = 10 × 30):
  query  : task1 (MRN, ships sol), task2 (age), task4 (recent Mg),
           task6 (avg CBG), task7 (recent CBG)
  action : task3  POST Observation  (record BP)
           task5  POST MedicationRequest (IV magnesium)        — conditional
           task8  POST ServiceRequest (orthopedic referral)
           task9  POST MedicationRequest + ServiceRequest (K + lab) — conditional
           task10 POST ServiceRequest (order HbA1C lab)        — conditional
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
from typing import Any

QUERY_TASKS = {"task1", "task2", "task4", "task6", "task7"}
ACTION_TASKS = {"task3", "task5", "task8", "task9", "task10"}
_CONDITIONAL = {"task5", "task9", "task10"}

# Code patterns (scanned over instruction+context so all 30 variants/task work).
_NDC = re.compile(r"\b\d{4,5}-\d{3,4}-\d{2}\b")          # 0338-1715-40, 40032-917-01
_LOINC = re.compile(r"\b\d{3,5}-\d\b")                   # 4548-4, 2823-3
_SNOMED = re.compile(r"\b\d{9,}\b")                      # 306181000000106
_BP = re.compile(r"\b\d{2,3}/\d{2,3}\b")                 # 118/77

_NOOP_NOTE = ("a conditional no-op (correctly ordering nothing) cannot be verified "
              "without live lab values — supply refsol_path for official grading")
_QVALUE_NOTE = ("query value correctness needs the official refsol / live FHIR gold; "
                "the built-in grader only verifies the agent issued no illegal write")


def parse_finish(result_str: str) -> list:
    """Parse the content of ``FINISH([...])`` into a Python list."""
    s = (result_str or "").strip()
    for loader in (ast.literal_eval, json.loads):
        try:
            v = loader(s)
            return list(v) if isinstance(v, (list, tuple)) else [v]
        except Exception:
            continue
    # bare "FINISH(x)" content without brackets
    s = s.strip("[]")
    return [p.strip().strip("'\"") for p in s.split(",")] if s else []


def _norm(x: Any) -> str:
    return re.sub(r"\s+", "", str(x)).strip().lower()


def _as_float(x: Any):
    try:
        return float(re.sub(r"[^0-9.\-]", "", str(x)))
    except (TypeError, ValueError):
        return None


def _query_match(answer: list, sol: list) -> bool:
    # numeric answers (e.g. averages): tolerant compare on the first element
    if len(sol) == 1 and len(answer) >= 1:
        fa, fs = _as_float(answer[0]), _as_float(sol[0])
        if fa is not None and fs is not None:
            return abs(fa - fs) <= max(0.01, abs(fs) * 0.01)
    a, s = {_norm(x) for x in answer}, {_norm(x) for x in sol}
    return s.issubset(a) and (a == s or len(a) <= len(s) + 1)


def _status(p: dict) -> int:
    try:
        return int(p.get("status", 0))
    except (TypeError, ValueError):
        return 0


def _matching_posts(posts: list[dict], resource: str, mrn: str,
                    need_codes: tuple = (), need_text: tuple = ()) -> list[dict]:
    """Successful POSTs whose body is the expected resource for the target patient
    and contains every required code / text fragment."""
    out = []
    for p in posts:
        if not (200 <= _status(p) < 300):
            continue
        body = p.get("body")
        rt = body.get("resourceType") if isinstance(body, dict) else None
        blob = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
        low = blob.lower()
        if resource and rt != resource:
            continue
        if mrn and f"patient/{mrn}".lower() not in low:
            continue
        if need_codes and not all(str(c) in blob for c in need_codes):
            continue
        if need_text and not all(str(t).lower() in low for t in need_text):
            continue
        out.append(p)
    return out


# --- per-task action validators -------------------------------------------
def _grade_action(tid: str, task: dict, mrn: str, text: str,
                  posts: list[dict]) -> tuple[bool, dict]:
    ok_posts = [p for p in posts if 200 <= _status(p) < 300]

    if tid == "task3":                       # record a BP Observation
        bp = _BP.search(text)
        need_text = ("BP",) + ((bp.group(0),) if bp else ())
        good = _matching_posts(posts, "Observation", mrn, need_text=need_text)
        return len(good) >= 1, {
            "mode": "action.task3(Observation/BP)", "bp": bp.group(0) if bp else None,
            "matched_posts": len(good), "successful_posts": len(ok_posts)}

    if tid == "task8":                       # orthopedic referral ServiceRequest
        sn = _SNOMED.search(text)
        good = _matching_posts(posts, "ServiceRequest", mrn,
                               need_codes=(sn.group(0),) if sn else ())
        note_ok = any(("situation" in json.dumps(p.get("body", {}), ensure_ascii=False).lower()
                       or "orthopedic" in json.dumps(p.get("body", {}), ensure_ascii=False).lower())
                      for p in good)
        return len(good) >= 1, {
            "mode": "action.task8(ServiceRequest/referral)",
            "snomed": sn.group(0) if sn else None, "matched_posts": len(good),
            "free_text_present": note_ok, "successful_posts": len(ok_posts)}

    if tid == "task5":                       # conditional: IV magnesium order
        ndc = _NDC.search(text)
        good = _matching_posts(posts, "MedicationRequest", mrn,
                               need_codes=(ndc.group(0),) if ndc else ())
        return _conditional_verdict("task5", good, ok_posts,
                                    {"ndc": ndc.group(0) if ndc else None})

    if tid == "task9":                       # conditional: K order + paired serum-K lab
        ndc, loinc = _NDC.search(text), _LOINC.search(text)
        med = _matching_posts(posts, "MedicationRequest", mrn,
                              need_codes=(ndc.group(0),) if ndc else ())
        lab = _matching_posts(posts, "ServiceRequest", mrn,
                              need_codes=(loinc.group(0),) if loinc else ())
        detail = {"ndc": ndc.group(0) if ndc else None,
                  "loinc": loinc.group(0) if loinc else None,
                  "med_posts": len(med), "lab_posts": len(lab),
                  "successful_posts": len(ok_posts)}
        if not ok_posts:                     # ordered nothing -> undecidable no-op
            return False, {"mode": "action.task9(conditional no-op)",
                           "undecidable_noop": True, "note": _NOOP_NOTE, **detail}
        # both the repletion order AND the paired morning-lab must be present
        return (len(med) >= 1 and len(lab) >= 1), {
            "mode": "action.task9(MedicationRequest + ServiceRequest)", **detail}

    if tid == "task10":                      # conditional: order HbA1C lab
        loinc = _LOINC.search(text)
        good = _matching_posts(posts, "ServiceRequest", mrn,
                               need_codes=(loinc.group(0),) if loinc else ())
        return _conditional_verdict("task10", good, ok_posts,
                                    {"loinc": loinc.group(0) if loinc else None})

    # unknown action task: fall back to "a successful write happened"
    return len(ok_posts) > 0, {"mode": "action(generic)",
                               "successful_posts": len(ok_posts)}


def _conditional_verdict(tid: str, good: list, ok_posts: list,
                         extra: dict) -> tuple[bool, dict]:
    if not ok_posts:                         # agent ordered nothing
        return False, {"mode": f"action.{tid}(conditional no-op)",
                       "undecidable_noop": True, "note": _NOOP_NOTE, **extra}
    return len(good) >= 1, {"mode": f"action.{tid}(order placed)",
                            "matched_posts": len(good),
                            "successful_posts": len(ok_posts), **extra}


def builtin_grade(task: dict, answer: list, fhir_base: str,
                  posts: list[dict]) -> tuple[bool, dict]:
    """Self-contained grader. Returns (success, detail). Conservative: never a
    false success — anything needing a live-FHIR gold is flagged, not passed."""
    tid = str(task.get("id", "")).split("_")[0]
    mrn = str(task.get("eval_MRN", ""))
    text = f"{task.get('instruction', '')} {task.get('context', '')}"

    # ---- query tasks: compare to gold (when shipped) + verify NO write -----
    if tid in QUERY_TASKS or (task.get("sol") and tid not in ACTION_TASKS):
        wrote = any(200 <= _status(p) < 300 for p in posts)
        if task.get("sol"):                  # task1 ships gold -> fully gradable
            val_ok = _query_match(answer, task["sol"])
            return (val_ok and not wrote), {
                "mode": "query", "answer": answer, "sol": task["sol"],
                "value_match": val_ok, "no_write": not wrote}
        # no shipped gold (task2/4/6/7): value is unverifiable offline. We enforce
        # the *checkable* official rule (a query must not POST) but do NOT fabricate
        # a pass — correctness of the value requires refsol.
        return False, {"mode": "query(value-unverified)", "answer": answer,
                       "no_write": not wrote, "value_unverified": True,
                       "note": _QVALUE_NOTE}

    # ---- action tasks: validate the POST payload per task ------------------
    if tid in ACTION_TASKS:
        return _grade_action(tid, task, mrn, text, posts)

    # ---- unknown task id: last-resort fallback ----------------------------
    ok_posts = [p for p in posts if 200 <= _status(p) < 300]
    if task.get("sol"):
        return _query_match(answer, task["sol"]), {
            "mode": "query(fallback)", "answer": answer, "sol": task["sol"]}
    return len(ok_posts) > 0, {"mode": "action(generic)",
                               "successful_posts": len(ok_posts)}


def load_refsol(path: str):
    """Import the official refsol.py from a file path (if the user supplies it)."""
    spec = importlib.util.spec_from_file_location("medagentbench_refsol", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load refsol from {path!r}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grade_with_refsol(refsol, task: dict, answer: list,
                      fhir_base: str) -> tuple[bool, dict]:
    task_id = str(task.get("id", "")).split("_")[0]
    fn = getattr(refsol, task_id, None)
    if fn is None:
        return False, {"error": f"refsol has no grader {task_id!r}"}
    try:
        return bool(fn(task, answer, fhir_base)), {"mode": f"refsol.{task_id}"}
    except Exception as e:  # the official eval also treats exceptions as failure
        return False, {"mode": f"refsol.{task_id}", "error": f"{type(e).__name__}: {e}"}
