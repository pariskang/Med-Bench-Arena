"""MedAgentBench grading.

Two paths, matching the official harness contract
``eval(case_data, results, fhir_api_base)`` where
``task_id = case_data['id'].split('_')[0]`` and
``grader_func = getattr(refsol, task_id)``:

1. **Official refsol** (gated, downloaded from the project's Box link). Point
   ``refsol_path`` at it and we call ``getattr(refsol, task_id)(task, results,
   fhir_base)`` exactly as the harness does.
2. **Built-in grader** (no gated file): query tasks (those with ``sol``) compare
   the agent's ``FINISH`` answer to the gold; action tasks verify a write
   actually happened on the FHIR server for the target patient. The action-task
   check is a documented approximation of the per-task refsol rules.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
from typing import Any


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


def builtin_grade(task: dict, answer: list, fhir_base: str,
                  posts: list[dict]) -> tuple[bool, dict]:
    """Self-contained grader. Returns (success, detail)."""
    if task.get("sol"):  # query task -> compare answer to gold
        ok = _query_match(answer, task["sol"])
        return ok, {"mode": "query", "answer": answer, "sol": task["sol"]}

    # action task -> a write must have hit FHIR for the target patient
    mrn = str(task.get("eval_MRN", ""))
    good_posts = [p for p in posts
                  if 200 <= int(p.get("status", 0)) < 300
                  and (not mrn or mrn in json.dumps(p.get("body", {}), ensure_ascii=False))]
    ok = len(good_posts) > 0
    return ok, {"mode": "action(approx)", "mrn": mrn,
                "successful_posts": len(good_posts), "total_posts": len(posts)}


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
