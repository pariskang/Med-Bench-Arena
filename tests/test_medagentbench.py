"""MedAgentBench FHIR loop + grader, validated against an in-process mock FHIR
server (no Docker needed). Confirms: real GET/POST execution, FINISH parsing,
query-task grading vs gold, and action-task write verification.
"""
from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from medeval.datasets.agent_env import MedAgentBenchEnv, fhir_request
from medeval.datasets.medagentbench_grader import parse_finish, builtin_grade


class _FHIRHandler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if "/metadata" in self.path:
            return self._send(200, {"resourceType": "CapabilityStatement"})
        if "/Patient" in self.path:
            return self._send(200, {"resourceType": "Bundle", "total": 1, "entry": [
                {"resource": {"resourceType": "Patient", "id": "S6534835",
                              "name": [{"text": "Peter Stafford"}], "birthDate": "1932-12-29"}}]})
        return self._send(200, {"resourceType": "Bundle", "total": 0, "entry": []})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        try:
            obj = json.loads(raw)
        except Exception:
            return self._send(400, {"error": "invalid"})
        return self._send(201, {"resourceType": obj.get("resourceType", "Observation"), "id": "new-1"})

    def log_message(self, *a):
        pass


def _server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FHIRHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


QUERY = {"id": "task1_1", "instruction": "MRN of Peter Stafford DOB 1932-12-29?",
         "context": "", "sol": ["S6534835"], "eval_MRN": "S6534835"}
ACTION = {"id": "task3_1", "instruction": "Record BP 118/77 for S2380121.",
          "context": "flowsheet BP", "eval_MRN": "S2380121"}


def test_parse_finish_and_builtin_grade():
    assert parse_finish('["S6534835"]') == ["S6534835"]
    assert parse_finish("[123.0]") == [123.0]
    # query task (task1 ships gold) -> exact match + no illegal write
    ok, d = builtin_grade(QUERY, ["S6534835"], "x", [])
    assert ok and d["mode"] == "query"
    ok2, _ = builtin_grade(QUERY, ["WRONG"], "x", [])
    assert not ok2
    # a query task that illegally POSTs must fail even with the right answer
    bad_write = [{"status": 201, "body": {"resourceType": "Observation"}}]
    assert not builtin_grade(QUERY, ["S6534835"], "x", bad_write)[0]
    # numeric tolerance on a real query task (task2 = age)
    assert builtin_grade({"id": "task2_1", "eval_MRN": "S1"}, ["98.60"],
                         "x", [], )[0] is False  # no shipped gold -> value-unverified
    assert builtin_grade({"id": "task2_1", "sol": ["98.6"], "eval_MRN": "S1"},
                         ["98.60"], "x", [])[0]   # explicit gold -> numeric match

    # --- action-task payload validation (the upgraded grader) -------------
    mrn = ACTION["eval_MRN"]
    good_bp = [{"status": 201, "body": {
        "resourceType": "Observation", "code": {"text": "BP"},
        "subject": {"reference": f"Patient/{mrn}"}, "valueString": "118/77 mmHg"}}]
    assert builtin_grade(ACTION, [], "x", good_bp)[0]                    # correct payload
    wrong_rt = [{"status": 201, "body": {
        "resourceType": "MedicationRequest", "code": {"text": "BP"},
        "subject": {"reference": f"Patient/{mrn}"}, "valueString": "118/77"}}]
    assert not builtin_grade(ACTION, [], "x", wrong_rt)[0]               # wrong resourceType
    wrong_pt = [{"status": 201, "body": {
        "resourceType": "Observation", "code": {"text": "BP"},
        "subject": {"reference": "Patient/SXXXX"}, "valueString": "118/77"}}]
    assert not builtin_grade(ACTION, [], "x", wrong_pt)[0]               # wrong patient

    # conditional action (task5 IV-Mg): a no-op is flagged undecidable, not passed
    t5 = {"id": "task5_1", "eval_MRN": "S9",
          "instruction": "order IV magnesium", "context": "NDC 0338-1715-40"}
    ok5, d5 = builtin_grade(t5, [], "x", [])
    assert (not ok5) and d5.get("undecidable_noop")
    good_mg = [{"status": 201, "body": {
        "resourceType": "MedicationRequest", "subject": {"reference": "Patient/S9"},
        "medicationCodeableConcept": {"coding": [{"code": "0338-1715-40"}]}}}]
    assert builtin_grade(t5, [], "x", good_mg)[0]                        # valid order shape


def test_real_get_post_finish_against_mock_fhir():
    srv, base = _server()
    try:
        # the helper actually speaks HTTP to the mock server
        status, data = fhir_request(f"{base}/metadata", "GET")
        assert status == 200 and data["resourceType"] == "CapabilityStatement"

        async def go():
            # --- query task: GET then FINISH ---
            env = MedAgentBenchEnv(QUERY, base, max_rounds=8)
            await env.reset()
            obs, _, _, _ = await env.step("GET Patient?name=Peter Stafford&birthdate=1932-12-29")
            assert "Patient" in obs and "_format=json" not in obs  # got the bundle back
            _, r, done, info = await env.step('FINISH(["S6534835"])')
            assert done and r == 1.0 and info["success"] and info["mode"] == "query"

            env_w = MedAgentBenchEnv(QUERY, base)
            await env_w.reset()
            _, rw, _, _ = await env_w.step('FINISH(["S0000000"])')
            assert rw == 0.0

            # --- action task: POST then FINISH ---
            env_a = MedAgentBenchEnv(ACTION, base, max_rounds=8)
            await env_a.reset()
            obs, _, _, _ = await env_a.step(
                'POST Observation\n{"resourceType":"Observation","code":{"text":"BP"},'
                '"subject":{"reference":"Patient/S2380121"},"valueString":"118/77 mmHg",'
                '"effectiveDateTime":"2023-11-13T10:15:00+00:00","status":"final"}')
            assert "successfully" in obs
            assert env_a.posts[0]["status"] == 201
            _, ra, _, ia = await env_a.step("FINISH([])")
            assert ra == 1.0 and ia["success"] and ia["successful_posts"] == 1

            # action task with no write -> fail
            env_n = MedAgentBenchEnv(ACTION, base)
            await env_n.reset()
            _, rn, _, _ = await env_n.step("FINISH([])")
            assert rn == 0.0

            # malformed POST body -> rejected, not counted
            env_b = MedAgentBenchEnv(ACTION, base)
            await env_b.reset()
            obs, _, _, _ = await env_b.step("POST Observation\n{not json}")
            assert obs == "Invalid POST request"

        asyncio.run(go())
    finally:
        srv.shutdown()


if __name__ == "__main__":
    test_parse_finish_and_builtin_grade()
    test_real_get_post_finish_against_mock_fhir()
    print("OK: MedAgentBench FHIR loop + grader tests passed")
