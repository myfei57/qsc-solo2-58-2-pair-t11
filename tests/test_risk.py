"""End-to-end tests for the re-identification risk assessment flow."""

from __future__ import annotations

import json
import tempfile
import unittest

from maskhub.console.errors import error_response
from maskhub.console.http import Request
from maskhub.console.routes import build_router
from maskhub.console.runtime import build_runtime
from maskhub.risk.assess import wilson_interval
from maskhub.store.kvstore import RecordStore

DATASET = "citizens"
SENTINELS = ("ZXQ", "SXA", "SXB", "REDACTED-DX")


def call(router, method, path, payload=None, query=None):
    body = json.dumps(payload).encode("utf-8") if payload is not None else b""
    request = Request(method=method, path=path, query=query or {}, headers={}, body=body)
    try:
        response = router.dispatch(request)
    except Exception as exc:  # mirror maskhub.console.server._Handler
        response = error_response(exc)
    return response.status, json.loads(response.payload)


def build_rows():
    rows = []
    for combo in range(18):
        zip_value = f"ZXQ-C{combo:02d}"
        year = 1950 + combo
        sex = "SXA" if combo % 2 == 0 else "SXB"
        for _ in range(10):
            rows.append(
                {"zip": zip_value, "birth_year": year, "sex": sex, "diagnosis": "REDACTED-DX"}
            )
    for unique in range(20):
        rows.append(
            {
                "zip": f"ZXQ-U{unique:03d}",
                "birth_year": 2001,
                "sex": "SXA",
                "diagnosis": "REDACTED-DX",
            }
        )
    return rows


class RiskFlowTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        store = RecordStore(self._tmp.name)
        store.load()
        self.runtime = build_runtime(store)
        self.router = build_router(self.runtime)
        status, _ = call(
            self.router,
            "POST",
            "/sources",
            {"identifier": "src1", "kind": "extract", "endpoint": "file:///tmp"},
        )
        self.assertEqual(status, 200)
        status, _ = call(
            self.router,
            "POST",
            "/datasets",
            {
                "identifier": DATASET,
                "source": "src1",
                "owner": "privacy-team",
                "fields": [
                    {"name": "zip", "sensitivity": "personal"},
                    {"name": "birth_year", "sensitivity": "personal"},
                    {"name": "sex", "sensitivity": "public"},
                    {"name": "diagnosis", "sensitivity": "special"},
                ],
            },
        )
        self.assertEqual(status, 200)
        for rule_id, pattern, action in (
            ("r-zip", "zip", "partial"),
            ("r-year", "birth_year", "hash"),
            ("r-dx", "diagnosis", "redact"),
        ):
            status, _ = call(
                self.router,
                "POST",
                "/rules",
                {"identifier": rule_id, "field_pattern": pattern, "action": action},
            )
            self.assertEqual(status, 200)
        status, _ = call(
            self.router,
            "POST",
            "/policies",
            {"identifier": "p1", "datasets": [DATASET], "rules": ["r-zip", "r-year", "r-dx"]},
        )
        self.assertEqual(status, 200)
        status, _ = call(self.router, "POST", "/policy/activate", {"identifier": "p1"})
        self.assertEqual(status, 200)
        status, body = call(self.router, "POST", "/dataset/sample", {"dataset": DATASET, "rows": build_rows()})
        self.assertEqual(status, 200)
        self.assertEqual(body["row_count"], 200)

    def tearDown(self):
        self._tmp.cleanup()

    def assess(self, **overrides):
        payload = {"dataset": DATASET, "seed": 12345, "sample_size": 200}
        payload.update(overrides)
        return call(self.router, "POST", "/risk/assess", payload)

    def test_high_grade_with_basis_and_intervals(self):
        status, report = self.assess()
        self.assertEqual(status, 200)
        self.assertEqual(report["grade"], "high")
        self.assertEqual(report["policy"], {"identifier": "p1", "state": "active"})
        self.assertEqual(
            report["inputs"]["quasi_identifiers"], ["zip", "birth_year", "sex"]
        )
        self.assertEqual(report["inputs"]["field_actions"]["diagnosis"], "redact")
        at_risk = report["estimates"]["at_risk"]
        self.assertAlmostEqual(at_risk["proportion"], 0.1, places=6)
        lo, hi = at_risk["interval"]
        self.assertLessEqual(lo, at_risk["proportion"])
        self.assertLessEqual(at_risk["proportion"], hi)
        self.assertGreaterEqual(lo, 0.05)
        classes = report["equivalence_classes"]
        self.assertEqual(classes["count"], 38)
        self.assertEqual(classes["min_size"], 1)
        self.assertEqual(classes["records_in_singletons"]["count"], 20)
        self.assertTrue(report["basis"])
        self.assertTrue(any(item["field"] == "zip" for item in report["drivers"]))
        self.assertTrue(report["hardening"])
        targeted = next(s for s in report["scenarios"] if s["name"] == "targeted")
        self.assertEqual(targeted["grade"], "high")
        self.assertEqual(targeted["coverage"], 1.0)

    def test_reproducible_sampling(self):
        _, first = self.assess(sample_size=120)
        _, second = self.assess(sample_size=120)
        for key in ("inputs", "equivalence_classes", "estimates", "scenarios"):
            self.assertEqual(first[key], second[key])
        _, defaulted_a = call(self.router, "POST", "/risk/assess", {"dataset": DATASET, "sample_size": 120})
        _, defaulted_b = call(self.router, "POST", "/risk/assess", {"dataset": DATASET, "sample_size": 120})
        self.assertEqual(defaulted_a["inputs"]["seed"], defaulted_b["inputs"]["seed"])
        self.assertEqual(defaulted_a["estimates"], defaulted_b["estimates"])

    def test_report_never_contains_raw_values(self):
        _, report = self.assess()
        serialized = json.dumps(report)
        for sentinel in SENTINELS:
            self.assertNotIn(sentinel, serialized)
        self.assertNotIn("rows", report)
        status, summary = call(self.router, "GET", "/dataset/sample", query={"dataset": [DATASET]})
        self.assertEqual(status, 200)
        self.assertNotIn("rows", summary)
        for sentinel in SENTINELS:
            self.assertNotIn(sentinel, json.dumps(summary))

    def test_assessment_does_not_touch_serving_policy(self):
        before = self.runtime.policies.get("p1").as_dict()
        self.assess()
        after = self.runtime.policies.get("p1").as_dict()
        self.assertEqual(before, after)
        self.assertEqual(after["state"], "active")
        self.assertEqual(len(self.runtime.policies.all()), 1)
        self.assertEqual(len(self.runtime.rules.all()), 3)

    def test_low_coverage_scenario_stays_low(self):
        status, report = self.assess(
            scenarios=[{"name": "blind", "availability": 0.1}]
        )
        self.assertEqual(status, 200)
        scenario = report["scenarios"][0]
        self.assertAlmostEqual(scenario["coverage"], 0.001, places=6)
        self.assertEqual(scenario["grade"], "low")
        self.assertEqual(report["grade"], "low")

    def test_quasi_identifier_override(self):
        status, report = self.assess(quasi_identifiers=["sex"])
        self.assertEqual(status, 200)
        self.assertEqual(report["inputs"]["quasi_identifiers"], ["sex"])
        self.assertEqual(report["grade"], "low")
        self.assertEqual(report["equivalence_classes"]["min_size"], 90)

    def test_reports_are_stored_and_listed(self):
        _, report = self.assess()
        run_id = report["run_id"]
        status, latest = call(self.router, "GET", "/risk/report", query={"dataset": [DATASET]})
        self.assertEqual(status, 200)
        self.assertEqual(latest["run_id"], run_id)
        status, fetched = call(
            self.router, "GET", "/risk/report", query={"dataset": [DATASET], "run": [run_id]}
        )
        self.assertEqual(status, 200)
        self.assertEqual(fetched["run_id"], run_id)
        status, listing = call(self.router, "GET", "/risk/reports", query={"dataset": [DATASET]})
        self.assertEqual(status, 200)
        self.assertTrue(any(item["run_id"] == run_id for item in listing["reports"]))

    def test_missing_sample_is_not_found(self):
        status, body = call(self.router, "POST", "/risk/assess", {"dataset": DATASET, "seed": 1})
        self.assertEqual(status, 200)  # sample exists in this fixture
        status, body = call(
            self.router,
            "POST",
            "/datasets",
            {
                "identifier": "empty-ds",
                "source": "src1",
                "owner": "privacy-team",
                "fields": [{"name": "zip", "sensitivity": "personal"}],
            },
        )
        self.assertEqual(status, 200)
        status, body = call(self.router, "POST", "/risk/assess", {"dataset": "empty-ds"})
        self.assertEqual(status, 404)
        self.assertIn("no sample on file", body["error"])

    def test_sample_validation(self):
        status, body = call(
            self.router,
            "POST",
            "/dataset/sample",
            {"dataset": DATASET, "rows": [{"zip": "ZXQ-1", "unknown": 1}]},
        )
        self.assertEqual(status, 400)
        self.assertIn("undeclared", body["error"])
        status, body = call(self.router, "POST", "/dataset/sample", {"dataset": DATASET, "rows": []})
        self.assertEqual(status, 400)

    def test_wilson_interval_bounds(self):
        phat, lo, hi = wilson_interval(0, 100, 0.95)
        self.assertEqual(phat, 0.0)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)
        _, lo, hi = wilson_interval(50, 100, 0.95)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)


if __name__ == "__main__":
    unittest.main()
