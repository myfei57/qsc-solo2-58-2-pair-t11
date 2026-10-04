"""Re-identification risk assessment endpoints."""

from __future__ import annotations

from maskhub.catalog.dataset import Dataset
from maskhub.console.http import Request, Response
from maskhub.console.validation import require_text
from maskhub.errors import InvalidRequest
from maskhub.risk.assess import (
    DEFAULT_CONFIDENCE,
    DEFAULT_K_THRESHOLD,
    DEFAULT_SAMPLE_SIZE,
    AssessOptions,
    assess,
)
from maskhub.risk.report import new_run_id
from maskhub.risk.sample import SampleFrame, validate_rows
from maskhub.risk.scenario import parse_scenarios


def submit_sample(runtime, request: Request) -> Response:
    payload = request.json_object()
    identifier = require_text(payload, "dataset")
    dataset = runtime.datasets.get(identifier)
    rows = validate_rows(dataset, payload.get("rows"))
    frame = SampleFrame.capture(identifier, rows)
    runtime.samples.save(frame)
    runtime.journal.append("sample.submit", identifier, f"rows={len(rows)} sha256={frame.digest[:12]}")
    return Response.json(200, _frame_summary(frame))


def sample_summary(runtime, request: Request) -> Response:
    identifier = request.query_one("dataset", "") or ""
    if not identifier:
        raise InvalidRequest("dataset query parameter is required")
    return Response.json(200, _frame_summary(runtime.samples.get(identifier)))


def _frame_summary(frame: SampleFrame) -> dict[str, object]:
    return {
        "dataset": frame.dataset,
        "row_count": len(frame.rows),
        "frame_sha256": frame.digest,
        "submitted_at": frame.submitted_at,
    }


def assess_risk(runtime, request: Request) -> Response:
    payload = request.json_object()
    identifier = require_text(payload, "dataset")
    dataset = runtime.datasets.get(identifier)
    frame = runtime.samples.get(identifier)
    options = _parse_options(payload, dataset)
    report = assess(dataset, frame, runtime.policies, runtime.rules, options)
    run_id = new_run_id(
        identifier,
        frame.digest,
        int(report["inputs"]["seed"]),
        int(report["inputs"]["sample_size"]),
        options.confidence,
        options.k_threshold,
    )
    report["run_id"] = run_id
    runtime.risk_reports.save(identifier, run_id, report)
    runtime.journal.append("risk.assess", identifier, f"run={run_id} grade={report['grade']}")
    return Response.json(200, report)


def get_report(runtime, request: Request) -> Response:
    identifier = request.query_one("dataset", "") or ""
    if not identifier:
        raise InvalidRequest("dataset query parameter is required")
    run_id = request.query_one("run", "") or ""
    if run_id:
        report = runtime.risk_reports.get(identifier, run_id)
    else:
        report = runtime.risk_reports.latest(identifier)
    return Response.json(200, report)


def list_reports(runtime, request: Request) -> Response:
    dataset = request.query_one("dataset", "") or None
    summaries = runtime.risk_reports.summaries(dataset)
    return Response.json(200, {"count": len(summaries), "reports": summaries})


def _parse_options(payload: dict[str, object], dataset: Dataset) -> AssessOptions:
    seed = payload.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise InvalidRequest("seed must be an integer")
    sample_size = _optional_int(payload, "sample_size", DEFAULT_SAMPLE_SIZE)
    if sample_size < 2:
        raise InvalidRequest("sample_size must be at least 2")
    confidence = payload.get("confidence", DEFAULT_CONFIDENCE)
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.5 < float(confidence) < 1.0
    ):
        raise InvalidRequest("confidence must be a number in (0.5, 1)")
    k_threshold = _optional_int(payload, "k_threshold", DEFAULT_K_THRESHOLD)
    if k_threshold < 2:
        raise InvalidRequest("k_threshold must be at least 2")
    fields = _optional_quasi_identifiers(payload, dataset)
    return AssessOptions(
        seed=seed,
        sample_size=sample_size,
        confidence=float(confidence),
        k_threshold=k_threshold,
        quasi_identifiers=fields,
        scenarios=parse_scenarios(payload.get("scenarios")),
    )


def _optional_int(payload: dict[str, object], key: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidRequest(f"{key} must be an integer")
    return value


def _optional_quasi_identifiers(
    payload: dict[str, object], dataset: Dataset
) -> tuple[str, ...] | None:
    raw = payload.get("quasi_identifiers")
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise InvalidRequest("quasi_identifiers must be a non-empty list of field names")
    if not all(isinstance(item, str) and item.strip() for item in raw):
        raise InvalidRequest("quasi_identifiers must be a non-empty list of field names")
    declared = {item.name for item in dataset.fields}
    unknown = sorted({item.strip() for item in raw} - declared)
    if unknown:
        raise InvalidRequest(f"unknown quasi-identifier fields {unknown}")
    return tuple(dict.fromkeys(item.strip() for item in raw))
