"""Stored re-identification risk reports."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from maskhub.errors import NotFound
from maskhub.store.kvstore import RecordStore

PREFIX = "risk.report."


def report_key(dataset: str, run_id: str) -> str:
    return f"{PREFIX}{dataset}.{run_id}"


def split_key(key: str) -> tuple[str, str]:
    dataset, _, run_id = key[len(PREFIX):].rpartition(".")
    return dataset, run_id


def new_run_id(
    dataset: str,
    digest: str,
    seed: int,
    sample_size: int,
    confidence: float,
    k_threshold: int,
) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw = f"{dataset}|{digest}|{seed}|{sample_size}|{confidence}|{k_threshold}"
    suffix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{stamp}-{suffix}"


class RiskReportRegistry:
    def __init__(self, store: RecordStore) -> None:
        self._store = store

    def save(self, dataset: str, run_id: str, report: dict[str, object]) -> dict[str, object]:
        self._store.put_json(report_key(dataset, run_id), report)
        return report

    def get(self, dataset: str, run_id: str) -> dict[str, object]:
        payload = self._store.get_json(report_key(dataset, run_id))
        if payload is None:
            raise NotFound(f"no risk report {run_id!r} for dataset {dataset!r}")
        return payload

    def latest(self, dataset: str) -> dict[str, object]:
        keys = [key for key in self._keys() if split_key(key)[0] == dataset]
        if not keys:
            raise NotFound(f"no risk report for dataset {dataset!r}; run POST /risk/assess")
        payload = self._store.get_json(sorted(keys)[-1])
        if payload is None:
            raise NotFound(f"no risk report for dataset {dataset!r}; run POST /risk/assess")
        return payload

    def summaries(self, dataset: str | None = None) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        for key in self._keys():
            name, run_id = split_key(key)
            if dataset is not None and name != dataset:
                continue
            payload = self._store.get_json(key) or {}
            estimates = payload.get("estimates") or {}
            at_risk = estimates.get("at_risk") or {}
            items.append(
                {
                    "dataset": name,
                    "run_id": run_id,
                    "created_at": payload.get("created_at", ""),
                    "grade": payload.get("grade", ""),
                    "at_risk": at_risk.get("proportion"),
                }
            )
        items.sort(key=lambda item: (str(item["dataset"]), str(item["run_id"])))
        return items

    def _keys(self) -> list[str]:
        from maskhub.store.query import keys_with_prefix

        return keys_with_prefix(self._store, PREFIX)
