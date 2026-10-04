"""Masked sample frames submitted for residual-risk assessment."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from maskhub.catalog.dataset import Dataset
from maskhub.errors import InvalidRequest, NotFound
from maskhub.store.kvstore import RecordStore

PREFIX = "sample."
MAX_ROWS = 50_000
SCALAR_TYPES = (str, int, float, bool)


def frame_digest(rows: tuple[dict[str, object], ...]) -> str:
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_rows(dataset: Dataset, raw: object) -> tuple[dict[str, object], ...]:
    if not isinstance(raw, list) or not raw:
        raise InvalidRequest("rows must be a non-empty list")
    if len(raw) > MAX_ROWS:
        raise InvalidRequest(f"rows must not exceed {MAX_ROWS}")
    declared = {field.name for field in dataset.fields}
    rows: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise InvalidRequest("each row must be an object")
        unknown = sorted(set(item) - declared)
        if unknown:
            raise InvalidRequest(f"row contains undeclared fields {unknown}")
        for key, value in item.items():
            if value is not None and not isinstance(value, SCALAR_TYPES):
                raise InvalidRequest(f"row field {key!r} must be a scalar")
            if isinstance(value, float) and not math.isfinite(value):
                raise InvalidRequest(f"row field {key!r} must be finite")
        rows.append({key: item[key] for key in sorted(item)})
    return tuple(rows)


@dataclass(frozen=True)
class SampleFrame:
    dataset: str
    rows: tuple[dict[str, object], ...]
    submitted_at: str
    digest: str

    @classmethod
    def capture(cls, dataset: str, rows: tuple[dict[str, object], ...]) -> "SampleFrame":
        submitted = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        return cls(
            dataset=dataset,
            rows=rows,
            submitted_at=submitted,
            digest=frame_digest(rows),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "rows": list(self.rows),
            "submitted_at": self.submitted_at,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "SampleFrame | None":
        if not isinstance(payload, dict):
            return None
        dataset = payload.get("dataset")
        rows = payload.get("rows")
        submitted_at = payload.get("submitted_at")
        digest = payload.get("digest")
        if not all(isinstance(item, str) for item in (dataset, submitted_at, digest)):
            return None
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            return None
        return cls(
            dataset=dataset,
            rows=tuple(rows),
            submitted_at=submitted_at,
            digest=digest,
        )


def sample_key(identifier: str) -> str:
    return PREFIX + identifier


class SampleRegistry:
    def __init__(self, store: RecordStore) -> None:
        self._store = store

    def save(self, frame: SampleFrame) -> SampleFrame:
        self._store.put_json(sample_key(frame.dataset), frame.as_dict())
        return frame

    def find(self, identifier: str) -> SampleFrame | None:
        return SampleFrame.from_dict(self._store.get_json(sample_key(identifier)))

    def get(self, identifier: str) -> SampleFrame:
        frame = self.find(identifier)
        if frame is None:
            raise NotFound(
                f"no sample on file for dataset {identifier!r}; submit one via POST /dataset/sample"
            )
        return frame

    def identifiers(self) -> list[str]:
        from maskhub.store.query import keys_with_prefix

        return [key[len(PREFIX):] for key in keys_with_prefix(self._store, PREFIX)]
