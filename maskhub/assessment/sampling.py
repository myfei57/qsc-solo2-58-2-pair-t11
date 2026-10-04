"""Deterministic, reproducible sampling and tabular input handling.

Sampling design (documented for auditability):

* The target frame is the full released table in its physical row order; the
  header row is not part of the frame.
* A simple random sample without replacement (SRSWOR) of target rows is
  drawn via a partial Fisher--Yates shuffle.  Randomness comes from a
  SHA-256 based generator seeded with
  ``sha256(f"{seed}|{dataset_id}|{purpose}|{counter}")``, so the procedure
  depends only on the manifest seed, the dataset identifier and the row
  count -- never on row contents -- and can be re-derived by a third party
  without any proprietary RNG.
* Equivalence-class sizes are looked up against the *whole* table, so
  sampled target rows are never biased by sampling their class-mates.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from maskhub.assessment.config import MISSING_TOKENS, DatasetSpec


class HashRandom:
    """A tiny deterministic PRNG with no hidden global state.

    Each call to :meth>::randbelow consumes a counter, hashing it together
    with the seed material.  This is slower than ``random.Random`` but is
    byte-stable across Python versions and transparent to auditors.
    """

    def __init__(self, seed_text: str) -> None:
        self._seed = seed_text.encode("utf-8")
        self._counter = 0

    def _next_u64(self) -> int:
        block = self._seed + b"|" + str(self._counter).encode("ascii")
        self._counter += 1
        return int.from_bytes(hashlib.sha256(block).digest()[:8], "big")

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("upper bound must be positive")
        # Rejection sampling to avoid modulo bias.
        limit = (1 << 64) - ((1 << 64) % upper)
        while True:
            draw = self._next_u64()
            if draw < limit:
                return draw % upper


def reproducible_indices(
    population: int, sample_size: int, seed: int, dataset_id: str
) -> list[int]:
    """Return ``sample_size`` distinct indices in ``range(population)``.

    The returned list is sorted.  Equal parameters always yield an equal
    sample, on every machine.
    """

    if sample_size > population:
        raise ValueError("sample size cannot exceed population size")
    rng = HashRandom(f"{seed}|{dataset_id}|target-rows")
    order = list(range(population))
    # Partial Fisher--Yates shuffle: only the tail positions we will keep.
    for position in range(population - 1, population - sample_size - 1, -1):
        j = rng.randbelow(position + 1)
        order[position], order[j] = order[j], order[position]
    return sorted(order[population - sample_size :])


def bootstrap_indices(
    population: int, sample_size: int, seed: int, dataset_id: str, replicate: int
) -> list[int]:
    """Indices into the target sample for one bootstrap replicate (SRSWR)."""

    rng = HashRandom(f"{seed}|{dataset_id}|bootstrap|{replicate}")
    return [rng.randbelow(sample_size) for _ in range(population)]


def normalize_value(raw: str) -> str:
    """Canonicalize a cell without discarding information.

    Values are whitespace-stripped but kept case-sensitive; recognized
    missing-value tokens are mapped to a single sentinel so missingness
    never silently forms a joinable pseudo-value.
    """

    text = raw.strip()
    if text.lower() in MISSING_TOKENS:
        return "__MISSING__"
    return text


def load_table(spec: DatasetSpec) -> tuple[list[str], list[tuple[str, ...]], str]:
    """Load the columns named in the manifest.

    Returns ``(header, rows, sha256)`` where every row contains the
    quasi-identifier columns followed by the sensitive columns, in manifest
    order.  No other columns are read into memory.
    """

    path = Path(spec.path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    with path.open("r", encoding=spec.encoding, newline="") as handle:
        reader = csv.reader(handle, delimiter=spec.delimiter)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"dataset {spec.identifier}: file is empty")
        header = [column.strip() for column in header]
        wanted = list(spec.quasi_identifiers) + list(spec.sensitive)
        positions: list[int] = []
        for name in wanted:
            if name not in header:
                raise ValueError(
                    f"dataset {spec.identifier}: declared column {name!r} not found "
                    f"in header"
                )
            positions.append(header.index(name))
        rows: list[tuple[str, ...]] = []
        for line, raw_row in enumerate(reader, start=2):
            if not raw_row or all(cell.strip() == "" for cell in raw_row):
                continue  # tolerate trailing blank lines
            if len(raw_row) < len(header):
                raw_row = raw_row + [""] * (len(header) - len(raw_row))
            if len(raw_row) > len(header):
                raise ValueError(
                    f"dataset {spec.identifier}: row {line} has {len(raw_row)} fields, "
                    f"expected {len(header)}"
                )
            rows.append(tuple(normalize_value(raw_row[index]) for index in positions))
    if not rows:
        raise ValueError(f"dataset {spec.identifier}: no data rows")
    return wanted, rows, digest.hexdigest()
