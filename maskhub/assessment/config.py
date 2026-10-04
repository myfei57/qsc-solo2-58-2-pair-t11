"""Configuration for a risk assessment run.

An assessment is described by a manifest (JSON).  Everything that influences
the result is explicit: input paths, quasi-identifiers, the attacker
knowledge profile (which combinations are externally obtainable), sampling
fraction and the RNG seed.  Two runs with the same manifest and the same
input bytes must produce identical numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MISSING_TOKENS = {"", "na", "n/a", "null", "none", "nan", "unknown", "未知", "缺失"}


@dataclass(frozen=True)
class AttackerProfile:
    """What an external attacker is assumed to know.

    ``available_combos`` lists quasi-identifier combinations whose values are
    obtainable from external sources (public registers, leaked databases,
    acquaintances...).  Combination entries are ``"a+b"`` strings.  A combo
    that is not listed is still measured for completeness but marked as not
    externally linkable.  ``pop_uniqueness`` optionally enables the
    population-uniqueness adjustment keyed by the sampling fraction of the
    released table relative to the target population (``pop_fraction``).
    """

    name: str = "external-acquirer"
    available_combos: tuple[frozenset[str], ...] = ()
    pop_fraction: float | None = None

    def combo_available(self, combo: tuple[str, ...]) -> bool:
        wanted = frozenset(combo)
        return any(wanted == item for item in self.available_combos)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "available_combos": ["+".join(sorted(item)) for item in self.available_combos],
            "pop_fraction": self.pop_fraction,
        }


@dataclass(frozen=True)
class Thresholds:
    """Band cut-offs for the headline conclusion.

    Bands are evaluated worst-case across feasible (externally available)
    combos and across the three risk families:
    ``max`` (small-class share, prosecutor), ``mean`` (average per-record
    risk, journalist) and ``unique`` (population-unique share).
    """

    max_high: float = 0.05
    max_medium: float = 0.01
    mean_high: float = 0.02
    mean_medium: float = 0.005
    unique_high: float = 0.02
    unique_medium: float = 0.005
    k_anonymity_floor: int = 5
    l_diversity_floor: int = 2

    def as_dict(self) -> dict[str, object]:
        return {
            "max_high": self.max_high,
            "max_medium": self.max_medium,
            "mean_high": self.mean_high,
            "mean_medium": self.mean_medium,
            "unique_high": self.unique_high,
            "unique_medium": self.unique_medium,
            "k_anonymity_floor": self.k_anonymity_floor,
            "l_diversity_floor": self.l_diversity_floor,
        }


@dataclass(frozen=True)
class DatasetSpec:
    identifier: str
    path: str
    quasi_identifiers: tuple[str, ...]
    sensitive: tuple[str, ...] = ()
    delimiter: str = ","
    encoding: str = "utf-8"
    note: str = ""

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise ValueError("dataset identifier is required")
        if not self.quasi_identifiers:
            raise ValueError(f"dataset {self.identifier}: at least one quasi-identifier required")
        if len(set(self.quasi_identifiers)) != len(self.quasi_identifiers):
            raise ValueError(f"dataset {self.identifier}: duplicate quasi-identifiers")


@dataclass(frozen=True)
class AssessmentConfig:
    datasets: tuple[DatasetSpec, ...]
    profile: AttackerProfile
    thresholds: Thresholds
    sample_fraction: float
    sample_size_cap: int
    min_sample_size: int
    seed: int
    max_combo_order: int
    bootstraps: int
    confidence: float

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("at least one dataset required")
        if not 0.0 < self.sample_fraction <= 1.0:
            raise ValueError("sample_fraction must be in (0, 1]")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.max_combo_order < 1:
            raise ValueError("max_combo_order must be >= 1")
        if self.bootstraps < 0:
            raise ValueError("bootstraps must be >= 0")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")


def _parse_combo(text: object) -> frozenset[str]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"invalid combo specifier: {text!r}")
    parts = frozenset(part.strip() for part in text.split("+") if part.strip())
    if not parts:
        raise ValueError(f"invalid combo specifier: {text!r}")
    return parts


def load_manifest(path: str | Path) -> AssessmentConfig:
    """Load and validate an assessment manifest.

    Structure (see docs/assessment/manifest.example.json)::

        {
          "seed": 20261004,
          "sample_fraction": 0.25,
          "max_combo_order": 3,
          "attacker": {"name": ..., "available_combos": ["dob+zip", ...],
                       "pop_fraction": 0.01},
          "thresholds": {...},
          "datasets": [{"identifier": ..., "path": ...,
                        "quasi_identifiers": [...], "sensitive": [...]}]
        }
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("manifest root must be an object")

    attacker_raw = raw.get("attacker", {})
    if not isinstance(attacker_raw, dict):
        raise ValueError("attacker must be an object")
    profile = AttackerProfile(
        name=str(attacker_raw.get("name", "external-acquirer")),
        available_combos=tuple(
            _parse_combo(item) for item in attacker_raw.get("available_combos", [])
        ),
        pop_fraction=_optional_float(attacker_raw.get("pop_fraction")),
    )

    thresholds_raw = raw.get("thresholds", {})
    if not isinstance(thresholds_raw, dict):
        raise ValueError("thresholds must be an object")
    known = {
        "max_high",
        "max_medium",
        "mean_high",
        "mean_medium",
        "unique_high",
        "unique_medium",
        "k_anonymity_floor",
        "l_diversity_floor",
    }
    unknown = set(thresholds_raw) - known
    if unknown:
        raise ValueError(f"unknown threshold keys: {sorted(unknown)}")
    thresholds = Thresholds(**{key: thresholds_raw[key] for key in thresholds_raw})

    datasets_raw = raw.get("datasets")
    if not isinstance(datasets_raw, list) or not datasets_raw:
        raise ValueError("datasets must be a non-empty list")
    max_order = int(raw.get("max_combo_order", 3))
    datasets: list[DatasetSpec] = []
    for item in datasets_raw:
        if not isinstance(item, dict):
            raise ValueError("each dataset entry must be an object")
        qis = item.get("quasi_identifiers")
        if not isinstance(qis, list) or not qis:
            raise ValueError("quasi_identifiers must be a non-empty list")
        sensitive = item.get("sensitive", [])
        if not isinstance(sensitive, list):
            raise ValueError("sensitive must be a list")
        if max_order > len(qis):
            raise ValueError(
                f"dataset {item.get('identifier')!r} declares only {len(qis)} "
                f"quasi-identifiers, cannot enumerate order {max_order}"
            )
        datasets.append(
            DatasetSpec(
                identifier=str(item["identifier"]),
                path=str(item["path"]),
                quasi_identifiers=tuple(str(name) for name in qis),
                sensitive=tuple(str(name) for name in sensitive),
                delimiter=str(item.get("delimiter", ",")),
                encoding=str(item.get("encoding", "utf-8")),
                note=str(item.get("note", "")),
            )
        )

    config = AssessmentConfig(
        datasets=tuple(datasets),
        profile=profile,
        thresholds=thresholds,
        sample_fraction=float(raw.get("sample_fraction", 0.25)),
        sample_size_cap=int(raw.get("sample_size_cap", 200_000)),
        min_sample_size=int(raw.get("min_sample_size", 2_000)),
        seed=int(raw.get("seed", 20261004)),
        max_combo_order=int(raw.get("max_combo_order", 3)),
        bootstraps=int(raw.get("bootstraps", 1000)),
        confidence=float(raw.get("confidence", 0.95)),
    )
    _validate_against_catalog(config)
    return config


def _validate_against_catalog(config: AssessmentConfig) -> None:
    for combo in config.profile.available_combos:
        matched = False
        for dataset in config.datasets:
            qi_set = set(dataset.quasi_identifiers)
            if combo <= qi_set:
                matched = True
            else:
                unknown = combo - qi_set - set(dataset.sensitive)
                if unknown:
                    raise ValueError(
                        f"attacker combo {'+'.join(sorted(combo))} references unknown "
                        f"columns of {dataset.identifier}: {sorted(unknown)}"
                    )
        if not matched:
            raise ValueError(
                f"attacker combo {'+'.join(sorted(combo))} is not a subset of any "
                "dataset's quasi-identifiers"
            )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)  # type: ignore[arg-type]
    if not 0.0 < number <= 1.0:
        raise ValueError("pop_fraction must be in (0, 1]")
    return number
