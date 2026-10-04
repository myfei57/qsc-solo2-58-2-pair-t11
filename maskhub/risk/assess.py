"""Sampling-based re-identification risk assessment over masked frames."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import NormalDist

from maskhub.catalog.dataset import Dataset
from maskhub.policy.registry import PolicyRegistry
from maskhub.risk.sample import SampleFrame
from maskhub.risk.scenario import Scenario
from maskhub.rule.registry import RuleRegistry

DEFAULT_SAMPLE_SIZE = 2000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_K_THRESHOLD = 5
SMALL_SAMPLE = 50

AT_RISK_HIGH = 0.05
COVERAGE_HIGH = 0.5
COVERAGE_MEDIUM = 0.3

GRADE_RANK = {"low": 0, "medium": 1, "high": 2}

ACTION_DIRECTIONS = {
    "partial": "shorten the retained fragment or move the field to 'redact'",
    "hash": "replace 'hash' with per-release rotating 'tokenize' or 'redact'; hashes stay linkable across releases",
    "tokenize": "rotate tokens per release or move rare categories to 'redact'",
    "none": "mask the field or exclude it from the release",
}


@dataclass(frozen=True)
class AssessOptions:
    seed: int | None
    sample_size: int
    confidence: float
    k_threshold: int
    quasi_identifiers: tuple[str, ...] | None
    scenarios: tuple[Scenario, ...]


def derive_quasi_identifiers(
    dataset: Dataset,
    policies: PolicyRegistry,
    rules: RuleRegistry,
) -> tuple[tuple[str, ...], dict[str, str], list[str]]:
    """Fields that keep residual signal under the serving policy."""

    serving = [policy for policy in policies.serving() if policy.covers(dataset.identifier)]
    warnings: list[str] = []
    matched = []
    for policy in serving:
        for rule_id in policy.rules:
            rule = rules.find(rule_id)
            if rule is None:
                warnings.append(
                    f"policy {policy.identifier!r} references missing rule {rule_id!r}; ignored"
                )
                continue
            matched.append(rule)
    actions: dict[str, str] = {}
    for item in dataset.fields:
        action = "none"
        for rule in matched:
            if rule.matches(item.name):
                action = rule.action
                break
        actions[item.name] = action
    if not serving:
        warnings.append("no active policy covers this dataset; every field is assessed as unmasked")
    fields: list[str] = []
    for item in dataset.fields:
        action = actions[item.name]
        if action == "redact":
            continue
        if action == "none" and item.sensitivity_enum.requires_masking:
            warnings.append(
                f"field {item.name!r} is declared {item.sensitivity!r} but no serving rule masks it; treated as unmasked"
            )
        fields.append(item.name)
    return tuple(fields), actions, warnings


def equivalence_keys(
    rows: tuple[dict[str, object], ...], fields: tuple[str, ...]
) -> list[tuple[object, ...]]:
    return [tuple(row.get(name) for name in fields) for row in rows]


def class_sizes(keys: list[tuple[object, ...]]) -> dict[tuple[object, ...], int]:
    sizes: dict[tuple[object, ...], int] = {}
    for key in keys:
        sizes[key] = sizes.get(key, 0) + 1
    return sizes


def wilson_interval(count: int, total: int, confidence: float) -> tuple[float, float, float]:
    if total <= 0:
        return 0.0, 0.0, 0.0
    z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    phat = count / total
    zz = z * z
    denom = 1.0 + zz / total
    center = (phat + zz / (2.0 * total)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / total + zz / (4.0 * total * total)) / denom
    return phat, max(0.0, center - half), min(1.0, center + half)


def mean_interval(values: list[float], confidence: float) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0, 1.0
    z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    half = z * statistics.stdev(values) / math.sqrt(len(values))
    return mean, max(0.0, mean - half), min(1.0, mean + half)


def draw_sample(population: int, size: int, seed: int) -> list[int]:
    if size >= population:
        return list(range(population))
    return sorted(random.Random(seed).sample(range(population), size))


def default_seed(dataset: str, digest: str) -> int:
    raw = hashlib.sha256(f"{dataset}|{digest}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big")


def size_histogram(sizes: list[int], k_threshold: int) -> dict[str, int]:
    buckets: dict[str, int] = {"1": 0}
    if k_threshold > 2:
        buckets[f"2..{k_threshold - 1}"] = 0
    buckets[f"{k_threshold}..{2 * k_threshold - 1}"] = 0
    buckets[f">={2 * k_threshold}"] = 0
    for size in sizes:
        if size == 1:
            buckets["1"] += 1
        elif size < k_threshold:
            buckets[f"2..{k_threshold - 1}"] += 1
        elif size < 2 * k_threshold:
            buckets[f"{k_threshold}..{2 * k_threshold - 1}"] += 1
        else:
            buckets[f">={2 * k_threshold}"] += 1
    return buckets


def field_drivers(
    keys: list[tuple[object, ...]],
    fields: tuple[str, ...],
    sizes: dict[tuple[object, ...], int],
) -> list[dict[str, object]]:
    """Leave-one-out: which fields shrink equivalence classes the most."""

    if len(fields) < 2:
        return []
    total = len(keys)
    base_singletons = sum(1 for size in sizes.values() if size == 1)
    drivers: list[dict[str, object]] = []
    for index, dropped in enumerate(fields):
        reduced: dict[tuple[object, ...], int] = {}
        for key in keys:
            sub = key[:index] + key[index + 1 :]
            reduced[sub] = reduced.get(sub, 0) + 1
        singletons = sum(1 for size in reduced.values() if size == 1)
        reduction = (base_singletons - singletons) / total
        if reduction > 0:
            drivers.append(
                {
                    "field": dropped,
                    "singleton_share_without": round(singletons / total, 6),
                    "singleton_share_reduction": round(reduction, 6),
                }
            )
    drivers.sort(key=lambda item: item["singleton_share_reduction"], reverse=True)
    return drivers[:3]


def grade_scenario(at_risk_lo: float, at_risk_hi: float, coverage: float) -> str:
    if at_risk_lo >= AT_RISK_HIGH:
        if coverage >= COVERAGE_HIGH:
            return "high"
        if coverage >= COVERAGE_MEDIUM:
            return "medium"
        return "low"
    if at_risk_hi >= AT_RISK_HIGH and coverage >= COVERAGE_MEDIUM:
        return "medium"
    return "low"


def hardening_directions(
    actions: dict[str, str],
    drivers: list[dict[str, object]],
    singleton_share: float,
    at_risk_hi: float,
    qi_count: int,
    k_threshold: int,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for driver in drivers:
        field_name = str(driver["field"])
        action = actions.get(field_name, "none")
        items.append(
            {
                "field": field_name,
                "direction": ACTION_DIRECTIONS.get(action, "strengthen generalization for the field"),
                "rationale": (
                    f"dropping this field lowers the singleton share by "
                    f"{driver['singleton_share_reduction']:.1%}; it drives small equivalence classes"
                ),
            }
        )
    if singleton_share > 0:
        items.append(
            {
                "field": None,
                "direction": "suppress records whose quasi-identifier combination is unique before release",
                "rationale": f"{singleton_share:.1%} of frame records are unique on the quasi-identifier combination",
            }
        )
    if at_risk_hi >= AT_RISK_HIGH:
        items.append(
            {
                "field": None,
                "direction": f"raise generalization until the smallest equivalence class reaches k={k_threshold}",
                "rationale": f"the at-risk share may reach {at_risk_hi:.1%} at the configured confidence",
            }
        )
    if qi_count >= 8:
        items.append(
            {
                "field": None,
                "direction": "release fewer quasi-identifier fields at once",
                "rationale": f"{qi_count} fields are combined, which fragments equivalence classes",
            }
        )
    items.append(
        {
            "field": None,
            "direction": "route any change through the draft → review → activate policy flow",
            "rationale": "assessment is advisory; no policy was modified",
        }
    )
    return items


def assess(
    dataset: Dataset,
    frame: SampleFrame,
    policies: PolicyRegistry,
    rules: RuleRegistry,
    options: AssessOptions,
) -> dict[str, object]:
    fields, actions, warnings = derive_quasi_identifiers(dataset, policies, rules)
    if options.quasi_identifiers is not None:
        fields = options.quasi_identifiers
    rows = frame.rows
    population = len(rows)
    sample_size = min(options.sample_size, population)
    seed = options.seed if options.seed is not None else default_seed(dataset.identifier, frame.digest)
    if sample_size < SMALL_SAMPLE:
        warnings.append(f"sample size {sample_size} is small; confidence intervals will be wide")
    if sample_size == population:
        warnings.append(
            "sample covers the whole submitted frame (census); intervals still express uncertainty beyond the frame"
        )
    serving = [policy for policy in policies.serving() if policy.covers(dataset.identifier)]
    policy_block = (
        {"identifier": serving[0].identifier, "state": serving[0].state} if serving else None
    )
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    confidence_pct = f"{options.confidence:.0%}"

    inputs = {
        "frame_rows": population,
        "frame_sha256": frame.digest,
        "frame_submitted_at": frame.submitted_at,
        "sample_size": sample_size,
        "seed": seed,
        "confidence": options.confidence,
        "k_threshold": options.k_threshold,
        "quasi_identifiers": list(fields),
        "field_actions": actions,
    }
    method = {
        "sampling": "uniform random without replacement over the submitted frame (random.Random(seed).sample)",
        "inference": "intervals generalize frame findings to the production dataset, assuming the submitted frame is representative",
        "reproducibility": "re-run POST /risk/assess with the same frame (see frame_sha256), seed, sample_size, confidence and k_threshold to reproduce",
    }
    guarantees = [
        "no individual record values appear in this report",
        "no policy, rule or dataset definition was modified",
    ]

    if not fields:
        return {
            "dataset": dataset.identifier,
            "created_at": created_at,
            "grade": "low",
            "policy": policy_block,
            "inputs": inputs,
            "method": method,
            "equivalence_classes": None,
            "estimates": None,
            "scenarios": [
                {
                    "name": scenario.name,
                    "coverage": 1.0,
                    "grade": "low",
                    "basis": ["no residual quasi-identifier fields; nothing to combine"],
                }
                for scenario in options.scenarios
            ],
            "basis": [
                "no residual quasi-identifier fields: every sensitive field is redacted or excluded"
            ],
            "drivers": [],
            "warnings": warnings,
            "hardening": hardening_directions(actions, [], 0.0, 0.0, 0, options.k_threshold),
            "guarantees": guarantees,
        }

    keys = equivalence_keys(rows, fields)
    sizes = class_sizes(keys)
    ordered = sorted(sizes.values())
    singleton_records = sum(1 for size in ordered if size == 1)
    below_records = sum(size for size in ordered if size < options.k_threshold)

    indices = draw_sample(population, sample_size, seed)
    sampled_sizes = [sizes[keys[index]] for index in indices]
    at_risk_count = sum(1 for size in sampled_sizes if size < options.k_threshold)
    singleton_count = sum(1 for size in sampled_sizes if size == 1)
    at_risk_p, at_risk_lo, at_risk_hi = wilson_interval(
        at_risk_count, sample_size, options.confidence
    )
    singleton_p, singleton_lo, singleton_hi = wilson_interval(
        singleton_count, sample_size, options.confidence
    )

    scenario_blocks: list[dict[str, object]] = []
    overall = "low"
    for scenario in options.scenarios:
        coverage = scenario.coverage(fields)
        scores = [coverage / size for size in sampled_sizes]
        mean, mean_lo, mean_hi = mean_interval(scores, options.confidence)
        grade = grade_scenario(at_risk_lo, at_risk_hi, coverage)
        basis = [
            f"external coverage {coverage:.2f}: probability that external sources supply all {len(fields)} quasi-identifier fields for a target",
            f"at-risk share {at_risk_p:.1%} (records in classes smaller than k={options.k_threshold}), {confidence_pct} CI [{at_risk_lo:.1%}, {at_risk_hi:.1%}]",
        ]
        if grade == "high":
            basis.append(
                f"grade 'high': lower confidence bound {at_risk_lo:.1%} >= {AT_RISK_HIGH:.0%} with coverage {coverage:.2f} >= {COVERAGE_HIGH:.2f}"
            )
        elif grade == "medium":
            basis.append(
                f"grade 'medium': at-risk share cannot be excluded below {AT_RISK_HIGH:.0%} ({confidence_pct} CI [{at_risk_lo:.1%}, {at_risk_hi:.1%}]) with coverage {coverage:.2f} >= {COVERAGE_MEDIUM:.2f}"
            )
        else:
            basis.append(
                f"grade 'low': at-risk share consistent with below {AT_RISK_HIGH:.0%} or coverage {coverage:.2f} below {COVERAGE_MEDIUM:.2f}"
            )
        scenario_blocks.append(
            {
                "name": scenario.name,
                "coverage": round(coverage, 6),
                "mean_record_risk": {
                    "estimate": round(mean, 6),
                    "interval": [round(mean_lo, 6), round(mean_hi, 6)],
                    "definition": "coverage / equivalence class size, averaged over sampled records",
                },
                "grade": grade,
                "basis": basis,
            }
        )
        if GRADE_RANK[grade] > GRADE_RANK[overall]:
            overall = grade

    drivers = field_drivers(keys, fields, sizes)
    basis = [
        f"{len(fields)} quasi-identifier fields assessed under policy {policy_block['identifier'] if policy_block else 'none'}",
        f"{len(sizes)} equivalence classes over {population} frame records; smallest class has {ordered[0]} record(s)",
        f"singleton share {singleton_p:.1%} ({confidence_pct} CI [{singleton_lo:.1%}, {singleton_hi:.1%}])",
        f"at-risk share {at_risk_p:.1%} ({confidence_pct} CI [{at_risk_lo:.1%}, {at_risk_hi:.1%}]) for k={options.k_threshold}",
        f"overall grade '{overall}' is the worst across {len(options.scenarios)} external-information scenario(s)",
    ]
    return {
        "dataset": dataset.identifier,
        "created_at": created_at,
        "grade": overall,
        "policy": policy_block,
        "inputs": inputs,
        "method": method,
        "equivalence_classes": {
            "count": len(sizes),
            "min_size": ordered[0],
            "median_size": statistics.median(ordered),
            "max_size": ordered[-1],
            "records_in_singletons": {
                "count": singleton_records,
                "share": round(singleton_records / population, 6),
            },
            "records_below_threshold": {
                "count": below_records,
                "share": round(below_records / population, 6),
            },
            "histogram": size_histogram(ordered, options.k_threshold),
        },
        "estimates": {
            "at_risk": {
                "proportion": round(at_risk_p, 6),
                "interval": [round(at_risk_lo, 6), round(at_risk_hi, 6)],
                "definition": "share of records whose quasi-identifier combination appears in fewer than k_threshold frame records",
            },
            "singleton": {
                "proportion": round(singleton_p, 6),
                "interval": [round(singleton_lo, 6), round(singleton_hi, 6)],
                "definition": "share of records with a unique quasi-identifier combination",
            },
        },
        "scenarios": scenario_blocks,
        "basis": basis,
        "drivers": drivers,
        "warnings": warnings,
        "hardening": hardening_directions(
            actions, drivers, singleton_records / population, at_risk_hi, len(fields), options.k_threshold
        ),
        "guarantees": guarantees,
    }
