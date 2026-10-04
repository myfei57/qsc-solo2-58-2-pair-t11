"""Equivalence-class statistics and risk estimators.

Risk model.  For a quasi-identifier combination Q, the released rows
partition into equivalence classes of identical Q-values.  For a target
record whose class contains ``k`` released records:

* prosecutor risk  = ``1`` if ``k == 1`` else ``0`` -- the attacker already
  knows the victim is in the data and merely needs to distinguish them;
* small-class risk = ``1`` if ``k < k_floor`` (default 5) -- classes below
  the anonymity floor are treated as effectively singletons once background
  knowledge narrows the class;
* journalist risk  = ``1/k`` -- the attacker also has to guess whether the
  victim is in the data.

Aggregates over rows give the *unique share*, *small-class share* and
*mean per-record risk*.  Population uniqueness (target population larger
than the released table) uses the standard Poisson rate approximation:
for sampling fraction ``f`` and class size ``k`` the population class rate
is ``lambda = k/f`` and the probability that the population class is a
singleton is ``lambda * exp(-lambda)``.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from maskhub.assessment.sampling import HashRandom


@dataclass(frozen=True)
class Interval:
    point: float
    lower: float
    upper: float

    def as_dict(self) -> dict[str, float]:
        return {"point": self.point, "lower": self.lower, "upper": self.upper}


@dataclass(frozen=True)
class ComboStats:
    combo: tuple[str, ...]
    externally_available: bool
    population: int
    classes: int
    class_size_histogram: dict[str, int]  # JSON keys; counts of ROWS at size k
    exact_singleton_share: float
    exact_small_class_share: float
    exact_mean_risk: float
    sample_size: int
    singleton_share: Interval
    small_class_share: Interval
    mean_risk: Interval
    population_unique_share: Interval | None
    l_diversity: dict[str, object] | None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "combo": list(self.combo),
            "externally_available": self.externally_available,
            "population": self.population,
            "classes": self.classes,
            "class_size_histogram": self.class_size_histogram,
            "k_anonymity": {
                "min_class_size": min((int(k) for k in self.class_size_histogram), default=0),
                "rows_below_floor_share": self.exact_small_class_share,
            },
            "exact_full_table": {
                "singleton_share": self.exact_singleton_share,
                "small_class_share": self.exact_small_class_share,
                "mean_risk": self.exact_mean_risk,
            },
            "sampled": {
                "sample_size": self.sample_size,
                "singleton_share": self.singleton_share.as_dict(),
                "small_class_share": self.small_class_share.as_dict(),
                "mean_risk": self.mean_risk.as_dict(),
            },
        }
        if self.population_unique_share is not None:
            payload["sampled"]["population_unique_share"] = self.population_unique_share.as_dict()
        if self.l_diversity is not None:
            payload["l_diversity"] = self.l_diversity
        return payload


def class_size_lookup(rows: list[tuple[str, ...]], positions: list[int]) -> list[int]:
    """Class size of every row under the selected column positions."""

    counts: Counter[tuple[str, ...]] = Counter()
    signatures: list[tuple[str, ...]] = []
    for row in rows:
        signature = tuple(row[index] for index in positions)
        signatures.append(signature)
        counts[signature] += 1
    return [counts[signature] for signature in signatures]


def wilson_interval(successes: int, n: int, confidence: float, fpc: float) -> Interval:
    """Wilson score interval for a proportion, with finite-population shrink.

    ``fpc`` is ``sqrt((N-n)/(N-1))`` for SRSWOR (1 when the whole frame is
    sampled).  It contracts the interval half-width while leaving the point
    estimate unchanged.
    """

    if n == 0:
        raise ValueError("empty sample")
    z = _normal_quantile(1 - (1 - confidence) / 2)
    p = successes / n
    center_denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / center_denom
    half = (z / center_denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    half *= fpc
    # Clip so the interval always contains the point estimate and stays in [0, 1].
    lower = min(p, max(0.0, center - half))
    upper = max(p, min(1.0, center + half))
    return Interval(p, lower, upper)


def bootstrap_mean_interval(
    values: list[float], confidence: float, fpc: float, seed_text: str, replicates: int
) -> Interval:
    """Percentile bootstrap interval for a mean, seeded and reproducible."""

    n = len(values)
    if n == 0:
        raise ValueError("empty sample")
    point = sum(values) / n
    if replicates == 0 or n == 1:
        return Interval(point, point, point)
    rng = HashRandom(seed_text)
    means: list[float] = []
    for _ in range(replicates):
        total = 0.0
        for _ in range(n):
            total += values[rng.randbelow(n)]
        means.append(total / n)
    means.sort()
    tail = (1 - confidence) / 2
    lower = means[min(len(means) - 1, int(tail * replicates))]
    upper = means[min(len(means) - 1, int((1 - tail) * replicates) - 1)]
    span = (upper - lower) * fpc
    return Interval(point, max(0.0, point - span / 2), min(1.0, point + span / 2))


def population_unique_rate(class_size: int, pop_fraction: float) -> float:
    """Poisson rate approximation of population-singleton probability."""

    rate = class_size / pop_fraction
    return rate * math.exp(-rate)


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF (Acklam's algorithm)."""

    if not 0.0 < p < 1.0:
        raise ValueError("probability out of range")
    a = (
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    )
    b = (
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    )
    c = (
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    )
    d = (
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    )
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    q = p - 0.5
    r = q * q
    return q * (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )