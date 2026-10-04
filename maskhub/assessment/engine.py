"""Dataset-level assessment engine.

Read-only by construction: it loads the released table, enumerates
quasi-identifier combinations up to the configured order, resolves class
sizes against the full table and estimates headline risks on a
reproducible target-row sample.  Nothing here writes to the record store
or touches policy state.
"""

from __future__ import annotations

import datetime as _dt
import itertools
import math
from collections import Counter
from dataclasses import dataclass

from maskhub.assessment.config import AssessmentConfig, DatasetSpec
from maskhub.assessment.sampling import load_table, reproducible_indices
from maskhub.assessment.stats import (
    ComboStats,
    Interval,
    bootstrap_mean_interval,
    class_size_lookup,
    population_unique_rate,
    wilson_interval,
)


@dataclass(frozen=True)
class BandedRisk:
    family: str
    combo: tuple[str, ...]
    interval: Interval
    band: str
    basis: str

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "combo": list(self.combo),
            "basis": self.basis,
            "band": self.band,
            "interval": self.interval.as_dict(),
        }


@dataclass(frozen=True)
class DatasetAssessment:
    spec: DatasetSpec
    n_rows: int
    sha256: str
    sample_size: int
    sample_fraction_realized: float
    sampled_indices_sha256: str
    columns: dict[str, dict[str, object]]
    combos: list[ComboStats]
    headline: dict[str, object]
    recommendations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "identifier": self.spec.identifier,
            "rows": self.n_rows,
            "input_sha256": self.sha256,
            "sampling": {
                "design": "SRSWOR, partial Fisher-Yates, SHA-256 seeded hash RNG",
                "sample_size": self.sample_size,
                "sample_fraction_realized": self.sample_fraction_realized,
                "sampled_indices_sha256": self.sampled_indices_sha256,
            },
            "columns": self.columns,
            "combos": [combo.as_dict() for combo in self.combos],
            "headline": self.headline,
            "recommendations": list(self.recommendations),
        }


def assess_dataset(spec: DatasetSpec, config: AssessmentConfig) -> DatasetAssessment:
    header, rows, digest = load_table(spec)
    n = len(rows)
    n_qi = len(spec.quasi_identifiers)
    qi_positions = list(range(n_qi))

    target_n = _target_sample_size(
        n, config.sample_fraction, config.min_sample_size, config.sample_size_cap
    )
    indices = reproducible_indices(n, target_n, config.seed, spec.identifier)
    import hashlib

    indices_digest = hashlib.sha256(
        ",".join(str(i) for i in indices).encode("ascii")
    ).hexdigest()
    fpc = math.sqrt(max(0.0, (n - target_n) / max(1, n - 1)))

    columns = _column_profiles(spec, rows, qi_positions, config.profile.pop_fraction)
    combos: list[ComboStats] = []
    for order in range(1, min(config.max_combo_order, n_qi) + 1):
        for combo in itertools.combinations(qi_positions, order):
            combos.append(
                _combo_stats(
                    spec=spec,
                    config=config,
                    qi_names=tuple(spec.quasi_identifiers[index] for index in combo),
                    rows=rows,
                    positions=list(combo),
                    sampled_indices=sorted(indices),
                    sample_size=target_n,
                    fpc=fpc,
                )
            )

    headline = _headline(spec, config, combos, n)
    recommendations = tuple(_recommendations(spec, config, combos, columns))
    return DatasetAssessment(
        spec=spec,
        n_rows=n,
        sha256=digest,
        sample_size=target_n,
        sample_fraction_realized=target_n / n,
        sampled_indices_sha256=indices_digest,
        columns=columns,
        combos=combos,
        headline=headline,
        recommendations=recommendations,
    )


def _target_sample_size(n: int, fraction: float, floor: int, cap: int) -> int:
    target = int(round(n * fraction))
    target = min(target, cap)
    target = min(n, target)
    target = max(target, min(floor, n))
    return target


def _column_profiles(
    spec: DatasetSpec,
    rows: list[tuple[str, ...]],
    qi_positions: list[int],
    pop_fraction: float | None,
) -> dict[str, dict[str, object]]:
    n = len(rows)
    profiles: dict[str, dict[str, object]] = {}
    for position in qi_positions:
        values = Counter(row[position] for row in rows)
        singleton_values = sum(1 for count in values.values() if count == 1)
        missing_rows = values.get("__MISSING__", 0)
        rare_values = sum(1 for count in values.values() if count <= 3)
        # Population uniqueness potential for a single attribute.
        if pop_fraction is not None:
            pop_unique_rows = sum(
                count * population_unique_rate(count, pop_fraction)
                for count in values.values()
            )
        else:
            pop_unique_rows = None
        name = spec.quasi_identifiers[position]
        profiles[name] = {
            "role": "quasi_identifier",
            "distinct_values": len(values),
            "distinct_values_per_row": len(values) / n,
            "singleton_value_share_of_distinct": singleton_values / len(values),
            "rare_value_share_of_distinct_le_3_occurrences": rare_values / len(values),
            "missing_row_share": missing_rows / n,
            "expected_population_unique_row_share": (
                pop_unique_rows / n if pop_unique_rows is not None else None
            ),
        }
    for offset, name in enumerate(spec.sensitive):
        position = len(spec.quasi_identifiers) + offset
        values = Counter(row[position] for row in rows)
        profiles[name] = {
            "role": "sensitive",
            "distinct_values": len(values),
            "missing_row_share": values.get("__MISSING__", 0) / n,
        }
    return profiles


def _combo_stats(
    spec: DatasetSpec,
    config: AssessmentConfig,
    qi_names: tuple[str, ...],
    rows: list[tuple[str, ...]],
    positions: list[int],
    sampled_indices: list[int],
    sample_size: int,
    fpc: float,
) -> ComboStats:
    n = len(rows)
    class_sizes = class_size_lookup(rows, positions)
    rows_at_size: Counter[int] = Counter(class_sizes)
    # Histogram expressed as ROW counts per class size (size k -> rows in size-k classes).
    row_histogram: dict[str, int] = {
        str(k): count for k, count in sorted(rows_at_size.items())
    }
    class_count = sum(count // k for k, count in rows_at_size.items())

    sampled_sizes = [class_sizes[i] for i in sampled_indices]
    singleton_successes = sum(1 for size in sampled_sizes if size == 1)
    small_successes = sum(
        1 for size in sampled_sizes if size < config.thresholds.k_anonymity_floor
    )
    journalist_values = [1.0 / size for size in sampled_sizes]

    singleton_ci = wilson_interval(
        singleton_successes, sample_size, config.confidence, fpc
    )
    small_ci = wilson_interval(small_successes, sample_size, config.confidence, fpc)
    mean_ci = bootstrap_mean_interval(
        journalist_values,
        config.confidence,
        fpc,
        seed_text=(
            f"{config.seed}|{spec.identifier}|{'|'.join(qi_names)}|mean-risk"
        ),
        replicates=config.bootstraps,
    )
    pop_ci: Interval | None = None
    if config.profile.pop_fraction is not None:
        pop_values = [
            population_unique_rate(size, config.profile.pop_fraction)
            for size in sampled_sizes
        ]
        pop_ci = bootstrap_mean_interval(
            pop_values,
            config.confidence,
            fpc,
            seed_text=(
                f"{config.seed}|{spec.identifier}|{'|'.join(qi_names)}|pop-unique"
            ),
            replicates=config.bootstraps,
        )

    diversity = None
    if spec.sensitive:
        diversity = _l_diversity(spec, rows, positions)

    return ComboStats(
        combo=qi_names,
        externally_available=config.profile.combo_available(qi_names),
        population=n,
        classes=class_count,
        class_size_histogram=row_histogram,
        exact_singleton_share=sum(1 for size in class_sizes if size == 1) / n,
        exact_small_class_share=(
            sum(1 for size in class_sizes if size < config.thresholds.k_anonymity_floor)
            / n
        ),
        exact_mean_risk=sum(1.0 / size for size in class_sizes) / n,
        sample_size=sample_size,
        singleton_share=singleton_ci,
        small_class_share=small_ci,
        mean_risk=mean_ci,
        population_unique_share=pop_ci,
        l_diversity=diversity,
    )


def _l_diversity(
    spec: DatasetSpec, rows: list[tuple[str, ...]], qi_positions: list[int]
) -> dict[str, object]:
    sensitive_offset = len(spec.quasi_identifiers)
    sensitive_positions = list(
        range(sensitive_offset, sensitive_offset + len(spec.sensitive))
    )
    groups: dict[tuple[str, ...], Counter[tuple[str, ...]]] = {}
    for row in rows:
        signature = tuple(row[index] for index in qi_positions)
        sensitive = tuple(row[index] for index in sensitive_positions)
        groups.setdefault(signature, Counter())[sensitive] += 1

    n_groups = len(groups)
    distinct_counts = [len(counter) for counter in groups.values()]
    # Entropy l-diversity is computed without retaining sensitive values.
    entropies: list[float] = []
    for counter in groups.values():
        total = sum(counter.values())
        entropy = -sum(
            (count / total) * math.log(count / total) for count in counter.values()
        )
        entropies.append(entropy)
    return {
        "groups": n_groups,
        "min_distinct_sensitive_values": min(distinct_counts),
        "mean_distinct_sensitive_values": sum(distinct_counts) / n_groups,
        "mean_entropy": sum(entropies) / n_groups,
    }


def _headline(
    spec: DatasetSpec,
    config: AssessmentConfig,
    combos: list[ComboStats],
    n: int,
) -> dict[str, object]:
    feasible = [combo for combo in combos if combo.externally_available]
    considered = feasible if feasible else combos
    basis = "external_combos" if feasible else "all_combos_no_external_profile_match"

    families: list[BandedRisk] = []
    for combo in considered:
        families.append(
            BandedRisk(
                "small_class_share", combo.combo, combo.small_class_share,
                _band_small(combo.small_class_share.upper, config.thresholds),
                basis,
            )
        )
        families.append(
            BandedRisk(
                "mean_per_record_risk", combo.combo, combo.mean_risk,
                _band_mean(combo.mean_risk.upper, config.thresholds),
                basis,
            )
        )
        if combo.population_unique_share is not None:
            families.append(
                BandedRisk(
                    "population_unique_share", combo.combo,
                    combo.population_unique_share,
                    _band_unique(
                        combo.population_unique_share.upper, config.thresholds
                    ),
                    basis,
                )
            )

    bands = {item.band for item in families}
    overall = "high" if "high" in bands else ("medium" if "medium" in bands else "low")
    worst = sorted(
        families, key=lambda item: item.interval.upper, reverse=True
    )
    return {
        "overall_band": overall,
        "banded_by": "抽样估计的置信区间上界（保守口径）",
        "feasible_combos_evaluated": len(considered),
        "total_combos_evaluated": len(combos),
        "confidence_level": config.confidence,
        "worst": [item.as_dict() for item in worst[: min(5, len(worst))]],
    }


def _band_small(value: float, thresholds) -> str:
    if value >= thresholds.max_high:
        return "high"
    if value >= thresholds.max_medium:
        return "medium"
    return "low"


def _band_mean(value: float, thresholds) -> str:
    if value >= thresholds.mean_high:
        return "high"
    if value >= thresholds.mean_medium:
        return "medium"
    return "low"


def _band_unique(value: float, thresholds) -> str:
    if value >= thresholds.unique_high:
        return "high"
    if value >= thresholds.unique_medium:
        return "medium"
    return "low"


def _recommendations(
    spec: DatasetSpec,
    config: AssessmentConfig,
    combos: list[ComboStats],
    columns: dict[str, dict[str, object]],
) -> list[str]:
    advice: list[str] = []
    feasible = [c for c in combos if c.externally_available] or combos
    worst_singleton = max(feasible, key=lambda c: c.small_class_share.upper)
    if worst_singleton.small_class_share.upper >= config.thresholds.max_medium:
        advice.append(
            "对风险最高的准标识符组合 "
            f"{'+'.join(worst_singleton.combo)} 加强泛化粒度"
            "（合并区间、截断到更粗的时间/地域层级）或对最小的若干等价类做抑制/聚合，"
                       f"使 k 达到 {config.thresholds.k_anonymity_floor} 匿名底线；"
            "先在影子策略上验证后再排期切换，当前在役策略不做变更。"
        )
    if any(
        profile.get("singleton_value_share_of_distinct", 0) and profile[
            "singleton_value_share_of_distinct"
        ]
        > 0.2
        for profile in columns.values()
        if profile.get("role") == "quasi_identifier"
    ):
        advice.append(
            "存在稀有取值占比偏高的准标识符列（报告中以列级统计标识），"
            "建议提高其泛化强度或采用微聚集（micro-aggregation），"
            "并在发布前复核该列与其他准标识符的交叉唯一性。"
        )
    for combo in feasible:
        if combo.l_diversity is not None and combo.l_diversity[
            "min_distinct_sensitive_values"
        ] < config.thresholds.l_diversity_floor:
            advice.append(
                f"组合 {'+'.join(combo.combo)} 的部分等价类内敏感属性取值少于 "
                f"{config.thresholds.l_diversity_floor} 种，建议对敏感属性增加"
                "泛化/分箱或在等价类层面做抑制，使 l-多样性达标。"
            )
            break
    if any(
        profile.get("missing_row_share", 0) > 0.01
        for profile in columns.values()
        if profile.get("role") == "quasi_identifier"
    ):
        advice.append(
            "准标识符列存在缺失值，缺失标记本身可能与外部数据形成连接键；"
            "建议统一缺失编码策略，并评估对缺失行单独施加更强泛化。"
        )
    advice.append(
        "复核外部可获取信息清单（公开登记册、历史泄露库、内部可交叉数据集），"
        "清单变化时重跑本评估；本次评估为只读操作，未修改任何在役掩码策略与规则。"
    )
    return advice


def build_report_metadata(config: AssessmentConfig) -> dict[str, object]:
    return {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "tool": "maskhub.assessment",
        "seed": config.seed,
        "sample_fraction_requested": config.sample_fraction,
        "sample_size_cap": config.sample_size_cap,
        "min_sample_size": config.min_sample_size,
        "max_combo_order": config.max_combo_order,
        "bootstraps": config.bootstraps,
        "confidence": config.confidence,
        "attacker_profile": config.profile.as_dict(),
        "thresholds": config.thresholds.as_dict(),
        "privacy_note": (
            "报告仅包含聚合统计，不包含任何个体的原始准标识符或敏感取值。"
        ),
    }
