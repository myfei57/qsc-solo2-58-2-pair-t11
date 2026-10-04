"""Report rendering: machine-readable JSON and a human-readable summary.

Both renderers are aggregation-only.  They receive statistics, never the
raw rows, and deliberately have no access to the input values.
"""

from __future__ import annotations

import json
from pathlib import Path

from maskhub.assessment.config import AssessmentConfig
from maskhub.assessment.engine import DatasetAssessment, build_report_metadata

BAND_LABELS = {"high": "高", "medium": "中", "low": "低"}


def render_json(
    config: AssessmentConfig, assessments: list[DatasetAssessment]
) -> str:
    document = build_report_metadata(config)
    document["datasets"] = [item.as_dict() for item in assessments]
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=False)


def render_markdown(
    config: AssessmentConfig, assessments: list[DatasetAssessment]
) -> str:
    lines: list[str] = []
    lines.append("# 数据集重识别风险评估报告")
    lines.append("")
    lines.append("> 本报告只包含聚合统计；不含任何个体的原始准标识符或敏感属性取值。")
    lines.append("> 评估全程只读：未修改任何在役掩码策略、规则或数据。")
    lines.append("")
    meta = build_report_metadata(config)
    lines.append("## 评估设置（可复现）")
    lines.append("")
    lines.append(f"- 生成时间（UTC）：{meta['generated_at_utc']}")
    lines.append(f"- 随机种子：`{config.seed}`")
    lines.append(
        f"- 抽样：无放回简单随机抽样（SRSWOR），请求比例 "
        f"{config.sample_fraction:.2%}，单数据集上限 {config.sample_size_cap:,} 行、"
        f"下限 {config.min_sample_size:,} 行"
    )
    lines.append(
        f"- 抽样实现：SHA-256 种子哈希驱动的部分 Fisher–Yates 洗牌，"
        f"仅依赖种子、数据集标识与行数"
    )
    lines.append(
        f"- 置信水平：{config.confidence:.0%}；均值类区间：{config.bootstraps} 次种子化"
        "自助法；比例类区间：Wilson 得分区间（含有限总体校正）"
    )
    profile = config.profile
    combos_rendered = [
        "+".join(sorted(item)) for item in profile.available_combos
    ]
    combos_text = "、".join(combos_rendered) if combos_rendered else "（未声明，按全部组合兜底评估）"
    lines.append(f"- 攻击者画像 `{profile.name}`，外部可获取组合：{combos_text}")
    if profile.pop_fraction is not None:
        lines.append(
            f"- 发布表占目标总体比例 f={profile.pop_fraction:g}，"
            "启用总体唯一性（泊松比率近似）估计"
        )
    lines.append(
        f"- 等价类组合枚举阶数：1 至 {config.max_combo_order} 阶；"
        f"k-匿名底线：{config.thresholds.k_anonymity_floor}"
    )
    lines.append("")

    lines.append("## 总体结论")
    lines.append("")
    lines.append("| 数据集 | 行数 | 实际抽样 | 结论 | 最高风险家族（上界） |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for assessment in assessments:
        worst_text = "—"
        if assessment.headline["worst"]:
            worst = assessment.headline["worst"][0]
            worst_text = (
                f"{worst['family']} [{'+'.join(worst['combo'])}] "
                f"上界 {worst['interval']['upper']:.3%}"
            )
        lines.append(
            f"| {assessment.spec.identifier} | {assessment.n_rows:,} | "
            f"{assessment.sample_size:,} "
            f"({assessment.sample_fraction_realized:.1%}) | "
            f"**{BAND_LABELS[assessment.headline['overall_band']]}** | {worst_text} |"
        )
    lines.append("")
    lines.append(
        "结论分级基于抽样估计的**置信区间上界**（保守口径）："
        "即使点估计不高，上界越过阈值仍按更高风险处理。"
    )
    lines.append("")

    for assessment in assessments:
        lines.extend(_render_dataset(config, assessment))

    lines.append("## 方法学说明与局限")
    lines.append("")
    lines.append(
        "1. **风险量**：对每个准标识符组合，按整表计算等价类规模；"
        "对目标行样本估计：唯一行占比（检察官模型）、"
        f"规模低于 k={config.thresholds.k_anonymity_floor} 的小类行占比、"
        "平均逐行重识别概率（记者模型 1/k）。"
    )
    lines.append(
        "2. **稀有度**：列级报告稀有取值（出现 ≤3 次）占比、单现取值占比与缺失率；"
        "组合级稀有度由等价类规模直方图给出。报告只给计数与比例，不给取值本身。"
    )
    lines.append(
        "3. **外部信息**：仅对攻击者画像中声明为外部可获取的组合做可行性判定与"
        "总体结论汇总；未声明时按全部枚举组合兜底，并明确标注。"
    )
    lines.append(
        "4. **置信区间**：比例类指标用 Wilson 得分区间，按抽样比例做有限总体校正；"
        "均值类指标用固定种子的百分位自助法。复跑同清单同输入应得到逐位一致的结果。"
    )
    lines.append(
        "5. **总体唯一性**为泊松比率近似，属于量级估计，不应解读为精确个体概率。"
    )
    lines.append(
        "6. 若发布表是目标总体的便利样本而非随机抽样，总体唯一性近似会系统性偏差，"
        "此时应以全表精确指标为准。"
    )
    lines.append("")
    return "\n".join(lines)


def _render_dataset(
    config: AssessmentConfig, assessment: DatasetAssessment
) -> list[str]:
    lines: list[str] = []
    spec = assessment.spec
    lines.append(f"## 数据集 {spec.identifier}")
    lines.append("")
    if spec.note:
        lines.append(f"备注：{spec.note}")
        lines.append("")
    lines.append(
        f"- 行数：{assessment.n_rows:,}；输入 SHA-256：`{assessment.sha256}`"
    )
    lines.append(
        f"- 抽样行数：{assessment.sample_size:,}"
        f"（实际比例 {assessment.sample_fraction_realized:.2%}）；"
        f"抽样索引摘要：`{assessment.sampled_indices_sha256[:16]}…`"
    )
    band = BAND_LABELS[assessment.headline["overall_band"]]
    lines.append(f"- **风险结论：{band}**（依据：{assessment.headline['banded_by']}）")
    lines.append("")

    lines.append("### 列级稀有度（聚合统计）")
    lines.append("")
    lines.append(
        "| 列 | 角色 | 不同取值数 | 单现取值占比 | 稀有取值占比(≤3) | 缺失行占比 |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for name, profile in assessment.columns.items():
        lines.append(
            f"| {name} | {profile['role']} | {profile['distinct_values']:,} | "
            f"{_pct(profile.get('singleton_value_share_of_distinct'))} | "
            f"{_pct(profile.get('rare_value_share_of_distinct_le_3_occurrences'))} | "
            f"{_pct(profile.get('missing_row_share'))} |"
        )
    lines.append("")

    lines.append("### 准标识符组合的等价类与风险")
    lines.append("")
    lines.append(
        "| 组合 | 外部可得 | 等价类数 | 最小类规模 | 全表唯一行占比 | 小类行占比[95%CI] | "
        "平均风险[95%CI] |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | --- | --- |")
    for combo in assessment.combos:
        kmin = combo.as_dict()["k_anonymity"]["min_class_size"]
        available = "是" if combo.externally_available else "否"
        pop = ""
        if combo.population_unique_share is not None:
            ci = combo.population_unique_share
            pop = f"；总体唯一≈{ci.point:.2%}"
        lines.append(
            f"| {'+'.join(combo.combo)} | {available} | {combo.classes:,} | {kmin} | "
            f"{combo.exact_singleton_share:.2%} | "
            f"{combo.small_class_share.point:.2%} "
            f"[{combo.small_class_share.lower:.2%}, {combo.small_class_share.upper:.2%}] | "
            f"{combo.mean_risk.point:.3%} "
            f"[{combo.mean_risk.lower:.3%}, {combo.mean_risk.upper:.3%}]{pop} |"
        )
    lines.append("")

    if any(combo.l_diversity is not None for combo in assessment.combos):
        lines.append("### 敏感属性多样性（l-多样性，聚合统计）")
        lines.append("")
        lines.append("| 组合 | 组内敏感取值种数(最小/均值) | 平均熵 |")
        lines.append("| --- | ---: | ---: |")
        for combo in assessment.combos:
            if combo.l_diversity is None:
                continue
            diversity = combo.l_diversity
            lines.append(
                f"| {'+'.join(combo.combo)} | "
                f"{diversity['min_distinct_sensitive_values']} / "
                f"{diversity['mean_distinct_sensitive_values']:.2f} | "
                f"{diversity['mean_entropy']:.3f} |"
            )
        lines.append("")

    lines.append("### 风险依据")
    lines.append("")
    for item in assessment.headline["worst"]:
        lines.append(
            f"- {item['family']}，组合 `{'+'.join(item['combo'])}`："
            f"点估计 {item['interval']['point']:.3%}，"
            f"{config.confidence:.0%} 区间 "
            f"[{item['interval']['lower']:.3%}, {item['interval']['upper']:.3%}]，"
            f"分级 **{BAND_LABELS[item['band']]}**（可行性口径：{item['basis']}）"
        )
    lines.append("")

    if assessment.recommendations:
        lines.append("### 加固方向（建议；未对在役策略做任何改动）")
        lines.append("")
        for recommendation in assessment.recommendations:
            lines.append(f"- {recommendation}")
        lines.append("")
    return lines


def _pct(value: object) -> str:
    if value is None:
        return "—"
    return f"{float(value):.2%}"  # type: ignore[arg-type]


def write_outputs(
    config: AssessmentConfig,
    assessments: list[DatasetAssessment],
    json_path: str | Path,
    markdown_path: str | Path | None,
) -> None:
    json_target = Path(json_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(render_json(config, assessments), encoding="utf-8")
    if markdown_path is not None:
        md_target = Path(markdown_path)
        md_target.parent.mkdir(parents=True, exist_ok=True)
        md_target.write_text(render_markdown(config, assessments), encoding="utf-8")
