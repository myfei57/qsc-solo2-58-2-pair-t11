"""Synthetic-data tests for the risk assessment tooling.

No real personal data is used anywhere: rows are generated from
enumerations and a fixed RNG seed.  Assertions also enforce that reports
never serialize individual raw values.
"""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from maskhub.assessment.config import (
    AssessmentConfig,
    AttackerProfile,
    DatasetSpec,
    Thresholds,
)
from maskhub.assessment.engine import assess_dataset
from maskhub.assessment.reporting import render_json, render_markdown
from maskhub.assessment.sampling import reproducible_indices
from maskhub.assessment.stats import _normal_quantile, wilson_interval


def _config(profile: AttackerProfile | None = None) -> AssessmentConfig:
    return AssessmentConfig(
        datasets=(DatasetSpec(
            identifier="synthetic-people",
            path="",
            quasi_identifiers=("age_band", "city", "job"),
            sensitive=("diagnosis_code",),
        ),),
        profile=profile or AttackerProfile(
            name="test",
            available_combos=(
                frozenset({"age_band", "city"}),
                frozenset({"age_band", "city", "job"}),
            ),
            pop_fraction=0.1,
        ),
        thresholds=Thresholds(),
        sample_fraction=0.5,
        sample_size_cap=100_000,
        min_sample_size=100,
        seed=4242,
        max_combo_order=3,
        bootstraps=200,
        confidence=0.95,
    )


def _write_synthetic_csv(path: Path) -> None:
    # 597 rows: many large classes + deliberately planted rare combinations.
    rows = []
    age_bands = ("20-29", "30-39", "40-49")
    cities = ("甲市", "乙市", "丙市")
    jobs = ("岗位A", "岗位B")
    for age in age_bands:
        for city in cities:
            for job in jobs:
                for _ in range(33):
                    rows.append((age, city, job, "D-COMMON"))
    # Three unique-looking planted records.
    rows.append(("60-69", "丁市", "稀有岗位X", "D-RARE-1"))
    rows.append(("20-29", "乙市", "稀有岗位Y", "D-RARE-2"))
    rows.append(("50-59", "甲市", "岗位A", "D-RARE-3"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["age_band", "city", "job", "diagnosis_code"])
        writer.writerows(rows)


class SamplingTests(unittest.TestCase):
    def test_indices_are_reproducible_and_distinct(self) -> None:
        first = reproducible_indices(597, 200, seed=7, dataset_id="d")
        second = reproducible_indices(597, 200, seed=7, dataset_id="d")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 200)
        self.assertEqual(len(set(first)), 200)
        self.assertEqual(first, sorted(first))
        other = reproducible_indices(597, 200, seed=8, dataset_id="d")
        self.assertNotEqual(first, other)
        other_dataset = reproducible_indices(597, 200, seed=7, dataset_id="e")
        self.assertNotEqual(first, other_dataset)

    def test_full_population_returns_everything(self) -> None:
        self.assertEqual(reproducible_indices(50, 50, 1, "d"), list(range(50)))


class StatsTests(unittest.TestCase):
    def test_normal_quantile_known_points(self) -> None:
        self.assertAlmostEqual(_normal_quantile(0.5), 0.0, places=9)
        self.assertAlmostEqual(_normal_quantile(0.975), 1.959963985, places=6)
        self.assertAlmostEqual(_normal_quantile(0.025), -1.959963985, places=6)
        self.assertAlmostEqual(_normal_quantile(0.841344746), 1.0, places=6)

    def test_wilson_interval_covers_point_and_respects_fpc(self) -> None:
        plain = wilson_interval(10, 100, 0.95, fpc=1.0)
        shrunk = wilson_interval(10, 100, 0.95, fpc=math.sqrt(0.5))
        self.assertLess(plain.lower, plain.point < plain.upper)
        self.assertEqual(plain.point, shrunk.point)
        self.assertGreater(plain.upper - plain.lower, shrunk.upper - shrunk.lower)

    def test_wilson_degenerate(self) -> None:
        interval = wilson_interval(0, 100, 0.95, fpc=1.0)
        self.assertEqual(interval.point, 0.0)
        self.assertGreater(interval.upper, 0.0)
        interval_all = wilson_interval(100, 100, 0.95, fpc=1.0)
        self.assertLess(interval_all.lower, 1.0)


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.csv_path = Path(self._tmp.name) / "synthetic.csv"
        _write_synthetic_csv(self.csv_path)
        self.config = _config()
        spec = DatasetSpec(
            identifier="synthetic-people",
            path=str(self.csv_path),
            quasi_identifiers=("age_band", "city", "job"),
            sensitive=("diagnosis_code",),
        )
        self.config = AssessmentConfig(
            datasets=(spec,),
            profile=self.config.profile,
            thresholds=Thresholds(),
            sample_fraction=0.5,
            sample_size_cap=100_000,
            min_sample_size=100,
            seed=4242,
            max_combo_order=3,
            bootstraps=200,
            confidence=0.95,
        )
        self.assessment = assess_dataset(spec, self.config)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_basic_shapes_and_histogram(self) -> None:
        self.assertEqual(self.assessment.n_rows, 597)
        self.assertEqual(self.assessment.sample_size, 298)
        # 1..3 order combos for 3 QIs = 7.
        self.assertEqual(len(self.assessment.combos), 7)
        full_combo = next(
            c for c in self.assessment.combos if len(c.combo) == 3
        )
        # 18 common combos + 3 planted singletons = 21 classes.
        self.assertEqual(full_combo.classes, 21)
        self.assertEqual(full_combo.class_size_histogram["1"], 3)
        self.assertEqual(full_combo.class_size_histogram["33"], 18 * 33)
        self.assertAlmostEqual(full_combo.exact_singleton_share, 3 / 597, places=12)

    def test_repeatability(self) -> None:
        spec = self.config.datasets[0]
        rerun = assess_dataset(spec, self.config)
        self.assertEqual(
            rerun.sampled_indices_sha256, self.assessment.sampled_indices_sha256
        )
        self.assertEqual(
            [c.as_dict() for c in rerun.combos],
            [c.as_dict() for c in self.assessment.combos],
        )

    def test_report_contains_no_individual_values(self) -> None:
        raw_csv = self.csv_path.read_text(encoding="utf-8")
        for secret in ("D-RARE-1", "D-RARE-2", "D-RARE-3", "稀有岗位X", "稀有岗位Y",
                       "D-COMMON", "丁市"):
            self.assertIn(secret, raw_csv)  # present in the source...
        json_report = render_json(self.config, [self.assessment])
        md_report = render_markdown(self.config, [self.assessment])
        for secret in ("D-RARE-1", "D-RARE-2", "D-RARE-3", "稀有岗位X", "稀有岗位Y",
                       "D-COMMON", "丁市"):
            self.assertNotIn(secret, json_report)
            self.assertNotIn(secret, md_report)

    def test_high_risk_detected_for_external_combo(self) -> None:
        self.assertEqual(self.assessment.headline["overall_band"], "high")
        worst_families = {
            (item["family"], tuple(item["combo"]))
            for item in self.assessment.headline["worst"]
        }
        self.assertIn(
            ("small_class_share", ("age_band", "city", "job")), worst_families
        )

    def test_l_diversity_aggregates_present(self) -> None:
        full_combo = next(
            c for c in self.assessment.combos if len(c.combo) == 3
        )
        self.assertIsNotNone(full_combo.l_diversity)
        self.assertEqual(full_combo.l_diversity["min_distinct_sensitive_values"], 1)

    def test_confidence_intervals_are_ordered(self) -> None:
        for combo in self.assessment.combos:
            for interval in (combo.singleton_share, combo.small_class_share,
                             combo.mean_risk, combo.population_unique_share):
                if interval is None:
                    continue
                self.assertLessEqual(interval.lower, interval.point + 1e-12)
                self.assertLessEqual(interval.point, interval.upper + 1e-12)
                self.assertTrue(0.0 <= interval.lower <= 1.0)
                self.assertTrue(0.0 <= interval.upper <= 1.0)


if __name__ == "__main__":
    unittest.main()
