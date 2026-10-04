"""Re-identification risk assessment (read-only).

This package measures how easily a released (masked) dataset can be linked
back to individuals, using quasi-identifier equivalence classes, rarity and a
stated attacker knowledge profile.  It never mutates the dataset or any
active policy and never emits individual-level values in its reports.
"""

from __future__ import annotations

from maskhub.assessment.config import (
    AttackerProfile,
    DatasetSpec,
    Thresholds,
    load_manifest,
)
from maskhub.assessment.engine import assess_dataset
from maskhub.assessment.sampling import reproducible_indices

__all__ = [
    "AttackerProfile",
    "DatasetSpec",
    "Thresholds",
    "assess_dataset",
    "load_manifest",
    "reproducible_indices",
]
