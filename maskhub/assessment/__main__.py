"""Command-line entry point: ``python -m maskhub.assessment``.

Example::

    python -m maskhub.assessment \\
        --manifest docs/assessment/manifest.example.json \\
        --json-out reports/risk.json --md-out reports/risk.md

The command is strictly read-only with respect to the maskhub data
directory and policy store.
"""

from __future__ import annotations

import argparse
import sys

from maskhub.assessment.config import load_manifest
from maskhub.assessment.engine import assess_dataset
from maskhub.assessment.reporting import write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maskhub.assessment",
        description="Per-dataset re-identification risk assessment (read-only).",
    )
    parser.add_argument("--manifest", required=True, help="path to the assessment manifest JSON")
    parser.add_argument("--json-out", required=True, help="machine-readable report path")
    parser.add_argument("--md-out", default=None, help="optional Markdown report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_manifest(args.manifest)
    assessments = [assess_dataset(spec, config) for spec in config.datasets]
    write_outputs(config, assessments, args.json_out, args.md_out)
    for assessment in assessments:
        print(
            f"{assessment.spec.identifier}: "
            f"{assessment.headline['overall_band']} "
            f"(rows={assessment.n_rows}, sampled={assessment.sample_size})",
            file=sys.stderr,
        )
    print(f"wrote {args.json_out}", file=sys.stderr)
    if args.md_out:
        print(f"wrote {args.md_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
