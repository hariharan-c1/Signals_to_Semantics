"""Command-line interface for repository demonstrations and validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from signals_to_semantics.demo import (
    DEFAULT_HERO_CASE,
    DemoValidationError,
    format_hero_trace,
    format_synthetic_demo,
    run_synthetic_signal_demo,
    validate_hero_trace,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sts",
        description="Inspect and validate the Signals to Semantics research artifact.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    hero = subparsers.add_parser(
        "demo",
        help="Validate the recorded real S0-S7 hero trace without external services.",
    )
    hero.add_argument(
        "--case-dir",
        type=Path,
        default=DEFAULT_HERO_CASE,
        help="Path to the published hero-scenario directory.",
    )
    hero.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    synthetic = subparsers.add_parser(
        "demo-synthetic",
        help="Run S0/S1A braking detection on deterministic synthetic kinematics.",
    )
    synthetic.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "demo":
            summary = validate_hero_trace(args.case_dir)
            print(json.dumps(summary.to_dict(), indent=2) if args.json else format_hero_trace(summary))
            return 0
        if args.command == "demo-synthetic":
            result = run_synthetic_signal_demo()
            print(json.dumps(result, indent=2) if args.json else format_synthetic_demo(result))
            return 0
    except DemoValidationError as exc:
        print(f"FAIL  {exc}")
        return 1
    return 2
