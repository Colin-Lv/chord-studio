#!/usr/bin/env python
"""Apply an offline candidate selector and then specialist gates."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from train_candidate_reranker import resolve_path


REPO_ROOT = Path(__file__).resolve().parent.parent


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--selector-model", type=Path, required=True)
    parser.add_argument("--target", default="target_exact_chord")
    parser.add_argument("--specialist-config", type=Path, default=Path("benchmarks/specialists/specialist_gate_config.json"))
    parser.add_argument("--specialist-candidates", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = resolve_path(args.output)
    work_dir = resolve_path(args.work_dir) if args.work_dir else output.parent / f"{output.stem}_pipeline"
    work_dir.mkdir(parents=True, exist_ok=True)

    selected_csv = work_dir / f"{output.stem}_selected.csv"
    selector_report = work_dir / f"{output.stem}_selector_report.json"
    selected_analysis = work_dir / f"{output.stem}_selected_analysis.json"
    selected_changes = work_dir / f"{output.stem}_selected_changes.csv"
    specialist_changes = work_dir / f"{output.stem}_specialist_changes.csv"
    specialist_candidates = resolve_path(args.specialist_candidates) if args.specialist_candidates else resolve_path(args.candidate_csv)

    run(
        [
            sys.executable,
            "app/evaluate_candidate_reranker.py",
            str(resolve_path(args.selector_model)),
            str(resolve_path(args.candidate_csv)),
            "--target",
            args.target,
            "--report",
            str(selector_report),
            "--selected",
            str(selected_csv),
        ]
    )
    run(
        [
            sys.executable,
            "app/apply_selected_candidates.py",
            str(resolve_path(args.analysis)),
            "--selected",
            str(selected_csv),
            "-o",
            str(selected_analysis),
            "--changes-output",
            str(selected_changes),
        ]
    )
    specialist_command = [
        sys.executable,
        "app/apply_specialist_gates.py",
        str(selected_analysis),
        "--candidates",
        str(specialist_candidates),
        "--config",
        str(resolve_path(args.specialist_config)),
        "-o",
        str(output),
        "--changes-output",
        str(specialist_changes),
    ]
    if args.html_output:
        specialist_command.extend(["--html-output", str(resolve_path(args.html_output))])
    run(specialist_command)
    print(f"Offline pipeline output: {output}")
    print(f"Offline pipeline work dir: {work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
