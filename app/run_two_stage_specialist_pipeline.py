#!/usr/bin/env python
"""Run roman-root selection, conservative spelling, then specialist gates."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from train_candidate_reranker import resolve_path


REPO_ROOT = Path(__file__).resolve().parent.parent


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def filter_candidate_csv(input_csv: Path, output_csv: Path, song_id: str) -> Path:
    if not song_id:
        return input_csv
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [row for row in reader if str(row.get("song_id") or "") == song_id]
    if not rows:
        raise SystemExit(f"No candidate rows found for song_id={song_id}: {input_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--song-id", default="", help="Filter the shared candidate table to one song before selecting/applying.")
    parser.add_argument("--key-timeline-from", type=Path, help="Apply trusted local-key sections from this analysis before root selection.")
    parser.add_argument("--key-section", action="append", default=[], help="Trusted key section passed to apply_key_timeline.py, e.g. 1-15:B or 50-:Ab.")
    parser.add_argument("--auto-candidate-key-timeline", action="store_true", help="Detect local-key sections from the candidate table before root selection.")
    parser.add_argument("--candidate-key-include-source", action="append", default=["topn", "key_v3"], help="Source substring for candidate key-timeline detection.")
    parser.add_argument("--candidate-key-exclude-source", action="append", default=["corrected_grid"], help="Source substring to exclude from candidate key-timeline detection.")
    parser.add_argument("--candidate-key-window-bars", type=int, default=4)
    parser.add_argument("--candidate-key-step-bars", type=int, default=1)
    parser.add_argument("--candidate-key-min-segment-bars", type=int, default=4)
    parser.add_argument("--candidate-key-min-margin", type=float, default=0.05)
    parser.add_argument("--candidate-key-score-weight", type=float, default=0.30)
    parser.add_argument("--candidate-key-rank-weight", type=float, default=0.03)
    parser.add_argument("--key-notation", choices=["auto", "sharps", "flats"], default="auto")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--root-model", type=Path, required=True)
    parser.add_argument("--specialist-config", type=Path, default=Path("benchmarks/specialists/specialist_gate_config.json"))
    parser.add_argument("--specialist-candidates", type=Path)
    parser.add_argument("--allow-extensions", action="store_true")
    parser.add_argument("--enable-chromatic-approach7", action="store_true")
    parser.add_argument("--extension-margin", type=float, default=0.12)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = resolve_path(args.output)
    work_dir = resolve_path(args.work_dir) if args.work_dir else output.parent / f"{output.stem}_pipeline"
    work_dir.mkdir(parents=True, exist_ok=True)

    selected_roots = work_dir / f"{output.stem}_selected_roots.csv"
    root_report = work_dir / f"{output.stem}_root_selector_report.json"
    root_analysis = work_dir / f"{output.stem}_root_analysis.json"
    root_changes = work_dir / f"{output.stem}_root_changes.csv"
    specialist_changes = work_dir / f"{output.stem}_specialist_changes.csv"
    grid_report = work_dir / f"{output.stem}_grid_coverage_report.json"
    grid_rows = work_dir / f"{output.stem}_grid_coverage_rows.csv"
    analysis_path = resolve_path(args.analysis)
    candidate_csv = filter_candidate_csv(
        resolve_path(args.candidate_csv),
        work_dir / f"{output.stem}_candidates_filtered.csv",
        args.song_id,
    )
    if args.key_timeline_from or args.key_section:
        keytimeline_analysis = work_dir / f"{output.stem}_keytimeline_analysis.json"
        key_command = [
            sys.executable,
            "app/apply_key_timeline.py",
            str(analysis_path),
            "-o",
            str(keytimeline_analysis),
            "--notation",
            args.key_notation,
        ]
        if args.key_timeline_from:
            key_command.extend(["--from-analysis", str(resolve_path(args.key_timeline_from))])
        for section in args.key_section:
            key_command.extend(["--section", str(section)])
        run(key_command)
        analysis_path = keytimeline_analysis
    elif args.auto_candidate_key_timeline:
        keytimeline_analysis = work_dir / f"{output.stem}_candidate_keytimeline_analysis.json"
        key_command = [
            sys.executable,
            "app/detect_candidate_key_timeline.py",
            str(analysis_path),
            "--candidate-csv",
            str(candidate_csv),
            "--song-id",
            args.song_id,
            "--window-bars",
            str(args.candidate_key_window_bars),
            "--step-bars",
            str(args.candidate_key_step_bars),
            "--min-segment-bars",
            str(args.candidate_key_min_segment_bars),
            "--min-margin",
            str(args.candidate_key_min_margin),
            "--score-weight",
            str(args.candidate_key_score_weight),
            "--rank-weight",
            str(args.candidate_key_rank_weight),
            "--notation",
            args.key_notation,
            "-o",
            str(keytimeline_analysis),
        ]
        for source in args.candidate_key_include_source:
            key_command.extend(["--include-source", source])
        for source in args.candidate_key_exclude_source:
            key_command.extend(["--exclude-source", source])
        run(key_command)
        analysis_path = keytimeline_analysis
    raw_specialist_candidates = resolve_path(args.specialist_candidates) if args.specialist_candidates else resolve_path(args.candidate_csv)
    specialist_candidates = filter_candidate_csv(
        raw_specialist_candidates,
        work_dir / f"{output.stem}_specialist_candidates_filtered.csv",
        args.song_id,
    )
    run(
        [
            sys.executable,
            "app/audit_candidate_grid_coverage.py",
            str(analysis_path),
            "--candidate-csv",
            str(candidate_csv),
            "--song-id",
            args.song_id,
            "--report",
            str(grid_report),
            "-o",
            str(grid_rows),
        ]
    )

    run(
        [
            sys.executable,
            "app/evaluate_candidate_reranker.py",
            str(resolve_path(args.root_model)),
            str(candidate_csv),
            "--target",
            "target_same_roman_root",
            "--report",
            str(root_report),
            "--selected",
            str(selected_roots),
        ]
    )
    root_command = [
        sys.executable,
        "app/apply_root_selector_conservative_extensions.py",
        str(analysis_path),
        "--selected-roots",
        str(selected_roots),
        "--candidates",
        str(candidate_csv),
        "--extension-margin",
        str(args.extension_margin),
        "-o",
        str(root_analysis),
        "--changes-output",
        str(root_changes),
    ]
    if args.allow_extensions:
        root_command.append("--allow-extensions")
    if args.enable_chromatic_approach7:
        root_command.append("--enable-chromatic-approach7")
    run(root_command)
    specialist_command = [
        sys.executable,
        "app/apply_specialist_gates.py",
        str(root_analysis),
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
    print(f"Two-stage specialist output: {output}")
    print(f"Two-stage specialist work dir: {work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
