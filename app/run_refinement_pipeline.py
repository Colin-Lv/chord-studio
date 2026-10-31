#!/usr/bin/env python
"""Run the main selector followed by configured specialist gates."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from train_candidate_reranker import resolve_path


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=Path(__file__).resolve().parent.parent, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--selector-model", type=Path, default=Path("benchmarks/models/user5_rich_scale_degree_audiofeat_exact_rf.pkl"))
    parser.add_argument("--specialist-candidates", type=Path, default=Path("benchmarks/reports/user6d_plus_gurenge_halfdim_sus9_harmctx_candidate_rows.csv"))
    parser.add_argument("--specialist-config", type=Path, default=Path("benchmarks/specialists/specialist_gate_config.json"))
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--mode", choices=["triads", "seventh", "rich"], default="rich")
    parser.add_argument("--notation", choices=["sharps", "flats"], default="sharps")
    parser.add_argument("--feature-mode", choices=["cqt", "fast"], default="fast")
    parser.add_argument("--hop-length", type=int, default=1024)
    parser.add_argument("--bass-weight", type=float, default=0.55)
    parser.add_argument("--same-root-alternatives", action="store_true")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = resolve_path(args.output)
    work_dir = resolve_path(args.work_dir) if args.work_dir else output.parent / f"{output.stem}_pipeline"
    work_dir.mkdir(parents=True, exist_ok=True)

    selector_analysis = work_dir / f"{output.stem}_selector_analysis.json"
    selector_html = work_dir / f"{output.stem}_selector_studio.html"
    selector_candidates = work_dir / f"{output.stem}_selector_candidates.csv"
    selector_audiofeat = work_dir / f"{output.stem}_selector_audiofeat_candidates.csv"
    selector_changes = work_dir / f"{output.stem}_selector_changes.csv"
    selector_report = work_dir / f"{output.stem}_selector_report.json"
    specialist_changes = work_dir / f"{output.stem}_specialist_changes.csv"

    selector_command = [
        sys.executable,
        "app/apply_audiofeat_selector.py",
        str(resolve_path(args.analysis)),
        "--audio",
        str(resolve_path(args.audio)),
        "--model",
        str(resolve_path(args.selector_model)),
        "--song-id",
        args.song_id,
        "--top-n",
        str(args.top_n),
        "--mode",
        args.mode,
        "--notation",
        args.notation,
        "--feature-mode",
        args.feature_mode,
        "--hop-length",
        str(args.hop_length),
        "--bass-weight",
        str(args.bass_weight),
        "-o",
        str(selector_analysis),
        "--html-output",
        str(selector_html),
        "--candidates-output",
        str(selector_candidates),
        "--audiofeat-output",
        str(selector_audiofeat),
        "--changes-output",
        str(selector_changes),
        "--report",
        str(selector_report),
    ]
    if args.same_root_alternatives:
        selector_command.append("--same-root-alternatives")
    run(selector_command)

    specialist_command = [
        sys.executable,
        "app/apply_specialist_gates.py",
        str(selector_analysis),
        "--candidates",
        str(resolve_path(args.specialist_candidates)),
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
    print(f"Pipeline output: {output}")
    print(f"Pipeline work dir: {work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
