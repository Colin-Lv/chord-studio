#!/usr/bin/env python
"""Run the current high-level precision diagnostics for Chord Studio."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


ATTRIBUTE_PAIRS = [
    (
        "gurenge_madmom",
        "benchmarks/predictions/user_tests/gurenge_corrected_analysis.json",
        "benchmarks/predictions/user_tests/gurenge_madmom_current_analysis.json",
    ),
    (
        "gurenge_halfdim_gated_legacy",
        "benchmarks/predictions/user_tests/gurenge_corrected_analysis.json",
        "benchmarks/predictions/user_tests/gurenge_user6d_halfdim_gated_analysis.json",
    ),
    (
        "gurenge_two_stage_specialist",
        "benchmarks/predictions/user_tests/gurenge_corrected_analysis.json",
        "benchmarks/predictions/user_tests/gurenge_two_stage_specialist_v1_analysis.json",
    ),
    (
        "gurenge_specialist_gates",
        "benchmarks/predictions/user_tests/gurenge_corrected_analysis.json",
        "benchmarks/predictions/user_tests/gurenge_specialist_gates_analysis.json",
    ),
    (
        "qimeidi_current",
        "benchmarks/predictions/user_tests/qimeidi_corrected_analysis.json",
        "benchmarks/predictions/user_tests/qimeidi_current_flats_extensions_analysis.json",
    ),
    (
        "starrail_madmom",
        "benchmarks/predictions/user_tests/starrail_heart_sorrow_corrected_analysis.json",
        "benchmarks/predictions/user_tests/starrail_heart_sorrow_madmom_current_analysis.json",
    ),
    (
        "laonanhai_selector",
        "benchmarks/predictions/user_tests/laonanhai_user_corrected_final_analysis.json",
        "benchmarks/predictions/user_tests/laonanhai_audiofeat_selector_seventh_withcurrent_analysis.json",
    ),
    (
        "laonanhai_functional_dom7_gate_context",
        "benchmarks/predictions/user_tests/laonanhai_user_corrected_final_analysis.json",
        "benchmarks/predictions/user_tests/laonanhai_functional_dom7_gate_context_analysis.json",
    ),
    (
        "laonanhai_dom7_sus9_gate",
        "benchmarks/predictions/user_tests/laonanhai_user_corrected_final_analysis.json",
        "benchmarks/predictions/user_tests/laonanhai_dom7_sus9_gate_analysis.json",
    ),
    (
        "laonanhai_specialist_gates",
        "benchmarks/predictions/user_tests/laonanhai_user_corrected_final_analysis.json",
        "benchmarks/predictions/user_tests/laonanhai_specialist_gates_analysis.json",
    ),
    (
        "laonanhai_two_stage_specialist",
        "benchmarks/predictions/user_tests/laonanhai_user_corrected_final_analysis.json",
        "benchmarks/predictions/user_tests/laonanhai_two_stage_specialist_v1_analysis.json",
    ),
    (
        "starrail_two_stage_specialist",
        "benchmarks/predictions/user_tests/starrail_heart_sorrow_corrected_analysis.json",
        "benchmarks/predictions/user_tests/starrail_two_stage_specialist_v1_analysis.json",
    ),
    (
        "starrail_keytimeline_two_stage",
        "benchmarks/predictions/user_tests/starrail_heart_sorrow_corrected_analysis.json",
        "benchmarks/predictions/user_tests/starrail_integrated_keytimeline_two_stage_v2_analysis.json",
    ),
    (
        "starrail_auto_candidate_keytimeline_two_stage",
        "benchmarks/predictions/user_tests/starrail_heart_sorrow_corrected_analysis.json",
        "benchmarks/predictions/user_tests/starrail_auto_candidate_keytimeline_two_stage_v1_analysis.json",
    ),
    (
        "yorushika_two_stage_specialist",
        "benchmarks/predictions/user_tests/yorushika_dakara_boku_corrected_latest_analysis.json",
        "benchmarks/predictions/user_tests/yorushika_two_stage_specialist_v1_analysis.json",
    ),
    (
        "yorushika_key_timeline",
        "benchmarks/predictions/user_tests/yorushika_dakara_boku_corrected_latest_analysis.json",
        "benchmarks/predictions/user_tests/yorushika_dakara_boku_key_timeline_analysis.json",
    ),
]


def resolve(path: str) -> Path:
    return REPO_ROOT / path


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-attributes", action="store_true")
    parser.add_argument("--skip-proposals", action="store_true")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "precision_diagnostics_summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports: dict[str, object] = {"attributes": {}, "candidate_proposals": {}}
    if not args.skip_attributes:
        for name, reference, prediction in ATTRIBUTE_PAIRS:
            ref_path = resolve(reference)
            pred_path = resolve(prediction)
            if not ref_path.exists() or not pred_path.exists():
                continue
            row_path = REPO_ROOT / "benchmarks" / "reports" / f"{name}_attribute_rows.csv"
            report_path = REPO_ROOT / "benchmarks" / "reports" / f"{name}_attribute_report.json"
            run(
                [
                    sys.executable,
                    "app/evaluate_chord_attributes.py",
                    str(ref_path),
                    str(pred_path),
                    "-o",
                    str(row_path),
                    "--report",
                    str(report_path),
                ]
            )
            reports["attributes"][name] = json.loads(report_path.read_text(encoding="utf-8"))
    if not args.skip_proposals:
        candidate_sets = [
            ("gurenge_halfdim", "benchmarks/reports/gurenge_halfdim_candidate_rows_expanded_full_v2.csv"),
            ("user6d", "benchmarks/reports/user6d_plus_gurenge_halfdim_candidate_rows_full.csv"),
            ("user6d_sus9", "benchmarks/reports/user6d_plus_gurenge_halfdim_sus9_candidate_rows.csv"),
            ("user6d_sus9_harmctx", "benchmarks/reports/user6d_plus_gurenge_halfdim_sus9_harmctx_candidate_rows.csv"),
        ]
        for name, candidate_csv in candidate_sets:
            path = resolve(candidate_csv)
            if not path.exists():
                continue
            report_path = REPO_ROOT / "benchmarks" / "reports" / f"{name}_candidate_proposals_report.json"
            run(
                [
                    sys.executable,
                    "app/evaluate_candidate_proposals.py",
                    str(path),
                    "--target",
                    "target_exact_chord",
                    "-o",
                    str(REPO_ROOT / "benchmarks" / "reports" / f"{name}_candidate_proposals"),
                    "--report",
                    str(report_path),
                ]
            )
            reports["candidate_proposals"][name] = json.loads(report_path.read_text(encoding="utf-8"))
    specialist_reports = [
        ("specialist_summary", "benchmarks/specialists/user6d_sus9_harmctx_specialist_summary.json"),
        ("sus9_thresholds", "benchmarks/specialists/user6d_sus9_vtoi_combo_thresholds_report.json"),
        ("functional_dom7_thresholds", "benchmarks/specialists/user6d_functional_dom7_combo_thresholds_report.json"),
        ("half_dim_thresholds", "benchmarks/specialists/user6d_half_dim_thresholds_report.json"),
    ]
    reports["specialists"] = {}
    for name, path in specialist_reports:
        report_path = resolve(path)
        if report_path.exists():
            reports["specialists"][name] = json.loads(report_path.read_text(encoding="utf-8"))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.report)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
