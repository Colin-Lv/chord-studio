#!/usr/bin/env python3
"""Build Chord Studio analysis JSONs from external chord labs and beat grids."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from chord_studio import normalize_key_label
from generate_madmom_chords_from_manifest import output_path_for_row
from generate_predictions_from_manifest import output_base_from_prediction


def resolve_path(base: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else base / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("external_chords_dir", type=Path)
    parser.add_argument(
        "--base-evidence-dir",
        type=Path,
        help="Directory containing *_analysis.json files used only for bass/template evidence.",
    )
    parser.add_argument(
        "--external-chords-dir-extra",
        type=Path,
        action="append",
        default=None,
        help="Additional directory containing per-row external chord labs.",
    )
    parser.add_argument("--external-chords-extra-weight", type=float, default=1.0)
    parser.add_argument("--extra-gating", choices=["off", "conservative", "soft"], default="off")
    parser.add_argument("--extra-gating-primary-strong", type=float, default=0.78)
    parser.add_argument("--extra-gating-primary-margin", type=float, default=0.18)
    parser.add_argument("--extra-gating-min-score", type=float, default=0.58)
    parser.add_argument("--extra-gating-min-track-agreement", type=float, default=0.0)
    parser.add_argument("--id-contains")
    parser.add_argument("--default-meter", default="4/4")
    parser.add_argument("--harmonic-unit", choices=["bar", "half-bar", "compound-half", "beat", "auto", "external"], default="auto")
    parser.add_argument("--boundary-snap-tolerance", type=float, default=0.0)
    parser.add_argument("--base-weight", type=float, default=0.0)
    parser.add_argument("--bass-root-weight", type=float, default=0.0)
    parser.add_argument("--infer-slash-bass", action=argparse.BooleanOptionalAction, default=False,
                        help="Opt into legacy inferred bass display; explicit source inversions are always preserved.")
    parser.add_argument("--dominant-seventh-threshold", type=float, default=1.01,
                        help="Default disables unvalidated template seventh inheritance; 0.45 reproduces legacy results.")
    parser.add_argument("--progression-prior", type=Path)
    parser.add_argument("--progression-prior-strength", type=float, default=0.045)
    parser.add_argument("--diatonic-shadow-weight", type=float, default=0.0)
    parser.add_argument("--diatonic-bonus", type=float, default=0.0)
    parser.add_argument("--auto-min-sub-score", type=float, default=0.56)
    parser.add_argument("--auto-min-gain", type=float, default=0.10)
    parser.add_argument("--auto-segment-window", type=int, default=0)
    parser.add_argument("--auto-segment-required", type=int, default=3)
    parser.add_argument("--edge-fill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edge-fill-include-no-chord", action=argparse.BooleanOptionalAction, default=True,
                        help="Preserve explicit source N in edge segment mode; disable to reproduce old coverage.")
    parser.add_argument("--edge-fill-min-duration", type=float, default=0.001)
    parser.add_argument("--edge-fill-min-score", type=float, default=0.0)
    parser.add_argument("--edge-fill-mode", choices=["summary", "segments"], default="segments")
    parser.add_argument("--repeat-consensus-refine", action="store_true")
    parser.add_argument("--repeat-phrase-bars", default="4,8")
    parser.add_argument("--repeat-min-support", type=int, default=3)
    parser.add_argument("--repeat-min-share", type=float, default=0.60)
    parser.add_argument("--repeat-max-score-gap", type=float, default=0.08)
    parser.add_argument("--repeat-max-current-score", type=float, default=0.86)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write experiment outputs to this directory instead of replacing the manifest prediction paths.",
    )
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def analysis_path_for_row(output_dir: Path, row_id: str) -> Path:
    safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in row_id)
    return output_dir / f"{safe_id}_analysis.json"


def output_base_for_row(output_dir: Path, row_id: str) -> Path:
    safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in row_id)
    return output_dir / safe_id


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    base = manifest.parent
    external_dir = args.external_chords_dir.resolve()
    base_evidence_dir = args.base_evidence_dir.resolve() if args.base_evidence_dir else None
    extra_external_dirs = [path.resolve() for path in args.external_chords_dir_extra or []]
    output_dir = args.output_dir.resolve() if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    fusion_script = Path(__file__).resolve().parent / "downbeat_external_fusion.py"

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected = []
    for row in rows:
        row_id = row.get("id", "")
        if args.id_contains and args.id_contains not in row_id:
            continue
        prediction = resolve_path(base, row.get("prediction", ""))
        beat_reference = resolve_path(base, row.get("beat_reference", ""))
        external = output_path_for_row(external_dir, row_id)
        if prediction and beat_reference and prediction.exists() and beat_reference.exists() and external.exists():
            selected.append((row, prediction, beat_reference, external))

    print(f"selected rows: {len(selected)}")
    failures = 0
    for index, (row, prediction, beat_reference, external) in enumerate(selected, start=1):
        row_id = row.get("id", "")
        print(f"[{index}/{len(selected)}] {row_id}")
        output_base = output_base_for_row(output_dir, row_id) if output_dir else output_base_from_prediction(prediction)
        key = normalize_key_label(row.get("key")) or "C"
        command = [
            args.python,
            str(fusion_script),
            str(prediction),
            str(beat_reference),
            str(external),
            "-o",
            str(output_base),
            "--key",
            key,
            "--beat-provenance",
            "reference",
            "--boundary-snap-tolerance",
            str(args.boundary_snap_tolerance),
            "--meter",
            row.get("meter") or args.default_meter,
            "--harmonic-unit",
            args.harmonic_unit,
            "--base-weight",
            f"{args.base_weight:.6f}",
            "--bass-root-weight",
            f"{args.bass_root_weight:.6f}",
            "--dominant-seventh-threshold",
            f"{args.dominant_seventh_threshold:.6f}",
            "--diatonic-shadow-weight",
            f"{args.diatonic_shadow_weight:.6f}",
            "--diatonic-bonus",
            f"{args.diatonic_bonus:.6f}",
            "--auto-min-sub-score",
            f"{args.auto_min_sub_score:.6f}",
            "--auto-min-gain",
            f"{args.auto_min_gain:.6f}",
            "--auto-segment-window",
            str(args.auto_segment_window),
            "--auto-segment-required",
            str(args.auto_segment_required),
            "--edge-fill-min-duration",
            f"{args.edge_fill_min_duration:.6f}",
            "--edge-fill-min-score",
            f"{args.edge_fill_min_score:.6f}",
            "--edge-fill-mode",
            args.edge_fill_mode,
            "--repeat-phrase-bars",
            args.repeat_phrase_bars,
            "--repeat-min-support",
            str(args.repeat_min_support),
            "--repeat-min-share",
            f"{args.repeat_min_share:.6f}",
            "--repeat-max-score-gap",
            f"{args.repeat_max_score_gap:.6f}",
            "--repeat-max-current-score",
            f"{args.repeat_max_current_score:.6f}",
            "--extra-gating",
            args.extra_gating,
            "--extra-gating-primary-strong",
            f"{args.extra_gating_primary_strong:.6f}",
            "--extra-gating-primary-margin",
            f"{args.extra_gating_primary_margin:.6f}",
            "--extra-gating-min-score",
            f"{args.extra_gating_min_score:.6f}",
            "--extra-gating-min-track-agreement",
            f"{args.extra_gating_min_track_agreement:.6f}",
        ]
        command.append("--edge-fill" if args.edge_fill else "--no-edge-fill")
        command.append("--infer-slash-bass" if args.infer_slash_bass else "--no-infer-slash-bass")
        command.append("--edge-fill-include-no-chord" if args.edge_fill_include_no_chord else "--no-edge-fill-include-no-chord")
        if args.repeat_consensus_refine:
            command.append("--repeat-consensus-refine")
        if args.progression_prior:
            command += [
                "--progression-prior",
                str(args.progression_prior.resolve()),
                "--progression-prior-strength",
                f"{args.progression_prior_strength:.6f}",
            ]
        if base_evidence_dir:
            evidence_analysis = analysis_path_for_row(base_evidence_dir, row_id)
            if evidence_analysis.exists():
                command += ["--base-evidence-analysis", str(evidence_analysis)]
        for extra_dir in extra_external_dirs:
            extra_external = output_path_for_row(extra_dir, row_id)
            if extra_external.exists():
                command += [
                    "--external-chords-extra",
                    str(extra_external),
                    "--external-chords-extra-weight",
                    f"{args.external_chords_extra_weight:.6f}",
                ]
        try:
            subprocess.run(command, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            failures += 1
            print(f"FAILED {row_id}: {exc}", file=sys.stderr)
    print(f"done: {len(selected) - failures} ok, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
