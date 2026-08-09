#!/usr/bin/env python3
"""Generate Chord Studio predictions for rows in a benchmark manifest."""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path


ANALYSIS_SUFFIX = "_analysis.json"


def resolve_path(base: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else base / path


def output_base_from_prediction(prediction: Path) -> Path:
    name = prediction.name
    if name.endswith(ANALYSIS_SUFFIX):
        return prediction.with_name(name[: -len(ANALYSIS_SUFFIX)])
    if prediction.suffix.lower() == ".json":
        return prediction.with_suffix("")
    return prediction


def external_lab_for_row(directory: Path, row_id: str) -> Path:
    safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in row_id)
    return directory / f"{safe_id}.lab"


def has_audio(row: dict[str, str], base: Path) -> bool:
    audio = resolve_path(base, row.get("audio", ""))
    return bool(audio and audio.exists())


def parse_seconds(value: object) -> float:
    text = str(value).strip()
    if ":" not in text:
        return float(text)
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60.0 + float(part)
    return seconds


def infer_tempo_meter_from_beats(path: Path) -> tuple[float | None, str | None]:
    times: list[float] = []
    labels: list[int] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if not parts:
            continue
        times.append(parse_seconds(parts[0]))
        if len(parts) > 1:
            labels.append(int(float(parts[1])))
    diffs = [
        right - left
        for left, right in zip(times, times[1:])
        if 0.18 <= right - left <= 1.8
    ]
    bpm = 60.0 / statistics.median(diffs) if diffs else None
    meter = None
    if labels:
        beats_per_bar = max(labels)
        if beats_per_bar in {2, 3, 4, 5, 6, 7, 8, 9, 12}:
            beat_unit = 8 if beats_per_bar in {6, 9, 12} else 4
            meter = f"{beats_per_bar}/{beat_unit}"
    return bpm, meter


def build_command(
    python_exe: str,
    studio_script: Path,
    row: dict[str, str],
    base: Path,
    beat_source: str,
    analysis_model: str,
    chord_mode: str,
    harmonic_unit: str,
    feature_mode: str,
    use_reference_key: bool,
    use_reference_meter: bool,
    bpm_from_reference_beats: bool,
    default_meter: str | None,
    use_reference_beat_grid: bool,
    bass_weight: float | None,
    transition_strength: float | None,
    external_chords_dir: Path | None,
    external_weight: float | None,
    no_html: bool,
) -> list[str]:
    audio = resolve_path(base, row.get("audio", ""))
    prediction = resolve_path(base, row.get("prediction", ""))
    if audio is None or not audio.exists():
        raise ValueError(f"audio not found: {row.get('audio', '')}")
    if prediction is None:
        raise ValueError("prediction path is empty")

    output_base = output_base_from_prediction(prediction)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python_exe,
        str(studio_script),
        str(audio),
        "-o",
        str(output_base),
        "--analysis-model",
        analysis_model,
        "--mode",
        chord_mode,
        "--beat-source",
        beat_source,
        "--harmonic-unit",
        harmonic_unit,
        "--feature-mode",
        feature_mode,
        "--display",
        "name",
    ]
    if use_reference_key and row.get("key", "").strip():
        command += ["--key", row["key"].strip()]
    if row.get("audio_offset", "").strip():
        command += ["--offset", row["audio_offset"].strip()]
    if bass_weight is not None:
        command += ["--bass-weight", f"{bass_weight:.6f}"]
    if transition_strength is not None:
        command += ["--transition-strength", f"{transition_strength:.6f}"]
    if external_chords_dir is not None:
        external = external_lab_for_row(external_chords_dir, row.get("id", ""))
        if external.exists():
            command += ["--external-chords", str(external)]
            if external_weight is not None:
                command += ["--external-weight", f"{external_weight:.6f}"]
    if use_reference_beat_grid:
        beat_reference = resolve_path(base, row.get("beat_reference", ""))
        if beat_reference and beat_reference.exists():
            command += ["--beat-grid", str(beat_reference)]
    ref_bpm = None
    ref_meter = None
    if bpm_from_reference_beats:
        beat_reference = resolve_path(base, row.get("beat_reference", ""))
        if beat_reference and beat_reference.exists():
            ref_bpm, ref_meter = infer_tempo_meter_from_beats(beat_reference)
    if ref_bpm is not None:
        command += ["--bpm", f"{ref_bpm:.6f}"]
    if use_reference_meter and row.get("meter", "").strip():
        command += ["--meter", row["meter"].strip()]
    elif default_meter:
        command += ["--meter", default_meter]
    elif use_reference_meter and ref_meter is not None:
        command += ["--meter", ref_meter]
    if no_html:
        command.append("--no-html")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Benchmark manifest CSV.")
    parser.add_argument("--dataset", help="Only generate rows from this dataset.")
    parser.add_argument("--id-contains", help="Only generate rows whose id contains this text.")
    parser.add_argument("--limit", type=int, help="Maximum rows to generate.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing predictions.")
    parser.add_argument("--beat-source", choices=["librosa", "madmom"], default="librosa")
    parser.add_argument("--analysis-model", choices=["functional", "template"], default="functional")
    parser.add_argument("--mode", choices=["triads", "seventh", "rich"], default="triads")
    parser.add_argument(
        "--harmonic-unit",
        choices=["auto", "bar", "half-bar", "compound-half", "beat"],
        default="auto",
    )
    parser.add_argument("--feature-mode", choices=["cqt", "fast"], default="cqt")
    parser.add_argument("--bass-weight", type=float)
    parser.add_argument("--transition-strength", type=float)
    parser.add_argument("--external-chords-dir", type=Path)
    parser.add_argument("--external-weight", type=float)
    parser.add_argument("--use-reference-key", action="store_true")
    parser.add_argument("--use-reference-meter", action="store_true")
    parser.add_argument("--default-meter", help="Meter to use when the manifest row has no meter, e.g. 4/4.")
    parser.add_argument(
        "--bpm-from-reference-beats",
        action="store_true",
        help="Use reference beat intervals only to seed BPM/meter for faster controlled runs.",
    )
    parser.add_argument(
        "--use-reference-beat-grid",
        action="store_true",
        help="Use the manifest beat_reference as the exact beat grid for diagnostic upper-bound runs.",
    )
    parser.add_argument("--with-html", action="store_true", help="Also write the interactive HTML pages.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for chord_studio.py.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    base = manifest.parent
    studio_script = (Path(__file__).resolve().parent / "chord_studio.py").resolve()

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected: list[dict[str, str]] = []
    for row in rows:
        if args.dataset and row.get("dataset") != args.dataset:
            continue
        if args.id_contains and args.id_contains not in row.get("id", ""):
            continue
        prediction = resolve_path(base, row.get("prediction", ""))
        if prediction and prediction.exists() and not args.force:
            continue
        if not has_audio(row, base):
            continue
        selected.append(row)
        if args.limit and len(selected) >= args.limit:
            break

    print(f"selected rows: {len(selected)}")
    failures = 0
    for index, row in enumerate(selected, start=1):
        row_id = row.get("id", f"row-{index}")
        print(f"[{index}/{len(selected)}] {row_id}")
        try:
            command = build_command(
                args.python,
                studio_script,
                row,
                base,
                args.beat_source,
                args.analysis_model,
                args.mode,
                args.harmonic_unit,
                args.feature_mode,
                args.use_reference_key,
                args.use_reference_meter,
                args.bpm_from_reference_beats,
                args.default_meter,
                args.use_reference_beat_grid,
                args.bass_weight,
                args.transition_strength,
                args.external_chords_dir.resolve() if args.external_chords_dir else None,
                args.external_weight,
                no_html=not args.with_html,
            )
            subprocess.run(command, check=True)
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            failures += 1
            print(f"FAILED {row_id}: {exc}", file=sys.stderr)

    print(f"done: {len(selected) - failures} ok, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
