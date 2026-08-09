#!/usr/bin/env python3
"""Build accepted/rejected manifests using time-shift scan results."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def parse_timed_label(line: str) -> tuple[float, float, str] | None:
    parts = line.strip().split()
    if len(parts) < 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), parts[2]
    except ValueError:
        return None


def shift_lab(source: Path, target: Path, shift: float) -> None:
    rows: list[str] = []
    for line in source.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        parsed = parse_timed_label(line)
        if parsed is None:
            continue
        start, end, label = parsed
        start += shift
        end += shift
        if end <= 0:
            continue
        start = max(0.0, start)
        if end <= start:
            continue
        rows.append(f"{start:.6f} {end:.6f} {label}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def shift_beats(source: Path, target: Path, shift: float) -> None:
    rows: list[str] = []
    for line in source.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 1:
            continue
        try:
            time = float(parts[0]) + shift
        except ValueError:
            continue
        if time < 0:
            continue
        beat = parts[1] if len(parts) > 1 else "1"
        rows.append(f"{time:.6f} {beat}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("shift_scan", type=Path)
    parser.add_argument("--output-accepted", type=Path, required=True)
    parser.add_argument("--output-rejected", type=Path, required=True)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--min-triad", type=float, default=0.50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    base = manifest.parent
    aligned_dir = args.aligned_dir.resolve()

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    scan_by_id = {}
    with args.shift_scan.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            scan_by_id[row.get("id", "")] = row

    for field in ("alignment_shift", "alignment_best_triad", "alignment_status", "alignment_reason"):
        if field not in fieldnames:
            fieldnames.append(field)

    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    for row in rows:
        row = dict(row)
        row_id = row.get("id", "")
        scan = scan_by_id.get(row_id, {})
        best_triad = float(scan.get("best_triad") or 0.0)
        best_shift = float(scan.get("best_time_shift") or 0.0)
        reference_shift = -best_shift
        row["alignment_shift"] = f"{reference_shift:.6f}"
        row["alignment_best_triad"] = f"{best_triad:.6f}"

        if best_triad < args.min_triad:
            row["alignment_status"] = "rejected"
            row["alignment_reason"] = f"best_triad_below_{args.min_triad:.2f}"
            rejected.append(row)
            continue

        safe_id = row_id.replace("/", "_").replace("\\", "_")
        target_dir = aligned_dir / safe_id
        chord_source = resolve_path(base, row.get("chord_reference", ""))
        beat_source = resolve_path(base, row.get("beat_reference", ""))
        chord_target = target_dir / "chords.lab"
        beat_target = target_dir / "beats.beats"
        if abs(reference_shift) < 1e-9:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(chord_source, chord_target)
            shutil.copyfile(beat_source, beat_target)
        else:
            shift_lab(chord_source, chord_target, reference_shift)
            shift_beats(beat_source, beat_target, reference_shift)

        row["chord_reference"] = str(chord_target.relative_to(base))
        row["beat_reference"] = str(beat_target.relative_to(base))
        row["prediction"] = f"predictions/{safe_id}_aligned_analysis.json"
        row["alignment_status"] = "accepted"
        row["alignment_reason"] = "ok"
        accepted.append(row)

    args.output_accepted.parent.mkdir(parents=True, exist_ok=True)
    args.output_rejected.parent.mkdir(parents=True, exist_ok=True)
    with args.output_accepted.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(accepted)
    with args.output_rejected.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rejected)

    print(f"accepted: {len(accepted)}")
    print(f"rejected: {len(rejected)}")
    print(f"wrote: {args.output_accepted}")
    print(f"wrote: {args.output_rejected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
