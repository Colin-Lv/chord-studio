#!/usr/bin/env python
"""Build accepted/rejected manifests from a generic version audit CSV."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from pathlib import Path


ACCEPTABLE_STATUSES = {"strong_match", "aligned_match"}


def resolve_path(base: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
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
    rows = []
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
        if end > start:
            rows.append(f"{start:.6f} {end:.6f} {label}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def shift_beats(source: Path, target: Path, shift: float) -> None:
    rows = []
    for line in source.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            time = float(parts[0]) + shift
        except ValueError:
            continue
        if time < 0:
            continue
        suffix = f" {parts[1]}" if len(parts) > 1 else ""
        rows.append(f"{time:.6f}{suffix}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def safe_id(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', value).strip('. ') or 'unnamed'


def manifest_path(path: Path, base: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(base.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("version_audit", type=Path)
    parser.add_argument("--output-accepted", type=Path, required=True)
    parser.add_argument("--output-rejected", type=Path, required=True)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--min-triad", type=float, default=0.50)
    parser.add_argument("--accept-status", action="append", default=None)
    parser.add_argument("--include-caution", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    base = manifest.parent
    aligned_dir = args.aligned_dir.resolve()
    accepted_statuses = set(args.accept_status or ACCEPTABLE_STATUSES)
    if args.include_caution:
        accepted_statuses.add("usable_with_caution")

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    with args.version_audit.open("r", encoding="utf-8-sig", newline="") as handle:
        audit_by_id = {row.get("id", ""): row for row in csv.DictReader(handle)}
    for field in ("alignment_shift", "alignment_best_triad", "alignment_status", "alignment_reason",
                  "alignment_method", "original_chord_reference", "original_beat_reference"):
        if field not in fieldnames:
            fieldnames.append(field)

    accepted = []
    rejected = []
    for row in rows:
        row = dict(row)
        row_id = row.get("id", "")
        audit = audit_by_id.get(row_id, {})
        best_triad = float(audit.get("best_triad") or 0.0)
        status = audit.get("status") or "missing_audit"
        shift = float(audit.get("best_time_shift") or 0.0)
        if not math.isfinite(shift) or not math.isfinite(best_triad):
            raise ValueError(f"Nonfinite alignment values for {row_id}")
        # scan_manifest_audio_versions evaluates shifted_reference(ref, shift).
        # Apply exactly that sign; negating it moves labels away from the audio.
        reference_shift = shift
        row["alignment_shift"] = f"{reference_shift:.6f}"
        row["alignment_best_triad"] = f"{best_triad:.6f}"
        row["alignment_status"] = status
        row["alignment_reason"] = audit.get("reason", "")
        row["alignment_method"] = "prediction_optimized_time_shift"
        row["original_chord_reference"] = row.get("chord_reference", "")
        row["original_beat_reference"] = row.get("beat_reference", "")
        if status not in accepted_statuses or best_triad < args.min_triad:
            rejected.append(row)
            continue

        target_dir = aligned_dir / safe_id(row_id)
        chord_source = resolve_path(base, row.get("chord_reference", ""))
        beat_source = resolve_path(base, row.get("beat_reference", ""))
        if chord_source and chord_source.exists():
            chord_target = target_dir / "chords.lab"
            if abs(reference_shift) < 1e-9:
                chord_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(chord_source, chord_target)
            else:
                shift_lab(chord_source, chord_target, reference_shift)
            row["chord_reference"] = manifest_path(chord_target, base)
        if beat_source and beat_source.exists():
            beat_target = target_dir / "beats.beats"
            if abs(reference_shift) < 1e-9:
                beat_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(beat_source, beat_target)
            else:
                shift_beats(beat_source, beat_target, reference_shift)
            row["beat_reference"] = manifest_path(beat_target, base)
        row["prediction"] = f"predictions/{safe_id(row_id)}_aligned_analysis.json"
        accepted.append(row)

    args.output_accepted.parent.mkdir(parents=True, exist_ok=True)
    args.output_rejected.parent.mkdir(parents=True, exist_ok=True)
    # Input paths are relative to the input manifest; each output manifest may
    # live in a different directory. Rebase all retained file references.
    for output_path, output_rows in ((args.output_accepted, accepted), (args.output_rejected, rejected)):
        for row in output_rows:
            for field in ("audio", "prediction", "chord_reference", "beat_reference",
                          "original_chord_reference", "original_beat_reference"):
                path = resolve_path(base, row.get(field, ""))
                if path is not None:
                    row[field] = manifest_path(path, output_path.parent)
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
    print("Alignment is optimized against model predictions; scores using these labels are diagnostic, not raw held-out accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
