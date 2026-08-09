#!/usr/bin/env python3
"""Write segment-level chord error diagnostics for benchmark predictions."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from chord_studio import (
    chord_to_roman_display,
    parse_chord_name,
    read_external_chord_hints,
)
from evaluate_chords import TimedChord, convert_annotation_chord, read_prediction, read_reference
from generate_madmom_chords_from_manifest import output_path_for_row


@dataclass(frozen=True)
class Source:
    name: str
    rows: list[TimedChord]


def resolve_path(base: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else base / path


def external_rows(path: Path) -> list[TimedChord]:
    return [TimedChord(item.start, item.end, item.chord) for item in read_external_chord_hints(path, "sharps")]


def active_at(rows: list[TimedChord], time: float) -> str:
    for row in rows:
        if row.start <= time < row.end:
            return row.chord
    return "N"


def chord_bass(label: str) -> tuple[str, str]:
    text = convert_annotation_chord(label)
    if "/" not in text:
        return text, "N"
    chord, bass = text.split("/", 1)
    return chord, bass


def chord_family(label: str) -> tuple[int | None, str, str]:
    chord, bass = chord_bass(label)
    root, suffix = parse_chord_name(chord)
    if root is None:
        return None, "N", bass
    if suffix.startswith("m") and not suffix.startswith("maj"):
        family = "m"
    elif "dim" in suffix:
        family = "dim"
    elif "aug" in suffix:
        family = "aug"
    elif "sus" in suffix:
        family = "sus"
    else:
        family = ""
    if "7" in suffix:
        family += "7"
    return root, family, bass


def same_triad(left: str, right: str) -> bool:
    left_root, left_family, _ = chord_family(left)
    right_root, right_family, _ = chord_family(right)
    if left_root is None or right_root is None:
        return left_root == right_root
    return left_root == right_root and left_family.replace("7", "") == right_family.replace("7", "")


def classify_error(reference: str, prediction: str) -> str:
    ref = convert_annotation_chord(reference)
    pred = convert_annotation_chord(prediction)
    if ref == pred:
        return "correct"
    ref_root, ref_family, ref_bass = chord_family(ref)
    pred_root, pred_family, pred_bass = chord_family(pred)
    if ref_root is None and pred_root is None:
        return "correct"
    if ref_root is None:
        return "false_chord"
    if pred_root is None:
        return "missed_chord"
    if ref_root != pred_root:
        return "root_error"
    if ref_family != pred_family:
        if ref_family.replace("7", "") == pred_family.replace("7", ""):
            return "seventh_color_error"
        return "quality_error"
    if ref_bass != pred_bass and ref_bass != "N":
        return "slash_bass_error"
    return "spelling_or_bass_display"


def is_error(reference: str, prediction: str, level: str) -> bool:
    if level == "root":
        return chord_family(reference)[0] != chord_family(prediction)[0]
    if level == "triad":
        return not same_triad(reference, prediction)
    return convert_annotation_chord(reference) != convert_annotation_chord(prediction)


def boundary_times(*sources: list[TimedChord]) -> list[float]:
    times: set[float] = set()
    for rows in sources:
        for row in rows:
            times.add(float(row.start))
            times.add(float(row.end))
    return sorted(times)


def coalesce_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    merged = [rows[0]]
    for row in rows[1:]:
        prev = merged[-1]
        same_labels = all(
            prev.get(key) == row.get(key)
            for key in ["reference", "prediction", "error_type", "madmom", "largevocab", "chordino"]
        )
        if same_labels and abs(float(prev["end"]) - float(row["start"])) < 1e-6:
            prev["end"] = row["end"]
            prev["duration"] = float(prev["end"]) - float(prev["start"])
        else:
            merged.append(row)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--madmom-dir", type=Path)
    parser.add_argument("--largevocab-dir", type=Path)
    parser.add_argument("--chordino-dir", type=Path)
    parser.add_argument("--level", choices=["root", "triad", "exact"], default="triad")
    parser.add_argument("--min-duration", type=float, default=0.25)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    base = manifest.parent
    rows_out: list[dict[str, object]] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for index, row in enumerate(rows, start=1):
        if args.limit and index > args.limit:
            break
        row_id = row.get("id", "")
        reference_path = resolve_path(base, row.get("chord_reference", ""))
        prediction_path = resolve_path(base, row.get("prediction", ""))
        if not reference_path or not prediction_path or not reference_path.exists() or not prediction_path.exists():
            continue
        reference = read_reference(reference_path)
        prediction = read_prediction(prediction_path)
        source_rows: dict[str, list[TimedChord]] = {}
        for name, directory in [
            ("madmom", args.madmom_dir),
            ("largevocab", args.largevocab_dir),
            ("chordino", args.chordino_dir),
        ]:
            if not directory:
                continue
            path = output_path_for_row(directory.resolve(), row_id)
            if path.exists():
                source_rows[name] = external_rows(path)

        times = boundary_times(reference, prediction, *source_rows.values())
        raw_rows: list[dict[str, object]] = []
        for start, end in zip(times, times[1:]):
            if end - start < args.min_duration:
                continue
            midpoint = (start + end) / 2.0
            ref_label = active_at(reference, midpoint)
            pred_label = active_at(prediction, midpoint)
            if not is_error(ref_label, pred_label, args.level):
                continue
            source_labels = {name: active_at(items, midpoint) for name, items in source_rows.items()}
            error_type = classify_error(ref_label, pred_label)
            raw_rows.append(
                {
                    "id": row_id,
                    "artist": row.get("artist", ""),
                    "title": row.get("title", ""),
                    "key": row.get("key", ""),
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "reference": convert_annotation_chord(ref_label),
                    "reference_roman": chord_to_roman_display(convert_annotation_chord(ref_label), row.get("key", "") or "C"),
                    "prediction": convert_annotation_chord(pred_label),
                    "prediction_roman": chord_to_roman_display(convert_annotation_chord(pred_label), row.get("key", "") or "C"),
                    "error_type": error_type,
                    "madmom": source_labels.get("madmom", ""),
                    "largevocab": source_labels.get("largevocab", ""),
                    "chordino": source_labels.get("chordino", ""),
                }
            )
        rows_out.extend(coalesce_rows(raw_rows))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "artist",
        "title",
        "key",
        "start",
        "end",
        "duration",
        "reference",
        "reference_roman",
        "prediction",
        "prediction_roman",
        "error_type",
        "madmom",
        "largevocab",
        "chordino",
    ]
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    summary: dict[str, float] = {}
    for row in rows_out:
        key = str(row["error_type"])
        summary[key] = summary.get(key, 0.0) + float(row["duration"])
    print(f"Wrote: {args.output}")
    print(json.dumps(dict(sorted(summary.items(), key=lambda item: item[1], reverse=True)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
