#!/usr/bin/env python
"""Add key-relative half-diminished seventh candidates.

This targets color chords such as D#m7b5 in G major.  They are uncommon enough
that generic top-N acoustic candidates often omit them, but common enough in
arranged pop/rock passing harmony that the reranker should be allowed to see
them.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from build_candidate_training_set import chord_family, chord_root
from chord_studio import chord_to_roman, note_name_for_pc, parse_chord_name
from train_candidate_reranker import resolve_path


MAJOR_ROMAN_TO_PC = {
    "I": 0,
    "bII": 1,
    "II": 2,
    "bIII": 3,
    "III": 4,
    "IV": 5,
    "#IV": 6,
    "V": 7,
    "#V": 8,
    "bVI": 8,
    "VI": 9,
    "bVII": 10,
    "VII": 11,
}


def group_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("song_id", ""),
        row.get("bar", ""),
        row.get("reference_start") or row.get("start", ""),
        row.get("reference_end") or row.get("end", ""),
    )


def roman_root(value: str) -> str:
    text = str(value or "")
    for suffix in ("maj7", "m7b5", "m7", "dim7", "7", "m", "dim", "aug", "sus4", "sus2"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def chord_for_roman_m7b5(key: str, roman: str) -> str | None:
    key_root, _ = parse_chord_name(key or "C")
    if key_root is None:
        return None
    root = roman_root(roman)
    offset = MAJOR_ROMAN_TO_PC.get(root)
    if offset is None:
        return None
    return f"{note_name_for_pc((key_root + offset) % 12, key)}m7b5"


def reference_fields(row: dict[str, str]) -> None:
    row["target_exact_chord"] = "1" if row.get("candidate_chord") == row.get("reference_chord") else "0"
    row["target_same_root"] = "1" if row.get("candidate_chord_root") and row.get("candidate_chord_root") == row.get("reference_chord_root") else "0"
    row["target_same_roman_root"] = "1" if row.get("candidate_roman_root") and row.get("candidate_roman_root") == row.get("reference_roman_root") else "0"
    row["target_same_chord_family"] = "1" if row.get("candidate_chord_family") == row.get("reference_chord_family") else "0"


def add_candidate(base: dict[str, str], source_name: str, roman: str, rank: int, score: float, confidence: float) -> dict[str, str] | None:
    key = str(base.get("local_key") or "").strip()
    if not key:
        return None
    chord = chord_for_roman_m7b5(key, roman)
    if not chord:
        return None
    root, _ = parse_chord_name(chord)
    row = dict(base)
    row["candidate_source"] = f"{source_name}_rank{rank}"
    row["candidate_chord"] = chord
    row["candidate_roman"] = chord_to_roman(chord, key)
    row["candidate_roman_root"] = roman_root(row["candidate_roman"])
    row["candidate_chord_root"] = "" if root is None else note_name_for_pc(root, key)
    row["candidate_chord_family"] = chord_family(chord)
    row["candidate_confidence"] = f"{confidence:.6f}"
    row["candidate_rank"] = str(rank)
    row["candidate_score"] = f"{score:.6f}"
    row["unit"] = base.get("unit") or "half_dim"
    reference_fields(row)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--romans", default="#Vm7b5,bVIm7b5,VIIm7b5,#IVm7b5,IIm7b5")
    parser.add_argument("--source-name", default="half_dim")
    parser.add_argument("--score", type=float, default=0.08)
    parser.add_argument("--confidence", type=float, default=0.22)
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.candidate_csv)
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    grouped: dict[tuple[str, str, str, str], dict[str, str]] = {}
    existing = {(group_key(row), row.get("candidate_chord", ""), row.get("candidate_roman", "")) for row in rows}
    for row in rows:
        grouped.setdefault(group_key(row), row)

    romans = [item.strip() for item in args.romans.split(",") if item.strip()]
    additions: list[dict[str, str]] = []
    for base in grouped.values():
        for rank, roman in enumerate(romans, start=1):
            row = add_candidate(base, args.source_name, roman, rank, args.score, args.confidence)
            if row is None:
                continue
            key = (group_key(row), row.get("candidate_chord", ""), row.get("candidate_roman", ""))
            if key in existing:
                continue
            existing.add(key)
            additions.append(row)

    output_rows = rows + additions
    for row in output_rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
        for field in fieldnames:
            row.setdefault(field, "")

    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} row(s), added {len(additions)} half-diminished candidate row(s): {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
