#!/usr/bin/env python
"""Add key-relative scale-degree chord candidates to candidate tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from build_candidate_training_set import chord_family, chord_root, row_for_candidate
from chord_studio import chord_to_roman, chord_to_roman_display, note_name_for_pc, parse_chord_name
from classify_boundary_segments import roman_root


REPO_ROOT = Path(__file__).resolve().parent.parent
MAJOR_ROMAN_TO_PC = {
    "I": 0,
    "IIm": 2,
    "IIIm": 4,
    "IV": 5,
    "V": 7,
    "VIm": 9,
    "VIIdim": 11,
    "bVII": 10,
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def parse_reference(row: dict[str, str]) -> dict[str, str]:
    return {
        "start": row.get("reference_start") or row.get("start") or "0",
        "end": row.get("reference_end") or row.get("end") or "0",
        "chord": row.get("reference_chord") or "N",
        "chord_display": row.get("reference_chord") or "N",
        "roman": row.get("reference_roman") or "N",
        "roman_display": row.get("reference_roman") or "N",
        "local_key": row.get("local_key") or "",
        "label_strength": row.get("reference_label_strength") or "",
        "root_locked": row.get("reference_root_locked") or "0",
    }


def chord_for_roman(key: str, roman: str) -> str | None:
    root, _ = parse_chord_name(key)
    if root is None:
        return None
    offset = MAJOR_ROMAN_TO_PC.get(roman)
    if offset is None:
        return None
    suffix = ""
    degree = roman
    if degree.endswith("dim"):
        suffix = "dim"
        degree = degree[:-3]
    elif degree.endswith("m"):
        suffix = "m"
        degree = degree[:-1]
    root_name = note_name_for_pc((root + offset) % 12, key)
    return root_name + suffix


def add_candidate(base: dict[str, str], source: str, roman: str, rank: int, score: float) -> dict[str, str] | None:
    key = base.get("local_key") or ""
    chord = chord_for_roman(key, roman)
    if chord is None:
        return None
    candidate = {
        "bar": base.get("bar", ""),
        "start": base.get("reference_start") or base.get("start") or "0",
        "end": base.get("reference_end") or base.get("end") or "0",
        "chord": chord,
        "chord_display": chord,
        "roman": chord_to_roman(chord, key),
        "roman_display": chord_to_roman_display(chord, key),
        "local_key": key,
        "confidence": f"{score:.6f}",
        "unit": base.get("unit") or "scale_degree",
        "candidate_rank": str(rank),
    }
    row = row_for_candidate(
        song_id=base.get("song_id", ""),
        candidate_source=f"{source}_rank{rank}",
        candidate=candidate,
        reference=parse_reference(base),
        overlap_seconds=float(base.get("overlap_seconds") or 0.0),
    )
    row["candidate_rank"] = str(rank)
    row["candidate_score"] = f"{score:.6f}"
    return row


def group_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("song_id", ""),
        row.get("bar", ""),
        row.get("reference_start") or row.get("start", ""),
        row.get("reference_end") or row.get("end", ""),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--romans", default="I,IIm,IIIm,IV,V,VIm,bVII")
    parser.add_argument("--source-name", default="scale_degree")
    parser.add_argument("--score", type=float, default=0.22)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])

    romans = [item.strip() for item in args.romans.split(",") if item.strip()]
    existing = {
        (group_key(row), row.get("candidate_chord", ""), row.get("candidate_roman", ""))
        for row in rows
    }
    additions: list[dict[str, str]] = []
    first_by_group: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        first_by_group.setdefault(group_key(row), row)
    for base in first_by_group.values():
        for rank, roman in enumerate(romans, start=1):
            candidate = add_candidate(base, args.source_name, roman, rank, args.score)
            if candidate is None:
                continue
            key = (group_key(candidate), candidate.get("candidate_chord", ""), candidate.get("candidate_roman", ""))
            if key in existing:
                continue
            existing.add(key)
            additions.append(candidate)

    for row in additions:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows + additions:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    print(f"Wrote {len(rows) + len(additions)} row(s), added {len(additions)}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
