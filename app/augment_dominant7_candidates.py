#!/usr/bin/env python
"""Augment candidate rows with same-root dominant seventh alternatives."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from build_candidate_training_set import chord_family, chord_root
from chord_studio import chord_to_roman, chord_to_roman_display, note_name_for_pc, parse_chord_name
from classify_boundary_segments import roman_root


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def is_plain_major(chord: str) -> bool:
    if not chord or chord == "N" or "/" in chord:
        return False
    root, suffix = parse_chord_name(chord)
    return root is not None and suffix == ""


def dominant7_name(chord: str) -> str:
    root, _ = parse_chord_name(chord)
    return f"{note_name_for_pc(root)}7" if root is not None else chord


def augment_row(row: dict[str, str], source_suffix: str, score_penalty: float) -> dict[str, str] | None:
    chord = row.get("candidate_chord", "")
    if not is_plain_major(chord):
        return None
    new_chord = dominant7_name(chord)
    if new_chord == chord:
        return None
    key = row.get("local_key", "")
    new_row = dict(row)
    new_row["candidate_source"] = f"{row.get('candidate_source', 'candidate')}_{source_suffix}"
    new_row["candidate_chord"] = new_chord
    if key:
        new_row["candidate_roman"] = chord_to_roman_display(new_chord, key) or chord_to_roman(new_chord, key)
    else:
        new_row["candidate_roman"] = new_chord
    new_row["candidate_roman_root"] = roman_root(new_row["candidate_roman"])
    new_row["candidate_chord_root"] = chord_root(new_chord)
    new_row["candidate_chord_family"] = chord_family(new_chord)
    if row.get("candidate_score", "") != "":
        try:
            new_row["candidate_score"] = f"{float(row['candidate_score']) - score_penalty:.6f}"
        except ValueError:
            pass
    if row.get("candidate_confidence", "") != "":
        try:
            new_row["candidate_confidence"] = f"{max(0.0, float(row['candidate_confidence']) * 0.88):.6f}"
        except ValueError:
            pass
    new_row["target_exact_chord"] = int(new_row["candidate_chord"] == row.get("reference_chord", ""))
    new_row["target_same_root"] = int(
        chord_root(new_row["candidate_chord"]) == row.get("reference_chord_root", "")
        and row.get("reference_chord", "") != "N"
    )
    new_row["target_same_roman_root"] = int(
        roman_root(new_row["candidate_roman"]) == row.get("reference_roman_root", "")
        and row.get("reference_roman", "") != "N"
    )
    return new_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--source-suffix", default="dom7")
    parser.add_argument("--score-penalty", type=float, default=0.035)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = resolve_path(args.input_csv)
    output = resolve_path(args.output)
    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    augmented: list[dict[str, str]] = []
    seen = {
        (
            row.get("song_id", ""),
            row.get("candidate_source", ""),
            row.get("bar", ""),
            row.get("reference_start", ""),
            row.get("reference_end", ""),
            row.get("candidate_chord", ""),
        )
        for row in rows
    }
    for row in rows:
        new_row = augment_row(row, args.source_suffix, args.score_penalty)
        if new_row is None:
            continue
        key = (
            new_row.get("song_id", ""),
            new_row.get("candidate_source", ""),
            new_row.get("bar", ""),
            new_row.get("reference_start", ""),
            new_row.get("reference_end", ""),
            new_row.get("candidate_chord", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        augmented.append(new_row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerows(augmented)
    print(f"Wrote {len(rows) + len(augmented)} row(s), added {len(augmented)} dominant7 candidate(s): {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
