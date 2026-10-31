#!/usr/bin/env python
"""Add same-root 9sus4 candidates from nearby dominant/add9/sus evidence.

Large-vocabulary ACR often proposes partial interpretations such as G9,
Gadd9, Gsus4, G7, or G for a true G9sus4.  This script keeps those acoustic
signals but makes the combined dominant-sus9 interpretation available to the
reranker.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from build_candidate_training_set import chord_family, chord_root
from chord_studio import chord_to_roman, chord_to_roman_display, note_name_for_pc, parse_chord_name
from classify_boundary_segments import roman_root
from train_candidate_reranker import resolve_path


def group_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("song_id", ""),
        row.get("bar", ""),
        row.get("reference_start") or row.get("start", ""),
        row.get("reference_end") or row.get("end", ""),
    )


def is_supporting_suffix(suffix: str) -> bool:
    return suffix in {"", "7", "9", "add9", "sus4", "7sus4", "9sus4"}


def support_score(row: dict[str, str]) -> float:
    try:
        return float(row.get("candidate_score") or 0.0)
    except ValueError:
        return 0.0


def support_confidence(row: dict[str, str]) -> float:
    try:
        return float(row.get("candidate_confidence") or 0.0)
    except ValueError:
        return 0.0


def source_rank(source: str) -> int:
    text = source.lower()
    if "topn_rank1" in text:
        return 0
    if "topn" in text:
        return 1
    if "current" in text:
        return 2
    if "functional" in text:
        return 4
    return 3


def sus9_name(root_pc: int, key: str) -> str:
    return f"{note_name_for_pc(root_pc, key or None)}9sus4"


def reference_fields(row: dict[str, str]) -> None:
    row["target_exact_chord"] = "1" if row.get("candidate_chord") == row.get("reference_chord") else "0"
    row["target_same_root"] = "1" if row.get("candidate_chord_root") and row.get("candidate_chord_root") == row.get("reference_chord_root") else "0"
    row["target_same_roman_root"] = "1" if row.get("candidate_roman_root") and row.get("candidate_roman_root") == row.get("reference_roman_root") else "0"
    row["target_same_chord_family"] = "1" if row.get("candidate_chord_family") == row.get("reference_chord_family") else "0"


def add_candidate(base: dict[str, str], root_pc: int, source_name: str, rank: int, score_penalty: float) -> dict[str, str]:
    key = str(base.get("local_key") or "").strip()
    chord = sus9_name(root_pc, key)
    row = dict(base)
    row["candidate_source"] = f"{source_name}_rank{rank}"
    row["candidate_chord"] = chord
    if key:
        row["candidate_roman"] = chord_to_roman_display(chord, key) or chord_to_roman(chord, key)
    else:
        row["candidate_roman"] = chord
    row["candidate_roman_root"] = roman_root(row["candidate_roman"])
    row["candidate_chord_root"] = chord_root(chord)
    row["candidate_chord_family"] = chord_family(chord)
    row["candidate_rank"] = str(rank)
    row["candidate_score"] = f"{max(0.0, support_score(base) - score_penalty):.6f}"
    row["candidate_confidence"] = f"{max(0.0, support_confidence(base) * 0.86):.6f}"
    row["audio_candidate_has_7"] = "1"
    row["audio_candidate_has_9"] = "1"
    row["audio_candidate_has_sus4"] = "1"
    row["audio_candidate_has_maj7"] = "0"
    row["audio_candidate_has_m7b5"] = "0"
    row["unit"] = base.get("unit") or "sus9"
    reference_fields(row)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--source-name", default="sus9")
    parser.add_argument("--score-penalty", type=float, default=0.045)
    parser.add_argument(
        "--all-dominants",
        action="store_true",
        help="Also add the key-relative V9sus4 even if no same-root support exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.candidate_csv)
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    existing = {(group_key(row), row.get("candidate_chord", ""), row.get("candidate_roman", "")) for row in rows}
    for row in rows:
        grouped.setdefault(group_key(row), []).append(row)

    additions: list[dict[str, str]] = []
    for group_rows in grouped.values():
        roots: dict[int, list[dict[str, str]]] = {}
        for row in group_rows:
            root, suffix = parse_chord_name(row.get("candidate_chord", ""))
            if root is None or not is_supporting_suffix(suffix):
                continue
            roots.setdefault(root, []).append(row)

        if args.all_dominants:
            key_root, _ = parse_chord_name(group_rows[0].get("local_key", ""))
            if key_root is not None:
                roots.setdefault((key_root + 7) % 12, [group_rows[0]])

        ranked_roots = sorted(
            roots.items(),
            key=lambda item: (
                source_rank(min((row.get("candidate_source", "") for row in item[1]), default="")),
                -max(support_score(row) for row in item[1]),
            ),
        )
        for rank, (root_pc, support_rows) in enumerate(ranked_roots, start=1):
            base = max(support_rows, key=lambda row: (support_score(row), support_confidence(row)))
            row = add_candidate(base, root_pc, args.source_name, rank, args.score_penalty)
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
    print(f"Wrote {len(output_rows)} row(s), added {len(additions)} 9sus4 candidate row(s): {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
