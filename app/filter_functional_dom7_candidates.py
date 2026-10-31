#!/usr/bin/env python
"""Filter speculative functional dominant seventh candidates by audio evidence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from train_candidate_reranker import resolve_path


FEATURES_TO_CLEAR = [
    "target_exact_chord",
    "target_same_root",
    "target_same_roman_root",
    "target_same_chord_family",
]


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_functional_dom7_candidate(row: dict[str, str], source_prefix: str) -> bool:
    source = str(row.get("candidate_source") or "")
    roman = str(row.get("candidate_roman") or "")
    return source.startswith(source_prefix) and roman.endswith("7")


def keep_candidate(
    row: dict[str, str],
    min_bass_root: float,
    min_dom7_vs_maj7: float,
    min_flat7_over_triad: float,
    allow_resolution_match: bool,
) -> bool:
    bass_root = to_float(row.get("audio_bass_root_energy"))
    dom7_vs_maj7 = to_float(row.get("audio_dom7_vs_maj7"))
    flat7_over_triad = to_float(row.get("audio_flat7_over_triad"))
    resolution_match = str(row.get("harm_next_acoustic_matches_resolution") or "0") == "1"
    color_ok = dom7_vs_maj7 >= min_dom7_vs_maj7 or flat7_over_triad >= min_flat7_over_triad
    if allow_resolution_match and resolution_match and color_ok:
        return True
    return bass_root >= min_bass_root and color_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--source-prefix", default="functional_dom7_direct")
    parser.add_argument("--min-bass-root", type=float, default=0.9)
    parser.add_argument("--min-dom7-vs-maj7", type=float, default=0.03)
    parser.add_argument("--min-flat7-over-triad", type=float, default=0.03)
    parser.add_argument("--allow-resolution-match", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = resolve_path(args.input_csv)
    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    kept = []
    removed = 0
    functional_seen = 0
    for row in rows:
        if is_functional_dom7_candidate(row, args.source_prefix):
            functional_seen += 1
            if not keep_candidate(
                row,
                args.min_bass_root,
                args.min_dom7_vs_maj7,
                args.min_flat7_over_triad,
                args.allow_resolution_match,
            ):
                removed += 1
                continue
        kept.append(row)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    print(
        f"Wrote {len(kept)} row(s): kept {functional_seen - removed}/{functional_seen} "
        f"{args.source_prefix} candidate(s), removed {removed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
