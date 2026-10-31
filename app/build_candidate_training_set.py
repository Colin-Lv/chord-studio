#!/usr/bin/env python
"""Build candidate-ranking rows from corrected and predicted analysis JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from chord_studio import chord_quality_family, parse_chord_name
from classify_boundary_segments import roman_root


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_spans(payload: dict[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for bar in payload.get("bars", []):
        for span in bar.get("spans") or []:
            item = dict(span)
            item["bar"] = int(bar.get("bar", item.get("bar", 0)))
            spans.append(item)
    return sorted(spans, key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))


def label_of(span: dict[str, Any], field: str) -> str:
    if field == "roman":
        return str(span.get("roman_display") or span.get("roman") or "N")
    return str(span.get("chord_display") or span.get("chord") or "N")


def overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    return max(
        0.0,
        min(float(left.get("end", 0.0)), float(right.get("end", 0.0)))
        - max(float(left.get("start", 0.0)), float(right.get("start", 0.0))),
    )


def best_reference_span(candidate: dict[str, Any], references: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    best = None
    best_overlap = 0.0
    for reference in references:
        amount = overlap(candidate, reference)
        if amount > best_overlap:
            best = reference
            best_overlap = amount
    return best, best_overlap


def chord_root(chord: str) -> str:
    root, _ = parse_chord_name(chord)
    return "" if root is None else str(root)


def chord_family(chord: str) -> str:
    _, suffix = parse_chord_name(chord)
    return chord_quality_family(suffix)


def row_for_candidate(
    song_id: str,
    candidate_source: str,
    candidate: dict[str, Any],
    reference: dict[str, Any] | None,
    overlap_seconds: float,
) -> dict[str, Any]:
    candidate_chord = label_of(candidate, "chord")
    candidate_roman = label_of(candidate, "roman")
    reference_chord = label_of(reference or {}, "chord")
    reference_roman = label_of(reference or {}, "roman")
    reference_label_strength = str((reference or {}).get("label_strength") or "")
    reference_root_locked = bool((reference or {}).get("root_locked"))
    root_only_reference = reference_label_strength == "root_only" or reference_root_locked
    same_root = int(chord_root(candidate_chord) == chord_root(reference_chord) and reference_chord != "N")
    same_roman_root = int(roman_root(candidate_roman) == roman_root(reference_roman) and reference_roman != "N")
    if root_only_reference:
        same_roman_root = same_root
    duration = max(float(candidate.get("end", 0.0)) - float(candidate.get("start", 0.0)), 0.0)
    return {
        "song_id": song_id,
        "candidate_source": candidate_source,
        "bar": candidate.get("bar", ""),
        "start": f"{float(candidate.get('start', 0.0)):.6f}",
        "end": f"{float(candidate.get('end', 0.0)):.6f}",
        "duration": f"{duration:.6f}",
        "position": candidate.get("position", ""),
        "unit": candidate.get("unit", ""),
        "local_key": candidate.get("local_key", ""),
        "candidate_chord": candidate_chord,
        "candidate_roman": candidate_roman,
        "candidate_roman_root": roman_root(candidate_roman),
        "candidate_chord_root": chord_root(candidate_chord),
        "candidate_chord_family": chord_family(candidate_chord),
        "candidate_confidence": candidate.get("confidence", ""),
        "reference_chord": reference_chord,
        "reference_roman": reference_roman,
        "reference_roman_root": roman_root(reference_roman),
        "reference_chord_root": chord_root(reference_chord),
        "reference_chord_family": chord_family(reference_chord),
        "reference_label_strength": reference_label_strength,
        "reference_root_locked": int(reference_root_locked),
        "reference_start": "" if reference is None else f"{float(reference.get('start', 0.0)):.6f}",
        "reference_end": "" if reference is None else f"{float(reference.get('end', 0.0)):.6f}",
        "overlap_seconds": f"{overlap_seconds:.6f}",
        "target_exact_chord": int((not root_only_reference) and candidate_chord == reference_chord),
        "target_same_root": same_root,
        "target_same_roman_root": same_roman_root,
    }


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), resolve_path(Path(path.strip()))
    path = resolve_path(Path(value))
    return path.stem.replace("_analysis", ""), path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate analysis JSON. Use name=path to label the source.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "reports" / "candidate_training_rows.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference_path = resolve_path(args.reference)
    reference_spans = iter_spans(load_json(reference_path))
    rows: list[dict[str, Any]] = []
    for candidate_value in args.candidate:
        source_name, candidate_path = parse_candidate(candidate_value)
        candidate_spans = iter_spans(load_json(candidate_path))
        for candidate in candidate_spans:
            reference, amount = best_reference_span(candidate, reference_spans)
            rows.append(row_for_candidate(args.song_id, source_name, candidate, reference, amount))

    args.output = resolve_path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "song_id",
        "candidate_source",
        "bar",
        "start",
        "end",
        "duration",
        "position",
        "unit",
        "local_key",
        "candidate_chord",
        "candidate_roman",
        "candidate_roman_root",
        "candidate_chord_root",
        "candidate_chord_family",
        "candidate_confidence",
        "reference_chord",
        "reference_roman",
        "reference_roman_root",
        "reference_chord_root",
        "reference_chord_family",
        "reference_label_strength",
        "reference_root_locked",
        "reference_start",
        "reference_end",
        "overlap_seconds",
        "target_exact_chord",
        "target_same_root",
        "target_same_roman_root",
    ]
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} candidate row(s): {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
