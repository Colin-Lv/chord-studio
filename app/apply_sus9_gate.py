#!/usr/bin/env python
"""Apply a conservative 9sus4 candidate gate to an analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from chord_studio import chord_to_roman, chord_to_roman_display, parse_chord_name, write_html
from classify_boundary_segments import roman_root
from refine_extensions import sync_beats_to_spans, sync_events_to_spans
from train_candidate_reranker import resolve_path


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def span_key(bar: int, start: float, end: float) -> tuple[int, float, float]:
    return (int(bar), round(float(start), 3), round(float(end), 3))


def is_candidate(row: dict[str, str], allowed_romans: set[str], source_contains: str) -> bool:
    source = str(row.get("candidate_source") or "").lower()
    roman = str(row.get("candidate_roman") or "")
    if source_contains and source_contains.lower() not in source:
        return False
    if allowed_romans and roman not in allowed_romans:
        return False
    return "9sus4" in roman or "9sus4" in str(row.get("candidate_chord") or "")


def passes_gate(
    row: dict[str, str],
    min_root: float,
    min_bass_root: float,
    min_fourth: float,
    min_ninth: float,
    min_flat7: float,
    min_score: float,
) -> bool:
    if to_float(row.get("candidate_score")) < min_score:
        return False
    if to_float(row.get("audio_root_energy")) < min_root:
        return False
    if to_float(row.get("audio_bass_root_energy")) < min_bass_root:
        return False
    if to_float(row.get("audio_4_energy")) < min_fourth:
        return False
    if to_float(row.get("audio_9_energy")) < min_ninth:
        return False
    if to_float(row.get("audio_flat7_energy")) < min_flat7:
        return False
    return True


def load_candidates(
    path: Path,
    allowed_romans: set[str],
    source_contains: str,
    min_root: float,
    min_bass_root: float,
    min_fourth: float,
    min_ninth: float,
    min_flat7: float,
    min_score: float,
) -> dict[tuple[int, float, float], dict[str, str]]:
    selected: dict[tuple[int, float, float], dict[str, str]] = {}
    selected_scores: dict[tuple[int, float, float], tuple[float, float, float, float, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not is_candidate(row, allowed_romans, source_contains):
                continue
            if not passes_gate(row, min_root, min_bass_root, min_fourth, min_ninth, min_flat7, min_score):
                continue
            key = span_key(
                int(float(row.get("bar") or 0)),
                float(row.get("reference_start") or row.get("start") or 0.0),
                float(row.get("reference_end") or row.get("end") or 0.0),
            )
            score = (
                to_float(row.get("audio_bass_root_energy")),
                to_float(row.get("audio_root_energy")),
                to_float(row.get("audio_4_energy")),
                to_float(row.get("audio_9_energy")),
                to_float(row.get("candidate_score")),
            )
            if key not in selected or score > selected_scores.get(key, (0.0, 0.0, 0.0, 0.0, 0.0)):
                selected[key] = row
                selected_scores[key] = score
    return selected


def apply_gate(payload: dict[str, Any], selected: dict[tuple[int, float, float], dict[str, str]]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    next_roman_by_key: dict[tuple[int, float, float], str] = {}
    all_spans: list[tuple[int, dict[str, Any]]] = []
    for bar in payload.get("bars", []) or []:
        bar_number = int(bar.get("bar") or bar.get("number") or 0)
        for span in bar.get("spans") or []:
            all_spans.append((bar_number, span))
    all_spans.sort(key=lambda item: (float(item[1].get("start") or 0.0), float(item[1].get("end") or 0.0)))
    for index, (bar_number, span) in enumerate(all_spans[:-1]):
        next_span = all_spans[index + 1][1]
        key = span_key(bar_number, float(span.get("start") or 0.0), float(span.get("end") or 0.0))
        next_label = str(next_span.get("roman_display") or next_span.get("roman") or "")
        if parse_chord_name(next_label)[0] is not None:
            next_key = str(next_span.get("local_key") or payload.get("key") or "C")
            next_label = chord_to_roman(next_label, next_key)
        next_roman_by_key[key] = roman_root(next_label)

    for bar in payload.get("bars", []) or []:
        bar_number = int(bar.get("bar") or bar.get("number") or 0)
        for span in bar.get("spans") or []:
            key = span_key(bar_number, float(span.get("start") or 0.0), float(span.get("end") or 0.0))
            row = selected.get(key)
            if row is None:
                continue
            if str(row.get("_require_next_roman_root") or ""):
                required = {item.strip() for item in str(row["_require_next_roman_root"]).split(",") if item.strip()}
                if next_roman_by_key.get(key) not in required:
                    continue
            old_chord = str(span.get("chord_display") or span.get("chord") or "N")
            new_chord = str(row.get("candidate_chord") or old_chord)
            local_key = str(span.get("local_key") or row.get("local_key") or payload.get("key") or "C")
            span["chord"] = new_chord
            span["chord_display"] = new_chord
            span["roman"] = chord_to_roman(new_chord, local_key)
            span["roman_display"] = chord_to_roman_display(new_chord, local_key)
            span["sus9_gate_refined"] = True
            span["sus9_gate"] = {
                "from": old_chord,
                "to": new_chord,
                "candidate_source": row.get("candidate_source"),
                "audio_root_energy": row.get("audio_root_energy"),
                "audio_bass_root_energy": row.get("audio_bass_root_energy"),
                "audio_4_energy": row.get("audio_4_energy"),
                "audio_9_energy": row.get("audio_9_energy"),
                "audio_flat7_energy": row.get("audio_flat7_energy"),
            }
            if old_chord != new_chord:
                changes.append(
                    {
                        "bar": bar_number,
                        "start": span.get("start"),
                        "end": span.get("end"),
                        "from": old_chord,
                        "to": new_chord,
                        "candidate_source": row.get("candidate_source"),
                        "audio_root_energy": row.get("audio_root_energy"),
                        "audio_bass_root_energy": row.get("audio_bass_root_energy"),
                        "audio_4_energy": row.get("audio_4_energy"),
                        "audio_9_energy": row.get("audio_9_energy"),
                        "audio_flat7_energy": row.get("audio_flat7_energy"),
                    }
                )
        sync_beats_to_spans(bar)
    sync_events_to_spans(payload)
    payload["sus9_gate_refinements"] = changes
    return changes


def write_changes(path: Path, changes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in changes for key in row.keys()}) if changes else ["bar", "start", "end", "from", "to"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(changes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--allowed-roman", action="append", default=["V9sus4"])
    parser.add_argument("--source-contains", default="sus9")
    parser.add_argument("--min-root", type=float, default=0.30)
    parser.add_argument("--min-bass-root", type=float, default=0.95)
    parser.add_argument("--min-fourth", type=float, default=0.18)
    parser.add_argument("--min-ninth", type=float, default=0.16)
    parser.add_argument("--min-flat7", type=float, default=0.06)
    parser.add_argument("--min-score", type=float, default=0.75)
    parser.add_argument(
        "--require-next-roman-root",
        default="I",
        help="Comma-separated expected next roman roots. Empty disables the resolution check.",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--changes-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = load_candidates(
        resolve_path(args.candidates),
        set(args.allowed_roman or []),
        args.source_contains,
        args.min_root,
        args.min_bass_root,
        args.min_fourth,
        args.min_ninth,
        args.min_flat7,
        args.min_score,
    )
    if args.require_next_roman_root:
        for row in selected.values():
            row["_require_next_roman_root"] = args.require_next_roman_root
    payload = json.loads(resolve_path(args.analysis).read_text(encoding="utf-8"))
    changes = apply_gate(payload, selected)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.html_output:
        write_html(resolve_path(args.html_output), payload)
    if args.changes_output:
        write_changes(resolve_path(args.changes_output), changes)
    print(f"Wrote {output} with {len(changes)} sus9 gated change(s)")
    if args.html_output:
        print(f"Wrote {resolve_path(args.html_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
