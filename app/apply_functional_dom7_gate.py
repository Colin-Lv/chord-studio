#!/usr/bin/env python
"""Apply a high-precision functional dominant seventh gate to an analysis.

This is a narrow specialist.  It is intentionally conservative: it only
overwrites the current chord when a functional-dom7 candidate has strong bass
root support and strong flat-seventh color evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from chord_studio import chord_to_roman, chord_to_roman_display, write_html
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
    return roman.endswith("7")


def passes_gate(
    row: dict[str, str],
    min_bass_root: float,
    min_dom7_vs_maj7: float,
    min_flat7_over_triad: float,
    require_both_color_tests: bool,
    require_predominant_context: bool,
) -> bool:
    bass = to_float(row.get("audio_bass_root_energy"))
    dom7 = to_float(row.get("audio_dom7_vs_maj7"))
    flat7 = to_float(row.get("audio_flat7_over_triad"))
    if bass < min_bass_root:
        return False
    if require_predominant_context:
        prev_ok = str(row.get("gate_dom7_prev_reference_is_tonic_or_predominant") or "0") == "1"
        next_ok = str(row.get("gate_dom7_next_reference_is_predominant") or "0") == "1"
        if not (prev_ok and next_ok):
            return False
    if require_both_color_tests:
        return dom7 >= min_dom7_vs_maj7 and flat7 >= min_flat7_over_triad
    return dom7 >= min_dom7_vs_maj7 or flat7 >= min_flat7_over_triad


def load_candidates(
    path: Path,
    allowed_romans: set[str],
    source_contains: str,
    min_bass_root: float,
    min_dom7_vs_maj7: float,
    min_flat7_over_triad: float,
    require_both_color_tests: bool,
    require_predominant_context: bool,
) -> dict[tuple[int, float, float], dict[str, str]]:
    selected: dict[tuple[int, float, float], dict[str, str]] = {}
    selected_scores: dict[tuple[int, float, float], tuple[float, float, float, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not is_candidate(row, allowed_romans, source_contains):
                continue
            if not passes_gate(
                row,
                min_bass_root,
                min_dom7_vs_maj7,
                min_flat7_over_triad,
                require_both_color_tests,
                require_predominant_context,
            ):
                continue
            key = span_key(
                int(float(row.get("bar") or 0)),
                float(row.get("reference_start") or row.get("start") or 0.0),
                float(row.get("reference_end") or row.get("end") or 0.0),
            )
            current = selected.get(key)
            score = (
                to_float(row.get("audio_bass_root_energy")),
                to_float(row.get("audio_dom7_vs_maj7")),
                to_float(row.get("audio_flat7_over_triad")),
                to_float(row.get("candidate_score")),
            )
            if current is None:
                selected[key] = row
                selected_scores[key] = score
                continue
            if score > selected_scores.get(key, (0.0, 0.0, 0.0, 0.0)):
                selected[key] = row
                selected_scores[key] = score
    return selected


def apply_gate(payload: dict[str, Any], selected: dict[tuple[int, float, float], dict[str, str]]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for bar in payload.get("bars", []) or []:
        bar_number = int(bar.get("bar") or bar.get("number") or 0)
        for span in bar.get("spans") or []:
            key = span_key(bar_number, float(span.get("start") or 0.0), float(span.get("end") or 0.0))
            row = selected.get(key)
            if row is None:
                continue
            old_chord = str(span.get("chord_display") or span.get("chord") or "N")
            new_chord = str(row.get("candidate_chord") or old_chord)
            local_key = str(span.get("local_key") or row.get("local_key") or payload.get("key") or "C")
            span["chord"] = new_chord
            span["chord_display"] = new_chord
            span["roman"] = chord_to_roman(new_chord, local_key)
            span["roman_display"] = chord_to_roman_display(new_chord, local_key)
            span["functional_dom7_gate_refined"] = True
            span["functional_dom7_gate"] = {
                "from": old_chord,
                "to": new_chord,
                "candidate_source": row.get("candidate_source"),
                "audio_bass_root_energy": row.get("audio_bass_root_energy"),
                "audio_dom7_vs_maj7": row.get("audio_dom7_vs_maj7"),
                "audio_flat7_over_triad": row.get("audio_flat7_over_triad"),
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
                        "audio_bass_root_energy": row.get("audio_bass_root_energy"),
                        "audio_dom7_vs_maj7": row.get("audio_dom7_vs_maj7"),
                        "audio_flat7_over_triad": row.get("audio_flat7_over_triad"),
                    }
                )
        sync_beats_to_spans(bar)
    sync_events_to_spans(payload)
    payload["functional_dom7_gate_refinements"] = changes
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
    parser.add_argument("--allowed-roman", action="append")
    parser.add_argument("--source-contains", default="functional_dom7_direct")
    parser.add_argument("--min-bass-root", type=float, default=0.99)
    parser.add_argument("--min-dom7-vs-maj7", type=float, default=0.60)
    parser.add_argument("--min-flat7-over-triad", type=float, default=0.60)
    parser.add_argument("--require-both-color-tests", action="store_true")
    parser.add_argument("--require-predominant-context", action="store_true")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--changes-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = load_candidates(
        resolve_path(args.candidates),
        set(args.allowed_roman or ["VI7"]),
        args.source_contains,
        args.min_bass_root,
        args.min_dom7_vs_maj7,
        args.min_flat7_over_triad,
        args.require_both_color_tests,
        args.require_predominant_context,
    )
    payload = json.loads(resolve_path(args.analysis).read_text(encoding="utf-8"))
    changes = apply_gate(payload, selected)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.html_output:
        write_html(resolve_path(args.html_output), payload)
    if args.changes_output:
        write_changes(resolve_path(args.changes_output), changes)
    print(f"Wrote {output} with {len(changes)} functional-dom7 gated change(s)")
    if args.html_output:
        print(f"Wrote {resolve_path(args.html_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
