#!/usr/bin/env python
"""Apply selected candidate CSV rows to a Chord Studio analysis JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from chord_studio import chord_to_roman, chord_to_roman_display, write_html
from refine_extensions import sync_beats_to_spans, sync_events_to_spans


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def span_key(bar: int, start: float, end: float) -> tuple[int, float, float]:
    return (int(bar), round(float(start), 3), round(float(end), 3))


def load_selected(path: Path) -> dict[tuple[int, float, float], dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected: dict[tuple[int, float, float], dict[str, str]] = {}
    for row in rows:
        selected[
            span_key(
                int(float(row.get("bar") or 0)),
                float(row.get("reference_start") or 0.0),
                float(row.get("reference_end") or 0.0),
            )
        ] = row
    return selected


def apply_selected(payload: dict[str, Any], selected: dict[tuple[int, float, float], dict[str, str]]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for bar in payload.get("bars", []) or []:
        bar_number = int(bar.get("bar") or bar.get("number") or 0)
        for span in bar.get("spans") or []:
            key = span_key(bar_number, float(span.get("start") or 0.0), float(span.get("end") or 0.0))
            row = selected.get(key)
            if not row:
                continue
            old_chord = str(span.get("chord_display") or span.get("chord") or "N")
            new_chord = str(row.get("candidate_chord") or old_chord)
            local_key = str(span.get("local_key") or payload.get("key") or "C")
            span["chord"] = new_chord
            span["chord_display"] = new_chord
            span["roman"] = chord_to_roman(new_chord, local_key)
            span["roman_display"] = chord_to_roman_display(new_chord, local_key)
            span["selector_refined"] = True
            span["selector_refinement"] = {
                "from": old_chord,
                "to": new_chord,
                "source": str(row.get("candidate_source") or ""),
                "score": float(row.get("reranker_score") or 0.0),
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
                        "reranker_score": row.get("reranker_score"),
                    }
                )
        sync_beats_to_spans(bar)
    sync_events_to_spans(payload)
    payload["selected_candidate_refinements"] = changes
    return changes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--changes-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis_path = resolve_path(args.analysis)
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    selected = load_selected(resolve_path(args.selected))
    changes = apply_selected(payload, selected)

    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.html_output:
        write_html(resolve_path(args.html_output), payload)
    if args.changes_output:
        path = resolve_path(args.changes_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in changes for key in row.keys()}) if changes else ["bar", "start", "end", "from", "to"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(changes)
    print(f"Wrote {output} with {len(changes)} change(s)")
    if args.html_output:
        print(f"Wrote {resolve_path(args.html_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
