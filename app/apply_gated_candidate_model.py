#!/usr/bin/env python
"""Apply a narrowly gated candidate-model correction to an analysis JSON.

This is meant for rare color chords where a general selector is too aggressive
or too conservative.  For example, apply only D#m7b5 candidates whose exact
reranker probability clears a high threshold, while leaving all other bars
untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from chord_studio import chord_to_roman, chord_to_roman_display, write_html
from refine_extensions import sync_beats_to_spans, sync_events_to_spans
from train_candidate_reranker import filter_evaluable_rows, load_rows


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def span_key(bar: int, start: float, end: float) -> tuple[int, float, float]:
    return (int(bar), round(float(start), 3), round(float(end), 3))


def load_model(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload["pipeline"], payload.get("target", "target_exact_chord")


def matches_filters(row: pd.Series, chords: set[str], romans: set[str], source_contains: str) -> bool:
    if chords and str(row.get("candidate_chord") or "") not in chords:
        return False
    if romans and str(row.get("candidate_roman") or "") not in romans:
        return False
    if source_contains and source_contains.lower() not in str(row.get("candidate_source") or "").lower():
        return False
    return True


def gated_rows(
    candidate_csv: Path,
    model_path: Path,
    chords: set[str],
    romans: set[str],
    source_contains: str,
    min_score: float,
) -> dict[tuple[int, float, float], dict[str, Any]]:
    pipeline, target = load_model(model_path)
    df = load_rows(candidate_csv)
    df = filter_evaluable_rows(df, target)
    scores = pipeline.predict_proba(df)[:, 1]
    df = df.copy()
    df["gated_model_score"] = scores
    mask = df["gated_model_score"] >= min_score
    if chords:
        mask &= df["candidate_chord"].fillna("").astype(str).isin(chords)
    if romans:
        mask &= df["candidate_roman"].fillna("").astype(str).isin(romans)
    if source_contains:
        mask &= df["candidate_source"].fillna("").astype(str).str.lower().str.contains(source_contains.lower(), regex=False)
    df = df[mask]
    selected: dict[tuple[int, float, float], dict[str, Any]] = {}
    for _, row in df.sort_values("gated_model_score", ascending=False).iterrows():
        key = span_key(int(row["bar"]), float(row["reference_start"]), float(row["reference_end"]))
        selected.setdefault(key, row.to_dict())
    return selected


def apply_gated(payload: dict[str, Any], selected: dict[tuple[int, float, float], dict[str, Any]]) -> list[dict[str, Any]]:
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
            local_key = str(span.get("local_key") or row.get("local_key") or payload.get("key") or "C")
            span["chord"] = new_chord
            span["chord_display"] = new_chord
            span["roman"] = chord_to_roman(new_chord, local_key)
            span["roman_display"] = chord_to_roman_display(new_chord, local_key)
            span["gated_candidate_refined"] = True
            span["gated_candidate_refinement"] = {
                "from": old_chord,
                "to": new_chord,
                "candidate_source": row.get("candidate_source"),
                "gated_model_score": float(row.get("gated_model_score") or 0.0),
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
                        "gated_model_score": f"{float(row.get('gated_model_score') or 0.0):.6f}",
                    }
                )
        sync_beats_to_spans(bar)
    sync_events_to_spans(payload)
    payload["gated_candidate_refinements"] = changes
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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidate-chord", action="append", default=[])
    parser.add_argument("--candidate-roman", action="append", default=[])
    parser.add_argument("--source-contains", default="")
    parser.add_argument("--min-score", type=float, default=0.42)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--changes-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(resolve_path(args.analysis).read_text(encoding="utf-8"))
    if not args.candidate_chord and not args.candidate_roman and not args.source_contains:
        raise SystemExit("At least one of --candidate-chord, --candidate-roman, or --source-contains is required.")
    selected = gated_rows(
        resolve_path(args.candidates),
        resolve_path(args.model),
        set(args.candidate_chord),
        set(args.candidate_roman),
        args.source_contains,
        args.min_score,
    )
    changes = apply_gated(payload, selected)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.html_output:
        write_html(resolve_path(args.html_output), payload)
    if args.changes_output:
        write_changes(resolve_path(args.changes_output), changes)
    print(f"Wrote {output} with {len(changes)} gated change(s)")
    if args.html_output:
        print(f"Wrote {resolve_path(args.html_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
