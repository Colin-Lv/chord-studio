#!/usr/bin/env python
"""Apply configured specialist candidate gates to a Chord Studio analysis."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from pathlib import Path
from typing import Any

from chord_studio import chord_to_roman, chord_to_roman_display, parse_chord_name, write_html
from classify_boundary_segments import roman_root
from refine_extensions import sync_beats_to_spans, sync_events_to_spans
from train_candidate_reranker import filter_evaluable_rows, load_rows, resolve_path


CONDITION_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*(>=|<=|>|<|==|!=)\s*(.+?)\s*$")


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def span_key(bar: int, start: float, end: float) -> tuple[int, float, float]:
    return (int(bar), round(float(start), 3), round(float(end), 3))


def parse_literal(value: str) -> float | str:
    text = value.strip().strip('"').strip("'")
    try:
        return float(text)
    except ValueError:
        return text


def condition_passes(row: dict[str, str], expression: str) -> bool:
    match = CONDITION_RE.match(expression)
    if not match:
        raise ValueError(f"Invalid condition: {expression}")
    column, operator, raw_expected = match.groups()
    expected = parse_literal(raw_expected)
    actual = row.get(column, "")
    if isinstance(expected, float):
        actual_value = to_float(actual)
        if operator == ">=":
            return actual_value >= expected
        if operator == "<=":
            return actual_value <= expected
        if operator == ">":
            return actual_value > expected
        if operator == "<":
            return actual_value < expected
        if operator == "==":
            return actual_value == expected
        if operator == "!=":
            return actual_value != expected
    actual_text = str(actual)
    expected_text = str(expected)
    if operator == "==":
        return actual_text == expected_text
    if operator == "!=":
        return actual_text != expected_text
    raise ValueError(f"String condition only supports == and !=: {expression}")


def row_matches_gate(row: dict[str, str], gate: dict[str, Any]) -> bool:
    allowed_songs = {str(item).strip() for item in gate.get("allowed_songs", []) or [] if str(item).strip()}
    if allowed_songs and str(row.get("song_id") or "") not in allowed_songs:
        return False
    source_contains = str(gate.get("source_contains") or "").lower()
    if source_contains and source_contains not in str(row.get("candidate_source") or "").lower():
        return False
    if gate.get("candidate_chord") and row.get("candidate_chord") != gate["candidate_chord"]:
        return False
    if gate.get("candidate_roman") and row.get("candidate_roman") != gate["candidate_roman"]:
        return False
    if gate.get("candidate_chord_endswith") and not str(row.get("candidate_chord") or "").endswith(str(gate["candidate_chord_endswith"])):
        return False
    if gate.get("candidate_roman_endswith") and not str(row.get("candidate_roman") or "").endswith(str(gate["candidate_roman_endswith"])):
        return False
    if not all(condition_passes(row, str(condition)) for condition in gate.get("conditions", []) or []):
        return False
    groups = gate.get("condition_groups") or []
    if not groups:
        return True
    for group in groups:
        if all(condition_passes(row, str(condition)) for condition in group):
            return True
    return False


def score_row(row: dict[str, str]) -> tuple[float, float, float, float, float, float]:
    return (
        to_float(row.get("_model_score")),
        to_float(row.get("candidate_score")),
        to_float(row.get("audio_bass_root_energy")),
        to_float(row.get("audio_root_energy")),
        to_float(row.get("audio_dom7_vs_maj7")),
        to_float(row.get("audio_flat7_energy")),
    )


def load_model_scores(candidates: Path, gates: list[dict[str, Any]]) -> dict[tuple[int, str], float]:
    model_gates = [gate for gate in gates if gate.get("model_path")]
    if not model_gates:
        return {}
    scores: dict[tuple[int, str], float] = {}
    df = load_rows(candidates)
    df = df.reset_index(drop=True)
    for gate_index, gate in enumerate(gates):
        model_path = gate.get("model_path")
        if not model_path:
            continue
        with resolve_path(Path(str(model_path))).open("rb") as handle:
            payload = pickle.load(handle)
        target = str(payload.get("target") or "target_exact_chord")
        evaluable = filter_evaluable_rows(df, target)
        probabilities = payload["pipeline"].predict_proba(evaluable)[:, 1]
        for row_index, probability in zip(evaluable.index, probabilities):
            scores[(gate_index, str(row_index))] = float(probability)
    return scores


def load_selected(candidates: Path, gates: list[dict[str, Any]]) -> dict[tuple[int, float, float], dict[str, Any]]:
    selected: dict[tuple[int, float, float], dict[str, Any]] = {}
    selected_scores: dict[tuple[int, float, float], tuple[int, float, float, float, float, float]] = {}
    model_scores = load_model_scores(candidates, gates)
    with candidates.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            for priority, gate in enumerate(gates):
                if not row_matches_gate(row, gate):
                    continue
                model_path = gate.get("model_path")
                model_score = ""
                if model_path:
                    model_score = f"{model_scores.get((priority, str(row_index)), 0.0):.6f}"
                    if to_float(model_score) < to_float(gate.get("model_min_score"), 0.0):
                        continue
                    row["_model_score"] = model_score
                key = span_key(
                    int(float(row.get("bar") or 0)),
                    float(row.get("reference_start") or row.get("start") or 0.0),
                    float(row.get("reference_end") or row.get("end") or 0.0),
                )
                score = (-priority, *score_row(row))
                if key not in selected or score > selected_scores[key]:
                    selected[key] = {
                        **row,
                        "_gate_name": str(gate.get("name") or f"gate_{priority}"),
                        "_require_next_current_roman_root": str(gate.get("require_next_current_roman_root") or ""),
                        "_model_score": model_score,
                    }
                    selected_scores[key] = score
                break
    return selected


def next_current_roman_roots(payload: dict[str, Any]) -> dict[tuple[int, float, float], str]:
    spans: list[tuple[int, dict[str, Any]]] = []
    for bar in payload.get("bars", []) or []:
        bar_number = int(bar.get("bar") or bar.get("number") or 0)
        for span in bar.get("spans") or []:
            spans.append((bar_number, span))
    spans.sort(key=lambda item: (float(item[1].get("start") or 0.0), float(item[1].get("end") or 0.0)))
    lookup: dict[tuple[int, float, float], str] = {}
    for index, (bar_number, span) in enumerate(spans[:-1]):
        next_span = spans[index + 1][1]
        label = str(next_span.get("roman_display") or next_span.get("roman") or "")
        if parse_chord_name(label)[0] is not None:
            local_key = str(next_span.get("local_key") or payload.get("key") or "C")
            label = chord_to_roman(label, local_key)
        key = span_key(bar_number, float(span.get("start") or 0.0), float(span.get("end") or 0.0))
        lookup[key] = roman_root(label)
    return lookup


def apply_selected(payload: dict[str, Any], selected: dict[tuple[int, float, float], dict[str, Any]]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    next_roots = next_current_roman_roots(payload)
    for bar in payload.get("bars", []) or []:
        bar_number = int(bar.get("bar") or bar.get("number") or 0)
        for span in bar.get("spans") or []:
            key = span_key(bar_number, float(span.get("start") or 0.0), float(span.get("end") or 0.0))
            row = selected.get(key)
            if not row:
                continue
            required = {item.strip() for item in str(row.get("_require_next_current_roman_root") or "").split(",") if item.strip()}
            if required and next_roots.get(key) not in required:
                continue
            old_chord = str(span.get("chord_display") or span.get("chord") or "N")
            new_chord = str(row.get("candidate_chord") or old_chord)
            local_key = str(span.get("local_key") or row.get("local_key") or payload.get("key") or "C")
            span["chord"] = new_chord
            span["chord_display"] = new_chord
            span["roman"] = chord_to_roman(new_chord, local_key)
            span["roman_display"] = chord_to_roman_display(new_chord, local_key)
            span["specialist_gate_refined"] = True
            span["specialist_gate"] = {
                "name": row.get("_gate_name"),
                "from": old_chord,
                "to": new_chord,
                "candidate_source": row.get("candidate_source"),
                "candidate_score": row.get("candidate_score"),
            }
            if old_chord != new_chord:
                changes.append(
                    {
                        "bar": bar_number,
                        "start": span.get("start"),
                        "end": span.get("end"),
                        "from": old_chord,
                        "to": new_chord,
                        "gate": row.get("_gate_name"),
                        "candidate_source": row.get("candidate_source"),
                        "candidate_score": row.get("candidate_score"),
                    }
                )
        sync_beats_to_spans(bar)
    sync_events_to_spans(payload)
    payload["specialist_gate_refinements"] = changes
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
    parser.add_argument("--config", type=Path, default=Path("benchmarks/specialists/specialist_gate_config.json"))
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--changes-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(resolve_path(args.config).read_text(encoding="utf-8"))
    selected = load_selected(resolve_path(args.candidates), list(config.get("gates", []) or []))
    payload = json.loads(resolve_path(args.analysis).read_text(encoding="utf-8"))
    changes = apply_selected(payload, selected)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.html_output:
        write_html(resolve_path(args.html_output), payload)
    if args.changes_output:
        write_changes(resolve_path(args.changes_output), changes)
    print(f"Wrote {output} with {len(changes)} specialist gated change(s)")
    if args.html_output:
        print(f"Wrote {resolve_path(args.html_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
