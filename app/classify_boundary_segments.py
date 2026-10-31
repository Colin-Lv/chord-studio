#!/usr/bin/env python
"""Classify chord segments created from harmonic boundary candidates."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from chord_studio import (
    SAMPLE_RATE,
    BeatInterval,
    bass_for_intervals,
    chroma_for_intervals,
    chord_to_roman,
    chord_to_roman_display,
    extract_bass_chroma,
    match_interval_chords,
    note_name_for_pc,
    parse_chord_name,
    rms_for_intervals,
    write_html,
)
from chord_recognizer import extract_chroma
from detect_chord_boundaries import detect as detect_boundaries
from detect_chord_boundaries import position_in_bar


ROMAN_ROOTS = ("bVII", "VII", "bVI", "VI", "#IV", "IV", "bIII", "III", "II", "I", "V")


def load_audio(path: Path) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return np.asarray(y, dtype=np.float32)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def local_key_for_bar(payload: dict[str, Any], bar_number: int) -> str:
    for section in payload.get("modulations", []):
        start = int(section.get("start_bar", 1))
        end_value = section.get("end_bar")
        end = int(end_value) if end_value is not None else 10**9
        if start <= bar_number <= end:
            return str(section.get("key") or payload.get("key") or "C")
    return str(payload.get("key") or "C").split("->", 1)[0].strip()


def spans_to_intervals(bar: dict[str, Any], cuts: list[float]) -> list[BeatInterval]:
    intervals: list[BeatInterval] = []
    for start, end in zip(cuts, cuts[1:]):
        if end - start < 0.08:
            continue
        intervals.append(
            BeatInterval(
                bar=int(bar.get("bar", 0)),
                beat=1,
                start=float(start),
                end=float(end),
            )
        )
    return intervals


def scale_degree_candidates(key: str) -> set[str]:
    root, suffix = parse_chord_name(key)
    if root is None:
        return set()
    if suffix == "m":
        degrees = [(0, "m"), (2, "dim"), (3, ""), (5, "m"), (7, ""), (8, ""), (10, "")]
    else:
        degrees = [(0, ""), (2, "m"), (4, "m"), (5, ""), (7, ""), (9, "m"), (11, "dim")]
    return {f"{note_name_for_pc((root + offset) % 12, key)}{quality}" for offset, quality in degrees}


def simplify_functional_chord(chord: str, key: str, duration: float, max_duration: float) -> str:
    if duration > max_duration or chord == "N":
        return chord
    root, suffix = parse_chord_name(chord)
    if root is None:
        return chord
    diatonic = scale_degree_candidates(key)
    root_name = note_name_for_pc(root, key)
    if suffix in {"add9", "6", "maj7", "7"} and f"{root_name}" in diatonic:
        return root_name
    if suffix in {"m7", "m9", "m6", "mMaj7", "madd9"} and f"{root_name}m" in diatonic:
        return f"{root_name}m"
    if suffix in {"m7b5", "dim7"} and f"{root_name}dim" in diatonic:
        return f"{root_name}dim"
    return chord


def relabel_span(template: dict[str, Any], chord: str, key: str, confidence: float, position: str) -> dict[str, Any]:
    span = deepcopy(template)
    span["chord"] = chord
    span["chord_display"] = chord
    span["roman"] = chord_to_roman(chord, key)
    span["roman_display"] = chord_to_roman_display(chord, key)
    span["local_key"] = key
    span["confidence"] = round(float(confidence), 3)
    span["unit"] = "boundary"
    span["position"] = position
    return span


def collapse_adjacent_dict_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not spans:
        return spans
    collapsed = [deepcopy(spans[0])]
    for span in spans[1:]:
        current = collapsed[-1]
        if (
            str(current.get("chord")) == str(span.get("chord"))
            and str(current.get("local_key")) == str(span.get("local_key"))
            and abs(float(current.get("end", 0.0)) - float(span.get("start", 0.0))) < 0.08
        ):
            total = max(float(span.get("end", 0.0)) - float(current.get("start", 0.0)), 1e-9)
            current_duration = max(float(current.get("end", 0.0)) - float(current.get("start", 0.0)), 0.0)
            span_duration = max(float(span.get("end", 0.0)) - float(span.get("start", 0.0)), 0.0)
            current["confidence"] = round(
                (
                    float(current.get("confidence", 0.0)) * current_duration
                    + float(span.get("confidence", 0.0)) * span_duration
                )
                / total,
                3,
            )
            current["end"] = span.get("end")
            current["beat_end"] = span.get("beat_end", current.get("beat_end"))
            continue
        collapsed.append(deepcopy(span))
    return collapsed


def classify_bar(
    payload: dict[str, Any],
    bar: dict[str, Any],
    candidates: list[dict[str, Any]],
    chroma: np.ndarray,
    bass_chroma: np.ndarray,
    y: np.ndarray,
    key: str,
    notation: str,
    feature_mode: str,
    hop_length: int,
    mode: str,
    transition_strength: float,
    subdivisions: int,
    simplify_short: bool,
    simplify_max_duration: float,
) -> list[dict[str, Any]]:
    bar_start = float(bar["start"])
    bar_end = float(bar["end"])
    cuts = [bar_start]
    cuts.extend(float(item["time"]) for item in candidates if bar_start + 0.08 < float(item["time"]) < bar_end - 0.08)
    cuts.append(bar_end)
    cuts = sorted(set(round(item, 3) for item in cuts))
    intervals = spans_to_intervals(bar, cuts)
    if not intervals:
        return []
    vectors = chroma_for_intervals(chroma, y, SAMPLE_RATE, hop_length, intervals)
    rms = rms_for_intervals(y, SAMPLE_RATE, intervals)
    bass_vectors, bass_roots, bass_conf = bass_for_intervals(bass_chroma, SAMPLE_RATE, hop_length, intervals, notation)
    result = match_interval_chords(
        vectors,
        rms,
        mode=mode,
        notation=notation,
        key=key,
        no_chord_threshold=0.02,
        confidence_threshold=0.0,
        bass_vectors=bass_vectors,
        bass_roots=bass_roots,
        bass_confidence=bass_conf,
        transition_strength=transition_strength,
        progression_strength=0.0,
    )
    diatonic = scale_degree_candidates(key)
    spans = []
    for interval, chord, confidence in zip(intervals, result.chords, result.confidence):
        if simplify_short:
            chord = simplify_functional_chord(chord, key, interval.end - interval.start, simplify_max_duration)
        _, position, _ = position_in_bar(bar, interval.start, subdivisions)
        template = {
            "bar": int(bar.get("bar", 0)),
            "beat_start": 1,
            "beat_end": 1,
            "start": interval.start,
            "end": interval.end,
        }
        span = relabel_span(template, chord, key, float(confidence), position)
        span["diatonic"] = chord in diatonic
        spans.append(span)
    return collapse_adjacent_dict_spans(spans)


def apply_segments(payload: dict[str, Any], bars_to_spans: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    output = deepcopy(payload)
    for bar in output.get("bars", []):
        number = int(bar.get("bar", 0))
        if number not in bars_to_spans:
            continue
        spans = bars_to_spans[number]
        bar["spans"] = spans
        for beat in bar.get("beats", []):
            center = (float(beat["start"]) + float(beat["end"])) / 2.0
            span = next((item for item in spans if float(item["start"]) <= center < float(item["end"])), spans[-1])
            beat.update({key: value for key, value in span.items() if key in {"chord", "chord_display", "roman", "roman_display", "local_key", "confidence"}})
    output["events"] = sorted(
        [
            {**deepcopy(span), "bar": int(bar.get("bar", span.get("bar", 0)))}
            for bar in output.get("bars", [])
            for span in bar.get("spans", [])
        ],
        key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))),
    )
    return output


def should_refine_bar(
    candidates: list[dict[str, Any]],
    min_candidates: int,
    min_score: float,
    min_offbeat_delta: float,
) -> bool:
    strong = [item for item in candidates if float(item.get("score", 0.0)) >= min_score]
    if len(strong) < min_candidates:
        return False
    if min_offbeat_delta <= 0:
        return True
    for item in strong:
        delta = item.get("nearest_beat_delta")
        if delta is not None and abs(float(delta)) >= min_offbeat_delta:
            return True
    return False


def roman_root(label: str) -> str:
    text = label.strip()
    for root in ROMAN_ROOTS:
        if text.startswith(root):
            return root
    return text


def should_accept_refined_spans(spans: list[dict[str, Any]], min_distinct_roman_roots: int) -> bool:
    if not spans:
        return False
    labels = [
        roman_root(str(span.get("roman_display") or span.get("roman") or span.get("chord") or "N"))
        for span in spans
        if str(span.get("chord") or "N") != "N"
    ]
    if len(labels) < min_distinct_roman_roots:
        return False
    return len(set(labels)) >= min_distinct_roman_roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("analysis_json", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--bar", type=int, action="append", dest="bars")
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--min-distance", type=float, default=0.12)
    parser.add_argument("--hop-length", type=int, default=1024)
    parser.add_argument("--feature-mode", choices=["cqt", "fast"], default="cqt")
    parser.add_argument("--notation", choices=["sharps", "flats"], default="sharps")
    parser.add_argument("--mode", choices=["triads", "seventh", "rich"], default="rich")
    parser.add_argument("--transition-strength", type=float, default=0.45)
    parser.add_argument("--subdivisions", type=int, default=16)
    parser.add_argument("--min-candidates", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=1.0)
    parser.add_argument("--min-offbeat-delta", type=float, default=0.08)
    parser.add_argument("--min-distinct-roman-roots", type=int, default=5)
    parser.add_argument("--no-simplify-short", action="store_true")
    parser.add_argument("--simplify-max-duration", type=float, default=0.55)
    parser.add_argument("--html", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_json(args.analysis_json)
    boundary_result = detect_boundaries(
        args.audio,
        args.analysis_json,
        args.hop_length,
        args.feature_mode,
        args.threshold,
        args.min_distance,
        args.subdivisions,
        include_bass=True,
    )
    wanted_bars = set(args.bars or [])
    y = load_audio(args.audio)
    chroma = extract_chroma(y, SAMPLE_RATE, hop_length=args.hop_length, mode=args.feature_mode)
    bass_chroma = extract_bass_chroma(y, SAMPLE_RATE, hop_length=args.hop_length, mode=args.feature_mode)

    bars_to_spans: dict[int, list[dict[str, Any]]] = {}
    for bar in payload.get("bars", []):
        number = int(bar.get("bar", 0))
        if wanted_bars and number not in wanted_bars:
            continue
        candidates = [item for item in boundary_result["candidates"] if int(item["bar"]) == number]
        if not candidates:
            continue
        if not wanted_bars and not should_refine_bar(
            candidates,
            min_candidates=args.min_candidates,
            min_score=args.min_score,
            min_offbeat_delta=args.min_offbeat_delta,
        ):
            continue
        key = local_key_for_bar(payload, number)
        spans = classify_bar(
            payload,
            bar,
            candidates,
            chroma,
            bass_chroma,
            y,
            key,
            args.notation,
            args.feature_mode,
            args.hop_length,
            args.mode,
            args.transition_strength,
            args.subdivisions,
            simplify_short=not args.no_simplify_short,
            simplify_max_duration=args.simplify_max_duration,
        )
        if spans and (wanted_bars or should_accept_refined_spans(spans, args.min_distinct_roman_roots)):
            bars_to_spans[number] = spans
            print(f"bar {number:03d} key={key}: " + " | ".join(f"{s['position']} {s['chord']} {s['roman']}" for s in spans))

    output = apply_segments(payload, bars_to_spans)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote: {args.output}")
        if args.html:
            html_path = args.output.with_name(args.output.stem.replace("_analysis", "") + "_studio.html")
            write_html(html_path, output)
            print(f"Wrote: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
