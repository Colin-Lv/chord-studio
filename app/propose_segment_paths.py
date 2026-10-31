#!/usr/bin/env python
"""Propose bar-level harmonic segmentation paths.

This is a diagnostic bridge between beat/downbeat tracking and chord selection.
It compares several possible segmentations for each bar:

- current spans from the existing analysis
- one chord for the whole bar
- half-bar split
- beat-level split
- strong chroma/bass novelty cuts inside the bar

The tool can write a proposal report only, or conservatively apply high-gain
paths to a copy of the analysis JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from chord_recognizer import build_templates, extract_chroma
from chord_studio import (
    SAMPLE_RATE,
    BeatInterval,
    apply_bass_bias,
    bass_for_intervals,
    chroma_for_intervals,
    chord_to_roman,
    chord_to_roman_display,
    extract_bass_chroma,
    match_interval_chords,
    parse_chord_name,
    progression_template_biases,
    rms_for_intervals,
    template_biases,
    write_html,
)
from classify_boundary_segments import local_key_for_bar, simplify_functional_chord
from detect_chord_boundaries import detect as detect_boundaries
from detect_chord_boundaries import position_in_bar


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_audio(path: Path) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return np.asarray(y, dtype=np.float32)


def unique_cuts(values: list[float], min_gap: float) -> list[float]:
    cuts: list[float] = []
    for value in sorted(values):
        if cuts and abs(value - cuts[-1]) < min_gap:
            if abs(value - cuts[-1]) < min_gap * 0.5:
                cuts[-1] = (cuts[-1] + value) * 0.5
            continue
        cuts.append(float(value))
    return cuts


def current_cuts(bar: dict[str, Any]) -> list[float]:
    start = float(bar.get("start", 0.0))
    end = float(bar.get("end", start))
    cuts = [start, end]
    for span in bar.get("spans") or []:
        value = float(span.get("start", start))
        if start < value < end:
            cuts.append(value)
    return unique_cuts(cuts, 0.04)


def beat_cuts(bar: dict[str, Any]) -> list[float]:
    beats = sorted(bar.get("beats") or [], key=lambda item: int(item.get("beat", 0)))
    if not beats:
        return [float(bar.get("start", 0.0)), float(bar.get("end", 0.0))]
    cuts = [float(beats[0].get("start", bar.get("start", 0.0)))]
    cuts.extend(float(beat.get("start", 0.0)) for beat in beats[1:])
    cuts.append(float(beats[-1].get("end", bar.get("end", cuts[-1]))))
    return unique_cuts(cuts, 0.04)


def half_bar_cuts(bar: dict[str, Any]) -> list[float]:
    cuts = beat_cuts(bar)
    if len(cuts) <= 2:
        start = float(bar.get("start", 0.0))
        end = float(bar.get("end", start))
        return [start, (start + end) * 0.5, end]
    midpoint_index = max(1, (len(cuts) - 1) // 2)
    return [cuts[0], cuts[midpoint_index], cuts[-1]]


def boundary_cuts(
    bar: dict[str, Any],
    candidates: list[dict[str, Any]],
    max_internal: int,
    min_gap: float,
) -> list[float]:
    start = float(bar.get("start", 0.0))
    end = float(bar.get("end", start))
    selected = sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:max_internal]
    cuts = [start, end]
    for item in selected:
        time = float(item.get("time", 0.0))
        if start + min_gap <= time <= end - min_gap:
            cuts.append(time)
    return unique_cuts(cuts, min_gap)


def intervals_from_cuts(bar: dict[str, Any], cuts: list[float], min_duration: float) -> list[BeatInterval]:
    intervals: list[BeatInterval] = []
    for start, end in zip(cuts, cuts[1:]):
        if end - start < min_duration:
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


def classify_intervals(
    intervals: list[BeatInterval],
    chroma: np.ndarray,
    bass_chroma: np.ndarray,
    y: np.ndarray,
    key: str,
    notation: str,
    mode: str,
    feature_mode: str,
    hop_length: int,
    transition_strength: float,
    bass_weight: float,
    simplify_short: bool,
    simplify_max_duration: float,
) -> tuple[list[str], list[float], list[str], list[float], list[float]]:
    if not intervals:
        return [], [], [], [], []
    vectors = chroma_for_intervals(chroma, y, SAMPLE_RATE, hop_length, intervals)
    rms = rms_for_intervals(y, SAMPLE_RATE, intervals)
    bass_vectors, bass_roots, bass_conf = bass_for_intervals(
        bass_chroma,
        SAMPLE_RATE,
        hop_length,
        intervals,
        notation,
    )
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
        bass_weight=bass_weight,
        transition_strength=transition_strength,
        progression_strength=0.0,
    )

    templates = build_templates(mode=mode, notation=notation)
    template_names = [template.name for template in templates]
    template_matrix = np.vstack([template.vector for template in templates])
    raw_scores = vectors @ template_matrix.T
    raw_scores += template_biases(template_names, key)
    raw_scores += progression_template_biases(template_names, key, None, 0.0)
    raw_scores = apply_bass_bias(raw_scores, bass_vectors, bass_conf, template_names, bass_weight)
    margins: list[float] = []
    for row in raw_scores:
        order = np.sort(row)
        margins.append(float(order[-1] - order[-2]) if order.size >= 2 else 0.0)

    chords: list[str] = []
    for chord, interval in zip(result.chords, intervals):
        if simplify_short:
            chord = simplify_functional_chord(chord, key, interval.end - interval.start, simplify_max_duration)
        chords.append(chord)
    return (
        chords,
        [float(value) for value in result.confidence],
        list(result.bass_roots),
        [float(value) for value in result.bass_confidence],
        margins,
    )


def chord_root_label(chord: str) -> str:
    root, _ = parse_chord_name(chord)
    return "" if root is None else str(root)


def boundary_support(cuts: list[float], candidates: list[dict[str, Any]], tolerance: float) -> float:
    if len(cuts) <= 2:
        return 0.0
    score = 0.0
    for cut in cuts[1:-1]:
        nearby = [
            float(item.get("score", 0.0))
            for item in candidates
            if abs(float(item.get("time", 0.0)) - cut) <= tolerance
        ]
        score += max(nearby) if nearby else -0.12
    return score / max(len(cuts) - 2, 1)


def path_score(
    intervals: list[BeatInterval],
    chords: list[str],
    confidences: list[float],
    margins: list[float],
    boundary_score: float,
    complexity_penalty: float,
    current_chords: list[str],
) -> float:
    if not intervals:
        return -999.0
    durations = np.asarray([max(item.end - item.start, 0.0) for item in intervals], dtype=np.float32)
    weights = durations / (float(np.sum(durations)) + 1e-9)
    confidence_mean = float(np.sum(weights * np.asarray(confidences, dtype=np.float32)))
    margin_mean = float(np.sum(weights * np.asarray(margins, dtype=np.float32))) if margins else 0.0
    roots = [chord_root_label(chord) for chord in chords if chord != "N"]
    root_changes = sum(1 for left, right in zip(roots, roots[1:]) if left and right and left != right)
    change_bonus = min(0.055 * root_changes, 0.12)
    same_as_current_penalty = 0.0
    if len(chords) == len(current_chords) and chords == current_chords:
        same_as_current_penalty = -0.015
    return (
        confidence_mean
        + 0.55 * margin_mean
        + 0.10 * boundary_score
        + change_bonus
        - complexity_penalty * max(0, len(intervals) - 1)
        + same_as_current_penalty
    )


def meaningful_root_change(best: dict[str, Any], current_chords: list[str]) -> bool:
    chords = [str(chord) for chord in best.get("chords") or []]
    if len(chords) <= 1:
        return False
    roots = [chord_root_label(chord) for chord in chords if chord != "N"]
    current_roots = [chord_root_label(chord) for chord in current_chords if chord != "N"]
    if len(set(root for root in roots if root)) < 2:
        return False
    if current_roots and set(roots) == set(current_roots) and len(roots) == len(current_roots):
        return False
    return True


def span_from_interval(
    bar: dict[str, Any],
    interval: BeatInterval,
    chord: str,
    confidence: float,
    bass_root: str,
    bass_confidence: float,
    key: str,
    subdivisions: int,
    source: str,
) -> dict[str, Any]:
    _, position, beat_start = position_in_bar(bar, interval.start, subdivisions)
    _, _, beat_end = position_in_bar(bar, max(interval.start, interval.end - 1e-4), subdivisions)
    return {
        "bar": int(bar.get("bar", 0)),
        "unit": source,
        "beat_start": round(float(beat_start), 3),
        "beat_end": round(float(beat_end), 3),
        "start": round(float(interval.start), 6),
        "end": round(float(interval.end), 6),
        "chord": chord,
        "chord_display": chord,
        "roman": chord_to_roman(chord, key),
        "roman_display": chord_to_roman_display(chord, key, bass_root, bass_confidence, confidence),
        "local_key": key,
        "bass_root": bass_root,
        "bass_confidence": round(float(bass_confidence), 6),
        "confidence": round(float(confidence), 6),
        "position": position,
        "segment_path_source": source,
    }


def proposal_for_path(
    name: str,
    bar: dict[str, Any],
    cuts: list[float],
    candidates: list[dict[str, Any]],
    chroma: np.ndarray,
    bass_chroma: np.ndarray,
    y: np.ndarray,
    key: str,
    notation: str,
    mode: str,
    feature_mode: str,
    hop_length: int,
    transition_strength: float,
    bass_weight: float,
    simplify_short: bool,
    simplify_max_duration: float,
    min_duration: float,
    subdivisions: int,
    complexity_penalty: float,
    current_chords: list[str],
) -> dict[str, Any] | None:
    intervals = intervals_from_cuts(bar, cuts, min_duration)
    if not intervals:
        return None
    chords, confidences, bass_roots, bass_conf, margins = classify_intervals(
        intervals,
        chroma,
        bass_chroma,
        y,
        key,
        notation,
        mode,
        feature_mode,
        hop_length,
        transition_strength,
        bass_weight,
        simplify_short,
        simplify_max_duration,
    )
    bscore = boundary_support(cuts, candidates, tolerance=0.13)
    score = path_score(intervals, chords, confidences, margins, bscore, complexity_penalty, current_chords)
    spans = [
        span_from_interval(
            bar,
            interval,
            chord,
            confidence,
            bass_root,
            bass_confidence,
            key,
            subdivisions,
            name,
        )
        for interval, chord, confidence, bass_root, bass_confidence in zip(
            intervals,
            chords,
            confidences,
            bass_roots,
            bass_conf,
        )
    ]
    return {
        "name": name,
        "score": round(float(score), 6),
        "boundary_score": round(float(bscore), 6),
        "span_count": len(spans),
        "chords": chords,
        "roman": [span["roman_display"] for span in spans],
        "avg_confidence": round(float(np.mean(confidences)) if confidences else 0.0, 6),
        "avg_margin": round(float(np.mean(margins)) if margins else 0.0, 6),
        "cuts": [round(float(value), 6) for value in cuts],
        "spans": spans,
    }


def candidate_paths_for_bar(
    bar: dict[str, Any],
    candidates: list[dict[str, Any]],
    max_boundary_cuts: int,
    min_gap: float,
) -> dict[str, list[float]]:
    start = float(bar.get("start", 0.0))
    end = float(bar.get("end", start))
    paths: dict[str, list[float]] = {
        "current": current_cuts(bar),
        "bar": [start, end],
        "half_bar": half_bar_cuts(bar),
        "beat": beat_cuts(bar),
    }
    strong = [item for item in candidates if start + min_gap <= float(item.get("time", 0.0)) <= end - min_gap]
    if strong:
        paths["boundary"] = boundary_cuts(bar, strong, max_boundary_cuts, min_gap)
        for index, item in enumerate(sorted(strong, key=lambda row: float(row.get("score", 0.0)), reverse=True)[:2], start=1):
            paths[f"single_boundary_{index}"] = unique_cuts([start, float(item["time"]), end], min_gap)
    return paths


def apply_selected_paths(payload: dict[str, Any], selected: dict[int, dict[str, Any]]) -> dict[str, Any]:
    output = deepcopy(payload)
    for bar in output.get("bars", []):
        number = int(bar.get("bar", 0))
        proposal = selected.get(number)
        if not proposal:
            continue
        spans = deepcopy(proposal["spans"])
        bar["spans"] = spans
        for beat in bar.get("beats") or []:
            midpoint = (float(beat.get("start", 0.0)) + float(beat.get("end", 0.0))) * 0.5
            match = next((span for span in spans if float(span["start"]) <= midpoint < float(span["end"]) + 1e-6), spans[-1])
            for field in (
                "chord",
                "chord_display",
                "roman",
                "roman_display",
                "local_key",
                "bass_root",
                "bass_confidence",
                "confidence",
            ):
                beat[field] = match[field]
    output["events"] = sorted(
        [
            {**deepcopy(span), "bar": int(bar.get("bar", span.get("bar", 0)))}
            for bar in output.get("bars", [])
            for span in bar.get("spans", [])
        ],
        key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))),
    )
    output["segment_path_refined"] = True
    output["segment_path_refined_bars"] = sorted(selected)
    return output


def build_proposals(
    audio: Path,
    analysis_json: Path,
    mode: str,
    notation: str,
    feature_mode: str,
    hop_length: int,
    transition_strength: float,
    bass_weight: float,
    boundary_threshold: float,
    boundary_min_distance: float,
    subdivisions: int,
    max_boundary_cuts: int,
    min_duration: float,
    complexity_penalty: float,
    simplify_short: bool,
    simplify_max_duration: float,
    bars: set[int],
) -> dict[str, Any]:
    payload = load_json(analysis_json)
    boundary_result = detect_boundaries(
        audio,
        analysis_json,
        hop_length,
        feature_mode,
        boundary_threshold,
        boundary_min_distance,
        subdivisions,
        include_bass=True,
    )
    y = load_audio(audio)
    chroma = extract_chroma(y, SAMPLE_RATE, hop_length=hop_length, mode=feature_mode)
    bass_chroma = extract_bass_chroma(y, SAMPLE_RATE, hop_length=hop_length, mode=feature_mode)

    proposals: list[dict[str, Any]] = []
    selected: dict[int, dict[str, Any]] = {}
    for bar in payload.get("bars") or []:
        number = int(bar.get("bar", 0))
        if bars and number not in bars:
            continue
        key = local_key_for_bar(payload, number)
        bar_candidates = [item for item in boundary_result["candidates"] if int(item["bar"]) == number]
        current_chords = [
            str(span.get("chord_display") or span.get("chord") or "N")
            for span in sorted(bar.get("spans") or [], key=lambda item: float(item.get("start", 0.0)))
        ]
        path_items = []
        seen_cuts: set[tuple[float, ...]] = set()
        for name, cuts in candidate_paths_for_bar(bar, bar_candidates, max_boundary_cuts, min_duration).items():
            cut_key = tuple(round(value, 3) for value in cuts)
            if cut_key in seen_cuts:
                continue
            seen_cuts.add(cut_key)
            item = proposal_for_path(
                name,
                bar,
                cuts,
                bar_candidates,
                chroma,
                bass_chroma,
                y,
                key,
                notation,
                mode,
                feature_mode,
                hop_length,
                transition_strength,
                bass_weight,
                simplify_short,
                simplify_max_duration,
                min_duration,
                subdivisions,
                complexity_penalty,
                current_chords,
            )
            if item:
                path_items.append(item)
        path_items.sort(key=lambda item: float(item["score"]), reverse=True)
        current = next((item for item in path_items if item["name"] == "current"), None)
        best = path_items[0] if path_items else None
        current_score = float(current["score"]) if current else None
        gain = None if best is None or current_score is None else float(best["score"]) - current_score
        accepted = bool(
            best
            and best["name"] != "current"
            and gain is not None
            and meaningful_root_change(best, current_chords)
        )
        proposal = {
            "bar": number,
            "start": bar.get("start"),
            "end": bar.get("end"),
            "key": key,
            "current_chords": current_chords,
            "boundary_candidates": bar_candidates,
            "best_name": None if best is None else best["name"],
            "current_score": None if current_score is None else round(current_score, 6),
            "best_score": None if best is None else best["score"],
            "gain_over_current": None if gain is None else round(gain, 6),
            "accepted_by_score_only": accepted,
            "meaningful_root_change": bool(best and meaningful_root_change(best, current_chords)),
            "paths": path_items,
        }
        proposals.append(proposal)
        if accepted and best is not None:
            selected[number] = best
    return {
        "audio": str(audio),
        "analysis": str(analysis_json),
        "settings": {
            "mode": mode,
            "notation": notation,
            "feature_mode": feature_mode,
            "hop_length": hop_length,
            "transition_strength": transition_strength,
            "bass_weight": bass_weight,
            "boundary_threshold": boundary_threshold,
            "boundary_min_distance": boundary_min_distance,
            "subdivisions": subdivisions,
            "max_boundary_cuts": max_boundary_cuts,
            "min_duration": min_duration,
            "complexity_penalty": complexity_penalty,
            "simplify_short": simplify_short,
            "simplify_max_duration": simplify_max_duration,
        },
        "proposals": proposals,
        "_payload": payload,
        "_selected": selected,
    }


def write_report_csv(path: Path, proposals: list[dict[str, Any]], min_gain: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bar",
        "key",
        "current_chords",
        "best_name",
        "best_chords",
        "best_roman",
        "current_score",
        "best_score",
        "gain_over_current",
        "accepted",
        "meaningful_root_change",
        "boundary_count",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in proposals:
            best = item["paths"][0] if item.get("paths") else {}
            gain = item.get("gain_over_current")
            accepted = bool(
                item.get("best_name") != "current"
                and gain is not None
                and float(gain) >= min_gain
                and item.get("meaningful_root_change")
            )
            writer.writerow(
                {
                    "bar": item.get("bar"),
                    "key": item.get("key"),
                    "current_chords": " ".join(item.get("current_chords") or []),
                    "best_name": item.get("best_name"),
                    "best_chords": " | ".join(best.get("chords") or []),
                    "best_roman": " | ".join(best.get("roman") or []),
                    "current_score": item.get("current_score"),
                    "best_score": item.get("best_score"),
                    "gain_over_current": gain,
                    "accepted": int(accepted),
                    "meaningful_root_change": int(bool(item.get("meaningful_root_change"))),
                    "boundary_count": len(item.get("boundary_candidates") or []),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("analysis_json", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="Proposal JSON output.")
    parser.add_argument("--applied-output", type=Path, help="Write a copy of the analysis with accepted paths applied.")
    parser.add_argument("--html-output", type=Path, help="Optional HTML for --applied-output.")
    parser.add_argument("--report-csv", type=Path)
    parser.add_argument("--bar", type=int, action="append", dest="bars")
    parser.add_argument("--mode", choices=["triads", "seventh", "rich"], default="seventh")
    parser.add_argument("--notation", choices=["sharps", "flats"], default="sharps")
    parser.add_argument("--feature-mode", choices=["cqt", "fast"], default="fast")
    parser.add_argument("--hop-length", type=int, default=1024)
    parser.add_argument("--transition-strength", type=float, default=0.45)
    parser.add_argument("--bass-weight", type=float, default=0.55)
    parser.add_argument("--boundary-threshold", type=float, default=0.42)
    parser.add_argument("--boundary-min-distance", type=float, default=0.16)
    parser.add_argument("--subdivisions", type=int, default=16)
    parser.add_argument("--max-boundary-cuts", type=int, default=2)
    parser.add_argument("--min-duration", type=float, default=0.22)
    parser.add_argument("--complexity-penalty", type=float, default=0.045)
    parser.add_argument("--min-gain", type=float, default=0.16)
    parser.add_argument("--no-simplify-short", action="store_true")
    parser.add_argument("--simplify-max-duration", type=float, default=0.55)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio = resolve_path(args.audio)
    analysis = resolve_path(args.analysis_json)
    result = build_proposals(
        audio=audio,
        analysis_json=analysis,
        mode=args.mode,
        notation=args.notation,
        feature_mode=args.feature_mode,
        hop_length=args.hop_length,
        transition_strength=args.transition_strength,
        bass_weight=args.bass_weight,
        boundary_threshold=args.boundary_threshold,
        boundary_min_distance=args.boundary_min_distance,
        subdivisions=args.subdivisions,
        max_boundary_cuts=args.max_boundary_cuts,
        min_duration=args.min_duration,
        complexity_penalty=args.complexity_penalty,
        simplify_short=not args.no_simplify_short,
        simplify_max_duration=args.simplify_max_duration,
        bars=set(args.bars or []),
    )
    payload = result.pop("_payload")
    selected_all = result.pop("_selected")
    selected = {
        bar: proposal
        for bar, proposal in selected_all.items()
        if any(
            item.get("bar") == bar
            and item.get("gain_over_current") is not None
            and float(item["gain_over_current"]) >= args.min_gain
            and item.get("meaningful_root_change")
            for item in result["proposals"]
        )
    }
    result["accepted_bars"] = sorted(selected)
    result["accepted_count"] = len(selected)

    if args.output:
        output = resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote proposals: {output}")

    if args.report_csv:
        report = resolve_path(args.report_csv)
        write_report_csv(report, result["proposals"], args.min_gain)
        print(f"Wrote report: {report}")

    if args.applied_output:
        applied = apply_selected_paths(payload, selected)
        output = resolve_path(args.applied_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(applied, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote applied analysis: {output}")
        if args.html_output:
            html_path = resolve_path(args.html_output)
            write_html(html_path, applied)
            print(f"Wrote HTML: {html_path}")

    print(f"proposal bars: {len(result['proposals'])}; accepted: {len(selected)}")
    for item in sorted(result["proposals"], key=lambda row: float(row.get("gain_over_current") or -999), reverse=True)[:12]:
        best = item["paths"][0] if item.get("paths") else {}
        print(
            f"bar {int(item['bar']):03d} gain={item.get('gain_over_current')} "
            f"path={item.get('best_name')} chords={' | '.join(best.get('chords') or [])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
