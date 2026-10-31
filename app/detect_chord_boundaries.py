#!/usr/bin/env python
"""Detect beat-independent harmonic change candidates from audio chroma."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from chord_recognizer import extract_chroma
from chord_studio import SAMPLE_RATE, extract_bass_chroma


def load_audio(path: Path, sr: int) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=True)
    return np.asarray(y, dtype=np.float32)


def normalize_columns(values: np.ndarray) -> np.ndarray:
    return values / (np.linalg.norm(values, axis=0, keepdims=True) + 1e-9)


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or values.size == 0:
        return values
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(values, kernel, mode="same")


def robust_scale(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    median = float(np.median(values))
    spread = float(np.quantile(values, 0.85) - np.quantile(values, 0.15))
    if spread <= 1e-9:
        spread = float(np.std(values)) + 1e-9
    return np.maximum((values - median) / spread, 0.0)


def novelty_curve(chroma: np.ndarray, lag: int = 2) -> np.ndarray:
    chroma = normalize_columns(np.maximum(chroma, 0.0))
    novelty = np.zeros(chroma.shape[1], dtype=np.float32)
    if chroma.shape[1] <= lag:
        return novelty
    similarity = np.sum(chroma[:, lag:] * chroma[:, :-lag], axis=0)
    novelty[lag:] = np.clip(1.0 - similarity, 0.0, 2.0)
    return novelty


def frame_times(frame_count: int, sr: int, hop_length: int) -> np.ndarray:
    return np.arange(frame_count, dtype=np.float32) * (hop_length / sr)


def load_analysis(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_bar(payload: dict[str, Any], time: float) -> dict[str, Any] | None:
    for bar in payload.get("bars", []):
        if float(bar.get("start", 0.0)) <= time < float(bar.get("end", 0.0)):
            return bar
    return None


def position_in_bar(bar: dict[str, Any], time: float, subdivisions: int) -> tuple[int, str, float]:
    start = float(bar.get("start", 0.0))
    end = float(bar.get("end", start))
    beats_per_bar = max(1, len(bar.get("beats") or []))
    if end <= start:
        return 0, "1.1/16", 0.0
    ratio = min(max((time - start) / (end - start), 0.0), 0.999999)
    grid = int(round(ratio * subdivisions))
    grid = max(0, min(subdivisions - 1, grid))
    beat_float = 1.0 + ratio * beats_per_bar
    beat = max(1, min(beats_per_bar, int(math.floor(beat_float))))
    within = beat_float - math.floor(beat_float)
    sub_per_beat = max(1, subdivisions // beats_per_bar)
    sub = max(1, min(sub_per_beat, int(math.floor(within * sub_per_beat)) + 1))
    return grid, f"{beat}.{sub}/16", beat_float


def nearest_beat_delta(bar: dict[str, Any], time: float) -> float | None:
    beats = bar.get("beats") or []
    if not beats:
        return None
    centers = [(float(item["start"]) + float(item["end"])) / 2.0 for item in beats]
    return min((time - center for center in centers), key=lambda value: abs(value))


def local_maxima(
    times: np.ndarray,
    scores: np.ndarray,
    analysis: dict[str, Any],
    threshold: float,
    min_distance: float,
    subdivisions: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if scores.size < 3:
        return candidates
    order = np.argsort(scores)[::-1]
    selected: list[float] = []
    for index in order:
        index = int(index)
        score = float(scores[index])
        if score < threshold:
            break
        if index <= 0 or index >= scores.size - 1:
            continue
        if score < float(scores[index - 1]) or score < float(scores[index + 1]):
            continue
        time = float(times[index])
        if any(abs(time - existing) < min_distance for existing in selected):
            continue
        bar = find_bar(analysis, time)
        if bar is None:
            continue
        bar_number = int(bar.get("bar", 0))
        bar_start = float(bar.get("start", 0.0))
        bar_end = float(bar.get("end", 0.0))
        if time - bar_start < 0.08 or bar_end - time < 0.08:
            continue
        grid, position, beat_float = position_in_bar(bar, time, subdivisions)
        delta = nearest_beat_delta(bar, time)
        selected.append(time)
        candidates.append(
            {
                "time": round(time, 3),
                "bar": bar_number,
                "position": position,
                "beat_position": round(beat_float, 3),
                "grid": grid,
                "score": round(score, 3),
                "nearest_beat_delta": None if delta is None else round(delta, 3),
            }
        )
    return sorted(candidates, key=lambda item: (int(item["bar"]), float(item["time"])))


def reference_boundaries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_chord: str | None = None
    for bar in payload.get("bars", []):
        for span in sorted(bar.get("spans") or [], key=lambda item: float(item.get("start", 0.0))):
            chord = str(span.get("chord_display") or span.get("chord") or "N")
            if chord == "N":
                previous_chord = chord
                continue
            if chord == previous_chord:
                continue
            rows.append(
                {
                    "time": float(span.get("start", 0.0)),
                    "bar": int(bar.get("bar", 0)),
                    "chord": chord,
                    "roman": str(span.get("roman_display") or span.get("roman") or ""),
                }
            )
            previous_chord = chord
    if rows:
        rows = rows[1:]
    return rows


def evaluate_candidates(
    candidates: list[dict[str, Any]],
    reference: dict[str, Any],
    tolerance: float,
    bar: int | None,
) -> dict[str, Any]:
    refs = reference_boundaries(reference)
    if bar is not None:
        refs = [item for item in refs if int(item["bar"]) == bar]
        candidates = [item for item in candidates if int(item["bar"]) == bar]
    candidate_times = [float(item["time"]) for item in candidates]
    hits = []
    missed = []
    used: set[int] = set()
    for ref in refs:
        best_index = None
        best_delta = None
        for index, candidate_time in enumerate(candidate_times):
            if index in used:
                continue
            delta = abs(candidate_time - float(ref["time"]))
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_index = index
        if best_index is not None and best_delta is not None and best_delta <= tolerance:
            used.add(best_index)
            hits.append({**ref, "candidate": candidates[best_index], "delta": round(best_delta, 3)})
        else:
            missed.append(ref)
    return {
        "reference_boundaries": len(refs),
        "candidate_boundaries": len(candidates),
        "hits": len(hits),
        "recall": round(len(hits) / len(refs), 3) if refs else 0.0,
        "hits_detail": hits,
        "missed": missed,
    }


def detect(
    audio: Path,
    analysis: Path,
    hop_length: int,
    feature_mode: str,
    threshold: float,
    min_distance: float,
    subdivisions: int,
    include_bass: bool,
) -> dict[str, Any]:
    y = load_audio(audio, SAMPLE_RATE)
    chroma = extract_chroma(y, SAMPLE_RATE, hop_length=hop_length, mode=feature_mode)
    score = robust_scale(moving_average(novelty_curve(chroma), 3))
    if include_bass:
        bass = extract_bass_chroma(y, SAMPLE_RATE, hop_length=hop_length, mode=feature_mode)
        score = score + 0.45 * robust_scale(moving_average(novelty_curve(bass), 3))
    times = frame_times(score.size, SAMPLE_RATE, hop_length)
    payload = load_analysis(analysis)
    candidates = local_maxima(times, score, payload, threshold, min_distance, subdivisions)
    return {
        "audio": str(audio),
        "analysis": str(analysis),
        "hop_length": hop_length,
        "feature_mode": feature_mode,
        "threshold": threshold,
        "min_distance": min_distance,
        "subdivisions": subdivisions,
        "include_bass": include_bass,
        "candidates": candidates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("analysis_json", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--reference", type=Path, help="Optional corrected analysis JSON for boundary recall.")
    parser.add_argument("--bar", type=int, help="Limit printed/evaluated candidates to one bar.")
    parser.add_argument("--hop-length", type=int, default=1024)
    parser.add_argument("--feature-mode", choices=["cqt", "fast"], default="cqt")
    parser.add_argument("--threshold", type=float, default=0.42)
    parser.add_argument("--min-distance", type=float, default=0.16)
    parser.add_argument("--subdivisions", type=int, default=16)
    parser.add_argument("--no-bass", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.12)
    parser.add_argument("--limit", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = detect(
        args.audio,
        args.analysis_json,
        args.hop_length,
        args.feature_mode,
        args.threshold,
        args.min_distance,
        args.subdivisions,
        include_bass=not args.no_bass,
    )
    candidates = result["candidates"]
    if args.bar is not None:
        candidates = [item for item in candidates if int(item["bar"]) == args.bar]
    print(f"candidates: {len(candidates)}")
    for item in candidates[: args.limit]:
        print(
            f"bar {int(item['bar']):03d} {float(item['time']):8.3f}s "
            f"{item['position']:>7} score={float(item['score']):.3f} "
            f"beat_delta={item['nearest_beat_delta']}"
        )
    if args.reference:
        reference = load_analysis(args.reference)
        metrics = evaluate_candidates(result["candidates"], reference, args.tolerance, args.bar)
        result["evaluation"] = metrics
        print(
            f"boundary recall: {metrics['hits']}/{metrics['reference_boundaries']} = "
            f"{metrics['recall']:.3f}"
        )
        if metrics["missed"]:
            print("missed:")
            for item in metrics["missed"][: args.limit]:
                print(f"  bar {item['bar']:03d} {item['time']:.3f}s {item['chord']} {item['roman']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
