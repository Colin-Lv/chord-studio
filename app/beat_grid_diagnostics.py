"""Inspect beat-level and bar-phase alternatives without changing an analysis.

Timing regularity alone cannot distinguish a beat from its subdivision or tell
which pulse begins a bar. Optional two-column madmom activations provide an
uncalibrated acoustic ranking. Every alternative remains a review candidate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def read_grid(payload):
    rows = []
    for bar in payload.get("bars", []):
        for beat in bar.get("beats", []):
            time, position = float(beat["start"]), int(beat["beat"])
            if not math.isfinite(time) or time < 0 or position < 1:
                raise ValueError("Beat times and positions must be finite and positive")
            rows.append((time, position))
    rows.sort()
    if len(rows) < 4 or any(right[0] <= left[0] for left, right in zip(rows, rows[1:])):
        raise ValueError("At least four distinct strictly increasing beats are required")
    count = int(payload.get("beats_per_bar") or max(position for _, position in rows))
    if count < 2 or count > 12 or any(position > count for _, position in rows):
        raise ValueError("Unsupported or inconsistent beats_per_bar")
    return rows, count


def alternatives(rows, beats_per_bar):
    times = np.asarray([time for time, _ in rows], dtype=float)
    original_numbers = np.asarray([position for _, position in rows], dtype=int)
    result = []
    continuous = bool(np.all((np.diff(original_numbers) % beats_per_bar) == 1))
    variants = [("detected", times, original_numbers)]
    if continuous:
        for parity in (0, 1):
            selected = times[parity::2]
            if len(selected) >= beats_per_bar * 2:
                variants.append((f"half{parity}", selected, np.arange(len(selected)) % beats_per_bar + 1))
        expanded = np.empty(2 * len(times) - 1)
        expanded[::2] = times
        expanded[1::2] = (times[:-1] + times[1:]) / 2
        variants.append(("double", expanded, np.arange(len(expanded)) % beats_per_bar + 1))
    for name, candidate_times, base_numbers in variants:
        for phase in range(beats_per_bar):
            numbers = (base_numbers - 1 - phase) % beats_per_bar + 1
            result.append({"id": f"{name}-phase{phase}", "tempo_variant": name,
                "phase_shift": phase, "is_original": name == "detected" and phase == 0,
                "beats_per_bar": beats_per_bar, "pulse_bpm": float(60 / np.median(np.diff(candidate_times))),
                "beat_count": len(candidate_times), "synthesized_midpoints": name == "double",
                "beats": [{"time": float(time), "beat": int(number)} for time, number in zip(candidate_times, numbers)]})
    return result, continuous


def acoustic_score(candidate, activations, fps=100):
    """Frame likelihood for a fixed grid; no learned score calibration implied.

The same audio interval is used for all hypotheses. A beat occupies 1/16 of
its neighboring period, following the upstream DBN observation partition.
This diagnostic omits the DBN's tempo-transition prior and is not its posterior.
"""
    values = np.asarray(activations, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Activations must be a finite [frames, beat/downbeat] array")
    if fps != 100:
        raise ValueError("Pretrained madmom downbeat activations have 100 fps")
    if np.min(values) < 0 or np.max(values.sum(axis=1)) > 1.0001:
        raise ValueError("Beat/downbeat activations must be mutually exclusive class probabilities")
    probabilities = np.column_stack([1 - values.sum(axis=1), values])
    probabilities = np.clip(probabilities, 1e-7, 1)
    labels = np.zeros(len(values), dtype=int)
    times = np.asarray([item["time"] for item in candidate["beats"]])
    periods = np.diff(times)
    for index, beat in enumerate(candidate["beats"]):
        adjacent = periods[max(0, index - 1):min(index + 1, len(periods))]
        half_width = max(1, int(round(float(np.median(adjacent)) * fps / 32)))
        frame = int(round(beat["time"] * fps))
        if frame < 0 or frame >= len(values):
            continue
        labels[max(0, frame - half_width):min(len(values), frame + half_width + 1)] = 2 if beat["beat"] == 1 else 1
    return float(np.log(probabilities[np.arange(len(values)), labels]).mean())


def diagnose(payload, activations=None, fps=100):
    rows, count = read_grid(payload)
    candidates, continuous = alternatives(rows, count)
    intervals = np.diff([time for time, _ in rows])
    warnings = ["Beat timing alone cannot determine metrical level or downbeat phase.",
                "Alternatives are not applied automatically; acoustic scores are not calibrated correctness probabilities."]
    if not continuous:
        warnings.append("Beat-position sequence has missing/repeated positions; half/double variants suppressed.")
    if count == 6:
        warnings.append("Six-pulse compound meter has additional notation-level ambiguity; half/double variants do not establish a new time signature.")
    if activations is None:
        warnings.append("No beat/downbeat activations supplied; alternatives are deliberately unranked.")
    else:
        values = np.asarray(activations)
        duration = float(payload.get("duration") or rows[-1][0])
        if abs(len(values) / fps - duration) > max(0.1, 2 / fps):
            raise ValueError("Activation duration does not match this analysis window")
        for candidate in candidates:
            candidate["acoustic_log_likelihood_per_frame"] = acoustic_score(candidate, values, fps)
        candidates.sort(key=lambda item: (-item["acoustic_log_likelihood_per_frame"], not item["is_original"], item["id"]))
    original = next(item for item in candidates if item["is_original"])
    return {"method": "metrical_alternatives_review", "version": 1, "applied": False,
        "source": payload.get("source"), "audio_offset": float(payload.get("audio_offset", 0)),
        "duration": payload.get("duration"), "beat_source": payload.get("beat_source"),
        "phase_verified": False, "beat_positions_continuous": continuous,
        "median_period_seconds": float(np.median(intervals)),
        "relative_tempo_variation": float(np.median(np.abs(intervals - np.median(intervals))) / np.median(intervals)),
        "score_kind": "uncalibrated_frame_log_likelihood" if activations is not None else None,
        "original_candidate_id": original["id"], "candidates": candidates, "warnings": warnings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--activations", type=Path, help="Optional madmom [frames,2] .npy for this exact audio window")
    parser.add_argument("--export-candidate", help="Explicit candidate ID to write as a separate beat file")
    parser.add_argument("--beat-file", type=Path)
    args = parser.parse_args()
    if bool(args.export_candidate) != bool(args.beat_file):
        parser.error("--export-candidate and --beat-file must be used together")
    payload = json.loads(args.analysis.read_text(encoding="utf-8-sig"))
    activations = np.load(args.activations, allow_pickle=False) if args.activations else None
    result = diagnose(payload, activations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if args.export_candidate:
        chosen = next((item for item in result["candidates"] if item["id"] == args.export_candidate), None)
        if chosen is None:
            parser.error("Unknown candidate ID")
        args.beat_file.parent.mkdir(parents=True, exist_ok=True)
        args.beat_file.write_text("".join(f"{item['time']:.6f}\t{item['beat']}\n" for item in chosen["beats"]), encoding="utf-8")


if __name__ == "__main__":
    main()
