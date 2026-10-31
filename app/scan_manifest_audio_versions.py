#!/usr/bin/env python
"""Check whether linked audio/predictions match benchmark chord annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import librosa

from evaluate_chords import TimedChord, evaluate, read_prediction, read_reference


def resolve_path(base: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else base / path


def audio_duration(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    try:
        return float(librosa.get_duration(path=str(path)))
    except Exception:  # noqa: BLE001 - version audit should keep going.
        return None


def reference_duration(rows: list[TimedChord]) -> float | None:
    if not rows:
        return None
    return max(item.end for item in rows)


def shifted_reference(rows: list[TimedChord], shift: float) -> list[TimedChord]:
    # Keep the original annotated duration in every trial. Clipping negative
    # times would reward shifts that discard hard-to-recognize opening chords.
    return [replace(item, start=item.start + shift, end=item.end + shift) for item in rows]


def shift_values(start: float, end: float, step: float) -> list[float]:
    count = int(round((end - start) / step))
    return [round(start + index * step, 6) for index in range(count + 1)]


def classify(duration_delta: float | None, best_shift: float | None, best_triad: float | None) -> tuple[str, str]:
    if best_triad is None:
        return "missing_prediction", "need predictions before version audit"
    delta = abs(duration_delta) if duration_delta is not None else 999.0
    shift = abs(best_shift or 0.0)
    triad = best_triad
    if triad >= 0.60 and delta <= 2.0 and shift <= 2.0:
        return "strong_match", "high chord agreement and small timing difference"
    if triad >= 0.50 and delta <= 6.0:
        return "aligned_match", "usable after time-shift check"
    if triad >= 0.50:
        return "usable_with_caution", "chord agreement is acceptable but duration differs"
    if triad >= 0.40 and delta <= 3.0:
        return "needs_listen", "duration is close but chord agreement is borderline"
    if triad >= 0.40:
        return "likely_different_edit", "some chord overlap, but arrangement or edit likely differs"
    if delta > 8.0:
        return "likely_mismatch", "low chord agreement and large duration difference"
    return "reject_for_training", "low chord agreement; exclude unless manually confirmed"


def audit_row(row: dict[str, str], base: Path, shifts: list[float], frame: float) -> dict[str, str]:
    audio = resolve_path(base, row.get("audio", ""))
    reference_path = resolve_path(base, row.get("chord_reference", ""))
    prediction_path = resolve_path(base, row.get("prediction", ""))
    result = {
        "id": row.get("id", ""),
        "dataset": row.get("dataset", ""),
        "artist": row.get("artist", ""),
        "title": row.get("title", ""),
        "audio": row.get("audio", ""),
        "prediction": row.get("prediction", ""),
        "chord_reference": row.get("chord_reference", ""),
        "audio_duration": "",
        "ref_duration": "",
        "duration_delta": "",
        "base_root": "",
        "base_triad": "",
        "best_time_shift": "",
        "best_root": "",
        "best_triad": "",
        "alignment_method": "prediction_optimized_time_shift",
        "status": "",
        "reason": "",
    }
    references = read_reference(reference_path) if reference_path and reference_path.exists() else []
    predictions = read_prediction(prediction_path) if prediction_path and prediction_path.exists() else []
    audio_len = audio_duration(audio)
    ref_len = reference_duration(references)
    duration_delta = audio_len - ref_len if audio_len is not None and ref_len is not None else None
    if audio_len is not None:
        result["audio_duration"] = f"{audio_len:.3f}"
    if ref_len is not None:
        result["ref_duration"] = f"{ref_len:.3f}"
    if duration_delta is not None:
        result["duration_delta"] = f"{duration_delta:.3f}"

    best_shift: float | None = None
    best_metrics: dict[str, float] | None = None
    base_metrics: dict[str, float] | None = None
    if references and predictions:
        for shift in shifts:
            try:
                metrics = evaluate(shifted_reference(references, shift), predictions, frame)
            except ValueError:
                continue
            if abs(shift) < 1e-9:
                base_metrics = metrics
            if best_metrics is None or metrics["triad_accuracy"] > best_metrics["triad_accuracy"]:
                best_shift = shift
                best_metrics = metrics
    if base_metrics:
        result["base_root"] = f"{base_metrics['root_accuracy']:.6f}"
        result["base_triad"] = f"{base_metrics['triad_accuracy']:.6f}"
    if best_metrics is not None and best_shift is not None:
        result["best_time_shift"] = f"{best_shift:.3f}"
        result["best_root"] = f"{best_metrics['root_accuracy']:.6f}"
        result["best_triad"] = f"{best_metrics['triad_accuracy']:.6f}"
    status, reason = classify(
        duration_delta,
        best_shift,
        best_metrics["triad_accuracy"] if best_metrics else None,
    )
    result["status"] = status
    result["reason"] = reason
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--shift-min", type=float, default=-12.0)
    parser.add_argument("--shift-max", type=float, default=12.0)
    parser.add_argument("--shift-step", type=float, default=0.5)
    parser.add_argument("--frame", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.shift_step <= 0 or args.frame <= 0:
        raise SystemExit("--shift-step and --frame must be positive")
    manifest = args.manifest.resolve()
    base = manifest.parent
    shifts = shift_values(args.shift_min, args.shift_max, args.shift_step)
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    results = [audit_row(row, base, shifts, args.frame) for row in rows]
    fieldnames = list(results[0].keys()) if results else []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    counts: dict[str, int] = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"Wrote: {args.output}")
    print("Prediction-optimized alignment is a diagnostic; it cannot certify independent test labels.")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")
    scored = [row for row in results if row.get("best_triad")]
    if scored:
        mean = sum(float(row["best_triad"]) for row in scored) / len(scored)
        print(f"mean_best_triad: {mean:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
