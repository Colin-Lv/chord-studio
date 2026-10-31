#!/usr/bin/env python
"""Measure sustained non-diatonic vocal melody evidence for corrected spans."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from chord_studio import ROOT_TO_PC, SAMPLE_RATE, parse_chord_name
from evaluate_pitch_evidence import chord_tones, iter_spans, load_json, resolve_path


MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}


def key_pcs(key: str) -> set[int]:
    text = str(key or "C").strip()
    minor = text.endswith("m")
    root_text = text[:-1] if minor else text
    root = ROOT_TO_PC.get(root_text, 0)
    scale = MINOR_SCALE if minor else MAJOR_SCALE
    return {(root + pc) % 12 for pc in scale}


def load_audio(path: Path) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return np.asarray(y, dtype=np.float32)


def f0_to_pc(f0: np.ndarray) -> np.ndarray:
    midi = np.rint(69 + 12 * np.log2(np.maximum(f0, 1e-9) / 440.0))
    return np.mod(midi.astype(int), 12)


def extract_vocal_pitch(vocal_audio: Path, frame_length: int, hop_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import librosa

    y = load_audio(vocal_audio)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=SAMPLE_RATE,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=SAMPLE_RATE, hop_length=hop_length)
    return times, f0, np.asarray(voiced_flag & np.isfinite(f0) & (voiced_prob >= 0.55), dtype=bool)


def longest_run_duration(mask: np.ndarray, frame_seconds: float) -> float:
    best = 0
    current = 0
    for value in mask:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best * frame_seconds


def chord_label(span: dict[str, Any]) -> str:
    return str(span.get("chord_display") or span.get("chord") or "N").split("/", 1)[0]


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = load_json(resolve_path(args.analysis))
    key = str(args.key or payload.get("key") or "C")
    diatonic = key_pcs(key)
    spans = iter_spans(payload)
    times, f0, voiced = extract_vocal_pitch(resolve_path(args.vocal), args.frame_length, args.hop_length)
    pcs = np.full(len(f0), -1, dtype=int)
    valid = voiced & np.isfinite(f0)
    pcs[valid] = f0_to_pc(f0[valid])
    frame_seconds = args.hop_length / SAMPLE_RATE
    rows = []
    totals = {
        "voiced_seconds": 0.0,
        "non_diatonic_seconds": 0.0,
        "chord_tone_seconds": 0.0,
        "sustained_non_diatonic_spans": 0,
        "spans_with_voiced": 0,
    }
    for span in spans:
        start = float(span.get("start", 0.0))
        end = float(span.get("end", 0.0))
        mask = (times >= start) & (times < end) & voiced
        voiced_count = int(np.sum(mask))
        voiced_seconds = voiced_count * frame_seconds
        chord = chord_label(span)
        tones = set(chord_tones(chord))
        non_diatonic = mask & np.array([pc >= 0 and pc not in diatonic for pc in pcs], dtype=bool)
        chord_tone = mask & np.array([pc in tones for pc in pcs], dtype=bool)
        non_diatonic_chord_tone = non_diatonic & chord_tone
        sustained_non_diatonic = longest_run_duration(non_diatonic, frame_seconds)
        sustained_non_diatonic_chord_tone = longest_run_duration(non_diatonic_chord_tone, frame_seconds)
        if voiced_seconds > 0:
            totals["spans_with_voiced"] += 1
            totals["voiced_seconds"] += voiced_seconds
            totals["non_diatonic_seconds"] += int(np.sum(non_diatonic)) * frame_seconds
            totals["chord_tone_seconds"] += int(np.sum(chord_tone)) * frame_seconds
        if sustained_non_diatonic >= args.sustain_threshold:
            totals["sustained_non_diatonic_spans"] += 1
        rows.append(
            {
                "bar": span.get("bar", ""),
                "beat_start": span.get("beat_start", ""),
                "beat_end": span.get("beat_end", ""),
                "start": f"{start:.3f}",
                "end": f"{end:.3f}",
                "chord": chord,
                "roman": span.get("roman_display") or span.get("roman") or "",
                "voiced_seconds": f"{voiced_seconds:.3f}",
                "non_diatonic_ratio": "" if voiced_seconds <= 0 else f"{int(np.sum(non_diatonic)) / max(voiced_count, 1):.6f}",
                "chord_tone_ratio": "" if voiced_seconds <= 0 else f"{int(np.sum(chord_tone)) / max(voiced_count, 1):.6f}",
                "non_diatonic_chord_tone_ratio": "" if voiced_seconds <= 0 else f"{int(np.sum(non_diatonic_chord_tone)) / max(voiced_count, 1):.6f}",
                "sustained_non_diatonic_seconds": f"{sustained_non_diatonic:.3f}",
                "sustained_non_diatonic_chord_tone_seconds": f"{sustained_non_diatonic_chord_tone:.3f}",
            }
        )
    voiced_total = max(totals["voiced_seconds"], 1e-9)
    summary = {
        "analysis": str(resolve_path(args.analysis)),
        "vocal": str(resolve_path(args.vocal)),
        "key": key,
        "spans": len(spans),
        "spans_with_voiced": totals["spans_with_voiced"],
        "voiced_seconds": totals["voiced_seconds"],
        "non_diatonic_seconds": totals["non_diatonic_seconds"],
        "non_diatonic_ratio": totals["non_diatonic_seconds"] / voiced_total,
        "chord_tone_seconds": totals["chord_tone_seconds"],
        "chord_tone_ratio": totals["chord_tone_seconds"] / voiced_total,
        "sustained_non_diatonic_spans": totals["sustained_non_diatonic_spans"],
        "sustain_threshold": args.sustain_threshold,
    }
    return rows, summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--vocal", type=Path, required=True, help="Vocal stem audio.")
    parser.add_argument("--key")
    parser.add_argument("--frame-length", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--sustain-threshold", type=float, default=0.35)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    rows, summary = evaluate(args)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if args.summary:
        summary_path = resolve_path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
