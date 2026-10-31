#!/usr/bin/env python
"""Build top-N acoustic chord candidates for reranker training/evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from build_candidate_training_set import best_reference_span, row_for_candidate
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
    progression_template_biases,
    rms_for_intervals,
    template_biases,
)
from classify_boundary_segments import local_key_for_bar, simplify_functional_chord


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_audio(path: Path) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return np.asarray(y, dtype=np.float32)


def spans_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for bar in payload.get("bars", []):
        for span in bar.get("spans") or []:
            item = dict(span)
            item["bar"] = int(bar.get("bar", item.get("bar", 0)))
            spans.append(item)
    return sorted(spans, key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))


def interval_from_span(span: dict[str, Any]) -> BeatInterval:
    return BeatInterval(
        bar=int(span.get("bar", 0)),
        beat=1,
        start=float(span.get("start", 0.0)),
        end=float(span.get("end", 0.0)),
    )


def stable_top_indices(scores: np.ndarray, top_n: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for row in scores:
        order = np.argsort(row)[::-1]
        rows.append([int(index) for index in order[:top_n]])
    return rows


def soft_confidence(row: np.ndarray, index: int) -> float:
    shifted = row - float(np.max(row))
    probs = np.exp(shifted / 0.12)
    probs = probs / (np.sum(probs) + 1e-9)
    return float(probs[index])


def build_rows(
    song_id: str,
    audio: Path,
    reference_payload: dict[str, Any],
    segment_payload: dict[str, Any],
    candidate_source: str,
    top_n: int,
    mode: str,
    notation: str,
    feature_mode: str,
    hop_length: int,
    bass_weight: float,
    progressions: str | None,
    progression_strength: float,
    simplify_short: bool,
    simplify_max_duration: float,
) -> list[dict[str, Any]]:
    y = load_audio(audio)
    chroma = extract_chroma(y, SAMPLE_RATE, hop_length=hop_length, mode=feature_mode)
    bass_chroma = extract_bass_chroma(y, SAMPLE_RATE, hop_length=hop_length, mode=feature_mode)
    segments = spans_from_payload(segment_payload)
    references = spans_from_payload(reference_payload)
    intervals = [interval_from_span(span) for span in segments]
    vectors = chroma_for_intervals(chroma, y, SAMPLE_RATE, hop_length, intervals)
    bass_vectors, _, bass_confidence = bass_for_intervals(bass_chroma, SAMPLE_RATE, hop_length, intervals, notation)
    templates = build_templates(mode=mode, notation=notation)
    template_matrix = np.vstack([template.vector for template in templates])
    template_names = [template.name for template in templates]

    scores = vectors @ template_matrix.T
    keys = [local_key_for_bar(segment_payload, int(span.get("bar", 0))) for span in segments]
    for row_index, key in enumerate(keys):
        scores[row_index] += template_biases(template_names, key)
        scores[row_index] += progression_template_biases(
            template_names, key, progressions, progression_strength
        )
    scores = apply_bass_bias(scores, bass_vectors, bass_confidence, template_names, bass_weight)

    rows: list[dict[str, Any]] = []
    for span_index, indices in enumerate(stable_top_indices(scores, top_n)):
        source_span = segments[span_index]
        reference, amount = best_reference_span(source_span, references)
        key = keys[span_index]
        duration = float(source_span.get("end", 0.0)) - float(source_span.get("start", 0.0))
        seen: set[str] = set()
        for rank, template_index in enumerate(indices, start=1):
            chord = template_names[template_index]
            if simplify_short:
                chord = simplify_functional_chord(chord, key, duration, simplify_max_duration)
            if chord in seen:
                continue
            seen.add(chord)
            candidate = dict(source_span)
            candidate["chord"] = chord
            candidate["chord_display"] = chord
            candidate["local_key"] = key
            candidate["roman"] = chord_to_roman(chord, key)
            candidate["roman_display"] = chord_to_roman_display(chord, key)
            candidate["confidence"] = round(soft_confidence(scores[span_index], template_index), 6)
            candidate["unit"] = source_span.get("unit") or "topn"
            candidate["candidate_rank"] = rank
            row = row_for_candidate(
                song_id=song_id,
                candidate_source=f"{candidate_source}_rank{rank}",
                candidate=candidate,
                reference=reference,
                overlap_seconds=amount,
            )
            row["candidate_rank"] = rank
            row["candidate_score"] = f"{float(scores[span_index, template_index]):.6f}"
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True, help="Analysis JSON whose spans define candidate intervals.")
    parser.add_argument("--source-name", default="topn")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--mode", choices=["triads", "seventh", "rich"], default="rich")
    parser.add_argument("--notation", choices=["sharps", "flats"], default="sharps")
    parser.add_argument("--feature-mode", choices=["cqt", "fast"], default="cqt")
    parser.add_argument("--hop-length", type=int, default=1024)
    parser.add_argument("--bass-weight", type=float, default=0.55)
    parser.add_argument("--progressions")
    parser.add_argument("--progression-strength", type=float, default=0.0)
    parser.add_argument("--no-simplify-short", action="store_true")
    parser.add_argument("--simplify-max-duration", type=float, default=0.55)
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio = resolve_path(args.audio)
    reference_payload = load_json(resolve_path(args.reference))
    segment_payload = load_json(resolve_path(args.segments))
    rows = build_rows(
        song_id=args.song_id,
        audio=audio,
        reference_payload=reference_payload,
        segment_payload=segment_payload,
        candidate_source=args.source_name,
        top_n=args.top_n,
        mode=args.mode,
        notation=args.notation,
        feature_mode=args.feature_mode,
        hop_length=args.hop_length,
        bass_weight=args.bass_weight,
        progressions=args.progressions,
        progression_strength=args.progression_strength,
        simplify_short=not args.no_simplify_short,
        simplify_max_duration=args.simplify_max_duration,
    )
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} top-N candidate row(s): {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
