#!/usr/bin/env python
"""Add bass-line slash chord candidates when bass chroma strongly supports them."""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from build_candidate_training_set import chord_family, chord_root
from build_topn_candidate_training_set import load_audio
from chord_recognizer import extract_chroma
from chord_studio import (
    SAMPLE_RATE,
    BeatInterval,
    chord_to_roman,
    extract_bass_chroma,
    note_name_for_pc,
    parse_chord_name,
)
from classify_boundary_segments import roman_root
from train_candidate_reranker import resolve_path


def load_manifest(paths: list[Path]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in paths:
        with resolve_path(path).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                song_id = row.get("id", "")
                audio = row.get("audio", "")
                if song_id and audio:
                    mapping[song_id] = resolve_path(Path("benchmarks") / audio)
    return mapping


@lru_cache(maxsize=8)
def song_features(audio_path: str, hop_length: int, feature_mode: str):
    y = load_audio(Path(audio_path))
    chroma = extract_chroma(y, SAMPLE_RATE, hop_length=hop_length, mode=feature_mode)
    bass = extract_bass_chroma(y, SAMPLE_RATE, hop_length=hop_length, mode=feature_mode)
    return y, chroma, bass


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def interval_for_row(row: dict[str, str]) -> BeatInterval:
    return BeatInterval(
        bar=int(float(row.get("bar") or 0)),
        beat=1,
        start=float(row.get("start") or row.get("reference_start") or 0.0),
        end=float(row.get("end") or row.get("reference_end") or 0.0),
    )


def vector_for(row: dict[str, str], y: np.ndarray, matrix: np.ndarray, hop_length: int) -> np.ndarray:
    interval = interval_for_row(row)
    start_frame = max(0, int(np.floor(interval.start * SAMPLE_RATE / hop_length)))
    end_frame = max(start_frame + 1, int(np.ceil(interval.end * SAMPLE_RATE / hop_length)))
    vector = np.median(matrix[:, start_frame:end_frame], axis=1)
    peak = float(np.max(vector))
    if peak > 0:
        vector = vector / peak
    return vector


def triad_tones(root: int, family: str) -> set[int]:
    if family == "m":
        third = root + 3
    elif family == "dim":
        third = root + 3
    else:
        third = root + 4
    fifth = root + (6 if family == "dim" else 7)
    return {root % 12, third % 12, fifth % 12}


def third_pc(root: int, family: str) -> int:
    return (root + (3 if family in {"m", "dim"} else 4)) % 12


def fifth_pc(root: int, family: str) -> int:
    return (root + (6 if family == "dim" else 7)) % 12


def triad_support(vector: np.ndarray, root: int, family: str) -> float:
    tones = [root % 12, third_pc(root, family), fifth_pc(root, family)]
    return float(np.mean([vector[tone] for tone in tones]))


def simplified_chord_name(root: int, family: str, key: str) -> str:
    name = note_name_for_pc(root, key or None)
    if family == "m":
        return name + "m"
    if family == "dim":
        return name + "dim"
    return name


def slash_candidate(row: dict[str, str], bass_pc: int, key: str, allow_fifth_bass: bool) -> str | None:
    chord = str(row.get("candidate_chord") or "")
    if not chord or chord == "N" or "/" in chord:
        return None
    root, suffix = parse_chord_name(chord)
    if root is None or bass_pc == root:
        return None
    family = chord_family(chord)
    if family not in {"maj", "m", "dim"}:
        return None
    allowed_bass = {third_pc(root, family)}
    if allow_fifth_bass:
        allowed_bass.add(fifth_pc(root, family))
    if bass_pc not in allowed_bass:
        return None
    bass_name = note_name_for_pc(bass_pc, key or None)
    return f"{simplified_chord_name(root, family, key)}/{bass_name}"


def add_reference_fields(row: dict[str, str]) -> None:
    row["candidate_roman_root"] = roman_root(row.get("candidate_roman"))
    row["candidate_chord_root"] = chord_root(row.get("candidate_chord", ""))
    row["candidate_chord_family"] = chord_family(row.get("candidate_chord", ""))
    row["target_exact_chord"] = str(int(row.get("candidate_chord") == row.get("reference_chord")))
    row["target_same_root"] = str(
        int(row.get("candidate_chord_root") == row.get("reference_chord_root") and row.get("reference_chord") != "N")
    )
    row["target_same_roman_root"] = str(
        int(row.get("candidate_roman_root") == row.get("reference_roman_root") and row.get("reference_roman") != "N")
    )
    row["target_same_chord_family"] = str(
        int(row.get("candidate_chord_family") == row.get("reference_chord_family") and row.get("reference_chord") != "N")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--feature-mode", choices=["cqt", "fast"], default="fast")
    parser.add_argument("--hop-length", type=int, default=1024)
    parser.add_argument("--min-bass-energy", type=float, default=0.82)
    parser.add_argument("--min-bass-margin", type=float, default=0.18)
    parser.add_argument("--min-triad-support", type=float, default=0.42)
    parser.add_argument("--max-candidate-rank", type=int, default=3)
    parser.add_argument("--min-candidate-confidence", type=float, default=0.35)
    parser.add_argument("--allow-fifth-bass", action="store_true")
    parser.add_argument("--source-name", default="slash_bass")
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio_by_song = load_manifest(args.manifest)
    input_csv = resolve_path(args.candidate_csv)
    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    added: list[dict[str, str]] = []
    seen = {
        (
            row.get("song_id", ""),
            row.get("bar", ""),
            row.get("reference_start") or row.get("start", ""),
            row.get("reference_end") or row.get("end", ""),
            row.get("candidate_chord", ""),
        )
        for row in rows
    }
    for row in rows:
        if to_float(row.get("candidate_rank"), 99) > args.max_candidate_rank:
            continue
        if to_float(row.get("candidate_confidence"), 0.0) < args.min_candidate_confidence:
            continue
        audio = audio_by_song.get(row.get("song_id", ""))
        if audio is None or not audio.exists():
            continue
        root, _ = parse_chord_name(str(row.get("candidate_chord") or ""))
        family = chord_family(str(row.get("candidate_chord") or ""))
        if root is None or family not in {"maj", "m", "dim"}:
            continue
        y, chroma, bass = song_features(str(audio), args.hop_length, args.feature_mode)
        full_vector = vector_for(row, y, chroma, args.hop_length)
        if triad_support(full_vector, root, family) < args.min_triad_support:
            continue
        bass_vector = vector_for(row, y, bass, args.hop_length)
        order = np.argsort(bass_vector)[::-1]
        bass_pc = int(order[0])
        second = float(bass_vector[int(order[1])]) if len(order) > 1 else 0.0
        if float(bass_vector[bass_pc]) < args.min_bass_energy or float(bass_vector[bass_pc]) - second < args.min_bass_margin:
            continue
        local_key = row.get("local_key") or ""
        chord = slash_candidate(row, bass_pc, local_key, args.allow_fifth_bass)
        if chord is None:
            continue
        key_tuple = (
            row.get("song_id", ""),
            row.get("bar", ""),
            row.get("reference_start") or row.get("start", ""),
            row.get("reference_end") or row.get("end", ""),
            chord,
        )
        if key_tuple in seen:
            continue
        seen.add(key_tuple)
        new_row = dict(row)
        new_row["candidate_source"] = f"{args.source_name}_rank1"
        new_row["candidate_chord"] = chord
        new_row["candidate_roman"] = chord_to_roman(chord, local_key) if local_key else chord
        new_row["candidate_confidence"] = f"{max(0.05, to_float(row.get('candidate_confidence'), 0.35) * 0.84):.6f}"
        new_row["candidate_score"] = f"{to_float(row.get('candidate_score'), 0.0) - 0.03:.6f}"
        new_row["candidate_rank"] = str(max(1, int(to_float(row.get("candidate_rank"), 1))))
        new_row["audio_candidate_root_name"] = note_name_for_pc(parse_chord_name(chord)[0] or 0, local_key or None)
        add_reference_fields(new_row)
        added.append(new_row)
    output_rows = rows + added
    for row in output_rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in output_rows)
    print(f"Wrote {len(output_rows)} row(s), added {len(added)} slash candidate(s): {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
