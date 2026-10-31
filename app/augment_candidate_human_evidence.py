#!/usr/bin/env python
"""Add stem-fusion and vocal-melody evidence features to candidate rows."""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from build_candidate_training_set import chord_family
from build_topn_candidate_training_set import load_audio
from chord_recognizer import extract_chroma
from chord_studio import SAMPLE_RATE, BeatInterval, chroma_for_intervals, extract_bass_chroma, parse_chord_name


REPO_ROOT = Path(__file__).resolve().parent.parent
MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def parse_mapping(values: list[str] | None) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"Expected mapping as song_id=path, got: {value}")
        song_id, path = value.split("=", 1)
        mapping[song_id.strip()] = resolve_path(Path(path.strip().strip('"')))
    return mapping


def interval_for_row(row: dict[str, str]) -> BeatInterval:
    return BeatInterval(
        bar=int(float(row.get("bar") or 0)),
        beat=1,
        start=float(row.get("start") or row.get("reference_start") or 0.0),
        end=float(row.get("end") or row.get("reference_end") or 0.0),
    )


def find_stem(stems_dir: Path, name: str) -> Path | None:
    if not stems_dir.exists():
        return None
    for path in stems_dir.rglob("*"):
        if path.is_file() and path.stem.lower() == name.lower() and path.suffix.lower() in {".wav", ".mp3", ".flac"}:
            return path
    return None


@lru_cache(maxsize=8)
def chroma_features(audio_path: str, hop_length: int, feature_mode: str):
    y = load_audio(Path(audio_path))
    chroma = extract_chroma(y, SAMPLE_RATE, hop_length=hop_length, mode=feature_mode)
    bass = extract_bass_chroma(y, SAMPLE_RATE, hop_length=hop_length, mode=feature_mode)
    return y, chroma, bass


@lru_cache(maxsize=4)
def vocal_pitch(vocal_path: str, frame_length: int, hop_length: int):
    import librosa

    y, _ = librosa.load(vocal_path, sr=SAMPLE_RATE, mono=True)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=SAMPLE_RATE,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    valid = np.asarray(voiced_flag, dtype=bool) & np.isfinite(f0) & (np.asarray(voiced_prob) >= 0.55)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=SAMPLE_RATE, hop_length=hop_length)
    pcs = np.full(len(f0), -1, dtype=np.int16)
    pcs[valid] = np.rint(12.0 * np.log2(f0[valid] / 440.0) + 69.0).astype(int) % 12
    return times, pcs, valid


def normalize(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32)
    peak = float(np.max(values)) if values.size else 0.0
    return values if peak <= 0 else values / peak


def row_vectors(audio_path: Path, rows: list[dict[str, str]], hop_length: int, feature_mode: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    y, chroma, bass = chroma_features(str(audio_path), hop_length, feature_mode)
    intervals = [interval_for_row(row) for row in rows]
    chroma_vectors = list(chroma_for_intervals(chroma, y, SAMPLE_RATE, hop_length, intervals))
    bass_vectors = []
    for interval in intervals:
        start_frame = max(0, int(np.floor(interval.start * SAMPLE_RATE / hop_length)))
        end_frame = max(start_frame + 1, int(np.ceil(interval.end * SAMPLE_RATE / hop_length)))
        bass_vector = np.median(bass[:, start_frame:end_frame], axis=1)
        bass_vectors.append(normalize(bass_vector))
    return chroma_vectors, bass_vectors


def chord_tones(chord: str) -> set[int]:
    root, suffix = parse_chord_name(str(chord or "N").split("/", 1)[0])
    if root is None:
        return set()
    suffix = suffix or ""
    lowered = suffix.lower()
    if lowered.startswith("m7b5"):
        intervals = [0, 3, 6, 10]
    elif lowered.startswith("dim7"):
        intervals = [0, 3, 6, 9]
    elif lowered.startswith("dim"):
        intervals = [0, 3, 6]
    elif lowered.startswith("aug"):
        intervals = [0, 4, 8]
    elif lowered.startswith("sus2"):
        intervals = [0, 2, 7]
    elif lowered.startswith("sus"):
        intervals = [0, 5, 7]
    elif lowered.startswith("m"):
        intervals = [0, 3, 7]
        extension = lowered[1:]
        if "maj7" in extension:
            intervals.append(11)
        elif "7" in extension or "9" in extension:
            intervals.append(10)
        if "6" in extension:
            intervals.append(9)
        if "9" in extension or "add9" in extension:
            intervals.append(2)
    else:
        intervals = [0, 4, 7]
        if "maj7" in lowered:
            intervals.append(11)
        elif "7" in lowered or "9" in lowered:
            intervals.append(10)
        if "6" in lowered:
            intervals.append(9)
        if "9" in lowered or "add9" in lowered:
            intervals.append(2)
    return {(root + interval) % 12 for interval in intervals}


def pc_energy(vector: np.ndarray, pc: int | None) -> float:
    if pc is None:
        return 0.0
    return float(vector[int(pc) % 12])


def feature_names() -> list[str]:
    return [
        "stem_fused_root_energy",
        "stem_fused_third_energy",
        "stem_fused_fifth_energy",
        "stem_fused_triad_mean",
        "stem_fused_flat7_energy",
        "stem_fused_maj7_energy",
        "stem_fused_9_energy",
        "stem_fused_flat7_over_triad",
        "stem_fused_maj7_over_triad",
        "stem_fused_9_over_triad",
        "stem_fused_dom7_vs_maj7",
        "stem_bass_root_energy",
        "melody_voiced_seconds",
        "melody_chord_tone_ratio",
        "melody_scale_tone_ratio",
        "melody_chromatic_ratio",
        "melody_chromatic_chord_tone_ratio",
        "melody_sustained_chromatic_seconds",
        "melody_candidate_contains_chromatic",
    ]


def add_stem_features(row: dict[str, str], other_vector: np.ndarray, bass_vector: np.ndarray, bass_weight: float) -> None:
    root, suffix = parse_chord_name(row.get("candidate_chord") or "N")
    if root is None:
        for name in feature_names():
            row.setdefault(name, "")
        return
    family = chord_family(row.get("candidate_chord", ""))
    third = (root + (3 if family in {"m", "dim"} else 4)) % 12
    fifth = (root + (6 if family == "dim" else 7)) % 12
    flat7 = (root + 10) % 12
    maj7 = (root + 11) % 12
    ninth = (root + 2) % 12
    fused = np.maximum(normalize(other_vector), normalize(bass_vector) * bass_weight)
    root_energy = pc_energy(fused, root)
    third_energy = pc_energy(fused, third)
    fifth_energy = pc_energy(fused, fifth)
    triad_mean = float(np.mean([root_energy, third_energy, fifth_energy]))
    values = {
        "stem_fused_root_energy": root_energy,
        "stem_fused_third_energy": third_energy,
        "stem_fused_fifth_energy": fifth_energy,
        "stem_fused_triad_mean": triad_mean,
        "stem_fused_flat7_energy": pc_energy(fused, flat7),
        "stem_fused_maj7_energy": pc_energy(fused, maj7),
        "stem_fused_9_energy": pc_energy(fused, ninth),
        "stem_fused_flat7_over_triad": pc_energy(fused, flat7) - triad_mean,
        "stem_fused_maj7_over_triad": pc_energy(fused, maj7) - triad_mean,
        "stem_fused_9_over_triad": pc_energy(fused, ninth) - triad_mean,
        "stem_fused_dom7_vs_maj7": pc_energy(fused, flat7) - pc_energy(fused, maj7),
        "stem_bass_root_energy": pc_energy(normalize(bass_vector), root),
    }
    for name, value in values.items():
        row[name] = f"{value:.6f}"


def key_scale_pcs(key: str) -> set[int]:
    root, suffix = parse_chord_name(key or "C")
    if root is None:
        root = 0
    scale = MINOR_SCALE if (suffix or "").lower().startswith("m") else MAJOR_SCALE
    return {(root + interval) % 12 for interval in scale}


def add_melody_features(row: dict[str, str], times: np.ndarray, pcs: np.ndarray, valid: np.ndarray, sustain_threshold: float) -> None:
    start = float(row.get("start") or row.get("reference_start") or 0.0)
    end = float(row.get("end") or row.get("reference_end") or 0.0)
    mask = (times >= start) & (times < end) & valid
    selected = pcs[mask]
    frame_seconds = 512 / SAMPLE_RATE
    voiced_seconds = float(len(selected) * frame_seconds)
    tones = chord_tones(row.get("candidate_chord") or "N")
    scale = key_scale_pcs(row.get("local_key") or "")
    if len(selected) == 0:
        values = {
            "melody_voiced_seconds": 0.0,
            "melody_chord_tone_ratio": 0.0,
            "melody_scale_tone_ratio": 0.0,
            "melody_chromatic_ratio": 0.0,
            "melody_chromatic_chord_tone_ratio": 0.0,
            "melody_sustained_chromatic_seconds": 0.0,
            "melody_candidate_contains_chromatic": 0.0,
        }
    else:
        chromatic = np.array([int(pc) not in scale for pc in selected], dtype=bool)
        chord_tone = np.array([int(pc) in tones for pc in selected], dtype=bool)
        chromatic_seconds = float(np.sum(chromatic) * frame_seconds)
        values = {
            "melody_voiced_seconds": voiced_seconds,
            "melody_chord_tone_ratio": float(np.mean(chord_tone)),
            "melody_scale_tone_ratio": float(np.mean([int(pc) in scale for pc in selected])),
            "melody_chromatic_ratio": float(np.mean(chromatic)),
            "melody_chromatic_chord_tone_ratio": float(np.mean(chromatic & chord_tone)),
            "melody_sustained_chromatic_seconds": chromatic_seconds if chromatic_seconds >= sustain_threshold else 0.0,
            "melody_candidate_contains_chromatic": float(any((pc not in scale) for pc in tones)),
        }
    for name, value in values.items():
        row[name] = f"{value:.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--stems", action="append", help="song_id=Demucs stems directory")
    parser.add_argument("--vocal", action="append", help="song_id=vocals.wav")
    parser.add_argument("--feature-mode", choices=["cqt", "fast"], default="fast")
    parser.add_argument("--hop-length", type=int, default=1024)
    parser.add_argument("--vocal-frame-length", type=int, default=2048)
    parser.add_argument("--vocal-hop-length", type=int, default=512)
    parser.add_argument("--bass-weight", type=float, default=0.75)
    parser.add_argument("--sustain-threshold", type=float, default=0.35)
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stems_by_song = parse_mapping(args.stems)
    vocal_by_song = parse_mapping(args.vocal)
    input_csv = resolve_path(args.candidate_csv)
    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for name in feature_names():
        if name not in fieldnames:
            fieldnames.append(name)
    by_song: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_song.setdefault(row.get("song_id", ""), []).append(index)

    stem_rows = 0
    melody_rows = 0
    for song_id, indices in by_song.items():
        song_rows = [rows[index] for index in indices]
        stems_dir = stems_by_song.get(song_id)
        if stems_dir:
            other = find_stem(stems_dir, "other")
            bass = find_stem(stems_dir, "bass")
            if other and bass:
                other_vectors, _ = row_vectors(other, song_rows, args.hop_length, args.feature_mode)
                _, bass_vectors = row_vectors(bass, song_rows, args.hop_length, args.feature_mode)
                for row, other_vector, bass_vector in zip(song_rows, other_vectors, bass_vectors):
                    add_stem_features(row, other_vector, bass_vector, args.bass_weight)
                    stem_rows += 1
        vocal = vocal_by_song.get(song_id)
        if vocal and vocal.exists():
            times, pcs, valid = vocal_pitch(str(vocal), args.vocal_frame_length, args.vocal_hop_length)
            for row in song_rows:
                add_melody_features(row, times, pcs, valid, args.sustain_threshold)
                melody_rows += 1
    for row in rows:
        for name in feature_names():
            row.setdefault(name, "")
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} row(s): {output}")
    print(f"stem evidence rows: {stem_rows}")
    print(f"melody evidence rows: {melody_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
