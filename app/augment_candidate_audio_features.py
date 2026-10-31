#!/usr/bin/env python
"""Add chroma/bass evidence features to candidate training rows."""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from build_candidate_training_set import chord_family
from build_topn_candidate_training_set import load_audio
from chord_recognizer import extract_chroma
from chord_studio import (
    SAMPLE_RATE,
    BeatInterval,
    chroma_for_intervals,
    extract_bass_chroma,
    note_name_for_pc,
    parse_chord_name,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


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


def interval_for_row(row: dict[str, str]) -> BeatInterval:
    return BeatInterval(
        bar=int(float(row.get("bar") or 0)),
        beat=1,
        start=float(row.get("start") or row.get("reference_start") or 0.0),
        end=float(row.get("end") or row.get("reference_end") or 0.0),
    )


def pc_energy(vector: np.ndarray, pc: int | None) -> float:
    if pc is None:
        return 0.0
    return float(vector[int(pc) % 12])


def add_features(row: dict[str, str], vector: np.ndarray, bass_vector: np.ndarray) -> dict[str, str]:
    chord = row.get("candidate_chord", "")
    root, suffix = parse_chord_name(chord)
    family = chord_family(chord)
    if root is None or chord == "N":
        for name in feature_names():
            row[name] = ""
        return row

    third = (root + (3 if family in {"m", "dim"} else 4)) % 12
    fifth = (root + (6 if family == "dim" else 7)) % 12
    flat7 = (root + 10) % 12
    maj7 = (root + 11) % 12
    ninth = (root + 2) % 12
    sixth = (root + 9) % 12
    fourth = (root + 5) % 12
    has_sus4 = "sus4" in suffix or suffix in {"sus", "7sus", "9sus"}
    root_energy = pc_energy(vector, root)
    third_energy = pc_energy(vector, third)
    fifth_energy = pc_energy(vector, fifth)
    triad_mean = float(np.mean([root_energy, third_energy, fifth_energy]))
    extension_values = {
        "audio_root_energy": root_energy,
        "audio_third_energy": third_energy,
        "audio_fifth_energy": fifth_energy,
        "audio_triad_mean": triad_mean,
        "audio_flat7_energy": pc_energy(vector, flat7),
        "audio_maj7_energy": pc_energy(vector, maj7),
        "audio_9_energy": pc_energy(vector, ninth),
        "audio_6_energy": pc_energy(vector, sixth),
        "audio_4_energy": pc_energy(vector, fourth),
        "audio_flat7_over_triad": pc_energy(vector, flat7) - triad_mean,
        "audio_maj7_over_triad": pc_energy(vector, maj7) - triad_mean,
        "audio_9_over_triad": pc_energy(vector, ninth) - triad_mean,
        "audio_dom7_vs_maj7": pc_energy(vector, flat7) - pc_energy(vector, maj7),
        "audio_bass_root_energy": pc_energy(bass_vector, root),
        "audio_bass_flat7_energy": pc_energy(bass_vector, flat7),
        "audio_candidate_has_7": int("7" in suffix or "9" in suffix),
        "audio_candidate_has_maj7": int("maj7" in suffix),
        "audio_candidate_has_9": int("9" in suffix or "add9" in suffix),
        "audio_candidate_has_sus4": int(has_sus4),
        "audio_candidate_has_m7b5": int("m7b5" in suffix),
    }
    for name, value in extension_values.items():
        row[name] = f"{float(value):.6f}" if isinstance(value, float) else str(value)
    row["audio_candidate_root_name"] = note_name_for_pc(root, row.get("local_key") or None)
    return row


def feature_names() -> list[str]:
    return [
        "audio_root_energy",
        "audio_third_energy",
        "audio_fifth_energy",
        "audio_triad_mean",
        "audio_flat7_energy",
        "audio_maj7_energy",
        "audio_9_energy",
        "audio_6_energy",
        "audio_4_energy",
        "audio_flat7_over_triad",
        "audio_maj7_over_triad",
        "audio_9_over_triad",
        "audio_dom7_vs_maj7",
        "audio_bass_root_energy",
        "audio_bass_flat7_energy",
        "audio_candidate_has_7",
        "audio_candidate_has_maj7",
        "audio_candidate_has_9",
        "audio_candidate_has_sus4",
        "audio_candidate_has_m7b5",
        "audio_candidate_root_name",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--feature-mode", choices=["cqt", "fast"], default="fast")
    parser.add_argument("--hop-length", type=int, default=1024)
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
    added = [name for name in feature_names() if name not in fieldnames]
    fieldnames.extend(added)
    missing_audio: set[str] = set()
    processed = 0
    for row in rows:
        song_id = row.get("song_id", "")
        audio = audio_by_song.get(song_id)
        if audio is None or not audio.exists():
            missing_audio.add(song_id)
            for name in added:
                row[name] = ""
            continue
        y, chroma, bass = song_features(str(audio), args.hop_length, args.feature_mode)
        interval = interval_for_row(row)
        vector = chroma_for_intervals(chroma, y, SAMPLE_RATE, args.hop_length, [interval])[0]
        frame_count = bass.shape[1]
        start_frame = max(0, min(frame_count - 1, int(np.floor(interval.start * SAMPLE_RATE / args.hop_length))))
        end_frame = max(start_frame + 1, min(frame_count, int(np.ceil(interval.end * SAMPLE_RATE / args.hop_length))))
        bass_slice = bass[:, start_frame:end_frame]
        bass_vector = np.median(bass_slice, axis=1) if bass_slice.size else np.zeros(12, dtype=float)
        peak = float(np.max(bass_vector))
        if peak > 0:
            bass_vector = bass_vector / peak
        add_features(row, vector, bass_vector)
        processed += 1
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} row(s), added audio features to {processed} row(s): {output}")
    if missing_audio:
        print(f"Missing audio for {len(missing_audio)} song(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
