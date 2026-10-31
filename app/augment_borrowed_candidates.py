#!/usr/bin/env python
"""Add evidence-gated borrowed-color candidates such as IVm."""

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
    PC_TO_FLAT,
    SAMPLE_RATE,
    BeatInterval,
    chord_to_roman,
    chroma_for_intervals,
    note_name_for_pc,
    parse_chord_name,
)
from classify_boundary_segments import roman_root
from train_candidate_reranker import resolve_path


ROMAN_MAJOR_TO_BORROWED = {
    "IV": "m",
}
BORROWED_MAJOR_DEGREES = {
    "bVII": 10,
    "bVI": 8,
    "bIII": 3,
}


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
def song_chroma(audio_path: str, hop_length: int, feature_mode: str):
    y = load_audio(Path(audio_path))
    return y, extract_chroma(y, SAMPLE_RATE, hop_length=hop_length, mode=feature_mode)


def interval_for_row(row: dict[str, str]) -> BeatInterval:
    return BeatInterval(
        bar=int(float(row.get("bar") or 0)),
        beat=1,
        start=float(row.get("start") or row.get("reference_start") or 0.0),
        end=float(row.get("end") or row.get("reference_end") or 0.0),
    )


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_plain_major_family(row: dict[str, str]) -> bool:
    chord = str(row.get("candidate_chord") or "")
    if not chord or chord == "N" or "/" in chord:
        return False
    root, suffix = parse_chord_name(chord)
    return root is not None and suffix in {"", "add9", "6", "maj7"}


def borrowed_candidate_name(row: dict[str, str]) -> str | None:
    if not is_plain_major_family(row):
        return None
    roman = roman_root(row.get("candidate_roman"))
    suffix = ROMAN_MAJOR_TO_BORROWED.get(roman)
    if suffix is None:
        return None
    root, _ = parse_chord_name(str(row.get("candidate_chord") or ""))
    return note_name_for_pc(root, row.get("local_key") or None) + suffix if root is not None else None


def borrowed_degree_chord(degree: str, key: str) -> str | None:
    key_root, key_suffix = parse_chord_name(key)
    if key_root is None or key_suffix == "m":
        return None
    semitone = BORROWED_MAJOR_DEGREES.get(degree)
    if semitone is None:
        return None
    # Flat-side modal mixture is more readable with flat spellings even in sharp keys.
    return PC_TO_FLAT[(key_root + semitone) % 12]


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


def should_add(row: dict[str, str], vector: np.ndarray, min_minor_third: float, min_minor_advantage: float) -> bool:
    root, _ = parse_chord_name(row.get("candidate_chord") or "N")
    if root is None:
        return False
    major_third = float(vector[(root + 4) % 12])
    minor_third = float(vector[(root + 3) % 12])
    return minor_third >= min_minor_third and (minor_third - major_third) >= min_minor_advantage


def major_triad_support(vector: np.ndarray, root: int) -> tuple[float, float, float, float]:
    root_energy = float(vector[root % 12])
    third_energy = float(vector[(root + 4) % 12])
    fifth_energy = float(vector[(root + 7) % 12])
    mean = float(np.mean([root_energy, third_energy, fifth_energy]))
    return root_energy, third_energy, fifth_energy, mean


def should_add_borrowed_degree(
    chord: str,
    vector: np.ndarray,
    min_root: float,
    min_third: float,
    min_fifth: float,
    min_triad_mean: float,
) -> bool:
    root, _ = parse_chord_name(chord)
    if root is None:
        return False
    root_energy, third_energy, fifth_energy, mean = major_triad_support(vector, root)
    return (
        root_energy >= min_root
        and third_energy >= min_third
        and fifth_energy >= min_fifth
        and mean >= min_triad_mean
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--feature-mode", choices=["cqt", "fast"], default="fast")
    parser.add_argument("--hop-length", type=int, default=1024)
    parser.add_argument("--min-minor-third", type=float, default=0.18)
    parser.add_argument("--min-minor-advantage", type=float, default=0.02)
    parser.add_argument("--enable-flat-borrowed", action="store_true")
    parser.add_argument("--min-borrowed-root", type=float, default=0.35)
    parser.add_argument("--min-borrowed-third", type=float, default=0.28)
    parser.add_argument("--min-borrowed-fifth", type=float, default=0.28)
    parser.add_argument("--min-borrowed-triad-mean", type=float, default=0.40)
    parser.add_argument("--source-name", default="borrowed_color")
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
        audio = audio_by_song.get(row.get("song_id", ""))
        if audio is None or not audio.exists():
            continue
        y, chroma = song_chroma(str(audio), args.hop_length, args.feature_mode)
        vector = chroma_for_intervals(chroma, y, SAMPLE_RATE, args.hop_length, [interval_for_row(row)])[0]
        local_key = row.get("local_key") or ""
        candidates: list[tuple[str, str, bool]] = []
        ivm = borrowed_candidate_name(row)
        if ivm and should_add(row, vector, args.min_minor_third, args.min_minor_advantage):
            candidates.append((ivm, "ivm", True))
        if args.enable_flat_borrowed:
            for degree in BORROWED_MAJOR_DEGREES:
                chord = borrowed_degree_chord(degree, local_key)
                if chord and should_add_borrowed_degree(
                    chord,
                    vector,
                    args.min_borrowed_root,
                    args.min_borrowed_third,
                    args.min_borrowed_fifth,
                    args.min_borrowed_triad_mean,
                ):
                    candidates.append((chord, degree.lower().replace("b", "flat"), False))
        for chord, source_suffix, is_ivm in candidates:
            key = (
                row.get("song_id", ""),
                row.get("bar", ""),
                row.get("reference_start") or row.get("start", ""),
                row.get("reference_end") or row.get("end", ""),
                chord,
            )
            if key in seen:
                continue
            seen.add(key)
            new_row = dict(row)
            new_row["candidate_source"] = f"{args.source_name}_{source_suffix}_rank1"
            new_row["candidate_chord"] = chord
            new_row["candidate_roman"] = chord_to_roman(chord, local_key) if local_key else chord
            penalty = 0.025 if is_ivm else 0.045
            multiplier = 0.86 if is_ivm else 0.78
            new_row["candidate_confidence"] = f"{max(0.05, to_float(row.get('candidate_confidence'), 0.35) * multiplier):.6f}"
            new_row["candidate_score"] = f"{to_float(row.get('candidate_score'), 0.0) - penalty:.6f}"
            new_row["candidate_rank"] = str(max(1, int(to_float(row.get("candidate_rank"), 1))))
            root, _ = parse_chord_name(chord)
            if root is not None:
                new_row["audio_candidate_root_name"] = note_name_for_pc(root, local_key or None)
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
    print(f"Wrote {len(output_rows)} row(s), added {len(added)} borrowed candidate(s): {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
