#!/usr/bin/env python
"""Infer coarse chord labels from Guitar Pro note content.

This is an experimental symbolic fallback for GP files that do not contain
explicit chord symbols. It reads notes from non-drum tracks, scores chord
templates over each bar or half-bar, and exports a `.lab` file plus diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chord_studio import note_name_for_pc, parse_chord_name
from import_guitarpro import (
    GP_EXTENSIONS,
    first_key,
    gpif_beat_count,
    gpif_id_map,
    gpif_key,
    gpif_read_score,
    gpif_rhythm_quarters,
    gpif_tempo,
    gpif_text,
    gpif_time_quarters,
    load_guitarpro_module,
    measure_meter,
    measure_quarters,
    safe_id,
    song_tempo,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class NoteEvent:
    start: float
    end: float
    midi: int
    track: str
    weight: float


@dataclass(frozen=True)
class InferredChord:
    start: float
    end: float
    chord: str
    confidence: float
    bass: str
    evidence: str


@dataclass(frozen=True)
class InferenceCandidate:
    mode: str
    chords: list[InferredChord]
    score: float
    diagnostics: dict[str, float | str]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def duration_quarters(duration: Any) -> float:
    value = int(getattr(duration, "value", 4) or 4)
    quarters = 4.0 / value
    if getattr(duration, "isDotted", False):
        quarters *= 1.5
    tuplet = getattr(duration, "tuplet", None)
    enters = float(getattr(tuplet, "enters", 1) or 1)
    times = float(getattr(tuplet, "times", 1) or 1)
    if enters > 0:
        quarters *= times / enters
    return quarters


def string_tuning(track: Any) -> dict[int, int]:
    return {int(string.number): int(string.value) for string in getattr(track, "strings", []) or []}


def is_drum_track(track: Any) -> bool:
    if bool(getattr(track, "isPercussionTrack", False)):
        return True
    text = f"{getattr(track, 'name', '')} {getattr(getattr(track, 'channel', None), 'instrument', '')}".lower()
    return any(word in text for word in ("drum", "percussion", "鼓", "敲击"))


def role_from_text(text: str) -> str:
    normalized = text.lower()
    if any(word in normalized for word in ("drum", "percussion", "鼓", "敲击")):
        return "drum"
    if any(word in normalized for word in ("bass", "贝斯")):
        return "bass"
    if any(
        word in normalized
        for word in ("rhythm", "节奏", "扫弦", "acoustic", "clean guitar", "overdriven guitar", "gtr", "guitar", "吉他", "和弦")
    ):
        return "rhythm"
    if any(word in normalized for word in ("piano", "keyboard", "organ", "strings", "pad", "synth", "钢琴", "键盘")):
        return "harmony"
    if any(word in normalized for word in ("vocal", "voice", "主唱", "旋律", "melody")):
        return "melody"
    if "solo" in normalized or "lead" in normalized:
        return "solo"
    return "other"


def track_role(track: Any) -> str:
    if is_drum_track(track):
        return "drum"
    name = str(getattr(track, "name", "")).lower()
    channel = str(getattr(getattr(track, "channel", None), "instrument", "")).lower()
    return role_from_text(f"{name} {channel}")


def track_weight(track: Any) -> float:
    role = track_role(track)
    return weight_for_role(role)


def weight_for_role(role: str) -> float:
    if role == "bass":
        return 1.35
    if role == "rhythm":
        return 1.20
    if role == "harmony":
        return 1.10
    if role == "melody":
        return 0.55
    if role == "solo":
        return 0.65
    return 1.0


def select_tracks(song: Any, mode: str) -> list[Any]:
    tracks = [track for track in (getattr(song, "tracks", []) or []) if not is_drum_track(track)]
    if mode == "all":
        return tracks
    role_groups: dict[str, list[Any]] = {}
    for track in tracks:
        role_groups.setdefault(track_role(track), []).append(track)
    if mode == "rhythm_guitar":
        selected = role_groups.get("rhythm", [])
        if selected:
            return selected
        name_matches = [
            track
            for track in tracks
            if any(
                word in str(getattr(track, "name", "")).lower()
                for word in ("rhythm", "gtr", "guitar", "clean", "overdriven", "acoustic", "和弦", "吉他")
            )
        ]
        return name_matches or role_groups.get("harmony", []) or tracks
    if mode == "rhythm_bass":
        selected = role_groups.get("rhythm", []) + role_groups.get("bass", [])
        return selected or [track for track in tracks if track_weight(track) >= 1.0] or tracks
    if mode == "bass_harmony":
        selected = role_groups.get("bass", []) + role_groups.get("rhythm", []) + role_groups.get("harmony", [])
        return selected or [track for track in tracks if track_weight(track) >= 1.0] or tracks
    if mode == "harmonic":
        selected = [track for track in tracks if track_role(track) in {"bass", "rhythm", "harmony", "other"} and track_weight(track) >= 1.0]
        return selected or tracks
    if mode.startswith("track:"):
        wanted = mode.split(":", 1)[1]
        return [track for track in tracks if str(getattr(track, "name", "")) == wanted]
    raise ValueError(f"Unsupported track mode: {mode}")


def collect_note_events(song: Any, mode: str) -> list[NoteEvent]:
    tempo = song_tempo(song)
    seconds_per_quarter = 60.0 / tempo
    first_tick = float(getattr(song.measureHeaders[0], "start", 0.0) or 0.0) if song.measureHeaders else 0.0
    events: list[NoteEvent] = []
    selected_tracks = select_tracks(song, mode)
    for track in selected_tracks:
        if is_drum_track(track):
            continue
        tuning = string_tuning(track)
        if not tuning:
            continue
        weight = track_weight(track)
        for measure in getattr(track, "measures", []) or []:
            for voice in getattr(measure, "voices", []) or []:
                for beat in getattr(voice, "beats", []) or []:
                    notes = getattr(beat, "notes", []) or []
                    if not notes:
                        continue
                    start_ticks = float(getattr(beat, "start", 0.0) or 0.0) - first_tick
                    start = start_ticks / 960.0 * seconds_per_quarter
                    dur = duration_quarters(getattr(beat, "duration", None)) * seconds_per_quarter
                    if dur <= 0:
                        continue
                    for note in notes:
                        string = int(getattr(note, "string", 0) or 0)
                        fret = int(getattr(note, "value", 0) or 0)
                        if string not in tuning:
                            continue
                        midi = tuning[string] + fret
                        events.append(NoteEvent(start, start + dur, midi, str(getattr(track, "name", "")), weight))
    return events


def available_track_modes(song: Any, requested: str) -> list[str]:
    if requested != "auto":
        if requested != "auto_per_track":
            return [requested]
    modes = ["rhythm_guitar", "harmonic", "rhythm_bass", "bass_harmony", "all"]
    if requested == "auto":
        return modes
    tracks = [track for track in (getattr(song, "tracks", []) or []) if not is_drum_track(track)]
    for track in tracks:
        role = track_role(track)
        if role in {"rhythm", "harmony", "bass", "other"}:
            modes.append(f"track:{getattr(track, 'name', '')}")
    return list(dict.fromkeys(modes))


def bar_grid(song: Any) -> tuple[list[tuple[float, float, str]], str]:
    tempo = song_tempo(song)
    seconds_per_quarter = 60.0 / tempo
    bars = []
    elapsed = 0.0
    meters = []
    for header in getattr(song, "measureHeaders", []) or []:
        meter = measure_meter(header)
        meters.append(meter)
        duration = measure_quarters(header) * seconds_per_quarter
        bars.append((elapsed, elapsed + duration, meter))
        elapsed += duration
    dominant_meter = max(set(meters), key=meters.count) if meters else "4/4"
    return bars, dominant_meter


def chord_templates(include_sevenths: bool) -> list[tuple[str, tuple[int, ...], str]]:
    qualities: list[tuple[str, tuple[int, ...], str]] = [
        ("", (0, 4, 7), "maj"),
        ("m", (0, 3, 7), "min"),
        ("dim", (0, 3, 6), "dim"),
        ("sus4", (0, 5, 7), "sus"),
        ("sus2", (0, 2, 7), "sus"),
        ("5", (0, 7), "power"),
    ]
    if include_sevenths:
        qualities.extend(
            [
                ("7", (0, 4, 7, 10), "dom7"),
                ("maj7", (0, 4, 7, 11), "maj7"),
                ("m7", (0, 3, 7, 10), "m7"),
                ("6", (0, 4, 7, 9), "six"),
                ("m6", (0, 3, 7, 9), "six"),
                ("add9", (0, 2, 4, 7), "add9"),
            ]
        )
    output = []
    for root in range(12):
        name = note_name_for_pc(root)
        for suffix, intervals, family in qualities:
            output.append((f"{name}{suffix}", tuple((root + interval) % 12 for interval in intervals), family))
    return output


def segment_vector(events: list[NoteEvent], start: float, end: float) -> tuple[list[float], str, float]:
    weights = [0.0] * 12
    bass_counter: Counter[int] = Counter()
    for event in events:
        overlap = max(0.0, min(end, event.end) - max(start, event.start))
        if overlap <= 0:
            continue
        pc = event.midi % 12
        # Longer notes and accompaniment/bass tracks matter more.
        amount = overlap * event.weight
        if event.start <= start + 0.08:
            amount *= 1.15
        weights[pc] += amount
        bass_counter[event.midi] += overlap
    total = sum(weights)
    if total > 0:
        weights = [value / total for value in weights]
    bass_pc = ""
    bass_strength = 0.0
    if bass_counter:
        lowest = min(bass_counter)
        bass_pc = note_name_for_pc(lowest % 12)
        bass_strength = bass_counter[lowest] / max(1e-9, sum(bass_counter.values()))
    return weights, bass_pc, bass_strength


def score_template(weights: list[float], tones: tuple[int, ...], bass: str, chord: str) -> float:
    if not tones:
        return 0.0
    tone_energy = sum(weights[pc] for pc in tones)
    non_chord = sum(value for pc, value in enumerate(weights) if pc not in tones)
    root_pc = tones[0]
    score = tone_energy - 0.55 * non_chord + 0.20 * weights[root_pc]
    if bass:
        bass_pc = parse_note_pc(bass)
        if bass_pc == root_pc:
            score += 0.12
        elif bass_pc in tones:
            score += 0.03
        else:
            score -= 0.10
    # Penalize rich labels unless their color tones are audible.
    if "maj7" in chord and weights[(root_pc + 11) % 12] < 0.045:
        score -= 0.10
    elif chord.endswith("7") and weights[(root_pc + 10) % 12] < 0.045:
        score -= 0.08
    elif "add9" in chord and weights[(root_pc + 2) % 12] < 0.045:
        score -= 0.08
    return score


def parse_note_pc(note: str) -> int | None:
    table = {
        "C": 0,
        "B#": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "Fb": 4,
        "E#": 5,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
        "Cb": 11,
    }
    return table.get(note)


def chord_suffix_family(chord: str) -> str:
    _, suffix = parse_chord_name(chord)
    low = suffix.lower()
    if low.startswith("m") and "maj" not in low:
        return "minor"
    if "dim" in low:
        return "dim"
    if "sus" in low or "5" in low:
        return "neutral"
    return "major"


def estimate_symbolic_key(chords: list[InferredChord]) -> str:
    weighted_roots: Counter[int] = Counter()
    weighted_families: dict[int, Counter[str]] = defaultdict(Counter)
    last_root: int | None = None
    for item in chords:
        root, _ = parse_chord_name(item.chord)
        if root is None:
            continue
        duration = max(0.0, item.end - item.start)
        if duration <= 0:
            continue
        weight = max(0.1, duration)
        weighted_roots[root] += weight
        weighted_families[root][chord_suffix_family(item.chord)] += weight
        last_root = root
    if not weighted_roots:
        return "C"

    major_degrees = {
        0: ("major", 1.25),
        2: ("minor", 0.75),
        4: ("minor", 0.85),
        5: ("major", 1.05),
        7: ("major", 1.15),
        9: ("minor", 1.05),
        11: ("dim", 0.35),
    }
    minor_degrees = {
        0: ("minor", 1.25),
        2: ("dim", 0.35),
        3: ("major", 1.05),
        5: ("minor", 0.75),
        7: ("minor", 0.95),
        8: ("major", 1.05),
        10: ("major", 1.00),
        11: ("major", 0.45),
    }

    candidates: list[tuple[float, str]] = []
    total = sum(weighted_roots.values()) or 1.0
    for key_root in range(12):
        for mode, profile in (("major", major_degrees), ("minor", minor_degrees)):
            score = 0.0
            for root, weight in weighted_roots.items():
                degree = (root - key_root) % 12
                family_counter = weighted_families[root]
                family = family_counter.most_common(1)[0][0] if family_counter else "neutral"
                if degree in profile:
                    expected, degree_weight = profile[degree]
                    score += weight * degree_weight
                    if family == expected:
                        score += weight * 0.28
                    elif family == "neutral":
                        score += weight * 0.08
                    elif expected != "dim":
                        score -= weight * 0.16
                else:
                    # Secondary dominants and borrowed chords are allowed, but
                    # a key with too many out-of-scale roots should lose.
                    score -= weight * 0.42
            tonic_family = "minor" if mode == "minor" else "major"
            score += weighted_families[key_root][tonic_family] * 0.45
            score += weighted_roots[key_root] * 0.18
            if last_root == key_root:
                score += total * 0.12
            key_name = note_name_for_pc(key_root) + ("m" if mode == "minor" else "")
            candidates.append((score / total, key_name))
    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates[0][1]


def infer_segment(events: list[NoteEvent], start: float, end: float, include_sevenths: bool, threshold: float) -> InferredChord:
    weights, bass, _ = segment_vector(events, start, end)
    if sum(weights) <= 0:
        return InferredChord(start, end, "N", 0.0, bass, "")
    candidates = []
    for name, tones, _ in chord_templates(include_sevenths):
        candidates.append((score_template(weights, tones, bass, name), name, tones))
    candidates.sort(reverse=True, key=lambda item: item[0])
    best_score, best_name, best_tones = candidates[0]
    second_score = candidates[1][0] if len(candidates) > 1 else 0.0
    confidence = max(0.0, min(1.0, (best_score - second_score) * 2.5 + best_score * 0.55))
    evidence = " ".join(f"{note_name_for_pc(pc)}:{weights[pc]:.2f}" for pc in sorted(range(12), key=lambda pc: weights[pc], reverse=True)[:5])
    if confidence < threshold or sum(weights[pc] for pc in best_tones) < 0.48:
        return InferredChord(start, end, "N", confidence, bass, evidence)
    return InferredChord(start, end, best_name, confidence, bass, evidence)


def merge_adjacent(chords: list[InferredChord]) -> list[InferredChord]:
    merged: list[InferredChord] = []
    for item in chords:
        if merged and merged[-1].chord == item.chord and item.chord != "N" and abs(merged[-1].end - item.start) < 0.035:
            previous = merged[-1]
            merged[-1] = InferredChord(
                previous.start,
                item.end,
                previous.chord,
                max(previous.confidence, item.confidence),
                previous.bass or item.bass,
                previous.evidence,
            )
        else:
            merged.append(item)
    return merged


def sequence_metrics(chords: list[InferredChord], total_duration: float) -> dict[str, float | str]:
    non_n = [item for item in chords if item.chord != "N"]
    covered = sum(max(0.0, item.end - item.start) for item in non_n)
    changes = sum(1 for previous, current in zip(non_n, non_n[1:]) if previous.chord != current.chord)
    unique = len({item.chord for item in non_n})
    mean_conf = sum(item.confidence for item in non_n) / max(1, len(non_n))
    coverage = covered / max(1e-9, total_duration)
    change_rate = changes / max(1e-9, total_duration / 60.0)
    return {
        "coverage": coverage,
        "mean_confidence": mean_conf,
        "unique_chords": float(unique),
        "changes_per_minute": change_rate,
        "segments": float(len(non_n)),
    }


def score_sequence(chords: list[InferredChord], total_duration: float, mode: str) -> InferenceCandidate:
    metrics = sequence_metrics(chords, total_duration)
    coverage = float(metrics["coverage"])
    mean_conf = float(metrics["mean_confidence"])
    unique = float(metrics["unique_chords"])
    change_rate = float(metrics["changes_per_minute"])
    segments = float(metrics["segments"])
    score = 0.58 * coverage + 0.35 * mean_conf
    if 3 <= unique <= 28:
        score += 0.08
    elif unique < 2 and total_duration > 45:
        score -= 0.20
    if change_rate > 35:
        score -= min(0.25, (change_rate - 35) / 120)
    if change_rate < 2 and total_duration > 45:
        score -= 0.10
    if segments < 4 and total_duration > 45:
        score -= 0.12
    if mode in {"harmonic", "rhythm_bass", "bass_harmony"}:
        score += 0.03
    if mode == "all":
        score -= 0.02
    diagnostics = dict(metrics)
    diagnostics["mode"] = mode
    diagnostics["score"] = score
    return InferenceCandidate(mode=mode, chords=chords, score=score, diagnostics=diagnostics)


def infer_chords_for_mode(
    song: Any,
    mode: str,
    bars: list[tuple[float, float, str]],
    subdivisions: int,
    include_sevenths: bool,
    threshold: float,
    merge_repeats: bool = True,
) -> list[InferredChord]:
    events = collect_note_events(song, mode)
    inferred = []
    for start, end, _ in bars:
        step = (end - start) / max(1, subdivisions)
        for index in range(max(1, subdivisions)):
            left = start + step * index
            right = start + step * (index + 1)
            inferred.append(infer_segment(events, left, right, include_sevenths, threshold))
    if merge_repeats:
        inferred = merge_adjacent(inferred)
    return [item for item in inferred if item.chord != "N"]


def parse_guitarpro_song(path: Path) -> Any:
    guitarpro = load_guitarpro_module()
    last_exc: Exception | None = None
    for encoding in ("cp1252", "gbk", "gb18030", "shift_jis", "latin1"):
        try:
            return guitarpro.parse(str(path), encoding=encoding)
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue
        except Exception as exc:
            if "UnicodeDecodeError" not in str(exc):
                raise
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    return guitarpro.parse(str(path))


def infer_song(
    path: Path,
    subdivisions: int,
    include_sevenths: bool,
    threshold: float,
    track_mode: str,
    merge_repeats: bool = True,
) -> tuple[dict[str, str], list[InferredChord], list[InferenceCandidate]]:
    song = parse_guitarpro_song(path)
    bars, meter = bar_grid(song)
    total_duration = bars[-1][1] if bars else 0.0
    candidates: list[InferenceCandidate] = []
    for mode in available_track_modes(song, track_mode):
        try:
            chords = infer_chords_for_mode(song, mode, bars, subdivisions, include_sevenths, threshold, merge_repeats)
        except Exception:
            continue
        candidates.append(score_sequence(chords, total_duration, mode))
    if not candidates:
        raise ValueError("No usable non-drum note events were found.")
    candidates.sort(key=lambda item: item.score, reverse=True)
    rhythm = next((candidate for candidate in candidates if candidate.mode == "rhythm_guitar"), None)
    harmonic = next((candidate for candidate in candidates if candidate.mode == "harmonic"), None)
    if track_mode == "auto" and rhythm is not None:
        best = candidates[0]
        rhythm_changes = float(rhythm.diagnostics["changes_per_minute"])
        best_changes = float(best.diagnostics["changes_per_minute"])
        rhythm_coverage = float(rhythm.diagnostics["coverage"])
        best_coverage = float(best.diagnostics["coverage"])
        if best.score > rhythm.score + 0.10 and best_coverage >= rhythm_coverage - 0.04 and best_changes <= rhythm_changes + 6:
            selected = best
        else:
            selected = rhythm
    elif track_mode == "auto" and harmonic is not None:
        # Symbolic confidence can be misleading when a melody-like track is clean.
        # Keep the accompaniment-oriented baseline unless another ensemble view is
        # clearly more complete and no more erratic.
        best = candidates[0]
        harmonic_changes = float(harmonic.diagnostics["changes_per_minute"])
        best_changes = float(best.diagnostics["changes_per_minute"])
        harmonic_coverage = float(harmonic.diagnostics["coverage"])
        best_coverage = float(best.diagnostics["coverage"])
        if best.score > harmonic.score + 0.08 and best_coverage >= harmonic_coverage - 0.05 and best_changes <= harmonic_changes + 8:
            selected = best
        else:
            selected = harmonic
    else:
        selected = candidates[0]
    metadata = {
        "title": str(getattr(song, "title", "") or path.stem),
        "artist": str(getattr(song, "artist", "") or ""),
        "key": estimate_symbolic_key(selected.chords) or first_key(song),
        "meter": meter,
        "tempo": str(song_tempo(song)),
        "track_mode": selected.mode,
        "track_mode_score": f"{selected.score:.4f}",
    }
    return metadata, selected.chords, candidates


def write_outputs(path: Path, metadata: dict[str, str], chords: list[InferredChord], candidates: list[InferenceCandidate], output_root: Path, dataset: str) -> dict[str, str]:
    source_hash = hashlib.sha1(str(path.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:10]
    item_id = safe_id(f"{dataset}_{metadata.get('artist','')}_{metadata.get('title','')}_{path.stem}_{source_hash}")
    item_dir = output_root / item_id
    item_dir.mkdir(parents=True, exist_ok=True)
    lab = item_dir / "inferred_chords.lab"
    lab.write_text(
        "\n".join(f"{item.start:.3f} {item.end:.3f} {item.chord}" for item in chords) + ("\n" if chords else ""),
        encoding="utf-8",
    )
    diag = item_dir / "inferred_diagnostics.csv"
    with diag.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["start", "end", "chord", "confidence", "bass", "evidence"])
        writer.writeheader()
        for item in chords:
            writer.writerow(
                {
                    "start": f"{item.start:.3f}",
                    "end": f"{item.end:.3f}",
                    "chord": item.chord,
                    "confidence": f"{item.confidence:.4f}",
                    "bass": item.bass,
                    "evidence": item.evidence,
                }
            )
    candidate_report = item_dir / "track_mode_candidates.csv"
    with candidate_report.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = ["rank", "mode", "score", "coverage", "mean_confidence", "unique_chords", "changes_per_minute", "segments"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, candidate in enumerate(candidates, start=1):
            row = {"rank": rank}
            row.update(candidate.diagnostics)
            writer.writerow(row)
    return {
        "id": item_id,
        "dataset": dataset,
        "split": "symbolic_inferred",
        "artist": metadata.get("artist", ""),
        "title": metadata.get("title", ""),
        "key": metadata.get("key", ""),
        "meter": metadata.get("meter", ""),
        "audio": "",
        "prediction": "",
        "chord_reference": str(lab.relative_to(REPO_ROOT / "benchmarks")).replace("\\", "/"),
        "beat_reference": "",
        "notes": f"inferred from Guitar Pro note content; track_mode={metadata.get('track_mode', '')}; score={metadata.get('track_mode_score', '')}",
    }


def find_files(input_path: Path, gp5_only: bool = False) -> list[Path]:
    extensions = {".gp5"} if gp5_only else GP_EXTENSIONS
    if input_path.is_file() and input_path.suffix.lower() in extensions:
        return [input_path]
    return sorted(path for path in input_path.rglob("*") if path.suffix.lower() in extensions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--dataset", default="guitarpro_inferred")
    parser.add_argument("-o", "--output-root", type=Path, default=REPO_ROOT / "benchmarks" / "raw" / "guitarpro_inferred")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "benchmarks" / "manifest.guitarpro_inferred.csv")
    parser.add_argument("--subdivisions", type=int, default=1, help="Segments per bar.")
    parser.add_argument("--include-sevenths", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument(
        "--track-mode",
        choices=["auto", "auto_per_track", "rhythm_guitar", "harmonic", "rhythm_bass", "bass_harmony", "all"],
        default="auto",
        help="Track selection strategy. auto evaluates ensemble strategies; auto_per_track also tries individual tracks.",
    )
    parser.add_argument("--gp5-only", action="store_true", help="Only infer from .gp5 files. Useful when .gp files duplicate the same charts.")
    parser.add_argument("--keep-repeats", action="store_true", help="Do not merge adjacent identical labels. Useful for bar/phrase n-gram training.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--error-report", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "guitarpro_infer_errors.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_root = resolve_path(args.output_root)
    rows = []
    errors = []
    for path in find_files(input_path, args.gp5_only):
        try:
            metadata, chords, candidates = infer_song(
                path,
                args.subdivisions,
                args.include_sevenths,
                args.threshold,
                args.track_mode,
                merge_repeats=not args.keep_repeats,
            )
            rows.append(write_outputs(path, metadata, chords, candidates, output_root, args.dataset))
        except Exception as exc:
            if not args.continue_on_error:
                raise
            errors.append({"source_file": str(path), "error_type": type(exc).__name__, "error": str(exc)})
    manifest = resolve_path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "dataset", "split", "artist", "title", "key", "meter", "audio", "prediction", "chord_reference", "beat_reference", "notes"]
    with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if errors:
        error_path = resolve_path(args.error_report)
        error_path.parent.mkdir(parents=True, exist_ok=True)
        with error_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source_file", "error_type", "error"])
            writer.writeheader()
            writer.writerows(errors)
    print(json.dumps({"files": len(rows) + len(errors), "inferred": len(rows), "errors": len(errors), "manifest": str(manifest)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
