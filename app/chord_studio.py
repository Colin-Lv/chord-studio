#!/usr/bin/env python
"""Beat-aligned chord analysis with an HTML playback viewer."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from audio_features import centered_magnitude_spectrum

from chord_recognizer import (
    SAMPLE_RATE,
    build_templates,
    decode_audio,
    estimate_key,
    extract_chroma,
    format_timestamp,
)
from downbeat_bridge import DEFAULT_MADMOM_PYTHON, run_madmom_downbeats


ROOT_TO_PC = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
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
}
ROOT_NAMES = sorted(ROOT_TO_PC, key=len, reverse=True)
PC_TO_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PC_TO_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
MAJOR_ROMAN_BY_SEMITONE = {
    0: "I",
    1: "bII",
    2: "II",
    3: "bIII",
    4: "III",
    5: "IV",
    6: "#IV",
    7: "V",
    8: "bVI",
    9: "VI",
    10: "bVII",
    11: "VII",
}
MINOR_ROMAN_BY_SEMITONE = {
    0: "I",
    1: "bII",
    2: "II",
    3: "III",
    4: "#III",
    5: "IV",
    6: "#IV",
    7: "V",
    8: "VI",
    9: "#VI",
    10: "VII",
    11: "#VII",
}

MINOR_MODE_NAMES = {"minor", "aeolian", "dorian", "phrygian", "locrian"}
MAJOR_MODE_NAMES = {"major", "ionian", "lydian", "mixolydian"}


@dataclass(frozen=True)
class MeterInfo:
    bpm: float
    meter_label: str
    beats_per_bar: int
    beat_unit: int
    beat_period: float
    first_bar_start: float
    confidence: float
    beat_times: tuple[float, ...] = ()
    beat_numbers: tuple[int, ...] = ()
    downbeat_offset: int = 0
    locked_phase: bool = False


@dataclass(frozen=True)
class BeatInterval:
    bar: int
    beat: int
    start: float
    end: float


@dataclass(frozen=True)
class HarmonicInterval:
    bar: int
    beat_start: int
    beat_end: int
    start: float
    end: float


@dataclass(frozen=True)
class BeatChord:
    bar: int
    beat: int
    start: float
    end: float
    chord: str
    confidence: float
    bass_root: str = "N"
    bass_confidence: float = 0.0


@dataclass(frozen=True)
class ChordSpan:
    unit: str
    bar: int
    beat_start: int
    beat_end: int
    start: float
    end: float
    chord: str
    confidence: float
    bass_root: str = "N"
    bass_confidence: float = 0.0


@dataclass(frozen=True)
class MatchResult:
    chords: list[str]
    confidence: np.ndarray
    bass_roots: list[str]
    bass_confidence: np.ndarray


@dataclass(frozen=True)
class FunctionalState:
    label: str
    chord: str
    root_pc: int
    family: str
    bass_pc: int | None = None
    prior: float = 0.0


@dataclass(frozen=True)
class FunctionalSequenceResult:
    score: float
    states: list[FunctionalState]
    path: np.ndarray
    state_scores: np.ndarray


@dataclass(frozen=True)
class ExternalChordHint:
    start: float
    end: float
    chord: str
    confidence: float = 1.0


@dataclass(frozen=True)
class MeterKeySelection:
    meter: MeterInfo
    key: str
    score: float
    tempo_variant: str
    key_candidates: list[tuple[str, str, float]]


def scalar_tempo(value: object) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return 120.0
    tempo = float(array[0])
    if not math.isfinite(tempo) or tempo <= 0:
        return 120.0
    while tempo < 45:
        tempo *= 2.0
    while tempo > 220:
        tempo /= 2.0
    return tempo


def normalize_key_label(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().replace("♭", "b").replace("♯", "#")
    if not text:
        return None
    match = re.fullmatch(r"([A-G][b#]?)(?:\s*:\s*|\s+)([A-Za-z]+)", text)
    if not match:
        return text
    root, mode = match.group(1), match.group(2).lower()
    if mode in MINOR_MODE_NAMES | {"min"}:
        return f"{root}m"
    if mode in MAJOR_MODE_NAMES | {"maj"}:
        return root
    return text


def parse_meter(value: str) -> tuple[int, int] | None:
    if value.lower() == "auto":
        return None
    if "/" not in value:
        raise ValueError("--meter must be auto or something like 4/4, 3/4, 6/8")
    top, bottom = value.split("/", 1)
    beats_per_bar = int(top)
    beat_unit = int(bottom)
    if beats_per_bar < 2 or beats_per_bar > 12 or beat_unit <= 0:
        raise ValueError("--meter is out of supported range")
    return beats_per_bar, beat_unit


def fixed_meter_info(bpm: float, requested: tuple[int, int]) -> MeterInfo:
    beats_per_bar, beat_unit = requested
    beat_period = 60.0 / bpm
    return MeterInfo(
        bpm=bpm,
        meter_label=f"{beats_per_bar}/{beat_unit}",
        beats_per_bar=beats_per_bar,
        beat_unit=beat_unit,
        beat_period=beat_period,
        first_bar_start=0.0,
        confidence=1.0,
        beat_times=(),
        beat_numbers=(),
        downbeat_offset=0,
        locked_phase=False,
    )


def display_bpm_for_meter(meter: MeterInfo) -> float:
    if meter.beats_per_bar == 6 and meter.beat_unit == 8 and meter.bpm > 120:
        return meter.bpm / 3.0
    return meter.bpm


def beat_strengths_from_onsets(onset_envelope: np.ndarray, beat_frames: np.ndarray) -> np.ndarray:
    strengths = []
    if onset_envelope.size == 0:
        return np.ones(max(1, len(beat_frames)), dtype=np.float32)
    last = len(onset_envelope) - 1
    for frame in beat_frames:
        center = int(np.clip(frame, 0, last))
        left = max(0, center - 2)
        right = min(last + 1, center + 3)
        strengths.append(float(np.max(onset_envelope[left:right])))
    values = np.asarray(strengths, dtype=np.float32)
    if values.size == 0:
        return np.ones(1, dtype=np.float32)
    return values


def meter_from_downbeat_rows(
    rows: list[tuple[float, int]],
    duration: float,
    requested: tuple[int, int] | None,
) -> MeterInfo | None:
    clean_rows = sorted(
        (float(time), int(beat))
        for time, beat in rows
        if 0.0 <= float(time) <= duration and int(beat) > 0
    )
    if len(clean_rows) < 4:
        return None

    if requested is not None:
        beats_per_bar, beat_unit = requested
    else:
        max_beat = max(beat for _, beat in clean_rows)
        beats_per_bar = max_beat if max_beat in (3, 4, 6) else 4
        beat_unit = 8 if beats_per_bar == 6 else 4

    rows_for_meter = [(time, beat) for time, beat in clean_rows if 1 <= beat <= beats_per_bar]
    if len(rows_for_meter) < 4:
        return None

    beat_times = np.asarray([time for time, _ in rows_for_meter], dtype=np.float32)
    diffs = np.diff(beat_times)
    diffs = diffs[(diffs > 0.12) & (diffs < 2.0)]
    if diffs.size == 0:
        return None

    median_period = float(np.median(diffs))
    bpm = scalar_tempo(60.0 / median_period)
    first_time, first_beat = rows_for_meter[0]
    downbeat_offset = int((1 - first_beat) % beats_per_bar)
    downbeat_times = [time for time, beat in rows_for_meter if beat == 1]
    bar_duration = median_period * beats_per_bar
    first_bar_start = float(downbeat_times[0]) if downbeat_times else float(first_time)
    while first_bar_start > 0.05:
        first_bar_start -= bar_duration
    while first_bar_start <= -bar_duration:
        first_bar_start += bar_duration

    return MeterInfo(
        bpm=bpm,
        meter_label=f"{beats_per_bar}/{beat_unit}",
        beats_per_bar=beats_per_bar,
        beat_unit=beat_unit,
        beat_period=median_period,
        first_bar_start=first_bar_start,
        confidence=0.95,
        beat_times=tuple(float(time) for time in beat_times),
        beat_numbers=tuple(int(beat) for _, beat in rows_for_meter),
        downbeat_offset=downbeat_offset,
        locked_phase=True,
    )


def estimate_meter_madmom(
    audio_path: Path,
    duration: float,
    meter_arg: str,
    madmom_python: Path,
    audio_offset: float = 0.0,
) -> MeterInfo | None:
    requested = parse_meter(meter_arg)
    candidates = [requested[0]] if requested is not None else [3, 4, 6]
    rows = run_madmom_downbeats(audio_path, candidates, madmom_python=madmom_python)
    rows = [(time - audio_offset, beat) for time, beat in rows]
    return meter_from_downbeat_rows(rows, duration, requested)


def read_downbeat_rows(path: Path) -> list[tuple[float, int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("beats", raw) if isinstance(raw, dict) else raw
    return [(float(row["time"]), int(row["beat"])) for row in rows]


def read_beat_grid_rows(path: Path) -> list[tuple[float, int]]:
    if path.suffix.lower() == ".json":
        return read_downbeat_rows(path)
    rows: list[tuple[float, int]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if not parts:
            continue
        beat = int(float(parts[1])) if len(parts) > 1 else 1
        rows.append((float(parts[0]), beat))
    return rows


def choose_meter(
    beat_times: np.ndarray,
    strengths: np.ndarray,
    requested: tuple[int, int] | None,
) -> tuple[int, int, int, float]:
    if requested is not None:
        return requested[0], requested[1], 0, 1.0
    if beat_times.size < 8 or strengths.size < 8:
        return 4, 4, 0, 0.0

    strengths = strengths[: beat_times.size]
    spread = float(np.std(strengths)) + 1e-6
    candidates: list[tuple[float, int, int, int]] = []
    for beats_per_bar, beat_unit in ((4, 4), (3, 4), (6, 8)):
        if strengths.size < beats_per_bar * 2:
            continue
        for offset in range(beats_per_bar):
            downbeats = strengths[offset::beats_per_bar]
            others = np.asarray(
                [
                    value
                    for index, value in enumerate(strengths)
                    if index % beats_per_bar != offset
                ],
                dtype=np.float32,
            )
            if downbeats.size == 0 or others.size == 0:
                continue
            contrast = (float(np.mean(downbeats)) - float(np.mean(others))) / spread
            prior = 0.16 if beats_per_bar == 4 else (-0.03 if beats_per_bar == 6 else 0.0)
            candidates.append((contrast + prior, beats_per_bar, beat_unit, offset))

    if not candidates:
        return 4, 4, 0, 0.0

    best = max(candidates, key=lambda item: item[0])
    best_44 = max(
        (item for item in candidates if item[1] == 4),
        key=lambda item: item[0],
        default=None,
    )
    if best_44 and best[1] != 4 and best[0] < best_44[0] + 0.24:
        best = best_44

    confidence = float(np.clip((best[0] + 0.5) / 2.5, 0.0, 1.0))
    return best[1], best[2], best[3], confidence


def estimate_meter(y: np.ndarray, sr: int, duration: float, meter_arg: str) -> MeterInfo:
    import librosa

    hop_length = 512
    onset_envelope = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length, aggregate=np.median
    )
    tempo_value, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_envelope,
        sr=sr,
        hop_length=hop_length,
        units="frames",
        trim=False,
    )
    bpm = scalar_tempo(tempo_value)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    beat_times = np.asarray(beat_times, dtype=np.float32)
    if beat_times.size >= 4:
        diffs = np.diff(beat_times)
        diffs = diffs[(diffs > 0.25) & (diffs < 1.5)]
        if diffs.size:
            median_period = float(np.median(diffs))
            if median_period > 0:
                bpm = scalar_tempo(60.0 / median_period)

    requested = parse_meter(meter_arg)
    strengths = beat_strengths_from_onsets(onset_envelope, beat_frames)
    beats_per_bar, beat_unit, downbeat_offset, confidence = choose_meter(
        beat_times, strengths, requested
    )

    beat_period = 60.0 / bpm
    bar_duration = beat_period * beats_per_bar
    if beat_times.size > downbeat_offset:
        first_bar_start = float(beat_times[downbeat_offset])
    else:
        first_bar_start = 0.0
    while first_bar_start > 0.05:
        first_bar_start -= bar_duration
    while first_bar_start <= -bar_duration:
        first_bar_start += bar_duration
    if requested is not None:
        first_bar_start = 0.0
        downbeat_offset = 0
    elif confidence < 0.65 or abs(first_bar_start) < beat_period * 0.25:
        first_bar_start = 0.0
        downbeat_offset = 0

    return MeterInfo(
        bpm=bpm,
        meter_label=f"{beats_per_bar}/{beat_unit}",
        beats_per_bar=beats_per_bar,
        beat_unit=beat_unit,
        beat_period=beat_period,
        first_bar_start=first_bar_start,
        confidence=confidence,
        beat_times=tuple(float(time) for time in beat_times if 0.0 <= float(time) <= duration),
        beat_numbers=(),
        downbeat_offset=downbeat_offset,
    )


def build_constant_beat_intervals(meter: MeterInfo, duration: float) -> list[BeatInterval]:
    intervals: list[BeatInterval] = []
    bar_duration = meter.beat_period * meter.beats_per_bar
    bar_start = meter.first_bar_start
    bar_number = 1
    while bar_start < duration:
        for beat_index in range(meter.beats_per_bar):
            raw_start = bar_start + beat_index * meter.beat_period
            raw_end = raw_start + meter.beat_period
            start = max(0.0, raw_start)
            end = min(duration, raw_end)
            if end - start >= 0.05:
                intervals.append(
                    BeatInterval(
                        bar=bar_number,
                        beat=beat_index + 1,
                        start=start,
                        end=end,
                    )
                )
        bar_start += bar_duration
        bar_number += 1
    return intervals


def build_beat_intervals(meter: MeterInfo, duration: float) -> list[BeatInterval]:
    if len(meter.beat_times) < 2:
        return build_constant_beat_intervals(meter, duration)

    if len(meter.beat_numbers) == len(meter.beat_times):
        intervals: list[BeatInterval] = []
        current_bar = 1
        previous_beat: int | None = None
        times = list(meter.beat_times)
        numbers = list(meter.beat_numbers)
        for index, (raw_start, beat_number) in enumerate(zip(times, numbers)):
            if raw_start >= duration:
                break
            raw_end = times[index + 1] if index + 1 < len(times) else raw_start + meter.beat_period
            if raw_end <= 0:
                continue
            start = max(0.0, raw_start)
            end = min(duration, raw_end)
            if end - start < 0.05:
                continue
            if beat_number == 1 and previous_beat is not None:
                current_bar += 1
            intervals.append(
                BeatInterval(
                    bar=current_bar,
                    beat=beat_number,
                    start=start,
                    end=end,
                )
            )
            previous_beat = beat_number
        if intervals:
            return intervals

    intervals: list[BeatInterval] = []
    period = meter.beat_period
    boundaries = list(meter.beat_times)
    prepend_count = 0

    previous = boundaries[0] - period
    while previous > -period:
        boundaries.insert(0, previous)
        prepend_count += 1
        previous -= period

    next_time = boundaries[-1] + period
    while next_time < duration + period:
        boundaries.append(next_time)
        next_time += period

    downbeat_index = prepend_count + meter.downbeat_offset
    raw_intervals: list[BeatInterval] = []
    for index in range(len(boundaries) - 1):
        raw_start = boundaries[index]
        raw_end = boundaries[index + 1]
        if raw_end <= 0:
            continue
        if raw_start >= duration:
            break
        start = max(0.0, raw_start)
        end = min(duration, raw_end)
        if end - start < 0.05:
            continue

        relative = index - downbeat_index
        beat_number = relative % meter.beats_per_bar + 1
        bar_number = math.floor(relative / meter.beats_per_bar) + 1
        raw_intervals.append(
            BeatInterval(
                bar=bar_number,
                beat=beat_number,
                start=start,
                end=end,
            )
        )

    if not raw_intervals:
        return build_constant_beat_intervals(meter, duration)

    if raw_intervals[0].start > 0.0 and raw_intervals[0].start < period * 0.55:
        first = raw_intervals[0]
        raw_intervals[0] = BeatInterval(
            bar=first.bar,
            beat=first.beat,
            start=0.0,
            end=first.end,
        )

    if (
        len(raw_intervals) > 1
        and raw_intervals[0].start == 0.0
        and raw_intervals[0].end - raw_intervals[0].start < period * 0.85
        and raw_intervals[0].beat != 1
    ):
        first_full = raw_intervals[1]
        raw_intervals[1] = BeatInterval(
            bar=first_full.bar,
            beat=first_full.beat,
            start=0.0,
            end=first_full.end,
        )
        raw_intervals = raw_intervals[1:]

    bar_shift = 1 - min(item.bar for item in raw_intervals)
    for item in raw_intervals:
        intervals.append(
            BeatInterval(
                bar=item.bar + bar_shift,
                beat=item.beat,
                start=item.start,
                end=item.end,
            )
        )
    return intervals


def meter_variant(meter: MeterInfo, variant: str) -> MeterInfo:
    if variant == "detected":
        return meter
    if variant.startswith("half"):
        parity = 0 if variant == "half0" else 1
        beat_times = tuple(meter.beat_times[parity::2])
        return MeterInfo(
            bpm=meter.bpm / 2.0,
            meter_label=meter.meter_label,
            beats_per_bar=meter.beats_per_bar,
            beat_unit=meter.beat_unit,
            beat_period=meter.beat_period * 2.0,
            first_bar_start=0.0,
            confidence=meter.confidence,
            beat_times=beat_times,
            downbeat_offset=0,
            locked_phase=False,
        )
    if variant == "double":
        expanded: list[float] = []
        beat_times = list(meter.beat_times)
        for index, time in enumerate(beat_times):
            expanded.append(time)
            if index + 1 < len(beat_times):
                expanded.append((time + beat_times[index + 1]) / 2.0)
        return MeterInfo(
            bpm=meter.bpm * 2.0,
            meter_label=meter.meter_label,
            beats_per_bar=meter.beats_per_bar,
            beat_unit=meter.beat_unit,
            beat_period=meter.beat_period / 2.0,
            first_bar_start=0.0,
            confidence=meter.confidence,
            beat_times=tuple(expanded),
            downbeat_offset=0,
            locked_phase=False,
        )
    return meter


def tempo_variants(meter: MeterInfo, mode: str) -> list[tuple[str, MeterInfo]]:
    if mode == "detected":
        return [("detected", meter)]
    if mode == "auto" and meter.locked_phase:
        return [("detected", meter)]
    if mode == "half":
        return [("half0", meter_variant(meter, "half0")), ("half1", meter_variant(meter, "half1"))]
    if mode == "double":
        return [("double", meter_variant(meter, "double"))]

    variants = [("detected", meter)]
    if meter.bpm >= 105 and len(meter.beat_times) >= 8:
        variants.extend(
            [
                ("half0", meter_variant(meter, "half0")),
                ("half1", meter_variant(meter, "half1")),
            ]
        )
    if meter.bpm <= 85 and len(meter.beat_times) >= 4:
        variants.append(("double", meter_variant(meter, "double")))
    return variants


def meter_phase_variants(meter: MeterInfo) -> list[tuple[str, MeterInfo]]:
    if meter.locked_phase:
        return [("p0", meter)]
    if len(meter.beat_times) < meter.beats_per_bar * 2:
        return [("p0", meter)]
    return [
        (
            f"p{phase}",
            replace(
                meter,
                first_bar_start=0.0,
                downbeat_offset=phase,
            ),
        )
        for phase in range(meter.beats_per_bar)
    ]


def bar_intervals_from_beats(intervals: list[BeatInterval]) -> list[BeatInterval]:
    bar_intervals = []
    for bar in sorted({item.bar for item in intervals}):
        items = [item for item in intervals if item.bar == bar]
        if not items:
            continue
        bar_intervals.append(
            BeatInterval(
                bar=bar,
                beat=1,
                start=items[0].start,
                end=items[-1].end,
            )
        )
    return bar_intervals


def harmonic_intervals_from_beats(
    intervals: list[BeatInterval],
    unit: str,
    beats_per_bar: int,
) -> list[HarmonicInterval]:
    cells: list[HarmonicInterval] = []
    for bar in sorted({item.bar for item in intervals}):
        beats = [item for item in intervals if item.bar == bar]
        if not beats:
            continue
        if unit == "beat":
            group_size = 1
        elif unit == "compound-half" and beats_per_bar == 6:
            group_size = 3
        elif unit == "half-bar":
            group_size = max(1, beats_per_bar // 2)
        else:
            group_size = len(beats)
        for start_index in range(0, len(beats), group_size):
            group = beats[start_index : start_index + group_size]
            if not group:
                continue
            cells.append(
                HarmonicInterval(
                    bar=bar,
                    beat_start=group[0].beat,
                    beat_end=group[-1].beat,
                    start=group[0].start,
                    end=group[-1].end,
                )
            )
    return cells


def harmonic_unit_for_meter(meter: MeterInfo, requested: str) -> str:
    if requested != "auto":
        return requested
    if meter.beats_per_bar == 6:
        return "bar"
    if meter.bpm < 85 and meter.beats_per_bar >= 4:
        return "half-bar"
    return "bar"


def leading_partial_bar_penalty(intervals: list[BeatInterval], beats_per_bar: int) -> float:
    if not intervals:
        return 0.0
    first_bar = min(item.bar for item in intervals)
    first_beats = [item.beat for item in intervals if item.bar == first_bar]
    if not first_beats:
        return 0.0
    missing = max(0, beats_per_bar - len(set(first_beats)))
    penalty = missing * 0.025
    if min(first_beats) != 1:
        penalty += 0.04
    return penalty


def chroma_for_intervals(
    chroma: np.ndarray,
    y: np.ndarray,
    sr: int,
    hop_length: int,
    intervals: list[BeatInterval],
) -> np.ndarray:
    frame_times = np.arange(chroma.shape[1], dtype=np.float32) * (hop_length / sr)
    vectors = []
    for interval in intervals:
        mask = (frame_times >= interval.start) & (frame_times < interval.end)
        if not np.any(mask):
            frame_idx = int(np.argmin(np.abs(frame_times - interval.start)))
            values = chroma[:, frame_idx : frame_idx + 1]
        else:
            values = chroma[:, mask]
        median = np.median(values, axis=1)
        upper = np.quantile(values, 0.75, axis=1)
        vector = 0.72 * median + 0.28 * upper
        noise_floor = float(np.quantile(vector, 0.18))
        vector = np.maximum(vector - 0.35 * noise_floor, 0.0)
        vector = np.power(vector, 1.15)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        vectors.append(vector)
    return np.asarray(vectors, dtype=np.float32)


def rms_for_intervals(y: np.ndarray, sr: int, intervals: list[BeatInterval]) -> np.ndarray:
    values = []
    for interval in intervals:
        left = max(0, int(interval.start * sr))
        right = min(len(y), int(interval.end * sr))
        if right <= left:
            values.append(0.0)
            continue
        chunk = y[left:right]
        values.append(float(np.sqrt(np.mean(chunk * chunk))))
    return np.asarray(values, dtype=np.float32)


def extract_bass_chroma(y: np.ndarray, sr: int, hop_length: int, mode: str = "cqt") -> np.ndarray:
    if mode == "fast":
        n_fft = 4096
        magnitude = centered_magnitude_spectrum(y, n_fft, hop_length)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        mask = (freqs >= 32.703) & (freqs <= 246.942)
        magnitude = magnitude[mask]
        freqs = freqs[mask]
        bass = np.zeros((12, magnitude.shape[1]), dtype=np.float32)
        if magnitude.size == 0:
            return bass
        midi = np.rint(69.0 + 12.0 * np.log2(np.maximum(freqs, 1e-6) / 440.0)).astype(int)
        weights = 1.0 / np.maximum(freqs / max(float(freqs[0]), 1.0), 1.0)
        for bin_index, pc in enumerate(midi % 12):
            bass[int(pc)] += magnitude[bin_index] * weights[bin_index]
        bass = np.maximum(bass, 0.0)
        bass /= np.max(bass, axis=0, keepdims=True) + 1e-9
        return bass
    if mode != "cqt":
        raise ValueError("mode must be 'cqt' or 'fast'")

    import librosa

    y = librosa.util.normalize(y)
    y_harmonic = librosa.effects.harmonic(y, margin=8.0)
    cqt = np.abs(
        librosa.cqt(
            y=y_harmonic,
            sr=sr,
            hop_length=hop_length,
            fmin=librosa.note_to_hz("C1"),
            n_bins=36,
            bins_per_octave=12,
        )
    )
    octave_weights = np.repeat(np.array([1.0, 0.78, 0.52], dtype=np.float32), 12)
    cqt = cqt * octave_weights[:, None]
    bass = np.zeros((12, cqt.shape[1]), dtype=np.float32)
    for bin_index in range(cqt.shape[0]):
        bass[bin_index % 12] += cqt[bin_index]
    bass = np.maximum(bass, 0.0)
    bass /= np.max(bass, axis=0, keepdims=True) + 1e-9
    return bass


def bass_for_intervals(
    bass_chroma: np.ndarray,
    sr: int,
    hop_length: int,
    intervals: list[BeatInterval],
    notation: str,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    notes = PC_TO_FLAT if notation == "flats" else PC_TO_SHARP
    frame_times = np.arange(bass_chroma.shape[1], dtype=np.float32) * (hop_length / sr)
    vectors = []
    names = []
    confidence = []
    for interval in intervals:
        mask = (frame_times >= interval.start) & (frame_times < interval.end)
        if not np.any(mask):
            frame_idx = int(np.argmin(np.abs(frame_times - interval.start)))
            values = bass_chroma[:, frame_idx : frame_idx + 1]
        else:
            values = bass_chroma[:, mask]
        vector = np.median(values, axis=1).astype(np.float32)
        total = float(np.sum(vector))
        if total <= 1e-9:
            vectors.append(np.zeros(12, dtype=np.float32))
            names.append("N")
            confidence.append(0.0)
            continue
        vector = vector / total
        order = np.argsort(vector)
        best = int(order[-1])
        second = int(order[-2])
        conf = float(np.clip((vector[best] - vector[second]) * 3.0 + vector[best], 0.0, 1.0))
        vectors.append(vector)
        names.append(notes[best])
        confidence.append(conf)
    return np.asarray(vectors, dtype=np.float32), names, np.asarray(confidence, dtype=np.float32)


def estimate_mode_from_key(key: str) -> str:
    if key == "Unknown":
        return "Unknown"
    return "minor" if key.endswith("m") else "major"


def parse_chord_name(chord: str) -> tuple[int | None, str]:
    if chord == "N":
        return None, ""
    chord = chord.split("/", 1)[0]
    for root in ROOT_NAMES:
        if chord.startswith(root):
            return ROOT_TO_PC[root], chord[len(root) :]
    return None, ""


def note_name_for_pc(pc: int, key: str | None = None) -> str:
    if key and "b" in key:
        return PC_TO_FLAT[pc % 12]
    return PC_TO_SHARP[pc % 12]


def chord_member_pcs(chord: str) -> set[int]:
    root, suffix = parse_chord_name(chord)
    if root is None:
        return set()
    # Keep the explicit vocabulary consistent with build_templates. A diminished
    # seventh is 9 semitones, minor-major seventh is 11, and add9 has no seventh.
    exact_intervals = {
        "": (0, 4, 7), "m": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
        "sus2": (0, 2, 7), "sus4": (0, 5, 7),
        "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10),
        "m7b5": (0, 3, 6, 10), "dim7": (0, 3, 6, 9), "mMaj7": (0, 3, 7, 11),
        "6": (0, 4, 7, 9), "m6": (0, 3, 7, 9),
        "add9": (0, 2, 4, 7), "madd9": (0, 2, 3, 7),
        "9": (0, 2, 4, 7, 10), "m9": (0, 2, 3, 7, 10),
        "maj9": (0, 2, 4, 7, 11), "mMaj9": (0, 2, 3, 7, 11),
    }
    if suffix in exact_intervals:
        return {(root + interval) % 12 for interval in exact_intervals[suffix]}
    family = chord_quality_family(suffix)
    if family == "m":
        intervals = [0, 3, 7]
        if "6" in suffix:
            intervals.append(9)
        if "7" in suffix or "9" in suffix:
            intervals.append(10)
        if "9" in suffix or "add9" in suffix:
            intervals.append(2)
    elif family == "dim":
        intervals = [0, 3, 6, 10] if "7" in suffix else [0, 3, 6]
    elif family == "sus":
        intervals = [0, 2 if "sus2" in suffix else 5, 7]
        if "7" in suffix or "9" in suffix:
            intervals.append(10)
        if "9" in suffix:
            intervals.append(2)
    elif family == "dom":
        intervals = [0, 4, 7, 10]
        if "9" in suffix:
            intervals.append(2)
    elif family == "aug":
        intervals = [0, 4, 8]
    else:
        intervals = [0, 4, 7]
        if "6" in suffix:
            intervals.append(9)
        if "maj7" in suffix:
            intervals.append(11)
        if "9" in suffix or "add9" in suffix:
            intervals.append(2)
    return {(root + interval) % 12 for interval in intervals}


def chord_interval_pcs(chord: str) -> tuple[int | None, int | None, int | None]:
    root, suffix = parse_chord_name(chord)
    if root is None:
        return None, None, None
    family = chord_quality_family(suffix)
    if family in {"m", "dim"}:
        third = (root + 3) % 12
    elif family == "sus":
        third = None
    else:
        third = (root + 4) % 12
    fifth_interval = 6 if family == "dim" else 8 if family == "aug" else 7
    fifth = (root + fifth_interval) % 12
    return root, third, fifth


def should_show_slash_bass(
    chord: str,
    bass_root: str,
    bass_confidence: float,
    chord_confidence: float = 1.0,
) -> bool:
    # An explicit inversion/pedal belongs to the source label. A separate
    # template bass estimate must never overwrite it or create C/E/G.
    if "/" in chord:
        return False
    chord_root, _ = parse_chord_name(chord)
    bass_pc, _ = parse_chord_name(bass_root)
    if chord == "N" or chord_root is None or bass_pc is None:
        return False
    if bass_pc == chord_root or bass_confidence < 0.68:
        return False
    if chord_confidence < 0.36:
        return False
    bass_interval = (bass_pc - chord_root) % 12
    if bass_interval == 7 and chord_confidence < 0.62:
        return False
    return bass_pc in chord_member_pcs(chord)


def chord_display_name(
    chord: str,
    bass_root: str = "N",
    bass_confidence: float = 0.0,
    chord_confidence: float = 1.0,
) -> str:
    if should_show_slash_bass(chord, bass_root, bass_confidence, chord_confidence):
        return f"{chord}/{bass_root}"
    return chord


def chord_to_roman(chord: str, key: str) -> str:
    """Convert a chord name to a scale-degree label in the current key.

    The display style follows the user's requested form: uppercase Roman
    degree plus the original chord suffix, e.g. Bm7 in D major -> VIm7.
    """
    if chord == "N":
        return "N"
    chord_root, chord_suffix = parse_chord_name(chord)
    key_root, key_suffix = parse_chord_name(key)
    if chord_root is None or key_root is None:
        return chord
    semitone = (chord_root - key_root) % 12
    roman_map = MINOR_ROMAN_BY_SEMITONE if key_suffix == "m" else MAJOR_ROMAN_BY_SEMITONE
    label = f"{roman_map[semitone]}{chord_suffix}"
    if "/" in chord:
        bass_pc, _ = parse_chord_name(chord.split("/", 1)[1])
        if bass_pc is not None:
            label += "/" + roman_map[(bass_pc - key_root) % 12]
    return label


def chord_to_roman_display(
    chord: str,
    key: str,
    bass_root: str = "N",
    bass_confidence: float = 0.0,
    chord_confidence: float = 1.0,
) -> str:
    label = chord_to_roman(chord, key)
    if not should_show_slash_bass(chord, bass_root, bass_confidence, chord_confidence):
        return label
    bass_pc, _ = parse_chord_name(bass_root)
    key_root, key_suffix = parse_chord_name(key)
    if bass_pc is None or key_root is None:
        return label
    roman_map = MINOR_ROMAN_BY_SEMITONE if key_suffix == "m" else MAJOR_ROMAN_BY_SEMITONE
    bass_degree = roman_map[(bass_pc - key_root) % 12]
    return f"{label}/{bass_degree}"


def chord_label(
    chord: str,
    key: str,
    display: str,
    bass_root: str = "N",
    bass_confidence: float = 0.0,
    chord_confidence: float = 1.0,
) -> str:
    if display == "roman":
        return chord_to_roman_display(chord, key, bass_root, bass_confidence, chord_confidence)
    return chord_display_name(chord, bass_root, bass_confidence, chord_confidence)


def diatonic_profile(key: str) -> dict[int, str]:
    root_pc, suffix = parse_chord_name(key)
    if root_pc is None:
        return {}
    if suffix == "m":
        intervals = [0, 2, 3, 5, 7, 8, 10]
        qualities = ["m", "dim", "", "m", "m", "", ""]
    else:
        intervals = [0, 2, 4, 5, 7, 9, 11]
        qualities = ["", "m", "m", "", "", "m", "dim"]
    return {(root_pc + interval) % 12: quality for interval, quality in zip(intervals, qualities)}


def chord_quality_family(suffix: str) -> str:
    if suffix in {"", "maj7", "maj9", "6", "add9"}:
        return ""
    if suffix in {"m", "m7", "m6", "m9", "madd9", "mMaj7", "mMaj9"}:
        return "m"
    if suffix in {"dim", "dim7", "m7b5"}:
        return "dim"
    if suffix.startswith("sus") or "sus" in suffix:
        return "sus"
    if suffix in {"7", "9"}:
        return "dom"
    if suffix == "aug":
        return "aug"
    return suffix


def template_biases(template_names: list[str], key: str) -> np.ndarray:
    profile = diatonic_profile(key)
    biases = []
    for name in template_names:
        root_pc, suffix = parse_chord_name(name)
        family = chord_quality_family(suffix)
        bias = 0.0
        if root_pc is not None and root_pc in profile:
            expected = profile[root_pc]
            if family == expected:
                bias += 0.09
            elif family in {"sus", "dom"} and expected == "":
                bias += 0.025
            else:
                bias -= 0.045
        elif root_pc is not None:
            bias -= 0.045
        if family in {"dim", "aug"}:
            bias -= 0.035
        if family == "sus":
            bias -= 0.015
        if suffix in {"m7b5", "dim7", "mMaj7"}:
            bias -= 0.025
        biases.append(bias)
    return np.asarray(biases, dtype=np.float32)


@dataclass(frozen=True)
class ProgressionHint:
    root_pc: int
    family: str
    bass_pc: int | None = None


def parse_progression_hints(value: str | None, key: str) -> list[ProgressionHint]:
    if not value:
        return []
    key_root, key_suffix = parse_chord_name(key)
    if key_root is None:
        return []

    if key_suffix == "m":
        degree_offsets = {1: 0, 2: 2, 3: 3, 4: 5, 5: 7, 6: 8, 7: 10}
        degree_families = {1: "m", 2: "dim", 3: "", 4: "m", 5: "m", 6: "", 7: ""}
    else:
        degree_offsets = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}
        degree_families = {1: "", 2: "m", 3: "m", 4: "", 5: "", 6: "m", 7: "dim"}

    hints: list[ProgressionHint] = []
    for digit in value:
        if digit not in "1234567":
            continue
        degree = int(digit)
        if key_suffix != "m" and degree == 7:
            # In pop number notation, 1-7-6 commonly means I - V/7 - vi:
            # the harmony is dominant, while the bass walks through scale degree 7.
            hints.append(
                ProgressionHint(
                    root_pc=(key_root + degree_offsets[5]) % 12,
                    family=degree_families[5],
                    bass_pc=(key_root + degree_offsets[7]) % 12,
                )
            )
            continue
        hints.append(
            ProgressionHint(
                root_pc=(key_root + degree_offsets[degree]) % 12,
                family=degree_families[degree],
                bass_pc=(key_root + degree_offsets[degree]) % 12,
            )
        )
    return hints


def progression_template_biases(
    template_names: list[str],
    key: str,
    progressions: str | None,
    strength: float,
) -> np.ndarray:
    if not progressions or strength <= 0:
        return np.zeros(len(template_names), dtype=np.float32)

    hints = parse_progression_hints(progressions, key)
    if not hints:
        return np.zeros(len(template_names), dtype=np.float32)

    allowed_roots = {hint.root_pc for hint in hints}
    allowed_pairs = {(hint.root_pc, hint.family) for hint in hints}
    biases = []
    for name in template_names:
        root_pc, suffix = parse_chord_name(name)
        family = chord_quality_family(suffix)
        bias = 0.0
        if root_pc is None:
            biases.append(0.0)
            continue
        if (root_pc, family) in allowed_pairs:
            bias += 0.34 * strength
        elif root_pc in allowed_roots and family in {"sus", "dom", ""}:
            bias += 0.16 * strength
        elif root_pc in allowed_roots:
            bias += 0.04 * strength
        else:
            bias -= 0.30 * strength
        if family in {"dim", "aug"} and (root_pc, family) not in allowed_pairs:
            bias -= 0.18 * strength
        biases.append(bias)
    return np.asarray(biases, dtype=np.float32)


def note_name(pc: int, notation: str) -> str:
    return (PC_TO_FLAT if notation == "flats" else PC_TO_SHARP)[pc % 12]


def parse_time_seconds(value: object) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError("empty time value")
    if ":" not in text:
        return float(text)
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60.0 + float(part)
    return seconds


def normalize_external_chord_label(label: object, notation: str) -> str:
    """Normalize spelling while retaining the source quality and explicit bass.

    Harte bass degrees are relative to the chord root, not the song key. See
    https://mir-eval.readthedocs.io/latest/api/chord.html#mir_eval.chord.split
    Unknown or malformed labels must not become evidence for silence (N).
    """
    text = str(label).strip().strip('"').strip("'")
    if not text:
        return "X"
    lowered = text.lower()
    if lowered in {"x", "unknown", "unrecognized"}:
        return "X"
    if lowered in {"n", "nc", "n.c.", "none", "no_chord", "no-chord", "silence"}:
        return "N"

    text = (
        text.replace("♯", "#")
        .replace("♭", "b")
        .replace("Δ", "maj")
        .replace("−", "-")
    )
    body, slash, bass = text.partition("/")
    words = body.split()
    if len(words) >= 2 and words[1].lower() in {"major", "maj", "minor", "min"}:
        body = f"{words[0]}:{words[1]}" + "".join(words[2:])
    body = body.replace(" ", "")
    match = re.fullmatch(r"([A-G])([b#]*)(.*)", body)
    if match is None:
        return "X"
    letter, accidentals, suffix = match.groups()
    root_pc = (ROOT_TO_PC[letter] + accidentals.count("#") - accidentals.count("b")) % 12

    if suffix.startswith(":"):
        quality = suffix[1:]
    else:
        quality = suffix
    quality = quality.strip()
    quality_lower = quality.lower()
    quality_map = {
        "": "",
        "maj": "",
        "major": "",
        "m": "m",
        "min": "m",
        "minor": "m",
        "-": "m",
        "maj7": "maj7",
        "major7": "maj7",
        "m7": "m7",
        "min7": "m7",
        "minor7": "m7",
        "-7": "m7",
        "7": "7",
        "dom7": "7",
        "9": "9",
        "6": "6",
        "maj6": "6",
        "m6": "m6",
        "min6": "m6",
        "dim": "dim",
        "o": "dim",
        "dim7": "dim7",
        "o7": "dim7",
        "hdim7": "m7b5",
        "ø": "m7b5",
        "ø7": "m7b5",
        "m7b5": "m7b5",
        "min7b5": "m7b5",
        "aug": "aug",
        "+": "aug",
        "sus": "sus4",
        "sus4": "sus4",
        "sus2": "sus2",
        "add9": "add9",
        "madd9": "madd9",
        "minmaj7": "mMaj7",
        "mmaj7": "mMaj7",
        "minmaj9": "mMaj9",
        "mmaj9": "mMaj9",
    }
    # Uppercase M means major; lowercasing it first changes CM7 into Cm7.
    major_aliases = {"M": "", "M7": "maj7", "M9": "maj9", "M11": "maj11", "M13": "maj13"}
    normalized_suffix = major_aliases.get(quality, quality_map.get(quality_lower))
    if normalized_suffix is None:
        if "(" in quality:
            # Retain Harte additions/omissions rather than reducing to a triad.
            # In particular, min(*5) must not become the unparseable m(*5).
            normalized_suffix = quality
        elif quality_lower.startswith("min"):
            normalized_suffix = "m" + quality_lower[3:]
        elif quality_lower.startswith("maj"):
            normalized_suffix = "maj" + quality_lower[3:]
            if normalized_suffix == "maj":
                normalized_suffix = ""
        else:
            normalized_suffix = quality.replace(":", "")

    normalized = f"{note_name(root_pc, notation)}{normalized_suffix}"
    if not slash:
        return normalized

    bass = bass.strip()
    absolute_bass = re.fullmatch(r"([A-G])([b#]*)", bass)
    relative_bass = re.fullmatch(r"([b#]*)([1-9][0-9]*)", bass)
    if absolute_bass:
        letter, accidentals = absolute_bass.groups()
        bass_pc = ROOT_TO_PC[letter] + accidentals.count("#") - accidentals.count("b")
    elif relative_bass:
        accidentals, degree = relative_bass.groups()
        interval = (0, 2, 4, 5, 7, 9, 11)[(int(degree) - 1) % 7]
        bass_pc = root_pc + interval + accidentals.count("#") - accidentals.count("b")
    else:
        return "X"
    return f"{normalized}/{note_name(bass_pc % 12, notation)}"


def external_hint_from_mapping(row: dict[str, object], notation: str) -> ExternalChordHint | None:
    start_value = (
        row.get("start")
        if row.get("start") not in {None, ""}
        else row.get("start_seconds", row.get("time", row.get("onset")))
    )
    end_value = (
        row.get("end")
        if row.get("end") not in {None, ""}
        else row.get("end_seconds", row.get("offset"))
    )
    if start_value is None:
        return None
    start = parse_time_seconds(start_value)
    if end_value is None:
        duration_value = row.get("duration")
        if duration_value is None:
            return None
        end = start + parse_time_seconds(duration_value)
    else:
        end = parse_time_seconds(end_value)
    chord_value = (
        row.get("chord")
        or row.get("chord_display")
        or row.get("label")
        or row.get("name")
        or row.get("value")
    )
    if chord_value is None:
        return None
    confidence_value = row.get("confidence", row.get("probability", row.get("score", 1.0)))
    try:
        confidence = float(confidence_value)
    except (TypeError, ValueError):
        confidence = 1.0
    confidence = float(np.clip(confidence, 0.0, 1.0))
    if end <= start:
        return None
    return ExternalChordHint(
        start=start,
        end=end,
        chord=normalize_external_chord_label(chord_value, notation),
        confidence=confidence,
    )


def read_external_chord_hints(path: Path, notation: str) -> list[ExternalChordHint]:
    suffix = path.suffix.lower()
    hints: list[ExternalChordHint] = []
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            rows = (
                data.get("events")
                or data.get("chords")
                or data.get("annotations")
                or data.get("segments")
                or []
            )
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        for row in rows:
            if isinstance(row, dict):
                hint = external_hint_from_mapping(row, notation)
                if hint is not None:
                    hints.append(hint)
        return sorted(hints, key=lambda item: (item.start, item.end))

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            if {"start", "end"} & fieldnames or {"start_seconds", "end_seconds"} & fieldnames:
                for row in reader:
                    hint = external_hint_from_mapping(dict(row), notation)
                    if hint is not None:
                        hints.append(hint)
                return sorted(hints, key=lambda item: (item.start, item.end))
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) < 3:
                    continue
                try:
                    start = parse_time_seconds(row[0])
                    end = parse_time_seconds(row[1])
                except ValueError:
                    continue
                if end <= start:
                    continue
                confidence = 1.0
                if len(row) >= 4:
                    try:
                        confidence = float(np.clip(float(row[3]), 0.0, 1.0))
                    except ValueError:
                        confidence = 1.0
                hints.append(
                    ExternalChordHint(
                        start,
                        end,
                        normalize_external_chord_label(row[2], notation),
                        confidence,
                    )
                )
        return sorted(hints, key=lambda item: (item.start, item.end))

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            start = parse_time_seconds(parts[0])
            end = parse_time_seconds(parts[1])
        except ValueError:
            continue
        if end <= start:
            continue
        confidence = 1.0
        if len(parts) >= 4:
            try:
                confidence = float(np.clip(float(parts[3]), 0.0, 1.0))
            except ValueError:
                confidence = 1.0
        hints.append(
            ExternalChordHint(
                start,
                end,
                normalize_external_chord_label(parts[2], notation),
                confidence,
            )
        )
    return sorted(hints, key=lambda item: (item.start, item.end))


def external_family_similarity(state_family: str, observed_family: str) -> float:
    state_family = "" if state_family == "dom" else state_family
    observed_family = "" if observed_family == "dom" else observed_family
    if state_family == observed_family:
        return 1.0
    if state_family == "" and observed_family == "sus":
        return 0.72
    if state_family == "sus" and observed_family == "":
        return 0.72
    if state_family == "" and observed_family in {"6", "add9"}:
        return 0.82
    return 0.0


def external_functional_scores(
    cells: list[HarmonicInterval] | None,
    hints: list[ExternalChordHint] | None,
    states: list[FunctionalState],
    weight: float,
) -> np.ndarray | None:
    if not cells or not hints or not states or weight <= 0:
        return None

    scores = np.zeros((len(cells), len(states)), dtype=np.float32)
    state_members = [chord_member_pcs(state.chord) for state in states]
    for cell_index, cell in enumerate(cells):
        duration = max(cell.end - cell.start, 1e-6)
        for hint in hints:
            if hint.end <= cell.start:
                continue
            if hint.start >= cell.end:
                break
            overlap = min(cell.end, hint.end) - max(cell.start, hint.start)
            if overlap <= 0:
                continue
            root, suffix = parse_chord_name(hint.chord)
            if root is None:
                continue
            observed_family = chord_quality_family(suffix)
            fraction = float(np.clip(overlap / duration, 0.0, 1.0))
            amount = weight * fraction * hint.confidence
            for state_index, state in enumerate(states):
                if state.root_pc == root:
                    family_score = external_family_similarity(state.family, observed_family)
                    scores[cell_index, state_index] += amount * (
                        0.34 * family_score if family_score > 0 else 0.12
                    )
                elif root in state_members[state_index]:
                    scores[cell_index, state_index] += amount * 0.035
    return scores


def make_state(label: str, key_root: int, offset: int, family: str, notation: str, bass_offset: int | None = None, prior: float = 0.0) -> FunctionalState:
    root_pc = (key_root + offset) % 12
    chord = note_name(root_pc, notation) + ("" if family == "" else family)
    bass_pc = None if bass_offset is None else (key_root + bass_offset) % 12
    return FunctionalState(
        label=label,
        chord=chord,
        root_pc=root_pc,
        family=family,
        bass_pc=bass_pc,
        prior=prior,
    )


def build_functional_states(key: str, notation: str) -> list[FunctionalState]:
    key_root, key_suffix = parse_chord_name(key)
    if key_root is None:
        return []
    if key_suffix == "m":
        return [
            make_state("i", key_root, 0, "m", notation, prior=0.13),
            make_state("VII", key_root, 10, "", notation, prior=0.12),
            make_state("VI", key_root, 8, "", notation, prior=0.12),
            make_state("iv", key_root, 5, "m", notation, prior=0.08),
            make_state("V", key_root, 7, "", notation, prior=0.09),
            make_state("III", key_root, 3, "", notation, prior=0.06),
            make_state("ii°", key_root, 2, "dim", notation, prior=-0.04),
        ]
    return [
        make_state("I", key_root, 0, "", notation, prior=0.14),
        make_state("V", key_root, 7, "", notation, prior=0.13),
        make_state("V/7", key_root, 7, "", notation, bass_offset=11, prior=0.12),
        make_state("vi", key_root, 9, "m", notation, prior=0.12),
        make_state("IV", key_root, 5, "", notation, prior=0.13),
        make_state("ii", key_root, 2, "m", notation, prior=0.04),
        make_state("iii", key_root, 4, "m", notation, prior=0.02),
        make_state("bVII", key_root, 10, "", notation, prior=-0.03),
    ]


def functional_transition_matrix(states: list[FunctionalState], strength: float) -> np.ndarray:
    count = len(states)
    matrix = np.full((count, count), -0.18 * strength, dtype=np.float32)

    pair_weights = {
        ("I", "V/7"): 0.48,
        ("V/7", "vi"): 0.50,
        ("I", "V"): 0.36,
        ("V", "vi"): 0.34,
        ("vi", "IV"): 0.44,
        ("IV", "V"): 0.50,
        ("IV", "V/7"): 0.34,
        ("V", "I"): 0.46,
        ("V/7", "I"): 0.40,
        ("IV", "I"): 0.26,
        ("I", "IV"): 0.26,
        ("I", "vi"): 0.24,
        ("vi", "ii"): 0.18,
        ("ii", "V"): 0.42,
        ("ii", "V/7"): 0.38,
        ("iii", "vi"): 0.34,
        ("bVII", "IV"): 0.18,
        ("bVII", "I"): 0.12,
        ("i", "VII"): 0.40,
        ("VII", "VI"): 0.38,
        ("VI", "iv"): 0.34,
        ("VI", "VII"): 0.24,
        ("iv", "V"): 0.38,
        ("V", "i"): 0.44,
        ("III", "VII"): 0.30,
        ("III", "VI"): 0.22,
    }
    sustain_weights = {
        "I": 0.17,
        "V": 0.13,
        "V/7": 0.10,
        "vi": 0.20,
        "IV": 0.17,
        "ii": 0.08,
        "iii": 0.06,
        "bVII": 0.06,
        "i": 0.12,
        "VII": 0.10,
        "VI": 0.10,
        "iv": 0.10,
        "III": 0.06,
    }
    discouraged_pairs = {
        ("V", "IV"),
        ("V/7", "IV"),
        ("vi", "V"),
        ("IV", "iii"),
        ("I", "bVII"),
    }

    for prev, prev_state in enumerate(states):
        for curr, curr_state in enumerate(states):
            pair = (prev_state.label, curr_state.label)
            if prev == curr:
                matrix[prev, curr] = sustain_weights.get(prev_state.label, 0.08) * strength
            if pair in pair_weights:
                matrix[prev, curr] = pair_weights[pair] * strength
            elif pair in discouraged_pairs:
                matrix[prev, curr] = -0.12 * strength
    return matrix


def common_functional_patterns() -> list[tuple[tuple[str, ...], ...]]:
    return [
        (("vi",), ("vi",), ("IV",), ("V", "V/7")),
        (("IV",), ("V", "V/7"), ("vi",), ("vi",)),
        (("I",), ("I",), ("IV",), ("V", "V/7")),
        (("I",), ("V/7", "V"), ("vi",), ("vi",)),
        (("IV",), ("V", "V/7"), ("I",), ("I",)),
        (("I",), ("V/7", "V"), ("vi",), ("IV",), ("V",), ("I",)),
        (("IV",), ("V",), ("vi",), ("IV",), ("V",), ("I",)),
        (("I",), ("V",), ("vi",), ("IV",)),
        (("I",), ("vi",), ("IV",), ("V",)),
        (("vi",), ("IV",), ("I",), ("V",)),
        (("I",), ("IV",), ("V",), ("I",)),
        (("ii",), ("V", "V/7"), ("I",), ("I",)),
        (("i",), ("VII",), ("VI",), ("VII",)),
        (("i",), ("VI",), ("III",), ("VII",)),
        (("VI",), ("VII",), ("i",), ("i",)),
        (("iv",), ("V",), ("i",), ("i",)),
    ]


def progression_library_bonus(
    scores: np.ndarray,
    states: list[FunctionalState],
    strength: float,
) -> np.ndarray:
    if scores.size == 0 or strength <= 0:
        return np.zeros_like(scores)

    label_indices: dict[str, tuple[int, ...]] = {}
    for index, state in enumerate(states):
        label_indices[state.label] = (*label_indices.get(state.label, ()), index)

    patterns: list[tuple[tuple[int, ...], ...]] = []
    for raw_pattern in common_functional_patterns():
        converted: list[tuple[int, ...]] = []
        for labels in raw_pattern:
            indices: list[int] = []
            for label in labels:
                indices.extend(label_indices.get(label, ()))
            if not indices:
                converted = []
                break
            converted.append(tuple(sorted(set(indices))))
        if converted:
            patterns.append(tuple(converted))

    if not patterns:
        return np.zeros_like(scores)

    frame_count = scores.shape[0]
    bonus = np.zeros_like(scores)

    for window, step, window_weight in ((4, 2, 1.15), (8, 4, 1.0)):
        if frame_count < window:
            continue
        for start in range(0, frame_count - window + 1, step):
            end = start + window
            local_scores = scores[start:end]
            local_best = float(np.mean(np.max(local_scores, axis=1)))
            best_plan: tuple[float, float, tuple[tuple[int, ...], ...], int] | None = None

            for pattern in patterns:
                pattern_len = len(pattern)
                for phase in range(pattern_len):
                    expected_total = 0.0
                    for local_index in range(window):
                        indices = pattern[(local_index + phase) % pattern_len]
                        expected_total += float(np.max(local_scores[local_index, list(indices)]))
                    expected_mean = expected_total / window
                    gap = local_best - expected_mean
                    rank = expected_mean - 0.004 * abs(pattern_len - window)
                    if best_plan is None or rank > best_plan[0]:
                        best_plan = (rank, gap, pattern, phase)

            if best_plan is None:
                continue
            _, gap, pattern, phase = best_plan
            confidence = float(np.clip((0.14 - gap) / 0.14, 0.0, 1.0))
            if confidence <= 0:
                continue
            amount = strength * confidence * window_weight
            for local_index in range(window):
                indices = pattern[(local_index + phase) % len(pattern)]
                frame = start + local_index
                for index in indices:
                    bonus[frame, index] = max(bonus[frame, index], amount)

    return np.clip(bonus, 0.0, strength * 1.5)


def repeated_section_bonus(
    scores: np.ndarray,
    vectors: np.ndarray,
    path: np.ndarray,
    strength: float,
) -> np.ndarray:
    if scores.size == 0 or vectors.size == 0 or path.size == 0 or strength <= 0:
        return np.zeros_like(scores)
    if vectors.shape[0] != scores.shape[0] or path.size != scores.shape[0]:
        return np.zeros_like(scores)

    time_count = scores.shape[0]
    if time_count < 12:
        return np.zeros_like(scores)

    row_norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / (row_norm + 1e-9)
    order = np.argsort(scores, axis=1)
    best = scores[np.arange(time_count), path]
    second = scores[np.arange(time_count), order[:, -2]]
    reliability = np.clip((best - second) / 0.12, 0.0, 1.0)
    bonus = np.zeros_like(scores)
    counts = np.zeros((time_count, 1), dtype=np.float32)

    for window, threshold in ((4, 0.93), (8, 0.88)):
        if time_count < window * 3:
            continue
        starts = list(range(0, time_count - window + 1, max(1, window // 2)))
        if len(starts) < 2:
            continue
        features = []
        for start in starts:
            feature = normalized[start : start + window].reshape(-1)
            feature = feature - float(np.mean(feature))
            norm = float(np.linalg.norm(feature))
            if norm <= 1e-9:
                features.append(np.zeros_like(feature))
            else:
                features.append(feature / norm)
        feature_matrix = np.asarray(features, dtype=np.float32)

        for left_index, left_start in enumerate(starts):
            for right_index in range(left_index + 1, len(starts)):
                right_start = starts[right_index]
                if right_start - left_start < window:
                    continue
                similarity = float(feature_matrix[left_index] @ feature_matrix[right_index])
                if similarity < threshold:
                    continue
                pair_weight = ((similarity - threshold) / max(1.0 - threshold, 1e-6)) * strength
                if pair_weight <= 0:
                    continue
                for offset in range(window):
                    left_pos = left_start + offset
                    right_pos = right_start + offset
                    left_state = int(path[left_pos])
                    right_state = int(path[right_pos])
                    left_reliability = float(reliability[left_pos])
                    right_reliability = float(reliability[right_pos])
                    if left_state == right_state:
                        amount = pair_weight * (0.45 + 0.55 * max(left_reliability, right_reliability))
                        bonus[left_pos, left_state] += amount
                        bonus[right_pos, right_state] += amount
                    else:
                        bonus[left_pos, right_state] += pair_weight * 0.45 * right_reliability
                        bonus[right_pos, left_state] += pair_weight * 0.45 * left_reliability
                    counts[left_pos, 0] += 1.0
                    counts[right_pos, 0] += 1.0

    if not np.any(counts):
        return bonus
    bonus = bonus / np.maximum(counts, 1.0)
    return np.clip(bonus, 0.0, strength)


def section_consensus_bonus(
    scores: np.ndarray,
    vectors: np.ndarray,
    strength: float,
) -> np.ndarray:
    if scores.size == 0 or vectors.size == 0 or strength <= 0:
        return np.zeros_like(scores)
    if vectors.shape[0] != scores.shape[0] or scores.shape[0] < 16:
        return np.zeros_like(scores)

    time_count, state_count = scores.shape
    row_norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / (row_norm + 1e-9)
    bonus = np.zeros_like(scores)
    counts = np.zeros((time_count, 1), dtype=np.float32)

    for window, threshold in ((8, 0.88), (12, 0.84)):
        if time_count < window * 2:
            continue
        starts = list(range(0, time_count - window + 1, max(2, window // 2)))
        if len(starts) < 2:
            continue
        features = []
        distributions = []
        for start in starts:
            feature = normalized[start : start + window].reshape(-1)
            feature = feature - float(np.mean(feature))
            feature_norm = float(np.linalg.norm(feature))
            features.append(feature / (feature_norm + 1e-9))

            local = scores[start : start + window]
            local = local - np.max(local, axis=1, keepdims=True)
            probs = np.exp(local / 0.16)
            probs = probs / (np.sum(probs, axis=1, keepdims=True) + 1e-9)
            distributions.append(probs.astype(np.float32))

        feature_matrix = np.asarray(features, dtype=np.float32)
        used_pairs: list[tuple[int, int, float]] = []
        for left_index, left_start in enumerate(starts):
            for right_index in range(left_index + 1, len(starts)):
                right_start = starts[right_index]
                if right_start - left_start < window:
                    continue
                similarity = float(feature_matrix[left_index] @ feature_matrix[right_index])
                if similarity >= threshold:
                    confidence = (similarity - threshold) / max(1.0 - threshold, 1e-6)
                    used_pairs.append((left_index, right_index, float(np.clip(confidence, 0.0, 1.0))))

        for left_index, right_index, confidence in used_pairs:
            if confidence <= 0:
                continue
            left_start = starts[left_index]
            right_start = starts[right_index]
            consensus = (distributions[left_index] + distributions[right_index]) * 0.5
            amount = strength * confidence
            for offset in range(window):
                left_pos = left_start + offset
                right_pos = right_start + offset
                centered = consensus[offset] - (1.0 / state_count)
                positive = np.maximum(centered, 0.0)
                bonus[left_pos] += amount * positive
                bonus[right_pos] += amount * positive
                counts[left_pos, 0] += 1.0
                counts[right_pos, 0] += 1.0

    if not np.any(counts):
        return bonus
    bonus = bonus / np.maximum(counts, 1.0)
    return np.clip(bonus, 0.0, strength)


def phrase_transition_modifiers(
    states: list[FunctionalState],
    cells: list[HarmonicInterval] | None,
    strength: float,
) -> np.ndarray | None:
    if not cells or len(cells) < 2 or not states or strength <= 0:
        return None
    count = len(states)
    modifiers = np.zeros((len(cells) - 1, count, count), dtype=np.float32)
    label_indices: dict[str, list[int]] = {}
    for index, state in enumerate(states):
        label_indices.setdefault(state.label, []).append(index)

    def add(step: int, prev_labels: tuple[str, ...], curr_labels: tuple[str, ...], value: float) -> None:
        for prev_label in prev_labels:
            for curr_label in curr_labels:
                for prev_index in label_indices.get(prev_label, []):
                    for curr_index in label_indices.get(curr_label, []):
                        modifiers[step, prev_index, curr_index] += value * strength

    def add_sustain(step: int, labels: tuple[str, ...], value: float) -> None:
        for label in labels:
            for index in label_indices.get(label, []):
                modifiers[step, index, index] += value * strength

    for step, (prev_cell, curr_cell) in enumerate(zip(cells, cells[1:])):
        same_bar = prev_cell.bar == curr_cell.bar
        next_bar = curr_cell.bar == prev_cell.bar + 1
        if same_bar:
            add_sustain(step, ("vi",), 0.07)
            add_sustain(step, ("I", "IV"), 0.04)
            add_sustain(step, ("V", "V/7"), 0.02)
            add(step, ("IV",), ("V", "V/7"), 0.06)
            add(step, ("vi",), ("IV",), -0.035)
            add(step, ("V", "V/7"), ("I",), -0.075)
        elif next_bar:
            add(step, ("vi",), ("IV",), 0.04)
            add(step, ("IV",), ("V", "V/7"), 0.03)
            add(step, ("V", "V/7"), ("I",), 0.035)
    return modifiers


def adaptive_transition_scales(scores: np.ndarray) -> np.ndarray | None:
    if scores.shape[0] < 2 or scores.shape[1] < 2:
        return None
    order = np.sort(scores, axis=1)
    margins = order[:, -1] - order[:, -2]
    current_confidence = np.clip(margins / 0.18, 0.0, 1.0)
    scales = 1.12 - 0.58 * current_confidence[1:]
    return np.clip(scales, 0.48, 1.12).astype(np.float32)


def viterbi_indices(
    scores: np.ndarray,
    transitions: np.ndarray,
    transition_modifiers: np.ndarray | None = None,
    transition_scales: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    if scores.size == 0:
        return 0.0, np.asarray([], dtype=np.int32)
    if scores.shape[0] == 1:
        index = int(np.argmax(scores[0]))
        return float(scores[0, index]), np.asarray([index], dtype=np.int32)

    time_count, state_count = scores.shape
    dp = np.empty((time_count, state_count), dtype=np.float32)
    back = np.zeros((time_count, state_count), dtype=np.int32)
    dp[0] = scores[0]
    for time_index in range(1, time_count):
        scale = 1.0
        if transition_scales is not None and transition_scales.size >= time_index:
            scale = float(transition_scales[time_index - 1])
        if transition_modifiers is not None and transition_modifiers.shape[0] >= time_index:
            step_transitions = (transitions + transition_modifiers[time_index - 1]) * scale
        else:
            step_transitions = transitions * scale
        candidates = dp[time_index - 1][:, None] + step_transitions
        back[time_index] = np.argmax(candidates, axis=0)
        dp[time_index] = scores[time_index] + candidates[
            back[time_index], np.arange(state_count)
        ]

    path = np.empty(time_count, dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))
    for time_index in range(time_count - 2, -1, -1):
        path[time_index] = back[time_index + 1, path[time_index + 1]]
    return float(np.max(dp[-1])), path


def functional_sequence(
    vectors: np.ndarray,
    bass_vectors: np.ndarray,
    bass_confidence: np.ndarray,
    key: str,
    notation: str,
    transition_strength: float,
    repeat_refine: bool = True,
    phrase_refine: bool = True,
    cells: list[HarmonicInterval] | None = None,
    external_hints: list[ExternalChordHint] | None = None,
    external_weight: float = 0.0,
    bass_weight: float = 0.15,
) -> FunctionalSequenceResult:
    states = build_functional_states(key, notation)
    if not states or vectors.size == 0:
        return FunctionalSequenceResult(0.0, states, np.asarray([], dtype=np.int32), np.zeros((0, 0), dtype=np.float32))

    templates = build_templates(mode="triads", notation=notation)
    template_matrix = np.vstack([template.vector for template in templates])
    template_names = [template.name for template in templates]
    name_to_index = {name: index for index, name in enumerate(template_names)}
    l1_vectors = vectors / (np.sum(vectors, axis=1, keepdims=True) + 1e-9)

    scores = np.empty((vectors.shape[0], len(states)), dtype=np.float32)
    for state_index, state in enumerate(states):
        template_index = name_to_index.get(state.chord)
        if template_index is None:
            scores[:, state_index] = -0.25
            continue
        state_scores = vectors @ template_matrix[template_index]
        state_scores = state_scores + state.prior
        member_mask = np.zeros(12, dtype=np.float32)
        for pc in chord_member_pcs(state.chord):
            member_mask[pc] = 1.0
        root_pc, third_pc, fifth_pc = chord_interval_pcs(state.chord)
        member_ratio = np.clip(l1_vectors @ member_mask, 0.0, 1.0)
        outside_ratio = np.clip(1.0 - member_ratio, 0.0, 1.0)
        structural_support = 0.10 * member_ratio - 0.075 * outside_ratio
        if root_pc is not None:
            structural_support = structural_support + 0.035 * l1_vectors[:, root_pc]
        if third_pc is not None:
            structural_support = structural_support + 0.04 * l1_vectors[:, third_pc]
        if fifth_pc is not None:
            structural_support = structural_support + 0.02 * l1_vectors[:, fifth_pc]
        state_scores = state_scores + structural_support
        if bass_vectors.shape[0] == vectors.shape[0] and bass_weight > 0:
            root_support = bass_vectors[:, state.root_pc]
            member_support = np.clip(bass_vectors @ member_mask, 0.0, 1.0)
            outside_support = np.clip(np.max(bass_vectors, axis=1) - member_support, 0.0, 1.0)
            state_scores = state_scores + (bass_weight / 0.15) * bass_confidence * (
                0.11 * root_support + 0.025 * member_support - 0.045 * outside_support
            )
            if state.bass_pc is not None:
                bass_support = bass_vectors[:, state.bass_pc]
                state_scores = state_scores + (bass_weight / 0.15) * 0.20 * bass_confidence * bass_support
        scores[:, state_index] = state_scores

    external_scores = external_functional_scores(cells, external_hints, states, external_weight)
    if external_scores is not None and external_scores.shape == scores.shape:
        scores = scores + external_scores

    progression_strength = min(0.09, max(0.035, 0.075 * transition_strength)) if transition_strength > 0 else 0.0
    scores = scores + progression_library_bonus(scores, states, progression_strength)
    consensus_strength = min(0.045, max(0.015, 0.035 * transition_strength)) if repeat_refine and transition_strength > 0 else 0.0
    scores = scores + section_consensus_bonus(scores, vectors, consensus_strength)
    transitions = functional_transition_matrix(states, transition_strength)
    transition_modifiers = (
        phrase_transition_modifiers(states, cells, transition_strength)
        if phrase_refine
        else None
    )
    transition_scales = adaptive_transition_scales(scores)
    total_score, path = viterbi_indices(
        scores,
        transitions,
        transition_modifiers,
        transition_scales,
    )
    if repeat_refine and transition_strength > 0 and path.size:
        repeat_strength = min(0.08, max(0.02, 0.055 * transition_strength))
        refined_scores = scores + repeated_section_bonus(scores, vectors, path, repeat_strength)
        refined_transition_scales = adaptive_transition_scales(refined_scores)
        refined_total_score, refined_path = viterbi_indices(
            refined_scores,
            transitions,
            transition_modifiers,
            refined_transition_scales,
        )
        if refined_total_score >= total_score - 0.01 * max(vectors.shape[0], 1):
            scores = refined_scores
            total_score = refined_total_score
            path = refined_path
    normalized_score = total_score / max(vectors.shape[0], 1)
    return FunctionalSequenceResult(normalized_score, states, path, scores)


def functional_key_score_bonus(result: FunctionalSequenceResult) -> float:
    if result.path.size == 0 or not result.states:
        return 0.0
    labels = [result.states[int(index)].label for index in result.path]
    total = max(len(labels), 1)
    ratios = {label: labels.count(label) / total for label in set(labels)}
    bonus = 0.0
    if ratios.get("IV", 0.0) >= 0.08:
        bonus += 0.045
    if ratios.get("vi", 0.0) >= 0.04:
        bonus += 0.04
    if ratios.get("I", 0.0) + ratios.get("V", 0.0) + ratios.get("V/7", 0.0) > 0.84:
        bonus -= 0.05
    if len(set(labels)) <= 2:
        bonus -= 0.08
    pair_counts: dict[tuple[str, str], int] = {}
    for prev, curr in zip(labels, labels[1:]):
        pair_counts[(prev, curr)] = pair_counts.get((prev, curr), 0) + 1
    pair_total = max(len(labels) - 1, 1)
    pop_pairs = {
        ("I", "V/7"),
        ("V/7", "vi"),
        ("I", "V"),
        ("V", "vi"),
        ("vi", "IV"),
        ("IV", "V"),
        ("V", "I"),
        ("IV", "I"),
    }
    pop_ratio = sum(pair_counts.get(pair, 0) for pair in pop_pairs) / pair_total
    bonus += min(0.075, pop_ratio * 0.11)
    detour_ratio = (
        pair_counts.get(("V", "IV"), 0)
        + pair_counts.get(("V/7", "IV"), 0)
        + pair_counts.get(("vi", "V"), 0)
    ) / pair_total
    bonus -= min(0.055, detour_ratio * 0.12)
    fourgram_counts: dict[tuple[str, str, str, str], int] = {}
    for a, b, c, d in zip(labels, labels[1:], labels[2:], labels[3:]):
        fourgram_counts[(a, b, c, d)] = fourgram_counts.get((a, b, c, d), 0) + 1
    fourgram_total = max(len(labels) - 3, 1)
    common_major_loops = {
        ("I", "V", "vi", "IV"),
        ("vi", "IV", "I", "V"),
        ("IV", "I", "V", "vi"),
        ("IV", "V", "vi", "IV"),
        ("I", "vi", "IV", "V"),
    }
    loop_ratio = sum(fourgram_counts.get(pattern, 0) for pattern in common_major_loops) / fourgram_total
    bonus += min(0.065, loop_ratio * 0.18)
    return bonus


def key_candidates_for_notation(notation: str) -> list[str]:
    notes = PC_TO_FLAT if notation == "flats" else PC_TO_SHARP
    return [*notes, *[f"{note}m" for note in notes]]


def select_meter_and_key(
    y: np.ndarray,
    chroma: np.ndarray,
    bass_chroma: np.ndarray,
    base_meter: MeterInfo,
    duration: float,
    notation: str,
    tempo_mode: str,
    requested_key: str | None,
    transition_strength: float,
    external_hints: list[ExternalChordHint] | None = None,
    external_weight: float = 0.0,
    bass_weight: float = 0.15,
    retain_all_key_candidates: bool = False,
) -> MeterKeySelection:
    if requested_key and base_meter.confidence >= 1.0 and not base_meter.locked_phase:
        return MeterKeySelection(
            base_meter,
            requested_key,
            0.0,
            "fixed",
            [("fixed", requested_key, 0.0)],
        )

    candidates: list[tuple[str, str, float, MeterInfo]] = []
    key_scores: list[tuple[str, str, float]] = []

    keys = [requested_key] if requested_key else key_candidates_for_notation(notation)
    for variant_name, tempo_meter in tempo_variants(base_meter, tempo_mode):
        for phase_name, candidate_meter in meter_phase_variants(tempo_meter):
            intervals = build_beat_intervals(candidate_meter, duration)
            leading_penalty = leading_partial_bar_penalty(
                intervals,
                candidate_meter.beats_per_bar,
            )
            harmonic_unit = harmonic_unit_for_meter(candidate_meter, "auto")
            cells = harmonic_intervals_from_beats(
                intervals,
                harmonic_unit,
                candidate_meter.beats_per_bar,
            )
            if len(cells) < 4:
                continue
            analysis_intervals = [
                BeatInterval(
                    bar=cell.bar,
                    beat=cell.beat_start,
                    start=cell.start,
                    end=cell.end,
                )
                for cell in cells
            ]
            vectors = chroma_for_intervals(chroma, y, SAMPLE_RATE, 2048, analysis_intervals)
            bass_vectors, _, bass_confidence = bass_for_intervals(
                bass_chroma, SAMPLE_RATE, 2048, analysis_intervals, notation
            )

            for key in keys:
                if key is None:
                    continue
                result = functional_sequence(
                    vectors,
                    bass_vectors,
                    bass_confidence,
                    key,
                    notation,
                    transition_strength=(3.75 * transition_strength if transition_strength < 0.2 else max(0.75, transition_strength * 3.0)),
                    repeat_refine=False,
                    cells=cells,
                    external_hints=external_hints,
                    external_weight=external_weight,
                    bass_weight=bass_weight,
                )
                score = result.score + functional_key_score_bonus(result)
                if variant_name.startswith("half") and base_meter.bpm >= 120:
                    score += 0.045
                if variant_name == "detected" and base_meter.bpm >= 120:
                    score -= 0.02
                if variant_name == "double" and base_meter.bpm <= 85:
                    score -= 0.025
                if phase_name != "p0":
                    score -= 0.008
                score -= leading_penalty
                if key.endswith("m"):
                    score -= 0.035
                candidate_name = f"{variant_name}/{phase_name}"
                candidates.append((candidate_name, key, score, candidate_meter))
                key_scores.append((candidate_name, key, score))

    if not candidates:
        fallback_key = requested_key or estimate_key(chroma, notation)
        return MeterKeySelection(base_meter, fallback_key, 0.0, "detected", [])

    candidates.sort(key=lambda item: item[2], reverse=True)
    best_variant, best_key, best_score, best_meter = candidates[0]
    key_scores.sort(key=lambda item: item[2], reverse=True)
    return MeterKeySelection(best_meter, best_key, best_score, best_variant,
                             key_scores if retain_all_key_candidates else key_scores[:8])


def template_roots_and_members(
    template_names: list[str],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    roots = []
    families = []
    members = []
    for name in template_names:
        root, suffix = parse_chord_name(name)
        roots.append(-1 if root is None else root)
        families.append(chord_quality_family(suffix))
        row = np.zeros(12, dtype=np.float32)
        for pc in chord_member_pcs(name):
            row[pc] = 1.0
        members.append(row)
    return np.asarray(roots, dtype=np.int16), families, np.asarray(members, dtype=np.float32)


def apply_bass_bias(
    scores: np.ndarray,
    bass_vectors: np.ndarray | None,
    bass_confidence: np.ndarray | None,
    template_names: list[str],
    bass_weight: float,
) -> np.ndarray:
    if bass_vectors is None or bass_confidence is None or bass_weight <= 0:
        return scores
    if bass_vectors.shape[0] != scores.shape[0]:
        return scores

    roots, _, members = template_roots_and_members(template_names)
    adjusted = scores.copy()
    for template_index, root in enumerate(roots):
        if root < 0:
            continue
        root_energy = bass_vectors[:, int(root)]
        member_energy = bass_vectors @ members[template_index]
        support = 0.22 * root_energy + 0.05 * np.clip(member_energy - root_energy, 0.0, 1.0)
        mismatch = np.clip(np.max(bass_vectors, axis=1) - member_energy, 0.0, 1.0)
        adjusted[:, template_index] += bass_weight * bass_confidence * (support - 0.08 * mismatch)
    return adjusted


def transition_matrix(template_names: list[str], key: str, strength: float) -> np.ndarray:
    roots, families, _ = template_roots_and_members(template_names)
    profile = diatonic_profile(key)
    count = len(template_names)
    matrix = np.full((count, count), -0.08 * strength, dtype=np.float32)
    for prev in range(count):
        for curr in range(count):
            if prev == curr:
                matrix[prev, curr] = 0.13 * strength
                continue
            prev_root = int(roots[prev])
            curr_root = int(roots[curr])
            if prev_root == curr_root and prev_root >= 0:
                if families[prev] == families[curr]:
                    matrix[prev, curr] = 0.045 * strength
                else:
                    matrix[prev, curr] = -0.005 * strength
                continue
            if curr_root in profile and prev_root in profile:
                movement = (curr_root - prev_root) % 12
                if movement in {2, 5, 7, 10}:
                    matrix[prev, curr] = -0.025 * strength
                else:
                    matrix[prev, curr] = -0.045 * strength
            elif curr_root in profile:
                matrix[prev, curr] = -0.06 * strength
            else:
                matrix[prev, curr] = -0.12 * strength
    return matrix


def viterbi_path(scores: np.ndarray, template_names: list[str], key: str, strength: float) -> np.ndarray:
    if scores.size == 0:
        return np.asarray([], dtype=np.int32)
    if scores.shape[0] == 1 or strength <= 0:
        return np.argmax(scores, axis=1).astype(np.int32)

    trans = transition_matrix(template_names, key, strength)
    time_count, state_count = scores.shape
    dp = np.empty((time_count, state_count), dtype=np.float32)
    back = np.zeros((time_count, state_count), dtype=np.int32)
    dp[0] = scores[0]
    for t in range(1, time_count):
        candidates = dp[t - 1][:, None] + trans
        back[t] = np.argmax(candidates, axis=0)
        dp[t] = scores[t] + candidates[back[t], np.arange(state_count)]

    path = np.empty(time_count, dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(time_count - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def match_interval_chords(
    vectors: np.ndarray,
    rms: np.ndarray,
    mode: str,
    notation: str,
    key: str,
    no_chord_threshold: float,
    confidence_threshold: float,
    bass_vectors: np.ndarray | None = None,
    bass_roots: list[str] | None = None,
    bass_confidence: np.ndarray | None = None,
    bass_weight: float = 0.55,
    transition_strength: float = 1.0,
    progressions: str | None = None,
    progression_strength: float = 1.0,
) -> MatchResult:
    templates = build_templates(mode=mode, notation=notation)
    template_matrix = np.vstack([template.vector for template in templates])
    template_names = [template.name for template in templates]
    scores = vectors @ template_matrix.T
    scores = scores + template_biases(template_names, key)
    scores = scores + progression_template_biases(
        template_names, key, progressions, progression_strength
    )
    scores = apply_bass_bias(scores, bass_vectors, bass_confidence, template_names, bass_weight)
    best_idx = viterbi_path(scores, template_names, key, transition_strength)

    order = np.argsort(scores, axis=1)
    second_idx = np.empty(scores.shape[0], dtype=np.int32)
    for i, best in enumerate(best_idx):
        row_order = order[i]
        second_idx[i] = row_order[-2] if row_order[-1] == best else row_order[-1]
    best_scores = scores[np.arange(scores.shape[0]), best_idx]
    second_scores = scores[np.arange(scores.shape[0]), second_idx]

    rms_floor = float(np.quantile(rms, 0.05)) if rms.size else 0.0
    rms_peak = float(np.quantile(rms, 0.95)) if rms.size else 1.0
    rms_norm = (rms - rms_floor) / max(rms_peak - rms_floor, 1e-9)
    rms_norm = np.clip(rms_norm, 0.0, 1.0)
    silence_cutoff = max(no_chord_threshold * max(rms_peak, 1e-9), 1e-5)

    margin = np.clip((best_scores - second_scores) * 3.0, 0.0, 1.0)
    confidence = np.clip(
        (best_scores * 0.80 + margin * 0.20) * (0.65 + 0.35 * rms_norm),
        0.0,
        1.0,
    )

    chords = []
    for i, template_index in enumerate(best_idx):
        if rms[i] < silence_cutoff or confidence[i] < confidence_threshold:
            chords.append("N")
        else:
            chords.append(template_names[int(template_index)])
    if bass_roots is None:
        bass_roots = ["N"] * len(chords)
    if bass_confidence is None:
        bass_confidence = np.zeros(len(chords), dtype=np.float32)
    return MatchResult(chords, confidence, bass_roots, bass_confidence)


def dominant_bass(items: list[BeatChord]) -> tuple[str, float]:
    votes: dict[str, float] = {}
    for item in items:
        if item.bass_root == "N":
            continue
        votes[item.bass_root] = votes.get(item.bass_root, 0.0) + item.bass_confidence
    if not votes:
        return "N", 0.0
    bass_root, score = max(votes.items(), key=lambda item: item[1])
    total = sum(votes.values())
    return bass_root, float(score / max(total, 1e-9))


def collapse_bar_chords(beat_chords: list[BeatChord]) -> list[ChordSpan]:
    if not beat_chords:
        return []

    spans: list[ChordSpan] = []
    current_chord = beat_chords[0].chord
    current_count = 1
    conf_sum = beat_chords[0].confidence
    start_beat = beat_chords[0].beat
    end_beat = beat_chords[0].beat
    start_time = beat_chords[0].start
    end_time = beat_chords[0].end

    for item in beat_chords[1:]:
        if item.chord == current_chord:
            end_beat = item.beat
            current_count += 1
            conf_sum += item.confidence
            end_time = item.end
            continue

        unit = "bar" if start_beat == beat_chords[0].beat and end_beat == beat_chords[-1].beat else "beat"
        span_beats = [
            beat for beat in beat_chords if start_beat <= beat.beat <= end_beat
        ]
        bass_root, bass_conf = dominant_bass(span_beats)
        spans.append(
            ChordSpan(
                unit=unit,
                bar=beat_chords[0].bar,
                beat_start=start_beat,
                beat_end=end_beat,
                start=start_time,
                end=end_time,
                chord=current_chord,
                confidence=conf_sum / max(current_count, 1),
                bass_root=bass_root,
                bass_confidence=bass_conf,
            )
        )
        current_chord = item.chord
        current_count = 1
        conf_sum = item.confidence
        start_beat = item.beat
        end_beat = item.beat
        start_time = item.start
        end_time = item.end

    unit = "bar" if start_beat == beat_chords[0].beat and end_beat == beat_chords[-1].beat else "beat"
    span_beats = [
        beat for beat in beat_chords if start_beat <= beat.beat <= end_beat
    ]
    bass_root, bass_conf = dominant_bass(span_beats)
    spans.append(
        ChordSpan(
            unit=unit,
            bar=beat_chords[0].bar,
            beat_start=start_beat,
            beat_end=end_beat,
            start=start_time,
            end=end_time,
            chord=current_chord,
            confidence=conf_sum / max(current_count, 1),
            bass_root=bass_root,
            bass_confidence=bass_conf,
        )
    )
    if len(spans) == 1:
        only = spans[0]
        spans[0] = ChordSpan(
            unit="bar",
            bar=only.bar,
            beat_start=beat_chords[0].beat,
            beat_end=beat_chords[-1].beat,
            start=beat_chords[0].start,
            end=beat_chords[-1].end,
            chord=only.chord,
            confidence=only.confidence,
            bass_root=only.bass_root,
            bass_confidence=only.bass_confidence,
        )
    return spans


def should_split_bar(bar_span: ChordSpan, beat_spans: list[ChordSpan]) -> bool:
    if not beat_spans:
        return False
    if bar_span.chord == "N":
        return any(span.chord != "N" for span in beat_spans)
    if len(beat_spans) == 1:
        bar_root, _ = parse_chord_name(bar_span.chord)
        beat_root, _ = parse_chord_name(beat_spans[0].chord)
        if (
            beat_spans[0].chord != "N"
            and beat_root is not None
            and beat_root != bar_root
            and (
                (
                    beat_spans[0].confidence >= max(0.52, bar_span.confidence + 0.08)
                )
                or (
                    beat_spans[0].bass_root != "N"
                    and parse_chord_name(beat_spans[0].bass_root)[0] == beat_root
                    and beat_spans[0].bass_confidence >= 0.55
                    and beat_spans[0].confidence >= 0.50
                )
            )
        ):
            return True
        return False

    bar_root, _ = parse_chord_name(bar_span.chord)
    strong_changes = 0
    for span in beat_spans:
        root, _ = parse_chord_name(span.chord)
        length = span.beat_end - span.beat_start + 1
        bass_root, _ = parse_chord_name(span.bass_root)
        bass_support = bass_root is not None and root == bass_root and span.bass_confidence >= 0.52
        one_beat_evidence = span.confidence >= max(0.64, bar_span.confidence + 0.12)
        sustained_evidence = length >= 2 and span.confidence >= max(0.50, bar_span.confidence + 0.02)
        bass_evidence = bass_support and span.confidence >= 0.60
        if (
            span.chord != "N"
            and root is not None
            and root != bar_root
            and (one_beat_evidence or sustained_evidence or bass_evidence)
        ):
            strong_changes += 1

    return strong_changes >= 2 or (strong_changes >= 1 and bar_span.confidence < 0.42)


def build_chord_spans(
    beat_chords: list[BeatChord],
    bar_chords: dict[int, ChordSpan],
) -> list[ChordSpan]:
    spans: list[ChordSpan] = []
    bars = sorted({item.bar for item in beat_chords})
    for bar in bars:
        items = [item for item in beat_chords if item.bar == bar]
        beat_spans = collapse_bar_chords(items)
        bar_span = bar_chords.get(bar)
        if bar_span is None or should_split_bar(bar_span, beat_spans):
            spans.extend(beat_spans)
        else:
            spans.append(bar_span)
    return spans


def dominant_slash_for_leading_dim(
    chord: str,
    bass_root: str,
    bass_confidence: float,
    key: str,
) -> tuple[str, str, float] | None:
    key_root, key_suffix = parse_chord_name(key)
    chord_root, chord_suffix = parse_chord_name(chord)
    if key_root is None or key_suffix == "m" or chord_root is None:
        return None
    if (chord_root - key_root) % 12 != 11:
        return None
    if chord_quality_family(chord_suffix) != "dim":
        return None

    leading_pc = (key_root + 11) % 12
    dominant_pc = (key_root + 7) % 12
    bass_pc, _ = parse_chord_name(bass_root)
    new_bass = bass_root
    new_bass_confidence = bass_confidence
    if bass_pc is None or bass_confidence < 0.70 or bass_pc in chord_member_pcs(chord):
        new_bass = note_name_for_pc(leading_pc, key)
        new_bass_confidence = max(bass_confidence, 0.72)
    return note_name_for_pc(dominant_pc, key), new_bass, new_bass_confidence


def apply_pop_harmony_cleanup(
    beat_chords: list[BeatChord],
    bar_chords: dict[int, ChordSpan],
    key: str,
    enabled: bool,
) -> tuple[list[BeatChord], dict[int, ChordSpan]]:
    if not enabled:
        return beat_chords, bar_chords

    cleaned_beats = []
    for item in beat_chords:
        replacement = dominant_slash_for_leading_dim(
            item.chord, item.bass_root, item.bass_confidence, key
        )
        if replacement is None:
            cleaned_beats.append(item)
            continue
        chord, bass_root, bass_confidence = replacement
        cleaned_beats.append(
            replace(item, chord=chord, bass_root=bass_root, bass_confidence=bass_confidence)
        )

    cleaned_bars: dict[int, ChordSpan] = {}
    for bar, span in bar_chords.items():
        replacement = dominant_slash_for_leading_dim(
            span.chord, span.bass_root, span.bass_confidence, key
        )
        if replacement is None:
            cleaned_bars[bar] = span
            continue
        chord, bass_root, bass_confidence = replacement
        cleaned_bars[bar] = replace(
            span, chord=chord, bass_root=bass_root, bass_confidence=bass_confidence
        )
    return cleaned_beats, cleaned_bars


def blank_intro_chords(
    beat_chords: list[BeatChord],
    bar_chords: dict[int, ChordSpan],
    intro_bars: int,
) -> tuple[list[BeatChord], dict[int, ChordSpan]]:
    if intro_bars <= 0:
        return beat_chords, bar_chords
    cleaned_beats = [
        replace(item, chord="N", confidence=1.0, bass_root="N", bass_confidence=0.0)
        if item.bar <= intro_bars
        else item
        for item in beat_chords
    ]
    cleaned_bars = {
        bar: (
            replace(span, chord="N", confidence=1.0, bass_root="N", bass_confidence=0.0)
            if bar <= intro_bars
            else span
        )
        for bar, span in bar_chords.items()
    }
    return cleaned_beats, cleaned_bars


def recognize_beat_chords(
    y: np.ndarray,
    chroma: np.ndarray,
    bass_chroma: np.ndarray,
    intervals: list[BeatInterval],
    key: str,
    mode: str,
    notation: str,
    smooth_width: int,
    no_chord_threshold: float,
    confidence_threshold: float,
    bass_weight: float,
    transition_strength: float,
    progressions: str | None,
    progression_strength: float,
) -> list[BeatChord]:
    vectors = chroma_for_intervals(chroma, y, SAMPLE_RATE, 2048, intervals)
    rms = rms_for_intervals(y, SAMPLE_RATE, intervals)
    bass_vectors, bass_roots, bass_confidence = bass_for_intervals(
        bass_chroma,
        SAMPLE_RATE,
        2048,
        intervals,
        notation,
    )
    continuity = transition_strength * max(0.0, smooth_width / 5.0)
    match = match_interval_chords(
        vectors,
        rms,
        mode=mode,
        notation=notation,
        key=key,
        no_chord_threshold=no_chord_threshold,
        confidence_threshold=confidence_threshold,
        bass_vectors=bass_vectors,
        bass_roots=bass_roots,
        bass_confidence=bass_confidence,
        bass_weight=bass_weight,
        transition_strength=continuity,
        progressions=progressions,
        progression_strength=progression_strength,
    )
    return [
        BeatChord(
            bar=interval.bar,
            beat=interval.beat,
            start=interval.start,
            end=interval.end,
            chord=chord,
            confidence=float(conf),
            bass_root=bass_root,
            bass_confidence=float(bass_conf),
        )
        for interval, chord, conf, bass_root, bass_conf in zip(
            intervals,
            match.chords,
            match.confidence,
            match.bass_roots,
            match.bass_confidence,
        )
    ]


def recognize_bar_chords(
    y: np.ndarray,
    chroma: np.ndarray,
    bass_chroma: np.ndarray,
    beat_chords: list[BeatChord],
    key: str,
    mode: str,
    notation: str,
    no_chord_threshold: float,
    confidence_threshold: float,
    bass_weight: float,
    transition_strength: float,
    progressions: str | None,
    progression_strength: float,
) -> dict[int, ChordSpan]:
    bars = sorted({item.bar for item in beat_chords})
    bar_intervals = []
    for bar in bars:
        items = [item for item in beat_chords if item.bar == bar]
        bar_intervals.append(
            BeatInterval(
                bar=bar,
                beat=1,
                start=items[0].start,
                end=items[-1].end,
            )
        )

    vectors = chroma_for_intervals(chroma, y, SAMPLE_RATE, 2048, bar_intervals)
    rms = rms_for_intervals(y, SAMPLE_RATE, bar_intervals)
    bass_vectors, bass_roots, bass_confidence = bass_for_intervals(
        bass_chroma,
        SAMPLE_RATE,
        2048,
        bar_intervals,
        notation,
    )
    match = match_interval_chords(
        vectors,
        rms,
        mode=mode,
        notation=notation,
        key=key,
        no_chord_threshold=no_chord_threshold,
        confidence_threshold=confidence_threshold,
        bass_vectors=bass_vectors,
        bass_roots=bass_roots,
        bass_confidence=bass_confidence,
        bass_weight=bass_weight,
        transition_strength=transition_strength,
        progressions=progressions,
        progression_strength=progression_strength,
    )

    result: dict[int, ChordSpan] = {}
    for interval, chord, conf, bass_root, bass_conf in zip(
        bar_intervals,
        match.chords,
        match.confidence,
        match.bass_roots,
        match.bass_confidence,
    ):
        visible_beats = [item.beat for item in beat_chords if item.bar == interval.bar]
        result[interval.bar] = ChordSpan(
            unit="bar",
            bar=interval.bar,
            beat_start=min(visible_beats),
            beat_end=max(visible_beats),
            start=interval.start,
            end=interval.end,
            chord=chord,
            confidence=float(conf),
            bass_root=bass_root,
            bass_confidence=float(bass_conf),
        )
    return result


def confidence_from_scores(scores: np.ndarray, path: np.ndarray, rms: np.ndarray) -> np.ndarray:
    if scores.size == 0 or path.size == 0:
        return np.asarray([], dtype=np.float32)
    order = np.argsort(scores, axis=1)
    second_idx = np.empty(scores.shape[0], dtype=np.int32)
    for index, best in enumerate(path):
        row_order = order[index]
        second_idx[index] = row_order[-2] if row_order[-1] == best else row_order[-1]
    best_scores = scores[np.arange(scores.shape[0]), path]
    second_scores = scores[np.arange(scores.shape[0]), second_idx]
    margin = np.clip((best_scores - second_scores) * 2.2, 0.0, 1.0)
    rms_floor = float(np.quantile(rms, 0.05)) if rms.size else 0.0
    rms_peak = float(np.quantile(rms, 0.95)) if rms.size else 1.0
    rms_norm = (rms - rms_floor) / max(rms_peak - rms_floor, 1e-9)
    rms_norm = np.clip(rms_norm, 0.0, 1.0)
    return np.clip(0.38 + 0.42 * margin + 0.20 * rms_norm, 0.0, 1.0).astype(np.float32)


def recognize_bar_chords_functional(
    y: np.ndarray,
    chroma: np.ndarray,
    bass_chroma: np.ndarray,
    intervals: list[BeatInterval],
    key: str,
    notation: str,
    transition_strength: float,
    no_chord_threshold: float,
    confidence_threshold: float,
    repeat_refine: bool = True,
    bass_weight: float = 0.15,
) -> dict[int, ChordSpan]:
    bar_intervals = bar_intervals_from_beats(intervals)
    vectors = chroma_for_intervals(chroma, y, SAMPLE_RATE, 2048, bar_intervals)
    rms = rms_for_intervals(y, SAMPLE_RATE, bar_intervals)
    bass_vectors, bass_roots, bass_confidence = bass_for_intervals(
        bass_chroma,
        SAMPLE_RATE,
        2048,
        bar_intervals,
        notation,
    )
    result = functional_sequence(
        vectors,
        bass_vectors,
        bass_confidence,
        key,
        notation,
        transition_strength=(4.0 * transition_strength if transition_strength < 0.2 else max(0.8, transition_strength * 3.0)),
        repeat_refine=repeat_refine,
        bass_weight=bass_weight,
    )
    confidence = confidence_from_scores(result.state_scores, result.path, rms)

    rms_peak = float(np.quantile(rms, 0.95)) if rms.size else 1.0
    silence_cutoff = max(no_chord_threshold * max(rms_peak, 1e-9), 1e-5)
    bar_map: dict[int, ChordSpan] = {}
    for index, interval in enumerate(bar_intervals):
        state = result.states[int(result.path[index])] if result.path.size else None
        conf = float(confidence[index]) if confidence.size else 0.0
        if state is None or rms[index] < silence_cutoff or conf < confidence_threshold:
            chord = "N"
            bass_root = "N"
            bass_conf = 0.0
        else:
            chord = state.chord
            bass_pc = state.bass_pc
            bass_root = bass_roots[index]
            bass_conf = float(bass_confidence[index])
            if bass_weight > 0 and bass_pc is not None and (
                bass_conf < 0.68 or parse_chord_name(bass_root)[0] in {None, state.root_pc}
            ):
                bass_root = note_name(bass_pc, notation)
                bass_conf = max(bass_conf, 0.72)
        bar_map[interval.bar] = ChordSpan(
            unit="bar",
            bar=interval.bar,
            beat_start=1,
            beat_end=max(
                [beat.beat for beat in intervals if beat.bar == interval.bar],
                default=1,
            ),
            start=interval.start,
            end=interval.end,
            chord=chord,
            confidence=conf,
            bass_root=bass_root,
            bass_confidence=bass_conf,
        )
    return bar_map


def recognize_functional_spans(
    y: np.ndarray,
    chroma: np.ndarray,
    bass_chroma: np.ndarray,
    intervals: list[BeatInterval],
    key: str,
    notation: str,
    transition_strength: float,
    no_chord_threshold: float,
    confidence_threshold: float,
    harmonic_unit: str,
    beats_per_bar: int,
    repeat_refine: bool = True,
    phrase_refine: bool = True,
    external_hints: list[ExternalChordHint] | None = None,
    external_weight: float = 0.0,
    bass_weight: float = 0.15,
) -> list[ChordSpan]:
    cells = harmonic_intervals_from_beats(intervals, harmonic_unit, beats_per_bar)
    analysis_intervals = [
        BeatInterval(
            bar=cell.bar,
            beat=cell.beat_start,
            start=cell.start,
            end=cell.end,
        )
        for cell in cells
    ]
    vectors = chroma_for_intervals(chroma, y, SAMPLE_RATE, 2048, analysis_intervals)
    rms = rms_for_intervals(y, SAMPLE_RATE, analysis_intervals)
    bass_vectors, bass_roots, bass_confidence = bass_for_intervals(
        bass_chroma,
        SAMPLE_RATE,
        2048,
        analysis_intervals,
        notation,
    )
    result = functional_sequence(
        vectors,
        bass_vectors,
        bass_confidence,
        key,
        notation,
        transition_strength=(4.0 * transition_strength if transition_strength < 0.2 else max(0.8, transition_strength * 3.0)),
        repeat_refine=repeat_refine,
        phrase_refine=phrase_refine,
        cells=cells,
        external_hints=external_hints,
        external_weight=external_weight,
        bass_weight=bass_weight,
    )
    confidence = confidence_from_scores(result.state_scores, result.path, rms)

    rms_peak = float(np.quantile(rms, 0.95)) if rms.size else 1.0
    silence_cutoff = max(no_chord_threshold * max(rms_peak, 1e-9), 1e-5)
    spans: list[ChordSpan] = []
    for index, cell in enumerate(cells):
        state = result.states[int(result.path[index])] if result.path.size else None
        conf = float(confidence[index]) if confidence.size else 0.0
        if state is None or rms[index] < silence_cutoff or conf < confidence_threshold:
            chord = "N"
            bass_root = "N"
            bass_conf = 0.0
        else:
            chord = state.chord
            bass_root = bass_roots[index]
            bass_conf = float(bass_confidence[index])
            if bass_weight > 0 and state.bass_pc is not None and (
                bass_conf < 0.68 or parse_chord_name(bass_root)[0] in {None, state.root_pc}
            ):
                bass_root = note_name(state.bass_pc, notation)
                bass_conf = max(bass_conf, 0.72)
        spans.append(
            ChordSpan(
                unit="bar" if cell.beat_start == 1 and cell.beat_end >= beats_per_bar else "beat",
                bar=cell.bar,
                beat_start=cell.beat_start,
                beat_end=cell.beat_end,
                start=cell.start,
                end=cell.end,
                chord=chord,
                confidence=conf,
                bass_root=bass_root,
                bass_confidence=bass_conf,
            )
        )
    return collapse_adjacent_spans(spans)


def collapse_adjacent_spans(spans: list[ChordSpan]) -> list[ChordSpan]:
    if not spans:
        return []
    collapsed: list[ChordSpan] = []
    current = spans[0]
    count = 1
    for span in spans[1:]:
        if span.bar == current.bar and span.chord == current.chord and span.bass_root == current.bass_root:
            total_conf = current.confidence * count + span.confidence
            count += 1
            current = replace(
                current,
                beat_end=span.beat_end,
                end=span.end,
                confidence=total_conf / count,
                unit="bar" if current.beat_start == 1 and span.unit == "bar" else "beat",
            )
            continue
        collapsed.append(current)
        current = span
        count = 1
    collapsed.append(current)
    return collapsed


def beat_chords_from_spans(
    intervals: list[BeatInterval],
    spans: list[ChordSpan],
) -> list[BeatChord]:
    beats: list[BeatChord] = []
    for interval in intervals:
        span = next(
            (
                item
                for item in spans
                if item.bar == interval.bar
                and item.beat_start <= interval.beat <= item.beat_end
            ),
            None,
        )
        if span is None:
            continue
        beats.append(
            BeatChord(
                bar=interval.bar,
                beat=interval.beat,
                start=interval.start,
                end=interval.end,
                chord=span.chord,
                confidence=span.confidence,
                bass_root=span.bass_root,
                bass_confidence=span.bass_confidence,
            )
        )
    return beats


def cleanup_spans(spans: list[ChordSpan], key: str, enabled: bool) -> list[ChordSpan]:
    if not enabled:
        return spans
    cleaned: list[ChordSpan] = []
    key_root, key_suffix = parse_chord_name(key)
    for index, span in enumerate(spans):
        replacement = dominant_slash_for_leading_dim(
            span.chord, span.bass_root, span.bass_confidence, key
        )
        if replacement is not None:
            chord, bass_root, bass_confidence = replacement
            cleaned.append(replace(span, chord=chord, bass_root=bass_root, bass_confidence=bass_confidence))
            continue
        chord_root, chord_suffix = parse_chord_name(span.chord)
        bass_pc, _ = parse_chord_name(span.bass_root)
        next_span = spans[index + 1] if index + 1 < len(spans) else None
        next_root, _ = parse_chord_name(next_span.chord) if next_span else (None, "")
        if (
            key_root is not None
            and key_suffix != "m"
            and chord_root == (key_root + 2) % 12
            and chord_quality_family(chord_suffix) == "m"
            and bass_pc == (key_root + 5) % 12
            and span.bass_confidence >= 0.52
            and (next_root == (key_root + 7) % 12 or span.confidence < 0.82)
        ):
            # Pop charts usually spell ii/IV ambiguity as IV when the bass is IV,
            # especially before V. This handles Am/C vs C6-style collisions.
            chord = note_name((key_root + 5) % 12, "flats" if "b" in key else "sharps")
            cleaned.append(replace(span, chord=chord))
            continue
        if (
            key_root is not None
            and key_suffix == "m"
            and chord_root == (key_root + 5) % 12
            and chord_quality_family(chord_suffix) == "m"
            and bass_pc == (key_root + 8) % 12
            and span.bass_confidence >= 0.52
            and (
                next_root in {(key_root + 7) % 12, (key_root + 10) % 12, key_root}
                or span.confidence < 0.82
            )
        ):
            # In minor keys, iv/VI often means the VI chord was heard through
            # shared chord tones and a clear VI bass.
            chord = note_name((key_root + 8) % 12, "flats" if "b" in key else "sharps")
            cleaned.append(replace(span, chord=chord))
            continue
        cleaned.append(span)
    return cleaned


def blank_intro_spans(spans: list[ChordSpan], intro_bars: int) -> list[ChordSpan]:
    if intro_bars <= 0:
        return spans
    return [
        replace(span, chord="N", confidence=1.0, bass_root="N", bass_confidence=0.0)
        if span.bar <= intro_bars
        else span
        for span in spans
    ]


def relative_minor_center_key(
    spans: list[ChordSpan],
    key: str,
    notation: str,
) -> str | None:
    key_root, key_suffix = parse_chord_name(key)
    if key_root is None or key_suffix == "m":
        return None

    relative_minor_root = (key_root + 9) % 12
    durations: dict[int, float] = {}
    first_roots: list[int] = []
    total_duration = 0.0
    for span in spans:
        root, _ = parse_chord_name(span.chord)
        if root is None:
            continue
        duration = max(0.0, span.end - span.start)
        if duration <= 0:
            continue
        durations[root] = durations.get(root, 0.0) + duration
        total_duration += duration
        if len(first_roots) < 8:
            first_roots.append(root)

    if total_duration <= 0:
        return None

    relative_ratio = durations.get(relative_minor_root, 0.0) / total_duration
    major_ratio = durations.get(key_root, 0.0) / total_duration
    early_relative = first_roots[:4].count(relative_minor_root)
    early_major = first_roots[:4].count(key_root)

    if (
        relative_ratio >= 0.30
        and relative_ratio >= major_ratio * 1.45
        and early_relative >= max(1, early_major)
    ):
        return f"{note_name(relative_minor_root, notation)}m"
    return None


def beat_chords_from_bar_chords(
    intervals: list[BeatInterval],
    bar_chords: dict[int, ChordSpan],
) -> list[BeatChord]:
    beats: list[BeatChord] = []
    for interval in intervals:
        span = bar_chords.get(interval.bar)
        if span is None:
            continue
        beats.append(
            BeatChord(
                bar=interval.bar,
                beat=interval.beat,
                start=interval.start,
                end=interval.end,
                chord=span.chord,
                confidence=span.confidence,
                bass_root=span.bass_root,
                bass_confidence=span.bass_confidence,
            )
        )
    return beats


def write_aligned_csv(path: Path, spans: list[ChordSpan], key: str) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "unit",
                "bar",
                "beat_start",
                "beat_end",
                "start",
                "end",
                "start_seconds",
                "end_seconds",
                "chord",
                "chord_display",
                "roman",
                "roman_display",
                "bass_root",
                "bass_confidence",
                "confidence",
            ]
        )
        for span in spans:
            writer.writerow(
                [
                    span.unit,
                    span.bar,
                    span.beat_start,
                    span.beat_end,
                    format_timestamp(span.start),
                    format_timestamp(span.end),
                    f"{span.start:.3f}",
                    f"{span.end:.3f}",
                    span.chord,
                    chord_display_name(
                        span.chord,
                        span.bass_root,
                        span.bass_confidence,
                        span.confidence,
                    ),
                    chord_to_roman(span.chord, key),
                    chord_to_roman_display(
                        span.chord,
                        key,
                        span.bass_root,
                        span.bass_confidence,
                        span.confidence,
                    ),
                    span.bass_root,
                    f"{span.bass_confidence:.3f}",
                    f"{span.confidence:.3f}",
                ]
            )


def format_bar_line(bar: int, spans: list[ChordSpan], key: str, display: str) -> str:
    if len(spans) == 1 and spans[0].unit == "bar":
        span = spans[0]
        return (
            f"{bar:03d} | {format_timestamp(span.start)}-{format_timestamp(span.end)} | "
            f"{chord_label(span.chord, key, display, span.bass_root, span.bass_confidence, span.confidence):<8} | conf={span.confidence:.2f}"
        )
    parts = []
    for span in spans:
        beat_label = (
            f"{span.beat_start}"
            if span.beat_start == span.beat_end
            else f"{span.beat_start}-{span.beat_end}"
        )
        label = chord_label(
            span.chord,
            key,
            display,
            span.bass_root,
            span.bass_confidence,
            span.confidence,
        )
        parts.append(f"beat {beat_label}: {label} ({span.confidence:.2f})")
    start = format_timestamp(spans[0].start)
    end = format_timestamp(spans[-1].end)
    return f"{bar:03d} | {start}-{end} | " + " / ".join(parts)


def write_aligned_txt(
    path: Path,
    source: Path,
    duration: float,
    key: str,
    mode: str,
    meter: MeterInfo,
    spans: list[ChordSpan],
    display: str,
) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write(f"Source: {source.name}\n")
        file.write(f"Duration: {format_timestamp(duration)}\n")
        file.write(f"Estimated BPM: {meter.bpm:.2f}\n")
        file.write(f"Estimated key: {key}\n")
        file.write(f"Estimated mode: {mode}\n")
        file.write(f"Estimated meter: {meter.meter_label} (confidence={meter.confidence:.2f})\n")
        file.write(f"Chord display: {display}\n")
        file.write("\n")
        bars = sorted({span.bar for span in spans})
        for bar in bars:
            items = [span for span in spans if span.bar == bar]
            file.write(format_bar_line(bar, items, key, display) + "\n")


def make_payload(
    source: Path,
    duration: float,
    audio_offset: float,
    key: str,
    mode: str,
    meter: MeterInfo,
    beat_chords: list[BeatChord],
    spans: list[ChordSpan],
    display: str,
    analysis_model: str,
    tempo_variant: str,
    harmonic_unit: str,
    repeat_refine: bool,
    phrase_refine: bool,
    key_candidates: list[tuple[str, str, float]],
    external_chords: list[str] | str | None = None,
    external_weight: float = 0.0,
    external_hint_count: int = 0,
    beat_source: str | None = None,
) -> dict[str, object]:
    bars = []
    for bar in sorted({item.bar for item in beat_chords}):
        beats = [item for item in beat_chords if item.bar == bar]
        bar_spans = [span for span in spans if span.bar == bar]
        bars.append(
            {
                "bar": bar,
                "start": beats[0].start,
                "end": beats[-1].end,
                "beats": [
                    {
                        "beat": item.beat,
                        "start": item.start,
                        "end": item.end,
                        "chord": item.chord,
                        "chord_display": chord_display_name(
                            item.chord,
                            item.bass_root,
                            item.bass_confidence,
                            item.confidence,
                        ),
                        "roman": chord_to_roman(item.chord, key),
                        "roman_display": chord_to_roman_display(
                            item.chord,
                            key,
                            item.bass_root,
                            item.bass_confidence,
                            item.confidence,
                        ),
                        "bass_root": item.bass_root,
                        "bass_confidence": item.bass_confidence,
                        "confidence": item.confidence,
                    }
                    for item in beats
                ],
                "spans": [
                    {
                        "unit": span.unit,
                        "beat_start": span.beat_start,
                        "beat_end": span.beat_end,
                        "start": span.start,
                        "end": span.end,
                        "chord": span.chord,
                        "chord_display": chord_display_name(
                            span.chord,
                            span.bass_root,
                            span.bass_confidence,
                            span.confidence,
                        ),
                        "roman": chord_to_roman(span.chord, key),
                        "roman_display": chord_to_roman_display(
                            span.chord,
                            key,
                            span.bass_root,
                            span.bass_confidence,
                            span.confidence,
                        ),
                        "bass_root": span.bass_root,
                        "bass_confidence": span.bass_confidence,
                        "confidence": span.confidence,
                    }
                    for span in bar_spans
                ],
            }
        )
    return {
        "source": str(source),
        "audio_file": source.name,
        "duration": duration,
        "audio_offset": audio_offset,
        "bpm": display_bpm_for_meter(meter),
        "pulse_bpm": meter.bpm,
        "key": key,
        "mode": mode,
        "meter": meter.meter_label,
        "meter_confidence": meter.confidence,
        "beat_period": meter.beat_period,
        "beats_per_bar": meter.beats_per_bar,
        "beat_grid": "tracked" if len(meter.beat_times) >= 2 else "constant",
        # A locked phase also occurs for human/reference grids. It cannot
        # establish which algorithm (if any) produced the supplied timestamps.
        "beat_source": beat_source or ("provided_locked_grid" if meter.locked_phase else "unknown"),
        "analysis_model": analysis_model,
        "tempo_variant": tempo_variant,
        "harmonic_unit": harmonic_unit,
        "repeat_refine": repeat_refine,
        "phrase_refine": phrase_refine,
        "external_chords": external_chords,
        "external_weight": external_weight,
        "external_hint_count": external_hint_count,
        "key_candidates": [
            {"tempo_variant": variant, "key": candidate_key, "score": score}
            for variant, candidate_key, score in key_candidates
        ],
        "display": display,
        "bars": bars,
        "events": [
            {
                "bar": span.bar,
                "unit": span.unit,
                "beat_start": span.beat_start,
                "beat_end": span.beat_end,
                "start": span.start,
                "end": span.end,
                "chord": span.chord,
                "chord_display": chord_display_name(
                    span.chord,
                    span.bass_root,
                    span.bass_confidence,
                    span.confidence,
                ),
                "roman": chord_to_roman(span.chord, key),
                "roman_display": chord_to_roman_display(
                    span.chord,
                    key,
                    span.bass_root,
                    span.bass_confidence,
                    span.confidence,
                ),
                "bass_root": span.bass_root,
                "bass_confidence": span.bass_confidence,
                "confidence": span.confidence,
            }
            for span in spans
        ],
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


STUDIO_ASSETS = Path(__file__).with_name("studio_assets")
HTML_TEMPLATE = (
    (STUDIO_ASSETS / "studio.html").read_text(encoding="utf-8")
    .replace("__STUDIO_CSS__", "\n".join((STUDIO_ASSETS / name).read_text(encoding="utf-8") for name in ("studio.css", "workbench.css")))
    .replace("__STUDIO_JS__", "\n".join((STUDIO_ASSETS / name).read_text(encoding="utf-8") for name in ("studio.js", "workbench.js")))
)


def write_html(path: Path, payload: dict[str, object]) -> None:
    from urllib.parse import quote

    source = Path(str(payload.get("source", "")))
    if source.is_file():
        payload = dict(payload)
        try:
            relative_source = Path(os.path.relpath(source.resolve(), path.parent.resolve())).as_posix()
            payload["audio_src"] = quote(relative_source, safe="/")
        except ValueError:
            # Windows paths on different drives cannot be made relative.
            payload["audio_src"] = source.resolve().as_uri()
    # Prevent a song title or imported label from ending the inline script.
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html_text = HTML_TEMPLATE.replace("__ANALYSIS_JSON__", data)
    path.write_text(html_text, encoding="utf-8")


def default_output_base(audio_path: Path) -> Path:
    return audio_path.with_suffix("")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize BPM, key, meter, beat-aligned chords, and build a playback viewer."
    )
    parser.add_argument("audio", type=Path, help="Input audio file: mp3, m4a, wav, flac...")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output base path. Defaults to the audio filename without extension.",
    )
    parser.add_argument(
        "--mode",
        choices=["triads", "seventh", "rich"],
        default="seventh",
        help="Chord vocabulary size. 'rich' detects more chord types but can be less stable.",
    )
    parser.add_argument(
        "--analysis-model",
        choices=["functional", "template", "acoustic"],
        default="functional",
        help="functional: restricted diatonic states; template: full --mode vocabulary; acoustic: experimental learned 24-chord model.",
    )
    parser.add_argument("--acoustic-model", type=Path, help="Trusted local model from train_acoustic_chords.py (required for acoustic).")
    parser.add_argument("--acoustic-transition-strength", type=float, default=None, help="Override learned model's validated continuity strength.")
    parser.add_argument("--no-structure", action="store_true", help="Skip lightweight acoustic section and harmonic-loop analysis.")
    parser.add_argument("--annotation-timebase", choices=["source", "clip"], default="source",
                        help="Time coordinates of --beat-grid, --downbeats and --external-chords when using --offset.")
    parser.add_argument(
        "--tempo-mode",
        choices=["auto", "detected", "half", "double"],
        default="auto",
        help="Resolve half/double tempo ambiguity. Auto evaluates the candidates jointly with key.",
    )
    parser.add_argument(
        "--harmonic-unit",
        choices=["auto", "bar", "half-bar", "compound-half", "beat"],
        default="auto",
        help="Time unit used by the functional harmony model. compound-half groups 6/8 as 1-3 and 4-6.",
    )
    parser.add_argument(
        "--no-repeat-refine",
        action="store_true",
        help="Disable the repeated-section consistency pass in the functional model.",
    )
    parser.add_argument(
        "--no-phrase-refine",
        action="store_true",
        help="Disable bar/half-bar phrase-position transition refinement in the functional model.",
    )
    parser.add_argument(
        "--notation",
        choices=["sharps", "flats"],
        default="sharps",
        help="Use sharp or flat note names.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["cqt", "fast"],
        default="cqt",
        help="Feature extractor. cqt is slower and richer; fast uses STFT chroma for batch evaluation.",
    )
    parser.add_argument(
        "--meter",
        default="auto",
        help="Time signature: auto, 4/4, 3/4, 6/8, etc. Default: auto.",
    )
    parser.add_argument(
        "--beat-source",
        choices=["librosa", "madmom"],
        default="librosa",
        help="Beat/downbeat source. Use madmom for neural downbeat tracking on difficult 3/4 or 6/8 songs.",
    )
    parser.add_argument(
        "--downbeats",
        type=Path,
        default=None,
        help="Use an existing madmom downbeat JSON instead of running downbeat tracking again.",
    )
    parser.add_argument(
        "--beat-grid",
        type=Path,
        default=None,
        help="Use an existing beat grid file with 'time beat_number' rows.",
    )
    parser.add_argument(
        "--madmom-python",
        type=Path,
        default=DEFAULT_MADMOM_PYTHON,
        help="Python executable where madmom is installed, used by --beat-source madmom.",
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=None,
        help="Use a fixed-BPM grid when no explicit --beat-grid/--downbeats is supplied.",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="Override the detected key, for example F#, F#m, Bb, Am.",
    )
    parser.add_argument("--all-key-candidates", action="store_true",
                        help="Export all independent key/meter scores for experimental key fusion.")
    parser.add_argument(
        "--display",
        choices=["name", "roman"],
        default="name",
        help="Default chord display for TXT and HTML: exact chord name or Roman numeral degree.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Continuity strength multiplier. Use 1 for looser changes, 5 for the default.",
    )
    parser.add_argument(
        "--bass-weight",
        type=float,
        default=0.15,
        help="How strongly low-frequency bass roots influence chord choice. Use 0 to disable.",
    )
    parser.add_argument(
        "--transition-strength",
        type=float,
        default=0.2,
        help="How strongly the Viterbi smoother prefers coherent chord progressions. Use 0 to disable.",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Start analysis after this many seconds.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Only analyze this many seconds.",
    )
    parser.add_argument(
        "--no-chord-threshold",
        type=float,
        default=0.025,
        help="Relative RMS threshold below which a beat is treated as silence/no chord.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.18,
        help="Minimum confidence required to emit a chord.",
    )
    parser.add_argument(
        "--progressions",
        default=None,
        help="Optional pop/functional progression hints, for example 176451,456451.",
    )
    parser.add_argument(
        "--progression-strength",
        type=float,
        default=1.0,
        help="How strongly --progressions biases candidate chords.",
    )
    parser.add_argument(
        "--external-chords",
        type=Path,
        action="append",
        default=None,
        help="Optional ACR output to fuse with the functional model. Can be repeated. Supports .lab/.txt/.csv/.json with start/end/chord columns.",
    )
    parser.add_argument(
        "--external-weight",
        type=float,
        default=0.35,
        help="Soft weight for --external-chords. Try 0.25-0.50; use 0 to ignore external hints.",
    )
    parser.add_argument(
        "--blank-intro-bars",
        type=int,
        default=0,
        help="Keep the first N bars in the timeline but force them to N/no chord.",
    )
    parser.add_argument(
        "--strict-diminished",
        action="store_true",
        help="Do not reinterpret major-key leading-tone diminished chords as dominant slash chords.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Do not write the interactive HTML viewer.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated HTML viewer in the default browser.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    for name in ("offset", "duration", "bpm", "bass_weight", "transition_strength", "progression_strength", "external_weight",
                 "no_chord_threshold", "confidence_threshold", "acoustic_transition_strength"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0 or (name in {"duration", "bpm"} and value == 0)):
            print(f"--{name.replace('_', '-')} must be finite and {'positive' if name in {'duration', 'bpm'} else 'nonnegative'}.", file=sys.stderr)
            return 2
    artifact = None
    if args.analysis_model == "acoustic":
        if args.acoustic_model is None or not args.acoustic_model.is_file():
            print("--analysis-model acoustic requires an existing --acoustic-model.", file=sys.stderr)
            return 2
        import joblib
        from acoustic_chord_model import FEATURE_SCHEMA
        artifact = joblib.load(args.acoustic_model)
        if artifact.get("feature_schema") != FEATURE_SCHEMA or artifact.get("feature_mode") != args.feature_mode:
            print("Acoustic model/extractor mismatch; use the feature mode specified in the model report.", file=sys.stderr)
            return 2
    audio_path = args.audio.resolve()
    if not audio_path.exists():
        print(f"Input file not found: {audio_path}", file=sys.stderr)
        return 2
    if args.smooth < 1:
        print("--smooth must be at least 1.", file=sys.stderr)
        return 2
    if args.smooth % 2 == 0:
        args.smooth += 1
    if args.bpm is not None and args.bpm <= 0:
        print("--bpm must be greater than 0.", file=sys.stderr)
        return 2
    if args.bass_weight < 0:
        print("--bass-weight must be zero or greater.", file=sys.stderr)
        return 2
    if args.transition_strength < 0:
        print("--transition-strength must be zero or greater.", file=sys.stderr)
        return 2
    if args.progression_strength < 0:
        print("--progression-strength must be zero or greater.", file=sys.stderr)
        return 2
    if args.external_weight < 0:
        print("--external-weight must be zero or greater.", file=sys.stderr)
        return 2
    if args.blank_intro_bars < 0:
        print("--blank-intro-bars must be zero or greater.", file=sys.stderr)
        return 2
    args.key = normalize_key_label(args.key)
    ignored_bpm = None
    if args.bpm is not None and (args.beat_grid is not None or args.downbeats is not None):
        ignored_bpm, args.bpm = args.bpm, None
        print("Using supplied beat timestamps; --bpm will not replace them with a constant grid.")
    external_hints: list[ExternalChordHint] = []
    external_chords_paths: list[Path] = []
    if args.external_chords:
        for external_arg in args.external_chords:
            external_chords_path = external_arg.resolve()
            if not external_chords_path.exists():
                print(f"External chord file not found: {external_chords_path}", file=sys.stderr)
                return 2
            try:
                file_hints = read_external_chord_hints(external_chords_path, args.notation)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"Could not read external chord file: {exc}", file=sys.stderr)
                return 2
            if not file_hints:
                print(
                    f"External chord file did not contain any usable timed chords: {external_chords_path}",
                    file=sys.stderr,
                )
                return 2
            external_hints.extend(file_hints)
            external_chords_paths.append(external_chords_path)
            print(f"Loaded {len(file_hints)} external chord hints: {external_chords_path.name}")
        external_hints.sort(key=lambda item: (item.start, item.end))

    try:
        requested_meter = parse_meter(args.meter)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    output_base = (
        args.output.resolve() if args.output is not None else default_output_base(audio_path)
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)

    print(f"Decoding: {audio_path.name}")
    y = decode_audio(
        audio_path,
        sample_rate=SAMPLE_RATE,
        offset=args.offset,
        duration=args.duration,
    )
    duration = len(y) / SAMPLE_RATE
    annotation_offset = args.offset if args.annotation_timebase == "source" else 0.0
    external_hints = [replace(hint, start=max(0.0, hint.start - annotation_offset), end=min(duration, hint.end - annotation_offset))
                      for hint in external_hints if hint.end > annotation_offset and hint.start < annotation_offset + duration]
    if duration > 20 * 60 and args.duration is None:
        print(
            "Notice: this audio is longer than 20 minutes. "
            "For better accuracy, split it by song or use --offset/--duration."
        )

    print("Estimating BPM, beats, and meter...")
    if args.beat_grid is not None:
        beat_grid_path = args.beat_grid.resolve()
        if not beat_grid_path.exists():
            print(f"Beat grid file not found: {beat_grid_path}", file=sys.stderr)
            return 2
        try:
            meter = meter_from_downbeat_rows(
                [(time - annotation_offset, beat) for time, beat in read_beat_grid_rows(beat_grid_path)],
                duration,
                requested_meter,
            )
        except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
            print(f"Could not read beat grid file: {exc}", file=sys.stderr)
            return 2
        if meter is None:
            print("Beat grid file did not contain a usable beat grid.", file=sys.stderr)
            return 2
    elif args.bpm is not None and requested_meter is not None:
        meter = fixed_meter_info(args.bpm, requested_meter)
    elif args.downbeats is not None:
        downbeat_path = args.downbeats.resolve()
        if not downbeat_path.exists():
            print(f"Downbeat file not found: {downbeat_path}", file=sys.stderr)
            return 2
        try:
            meter = meter_from_downbeat_rows(
                [(time - annotation_offset, beat) for time, beat in read_downbeat_rows(downbeat_path)],
                duration,
                requested_meter,
            )
        except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
            print(f"Could not read downbeat file: {exc}", file=sys.stderr)
            return 2
        if meter is None:
            print("Downbeat file did not contain a usable beat grid.", file=sys.stderr)
            return 2
    elif args.beat_source == "madmom":
        try:
            meter = estimate_meter_madmom(audio_path, duration, args.meter, args.madmom_python.resolve(), args.offset)
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            print(f"Could not run madmom downbeat tracking: {exc}", file=sys.stderr)
            return 2
        if meter is None:
            print("madmom downbeat tracking did not return a usable beat grid.", file=sys.stderr)
            return 2
    else:
        meter = estimate_meter(y, SAMPLE_RATE, duration, args.meter)
    if args.bpm is not None and requested_meter is None:
        meter = MeterInfo(
            bpm=args.bpm,
            meter_label=meter.meter_label,
            beats_per_bar=meter.beats_per_bar,
            beat_unit=meter.beat_unit,
            beat_period=60.0 / args.bpm,
            first_bar_start=0.0,
            confidence=meter.confidence,
            beat_times=(),
            downbeat_offset=0,
            locked_phase=False,
        )

    print("Extracting chroma features...")
    chroma = extract_chroma(y, SAMPLE_RATE, hop_length=2048, mode=args.feature_mode)
    print("Extracting bass-root features...")
    bass_chroma = extract_bass_chroma(y, SAMPLE_RATE, hop_length=2048, mode=args.feature_mode)

    print("Selecting tempo layer and key...")
    selection = select_meter_and_key(
        y,
        chroma,
        bass_chroma,
        meter,
        duration,
        args.notation,
        tempo_mode="detected" if args.bpm is not None else args.tempo_mode,
        requested_key=args.key,
        retain_all_key_candidates=args.all_key_candidates,
        transition_strength=args.transition_strength,
        external_hints=external_hints,
        external_weight=args.external_weight,
        bass_weight=args.bass_weight,
    )
    meter = selection.meter
    key = selection.key
    tonal_mode = estimate_mode_from_key(key)
    intervals = build_beat_intervals(meter, duration)

    print("Matching chords to beats and bars...")
    harmonic_unit = "mixed"
    acoustic_candidates = []
    if args.analysis_model == "acoustic":
        from acoustic_chord_model import recognize_acoustic
        harmonic_unit = artifact["harmonic_unit"] if args.harmonic_unit == "auto" else args.harmonic_unit
        if harmonic_unit != artifact["harmonic_unit"]:
            print("Acoustic model harmonic unit must match its training report.", file=sys.stderr)
            return 2
        spans, acoustic_candidates = recognize_acoustic(
            y, chroma, bass_chroma, intervals, artifact=artifact, key=key, notation=args.notation,
            harmonic_unit=harmonic_unit, beats_per_bar=meter.beats_per_bar,
            transition_strength=(artifact["transition_strength"] if args.acoustic_transition_strength is None else args.acoustic_transition_strength),
            no_chord_threshold=args.no_chord_threshold)
        spans = blank_intro_spans(spans, args.blank_intro_bars)
        beat_chords = beat_chords_from_spans(intervals, spans)
    elif args.analysis_model == "functional":
        harmonic_unit = harmonic_unit_for_meter(meter, args.harmonic_unit)
        spans = recognize_functional_spans(
            y,
            chroma,
            bass_chroma,
            intervals,
            key=key,
            notation=args.notation,
            transition_strength=args.transition_strength,
            no_chord_threshold=args.no_chord_threshold,
            confidence_threshold=args.confidence_threshold,
            harmonic_unit=harmonic_unit,
            beats_per_bar=meter.beats_per_bar,
            repeat_refine=not args.no_repeat_refine,
            phrase_refine=not args.no_phrase_refine,
            external_hints=external_hints,
            external_weight=args.external_weight,
            bass_weight=args.bass_weight,
        )
        spans = cleanup_spans(spans, key, enabled=not args.strict_diminished)
        if args.key is None:
            relative_key = relative_minor_center_key(spans, key, args.notation)
            if relative_key is not None:
                print(f"Relative minor center detected; rerunning key as {relative_key}...")
                key = relative_key
                tonal_mode = estimate_mode_from_key(key)
                spans = recognize_functional_spans(
                    y,
                    chroma,
                    bass_chroma,
                    intervals,
                    key=key,
                    notation=args.notation,
                    transition_strength=args.transition_strength,
                    no_chord_threshold=args.no_chord_threshold,
                    confidence_threshold=args.confidence_threshold,
                    harmonic_unit=harmonic_unit,
                    beats_per_bar=meter.beats_per_bar,
                    repeat_refine=not args.no_repeat_refine,
                    phrase_refine=not args.no_phrase_refine,
                    external_hints=external_hints,
                    external_weight=args.external_weight,
                    bass_weight=args.bass_weight,
                )
                spans = cleanup_spans(spans, key, enabled=not args.strict_diminished)
        spans = blank_intro_spans(spans, args.blank_intro_bars)
        beat_chords = beat_chords_from_spans(intervals, spans)
    else:
        beat_chords = recognize_beat_chords(
            y,
            chroma,
            bass_chroma,
            intervals,
            key=key,
            mode=args.mode,
            notation=args.notation,
            smooth_width=args.smooth,
            no_chord_threshold=args.no_chord_threshold,
            confidence_threshold=args.confidence_threshold,
            bass_weight=args.bass_weight,
            transition_strength=args.transition_strength,
            progressions=args.progressions,
            progression_strength=args.progression_strength,
        )
        bar_chords = recognize_bar_chords(
            y,
            chroma,
            bass_chroma,
            beat_chords,
            key=key,
            mode=args.mode,
            notation=args.notation,
            no_chord_threshold=args.no_chord_threshold,
            confidence_threshold=args.confidence_threshold,
            bass_weight=args.bass_weight,
            transition_strength=args.transition_strength,
            progressions=args.progressions,
            progression_strength=args.progression_strength,
        )
        beat_chords, bar_chords = apply_pop_harmony_cleanup(
            beat_chords,
            bar_chords,
            key,
            enabled=not args.strict_diminished,
        )
        beat_chords, bar_chords = blank_intro_chords(
            beat_chords,
            bar_chords,
            args.blank_intro_bars,
        )
        spans = build_chord_spans(beat_chords, bar_chords)
    payload = make_payload(
        audio_path,
        duration,
        args.offset,
        key,
        tonal_mode,
        meter,
        beat_chords,
        spans,
        display=args.display,
        analysis_model=args.analysis_model,
        tempo_variant=selection.tempo_variant,
        harmonic_unit=harmonic_unit,
        repeat_refine=args.analysis_model == "functional" and not args.no_repeat_refine,
        phrase_refine=args.analysis_model == "functional" and not args.no_phrase_refine,
        key_candidates=selection.key_candidates,
        external_chords=[str(path) for path in external_chords_paths] if external_chords_paths else None,
        external_weight=args.external_weight if external_hints else 0.0,
        external_hint_count=len(external_hints),
    )
    payload["feature_schema"] = "centered_chroma_bass_v1"
    from audio_waveform import attach_waveform
    attach_waveform(payload, samples=y, sample_rate=SAMPLE_RATE)
    payload["feature_mode"] = args.feature_mode
    payload["annotation_timebase"] = args.annotation_timebase
    payload["beat_source"] = ("reference_grid" if args.beat_grid else "provided_downbeats" if args.downbeats
                              else "fixed_bpm" if args.bpm is not None else args.beat_source)
    payload["confidence_kind"] = "uncalibrated_evidence_score"
    payload["key_source"] = "provided" if args.key else "functional_joint"
    if ignored_bpm is not None:
        payload["ignored_fixed_bpm"] = ignored_bpm
    if artifact is not None:
        payload["acoustic_candidates"] = acoustic_candidates
        payload["acoustic_model"] = {"path": str(args.acoustic_model.resolve()), "schema": artifact["feature_schema"],
                                     "training_songs": len(artifact["training_song_ids"]), "vocabulary": "24_major_minor"}
    if args.analysis_model == "functional":
        payload["analysis_warnings"] = ["Functional vocabulary is restricted to diatonic triads; --mode applies to template inference."]
        print("Note: functional uses restricted diatonic triads; use template for --mode vocabulary or an external neural recognizer.")
    if not args.no_structure:
        print("Analyzing repeated sections and harmonic loops...")
        from song_structure import analyze_song_structure
        payload["structure"] = analyze_song_structure(payload, y=y, sr=SAMPLE_RATE)

    csv_path = output_base.with_name(output_base.name + "_bar_chords.csv")
    txt_path = output_base.with_name(output_base.name + "_bar_chords.txt")
    json_path = output_base.with_name(output_base.name + "_analysis.json")
    write_aligned_csv(csv_path, spans, key)
    write_aligned_txt(
        txt_path,
        audio_path,
        duration,
        key,
        tonal_mode,
        meter,
        spans,
        display=args.display,
    )
    write_json(json_path, payload)

    display_bpm = display_bpm_for_meter(meter)
    if abs(display_bpm - meter.bpm) > 0.01:
        print(f"Estimated BPM: {display_bpm:.2f} (pulse={meter.bpm:.2f}, {selection.tempo_variant})")
    else:
        print(f"Estimated BPM: {meter.bpm:.2f} ({selection.tempo_variant})")
    print(f"Estimated key/mode: {key} {tonal_mode}")
    print(f"Estimated meter: {meter.meter_label} (confidence={meter.confidence:.2f})")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {txt_path}")
    print(f"Wrote: {json_path}")

    if not args.no_html:
        html_path = output_base.with_name(output_base.name + "_studio.html")
        write_html(html_path, payload)
        print(f"Wrote: {html_path}")
        if args.open:
            import webbrowser

            webbrowser.open(html_path.resolve().as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
