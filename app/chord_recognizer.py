#!/usr/bin/env python
"""Local chord recognition from audio files.

This is a practical template-matching recognizer:
1. Decode audio with ffmpeg.
2. Extract harmonic chroma features with librosa.
3. Match chroma vectors against chord templates.
4. Smooth and merge adjacent predictions.

It is meant to create a useful first draft, not a commercial-grade result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from audio_features import centered_magnitude_spectrum


SAMPLE_RATE = 22050
NOTE_SHARPS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_FLATS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


@dataclass(frozen=True)
class ChordTemplate:
    name: str
    pitch_classes: tuple[int, ...]
    vector: np.ndarray


@dataclass
class ChordSegment:
    start: float
    end: float
    chord: str
    confidence: float


def find_ffmpeg() -> str:
    """Find an ffmpeg executable.

    The user previously installed imageio-ffmpeg in this environment, so this
    also falls back to that package when ffmpeg is not on PATH.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Could not find ffmpeg. Install ffmpeg or run: "
            "python -m pip install --user imageio-ffmpeg"
        ) from exc


def decode_audio(
    audio_path: Path,
    sample_rate: int = SAMPLE_RATE,
    offset: float = 0.0,
    duration: float | None = None,
) -> np.ndarray:
    """Decode audio to mono float32 samples using ffmpeg."""
    if not math.isfinite(offset) or offset < 0:
        raise ValueError("Audio offset must be finite and nonnegative")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("Audio duration must be finite and positive")
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer")
    ffmpeg = find_ffmpeg()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    command.extend(["-i", str(audio_path)])
    # Output-side seeking decodes the codec's preceding frames and preserves
    # resampler state. Input-side -ss produced up to ~40 ms of warm-up zeros in
    # MP3 excerpts, so a clip did not match the same samples in a full decode.
    if offset > 0:
        command.extend(["-ss", f"{offset:.6f}"])
    if duration is not None:
        command.extend(["-t", f"{duration:.6f}"])
    command.extend(
        [
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ]
    )

    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg could not decode this file:\n{message}")

    samples = np.frombuffer(completed.stdout, dtype=np.float32)
    if samples.size == 0:
        raise RuntimeError("Decoded audio is empty.")
    return samples


def normalize_template(values: Iterable[float]) -> np.ndarray:
    vector = np.asarray(list(values), dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm


def rotate_template(interval_weights: dict[int, float], root: int) -> np.ndarray:
    values = np.full(12, -0.08, dtype=np.float32)
    for interval, weight in interval_weights.items():
        values[(root + interval) % 12] = weight
    values = np.maximum(values, 0.0)
    return normalize_template(values)


def build_templates(mode: str = "seventh", notation: str = "sharps") -> list[ChordTemplate]:
    notes = NOTE_FLATS if notation == "flats" else NOTE_SHARPS
    qualities: list[tuple[str, dict[int, float]]] = [
        ("", {0: 1.0, 4: 0.82, 7: 0.9}),
        ("m", {0: 1.0, 3: 0.82, 7: 0.9}),
        ("dim", {0: 1.0, 3: 0.78, 6: 0.82}),
        ("aug", {0: 1.0, 4: 0.78, 8: 0.82}),
        ("sus2", {0: 1.0, 2: 0.72, 7: 0.9}),
        ("sus4", {0: 1.0, 5: 0.72, 7: 0.9}),
    ]

    if mode in {"seventh", "rich"}:
        qualities.extend(
            [
                ("7", {0: 1.0, 4: 0.75, 7: 0.86, 10: 0.62}),
                ("maj7", {0: 1.0, 4: 0.75, 7: 0.86, 11: 0.62}),
                ("m7", {0: 1.0, 3: 0.75, 7: 0.86, 10: 0.62}),
                ("m7b5", {0: 1.0, 3: 0.72, 6: 0.78, 10: 0.62}),
                ("dim7", {0: 1.0, 3: 0.72, 6: 0.78, 9: 0.62}),
            ]
        )

    if mode == "rich":
        qualities.extend(
            [
                ("mMaj7", {0: 1.0, 3: 0.75, 7: 0.86, 11: 0.62}),
                ("6", {0: 1.0, 4: 0.72, 7: 0.82, 9: 0.55}),
                ("m6", {0: 1.0, 3: 0.72, 7: 0.82, 9: 0.55}),
                ("add9", {0: 1.0, 2: 0.45, 4: 0.72, 7: 0.82}),
                ("madd9", {0: 1.0, 2: 0.45, 3: 0.72, 7: 0.82}),
                ("9", {0: 1.0, 2: 0.45, 4: 0.68, 7: 0.8, 10: 0.56}),
                ("m9", {0: 1.0, 2: 0.45, 3: 0.68, 7: 0.8, 10: 0.56}),
            ]
        )

    templates: list[ChordTemplate] = []
    for root in range(12):
        for suffix, weights in qualities:
            name = f"{notes[root]}{suffix}"
            pitch_classes = tuple((root + interval) % 12 for interval in weights)
            templates.append(
                ChordTemplate(
                    name=name,
                    pitch_classes=pitch_classes,
                    vector=rotate_template(weights, root),
                )
            )
    return templates


def estimate_key(chroma: np.ndarray, notation: str = "sharps") -> str:
    """Estimate a broad key from mean chroma using Krumhansl profiles."""
    notes = NOTE_FLATS if notation == "flats" else NOTE_SHARPS
    major_profile = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        dtype=np.float32,
    )
    minor_profile = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
        dtype=np.float32,
    )
    mean_chroma = np.mean(chroma, axis=1)
    if np.linalg.norm(mean_chroma) == 0:
        return "Unknown"
    mean_chroma = (mean_chroma - np.mean(mean_chroma)) / (np.std(mean_chroma) + 1e-9)

    best_score = -float("inf")
    best_key = "Unknown"
    for root in range(12):
        for label, profile in (("major", major_profile), ("minor", minor_profile)):
            rolled = np.roll(profile, root)
            rolled = (rolled - np.mean(rolled)) / (np.std(rolled) + 1e-9)
            score = float(np.dot(mean_chroma, rolled))
            if score > best_score:
                best_score = score
                suffix = "" if label == "major" else "m"
                best_key = f"{notes[root]}{suffix}"
    return best_key


def extract_chroma(y: np.ndarray, sr: int, hop_length: int, mode: str = "cqt") -> np.ndarray:
    if mode == "fast":
        n_fft = 4096
        magnitude = centered_magnitude_spectrum(y, n_fft, hop_length)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        mask = (freqs >= 65.0) & (freqs <= 2093.0)
        magnitude = magnitude[mask]
        freqs = freqs[mask]
        chroma = np.zeros((12, magnitude.shape[1]), dtype=np.float32)
        if magnitude.size:
            midi = np.rint(69.0 + 12.0 * np.log2(np.maximum(freqs, 1e-6) / 440.0)).astype(int)
            for bin_index, pc in enumerate(midi % 12):
                chroma[int(pc)] += magnitude[bin_index]
        chroma = np.maximum(chroma, 0.0)
        chroma /= np.max(chroma, axis=0, keepdims=True) + 1e-9
        return chroma
    if mode != "cqt":
        raise ValueError("mode must be 'cqt' or 'fast'")

    import librosa

    y = librosa.util.normalize(y)
    y_harmonic = librosa.effects.harmonic(y, margin=8.0)
    chroma = librosa.feature.chroma_cqt(
        y=y_harmonic,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=36,
        n_chroma=12,
    )
    chroma = np.maximum(chroma, 0.0)
    chroma /= np.max(chroma, axis=0, keepdims=True) + 1e-9
    return chroma


def segment_chroma(
    chroma: np.ndarray,
    y: np.ndarray,
    sr: int,
    hop_length: int,
    segment_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import librosa

    frame_times = librosa.frames_to_time(
        np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length
    )
    duration = len(y) / sr
    edges = np.arange(0.0, duration + segment_seconds, segment_seconds)
    if edges[-1] < duration:
        edges = np.append(edges, duration)
    if edges.size < 2:
        edges = np.array([0.0, duration], dtype=np.float32)

    segment_vectors = []
    starts = []
    ends = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        mask = (frame_times >= start) & (frame_times < end)
        if not np.any(mask):
            frame_idx = int(np.argmin(np.abs(frame_times - start)))
            values = chroma[:, frame_idx : frame_idx + 1]
        else:
            values = chroma[:, mask]
        vector = np.median(values, axis=1)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        segment_vectors.append(vector)
        starts.append(float(start))
        ends.append(float(min(end, duration)))
    return np.asarray(segment_vectors), np.asarray(starts), np.asarray(ends)


def rms_by_segment(
    y: np.ndarray, sr: int, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    values = []
    for start, end in zip(starts, ends):
        left = max(0, int(start * sr))
        right = min(len(y), int(end * sr))
        if right <= left:
            values.append(0.0)
            continue
        chunk = y[left:right]
        values.append(float(np.sqrt(np.mean(chunk * chunk))))
    return np.asarray(values, dtype=np.float32)


def match_chords(
    segment_vectors: np.ndarray,
    rms: np.ndarray,
    templates: list[ChordTemplate],
    no_chord_threshold: float,
    confidence_threshold: float,
) -> tuple[list[str], np.ndarray]:
    template_matrix = np.vstack([template.vector for template in templates])
    scores = segment_vectors @ template_matrix.T
    order = np.argsort(scores, axis=1)
    best_idx = order[:, -1]
    second_idx = order[:, -2]
    best_scores = scores[np.arange(scores.shape[0]), best_idx]
    second_scores = scores[np.arange(scores.shape[0]), second_idx]

    rms_floor = float(np.quantile(rms, 0.05)) if rms.size else 0.0
    rms_peak = float(np.quantile(rms, 0.95)) if rms.size else 1.0
    rms_norm = (rms - rms_floor) / max(rms_peak - rms_floor, 1e-9)
    rms_norm = np.clip(rms_norm, 0.0, 1.0)
    silence_cutoff = max(no_chord_threshold * max(rms_peak, 1e-9), 1e-5)

    margin = np.clip((best_scores - second_scores) * 3.0, 0.0, 1.0)
    confidence = np.clip(
        (best_scores * 0.82 + margin * 0.18) * (0.65 + 0.35 * rms_norm),
        0.0,
        1.0,
    )

    chords = []
    for i, template_index in enumerate(best_idx):
        if rms[i] < silence_cutoff or confidence[i] < confidence_threshold:
            chords.append("N")
        else:
            chords.append(templates[int(template_index)].name)
    return chords, confidence


def smooth_chords(chords: list[str], confidence: np.ndarray, width: int = 3) -> list[str]:
    if width <= 1 or len(chords) < 3:
        return chords
    radius = width // 2
    smoothed = chords[:]
    for i in range(len(chords)):
        left = max(0, i - radius)
        right = min(len(chords), i + radius + 1)
        votes: dict[str, float] = {}
        for chord, weight in zip(chords[left:right], confidence[left:right]):
            votes[chord] = votes.get(chord, 0.0) + float(weight) + 0.05
        smoothed[i] = max(votes.items(), key=lambda item: item[1])[0]
    return smoothed


def merge_segments(
    starts: np.ndarray,
    ends: np.ndarray,
    chords: list[str],
    confidence: np.ndarray,
    min_duration: float,
) -> list[ChordSegment]:
    if not chords:
        return []

    merged: list[ChordSegment] = []
    current = ChordSegment(
        start=float(starts[0]),
        end=float(ends[0]),
        chord=chords[0],
        confidence=float(confidence[0]),
    )
    current_count = 1

    for start, end, chord, conf in zip(starts[1:], ends[1:], chords[1:], confidence[1:]):
        if chord == current.chord:
            current.end = float(end)
            current.confidence = (
                current.confidence * current_count + float(conf)
            ) / (current_count + 1)
            current_count += 1
            continue
        merged.append(current)
        current = ChordSegment(
            start=float(start),
            end=float(end),
            chord=chord,
            confidence=float(conf),
        )
        current_count = 1
    merged.append(current)

    if min_duration <= 0 or len(merged) < 2:
        return merged

    cleaned: list[ChordSegment] = []
    for segment in merged:
        duration = segment.end - segment.start
        if (
            cleaned
            and duration < min_duration
            and segment.chord != "N"
            and cleaned[-1].chord != "N"
        ):
            previous = cleaned[-1]
            total_duration = (previous.end - previous.start) + duration
            previous.confidence = (
                previous.confidence * (previous.end - previous.start)
                + segment.confidence * duration
            ) / max(total_duration, 1e-9)
            previous.end = segment.end
        else:
            cleaned.append(segment)
    return cleaned


def format_timestamp(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def write_csv(path: Path, segments: list[ChordSegment]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["start", "end", "start_seconds", "end_seconds", "chord", "confidence"])
        for segment in segments:
            writer.writerow(
                [
                    format_timestamp(segment.start),
                    format_timestamp(segment.end),
                    f"{segment.start:.3f}",
                    f"{segment.end:.3f}",
                    segment.chord,
                    f"{segment.confidence:.3f}",
                ]
            )


def write_txt(path: Path, segments: list[ChordSegment], source: Path, key: str) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write(f"Source: {source.name}\n")
        file.write(f"Estimated key: {key}\n")
        file.write("\n")
        for segment in segments:
            file.write(
                f"{format_timestamp(segment.start)} - {format_timestamp(segment.end)}  "
                f"{segment.chord:<8}  confidence={segment.confidence:.2f}\n"
            )


def write_json(path: Path, segments: list[ChordSegment], source: Path, key: str) -> None:
    payload = {
        "source": str(source),
        "estimated_key": key,
        "segments": [
            {
                "start": segment.start,
                "end": segment.end,
                "chord": segment.chord,
                "confidence": segment.confidence,
            }
            for segment in segments
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def default_output_base(audio_path: Path) -> Path:
    return audio_path.with_suffix("")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize chord progressions from a local audio file."
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
        "--notation",
        choices=["sharps", "flats"],
        default="sharps",
        help="Use sharp or flat note names.",
    )
    parser.add_argument(
        "--segment",
        type=float,
        default=1.0,
        help="Analysis segment length in seconds.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.2,
        help="Merge very short chord segments into the previous chord when possible.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=3,
        help="Odd smoothing width in segments. Use 1 to disable.",
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
        help="Relative RMS threshold below which a segment is treated as silence/no chord.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.18,
        help="Minimum confidence required to emit a chord.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also write a JSON result file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    audio_path = args.audio.resolve()
    if not audio_path.exists():
        print(f"Input file not found: {audio_path}", file=sys.stderr)
        return 2
    if args.segment <= 0:
        print("--segment must be greater than 0.", file=sys.stderr)
        return 2
    if args.smooth < 1:
        print("--smooth must be at least 1.", file=sys.stderr)
        return 2
    if args.smooth % 2 == 0:
        args.smooth += 1

    output_base = (
        args.output.resolve() if args.output is not None else default_output_base(audio_path)
    )

    print(f"Decoding: {audio_path.name}")
    y = decode_audio(
        audio_path,
        sample_rate=SAMPLE_RATE,
        offset=args.offset,
        duration=args.duration,
    )
    duration = len(y) / SAMPLE_RATE
    if duration > 20 * 60 and args.duration is None:
        print(
            "Notice: this audio is longer than 20 minutes. "
            "For better accuracy, split it by song or use --offset/--duration."
        )

    hop_length = 2048
    print("Extracting chroma features...")
    chroma = extract_chroma(y, SAMPLE_RATE, hop_length=hop_length)
    key = estimate_key(chroma, args.notation)

    print("Matching chord templates...")
    vectors, starts, ends = segment_chroma(
        chroma,
        y,
        SAMPLE_RATE,
        hop_length=hop_length,
        segment_seconds=args.segment,
    )
    rms = rms_by_segment(y, SAMPLE_RATE, starts, ends)
    templates = build_templates(mode=args.mode, notation=args.notation)
    chords, confidence = match_chords(
        vectors,
        rms,
        templates,
        no_chord_threshold=args.no_chord_threshold,
        confidence_threshold=args.confidence_threshold,
    )
    chords = smooth_chords(chords, confidence, width=args.smooth)
    segments = merge_segments(
        starts,
        ends,
        chords,
        confidence,
        min_duration=args.min_duration,
    )

    csv_path = output_base.with_name(output_base.name + "_chords.csv")
    txt_path = output_base.with_name(output_base.name + "_chords.txt")
    write_csv(csv_path, segments)
    write_txt(txt_path, segments, audio_path, key)
    print(f"Estimated key: {key}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {txt_path}")

    if args.json:
        json_path = output_base.with_name(output_base.name + "_chords.json")
        write_json(json_path, segments, audio_path, key)
        print(f"Wrote: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
