from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chord_recognizer import SAMPLE_RATE, decode_audio
from chord_studio import (
    chord_to_roman,
    estimate_mode_from_key,
    note_name_for_pc,
    parse_chord_name,
    parse_progression_hints,
    write_html,
)


COMMON_PROGRESSIONS = [
    "6415",
    "6545",
    "1564",
    "1645",
    "4536",
    "456451",
    "176451",
]
DEFAULT_KEYS = ["C", "D", "E", "F", "G", "A", "Bb", "B", "Db", "Eb", "F#", "Ab"]
PROGRESSION_PRIOR = {
    "6415": 0.18,
    "6545": 0.16,
    "1564": 0.11,
    "1645": 0.08,
    "4536": 0.02,
    "456451": -0.08,
    "176451": 0.04,
}


@dataclass(frozen=True)
class ExternalChord:
    start: float
    end: float
    chord: str
    root_pc: int | None


@dataclass(frozen=True)
class SectionCandidate:
    anchor: float
    key: str
    progression: str
    score: float
    energy_score: float
    loop_score: float
    downbeat_score: float
    anchor_score: float


@dataclass(frozen=True)
class MeterSectionCandidate:
    meter_label: str
    beats_per_bar: int
    section: SectionCandidate


def read_lab(path: Path | None) -> list[ExternalChord]:
    if path is None or not path.exists():
        return []
    chords: list[ExternalChord] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        start = float(parts[0])
        end = float(parts[1])
        chord = parts[2].replace(":maj", "").replace(":min", "m")
        root_pc, _ = parse_chord_name(chord)
        chords.append(ExternalChord(start, end, chord, root_pc))
    return chords


def frame_rms(y: np.ndarray, sr: int, frame_seconds: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    frame = max(1, int(sr * frame_seconds))
    hop = frame
    if y.size < frame:
        return np.asarray([0.0]), np.asarray([float(np.sqrt(np.mean(y**2)))])
    starts = np.arange(0, y.size - frame + 1, hop)
    values = np.asarray([float(np.sqrt(np.mean(y[start : start + frame] ** 2))) for start in starts])
    times = starts / sr
    return times, values


def energy_at(times: np.ndarray, values: np.ndarray, time: float, window: float = 1.2) -> float:
    mask = (times >= time) & (times < time + window)
    if not np.any(mask):
        return 0.0
    return float(np.median(values[mask]))


def energy_jump_score(times: np.ndarray, values: np.ndarray, time: float) -> float:
    before = energy_at(times, values, max(0.0, time - 2.5), 2.0)
    after = energy_at(times, values, time, 2.0)
    floor = float(np.percentile(values, 15)) + 1e-8
    return float(np.clip((after - before) / (after + floor), -1.0, 1.0))


def stable_external_starts(chords: list[ExternalChord], max_time: float) -> list[float]:
    starts = []
    for item in chords:
        if item.start > max_time:
            break
        if item.root_pc is None:
            continue
        if item.end - item.start < 0.45:
            continue
        starts.append(item.start)
    return starts


def chord_anchor_score(chords: list[ExternalChord], anchor: float, max_time: float) -> float:
    starts = stable_external_starts(chords, max_time)
    if not starts:
        return 0.0
    distance = min(abs(anchor - start) for start in starts)
    return float(np.clip(1.0 - distance / 0.18, 0.0, 1.0))


def chord_root_coverage(chords: list[ExternalChord], start: float, end: float) -> dict[int, float]:
    coverage: dict[int, float] = {}
    for item in chords:
        if item.root_pc is None or item.end <= start or item.start >= end:
            continue
        overlap = min(end, item.end) - max(start, item.start)
        if overlap > 0:
            coverage[item.root_pc] = coverage.get(item.root_pc, 0.0) + overlap
    return coverage


def progression_roots(key: str, progression: str) -> list[int]:
    hints = parse_progression_hints(progression, key)
    return [hint.root_pc for hint in hints]


def major_scale_pcs(key: str) -> set[int]:
    root, suffix = parse_chord_name(key)
    if root is None:
        return set()
    if suffix == "m":
        offsets = [0, 2, 3, 5, 7, 8, 10]
    else:
        offsets = [0, 2, 4, 5, 7, 9, 11]
    return {(root + offset) % 12 for offset in offsets}


def score_key_fit(
    chords: list[ExternalChord],
    anchor: float,
    beat_period: float,
    beats_per_bar: int,
    key: str,
    bars: int,
) -> float:
    scale = major_scale_pcs(key)
    if not scale:
        return 0.0
    start = anchor
    end = anchor + bars * beats_per_bar * beat_period
    coverage = chord_root_coverage(chords, start, end)
    total = sum(coverage.values())
    if total <= 0:
        return 0.0
    in_key = sum(value for pc, value in coverage.items() if pc in scale)
    return float((in_key / total) * 2.0 - 1.0)


def score_loop(
    chords: list[ExternalChord],
    anchor: float,
    beat_period: float,
    beats_per_bar: int,
    key: str,
    progression: str,
    bars: int,
) -> float:
    roots = progression_roots(key, progression)
    if not roots:
        return 0.0
    bar_duration = beat_period * beats_per_bar
    scores = []
    for index in range(bars):
        start = anchor + index * bar_duration
        end = start + bar_duration
        coverage = chord_root_coverage(chords, start, end)
        total = sum(coverage.values())
        if total <= 0:
            continue
        expected = roots[index % len(roots)]
        direct = coverage.get(expected, 0.0) / total
        # ACR often hears iii for V or vi/iii in sparse acoustic textures; keep
        # this weak instead of letting it dominate the structural decision.
        scores.append(direct)
    if not scores:
        return 0.0
    return float(np.mean(scores))


def score_first_root(
    chords: list[ExternalChord],
    anchor: float,
    beat_period: float,
    beats_per_bar: int,
    key: str,
    progression: str,
) -> float:
    roots = progression_roots(key, progression)
    if not roots:
        return 0.0
    coverage = chord_root_coverage(chords, anchor, anchor + beat_period * beats_per_bar * 0.75)
    total = sum(coverage.values())
    if total <= 0:
        return 0.0
    expected = roots[0]
    return float(coverage.get(expected, 0.0) / total * 2.0 - 1.0)


def beat_strength_from_audio(times: np.ndarray, values: np.ndarray, beat_time: float) -> float:
    return energy_at(times, values, beat_time, 0.18)


def score_downbeat_pattern(
    times: np.ndarray,
    values: np.ndarray,
    anchor: float,
    beat_period: float,
    beats_per_bar: int,
    bars: int,
) -> float:
    downbeats = []
    offbeats = []
    for bar in range(bars):
        for beat in range(beats_per_bar):
            strength = beat_strength_from_audio(times, values, anchor + (bar * beats_per_bar + beat) * beat_period)
            if beat == 0:
                downbeats.append(strength)
            else:
                offbeats.append(strength)
    if not downbeats or not offbeats:
        return 0.0
    spread = float(np.std(downbeats + offbeats)) + 1e-8
    return float(np.clip((np.median(downbeats) - np.median(offbeats)) / spread, -1.0, 1.0))


def candidate_anchors(
    chords: list[ExternalChord],
    times: np.ndarray,
    values: np.ndarray,
    beat_period: float,
    max_time: float,
) -> list[float]:
    starts = stable_external_starts(chords, max_time)
    if not starts:
        threshold = np.percentile(values, 20) + (np.percentile(values, 85) - np.percentile(values, 20)) * 0.35
        starts = [float(times[index]) for index in np.flatnonzero(values > threshold)[:12]]
    anchors = set()
    for start in starts[:18]:
        for shift in range(-4, 5):
            anchor = start + shift * beat_period
            if 0.0 <= anchor <= max_time:
                anchors.add(round(anchor, 3))
    return sorted(anchors)


def find_best_section_start(
    chords: list[ExternalChord],
    times: np.ndarray,
    values: np.ndarray,
    beat_period: float,
    beats_per_bar: int,
    keys: list[str],
    progressions: list[str],
    max_time: float = 35.0,
    bars: int = 12,
) -> SectionCandidate:
    candidates = []
    for anchor in candidate_anchors(chords, times, values, beat_period, max_time):
        energy = energy_jump_score(times, values, anchor)
        downbeat = score_downbeat_pattern(times, values, anchor, beat_period, beats_per_bar, bars)
        anchor_fit = chord_anchor_score(chords, anchor, max_time)
        for key in keys:
            key_fit = score_key_fit(chords, anchor, beat_period, beats_per_bar, key, bars)
            for progression in progressions:
                loop = score_loop(chords, anchor, beat_period, beats_per_bar, key, progression, bars)
                first_root = score_first_root(chords, anchor, beat_period, beats_per_bar, key, progression)
                cycle_penalty = 0.018 * max(0, len(progression) - 4)
                score = (
                    loop * 2.6
                    + first_root * 0.48
                    + key_fit * 0.55
                    + PROGRESSION_PRIOR.get(progression, 0.0)
                    + energy * 0.45
                    + downbeat * 0.25
                    + anchor_fit * 0.22
                    - cycle_penalty
                    - max(0.0, anchor - 12.0) * 0.01
                )
                candidates.append(SectionCandidate(anchor, key, progression, score, energy, loop, downbeat, anchor_fit))
    if not candidates:
        return SectionCandidate(0.0, keys[0], progressions[0], 0.0, 0.0, 0.0, 0.0, 0.0)
    best = max(candidates, key=lambda item: item.score)
    eligible = [
        item
        for item in candidates
        if item.score >= best.score - 0.34
        and item.energy_score > 0.20
        and item.anchor_score > 0.60
        and item.anchor <= best.anchor
    ]
    if eligible:
        earliest_anchor = min(item.anchor for item in eligible)
        same_anchor = [item for item in eligible if abs(item.anchor - earliest_anchor) < 1e-6]
        return max(same_anchor, key=lambda item: item.score)
    return best


def meter_candidates(value: str) -> list[tuple[str, int]]:
    text = value.strip().lower()
    if text == "auto":
        return [("3/4", 3), ("4/4", 4), ("6/8", 6)]
    if "/" not in text:
        raise ValueError("--meter must be auto or something like 3/4, 4/4, 6/8")
    top, _ = text.split("/", 1)
    return [(value, int(top))]


def choose_meter_and_section(
    chords: list[ExternalChord],
    times: np.ndarray,
    rms: np.ndarray,
    beat_period: float,
    keys: list[str],
    progressions: list[str],
    max_entry_time: float,
    meter_arg: str,
) -> MeterSectionCandidate:
    candidates = []
    for meter_label, beats_per_bar in meter_candidates(meter_arg):
        section = find_best_section_start(
            chords,
            times,
            rms,
            beat_period,
            beats_per_bar,
            keys,
            progressions,
            max_time=max_entry_time,
        )
        # Prefer the meter that explains the most stable repeated harmony. A
        # wrong beats-per-bar can still have plausible BPM, but its loop score
        # collapses because the chord cycle is stretched over the wrong bars.
        score = section.score + section.loop_score * 1.2
        if meter_label == "4/4":
            score += 0.03
        candidates.append((score, meter_label, beats_per_bar, section))
    if not candidates:
        fallback = SectionCandidate(0.0, keys[0], progressions[0], 0.0, 0.0, 0.0, 0.0, 0.0)
        return MeterSectionCandidate("4/4", 4, fallback)
    _, meter_label, beats_per_bar, section = max(candidates, key=lambda item: item[0])
    return MeterSectionCandidate(meter_label, beats_per_bar, section)


def detect_modulations(
    chords: list[ExternalChord],
    anchor: float,
    beat_period: float,
    beats_per_bar: int,
    base_key: str,
    progression: str,
    keys: list[str],
    duration: float,
    window_bars: int = 8,
    min_modulation_bar: int = 1,
) -> list[dict[str, object]]:
    bar_duration = beat_period * beats_per_bar
    total_bars = max(1, int(math.floor((duration - anchor) / bar_duration)))
    sections = [
        {
            "start_bar": 1,
            "end_bar": None,
            "key": base_key,
            "mode": estimate_mode_from_key(base_key),
            "progression": progression,
            "unit": "bar",
            "phase": 0,
            "reason": "best body-entry progression fit",
        }
    ]
    current_key = base_key
    bar = 1 + window_bars
    while bar < total_bars - window_bars + 1:
        section_anchor = anchor + (bar - 1) * bar_duration
        base_score = score_loop(chords, section_anchor, beat_period, beats_per_bar, current_key, progression, window_bars)
        scored = [
            (key, score_loop(chords, section_anchor, beat_period, beats_per_bar, key, progression, window_bars))
            for key in keys
            if key != current_key
        ]
        if not scored:
            bar += 1
            continue
        next_key, next_score = max(scored, key=lambda item: item[1])
        future_edges = []
        for lookahead in range(0, 8):
            future_anchor = anchor + (bar + lookahead - 1) * bar_duration
            future_base = score_loop(chords, future_anchor, beat_period, beats_per_bar, current_key, progression, window_bars)
            future_next = score_loop(chords, future_anchor, beat_period, beats_per_bar, next_key, progression, window_bars)
            future_edges.append(future_next - future_base)
        support_count = sum(1 for edge in future_edges if edge > 0.06)
        average_edge = float(np.mean(future_edges)) if future_edges else 0.0
        first_root = score_first_root(chords, section_anchor, beat_period, beats_per_bar, next_key, progression)
        if (
            bar >= min_modulation_bar
            and next_score >= 0.34
            and next_score > base_score + 0.10
            and support_count >= 4
            and average_edge > 0.06
            and (first_root > -0.20 or (support_count >= 5 and average_edge > 0.10))
        ):
            if bar - int(sections[-1]["start_bar"]) < 16:
                bar += 1
                continue
            sections[-1]["end_bar"] = bar - 1
            sections.append(
                {
                    "start_bar": bar,
                    "end_bar": None,
                    "key": next_key,
                    "mode": estimate_mode_from_key(next_key),
                    "progression": progression,
                    "unit": "bar",
                    "phase": 0,
                    "reason": f"sliding {window_bars}-bar local-key vote {next_score:.2f} > {base_score:.2f}, support={support_count}/8",
                }
            )
            current_key = next_key
            bar += max(window_bars, 16)
            continue
        bar += 1
    return sections


def write_body_beat_grid(path: Path, anchor: float, beat_period: float, beats_per_bar: int, duration: float) -> None:
    lines = []
    beat_index = 0
    time = anchor
    while time <= duration + beat_period:
        beat = beat_index % beats_per_bar + 1
        lines.append(f"{time:.6f} {beat}")
        beat_index += 1
        time = anchor + beat_index * beat_period
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_duration(audio_path: Path) -> tuple[np.ndarray, float]:
    y = decode_audio(audio_path)
    return y, float(len(y) / SAMPLE_RATE)


def run_analysis(
    audio_path: Path,
    output_base: Path,
    beat_grid: Path,
    section: SectionCandidate,
    external_lab: Path | None,
    meter: str,
) -> Path:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "chord_studio.py"),
        str(audio_path),
        "-o",
        str(output_base),
        "--analysis-model",
        "functional",
        "--mode",
        "triads",
        "--harmonic-unit",
        "bar",
        "--feature-mode",
        "fast",
        "--display",
        "roman",
        "--meter",
        meter,
        "--beat-grid",
        str(beat_grid),
        "--key",
        section.key,
        "--progressions",
        section.progression,
        "--progression-strength",
        "3.0",
        "--transition-strength",
        "0.8",
    ]
    if external_lab is not None:
        command += ["--external-chords", str(external_lab), "--external-weight", "0.35"]
    subprocess.run(command, check=True)
    return output_base.with_name(output_base.name + "_analysis.json")


def apply_sections(payload_path: Path, output_path: Path, sections: list[dict[str, object]]) -> None:
    from apply_progression_sections import apply_section

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["modulations"] = []
    for item in sections:
        start_bar = int(item["start_bar"])
        end_bar = int(item["end_bar"]) if item.get("end_bar") is not None else None
        key = str(item["key"])
        progression = str(item["progression"])
        apply_section(payload, start_bar, end_bar, key, progression, "bar", 0)
        payload["modulations"].append(item)
    labels = [f"{item['key']} {item.get('mode') or estimate_mode_from_key(str(item['key']))}" for item in payload["modulations"]]
    payload["key"] = " -> ".join(dict.fromkeys(labels))
    payload["mode"] = "modulating" if len(labels) > 1 else payload["modulations"][0].get("mode", "major")
    payload["structure_refine"] = {
        "method": "energy+external-chord progression-grid search",
        "source_analysis": str(payload_path),
    }
    payload["events"] = sorted(
        [
            {**span, "bar": int(bar.get("bar", span.get("bar", 0)))}
            for bar in payload.get("bars", [])
            for span in bar.get("spans", [])
        ],
        key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))),
    )
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(output_path.with_name(output_path.stem.replace("_analysis", "") + "_studio.html"), payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect body entrance, rebuild a stable beat grid, and optionally render a refined Chord Studio analysis.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output base path.")
    parser.add_argument("--external-chords", type=Path)
    parser.add_argument("--bpm", type=float, required=True)
    parser.add_argument("--meter", default="auto")
    parser.add_argument("--keys", default=",".join(DEFAULT_KEYS))
    parser.add_argument("--progressions", default=",".join(COMMON_PROGRESSIONS))
    parser.add_argument("--max-entry-time", type=float, default=35.0)
    parser.add_argument("--min-modulation-bar", type=int, default=1)
    parser.add_argument("--run-analysis", action="store_true")
    args = parser.parse_args()

    beat_period = 60.0 / float(args.bpm)
    keys = [item.strip() for item in args.keys.split(",") if item.strip()]
    progressions = [item.strip() for item in args.progressions.split(",") if item.strip()]
    y, duration = load_duration(args.audio)
    times, rms = frame_rms(y, SAMPLE_RATE)
    chords = read_lab(args.external_chords)

    meter_section = choose_meter_and_section(
        chords,
        times,
        rms,
        beat_period,
        keys,
        progressions,
        args.max_entry_time,
        args.meter,
    )
    section = meter_section.section
    beats_per_bar = meter_section.beats_per_bar
    meter_label = meter_section.meter_label
    sections = detect_modulations(
        chords,
        section.anchor,
        beat_period,
        beats_per_bar,
        section.key,
        section.progression,
        keys,
        duration,
        min_modulation_bar=args.min_modulation_bar,
    )

    output_base = args.output
    output_base.parent.mkdir(parents=True, exist_ok=True)
    beat_grid = output_base.with_name(output_base.name + "_body_grid.beats")
    report_path = output_base.with_name(output_base.name + "_structure_report.json")
    write_body_beat_grid(beat_grid, section.anchor, beat_period, beats_per_bar, duration)
    report = {
        "audio": str(args.audio),
        "duration": duration,
        "bpm": args.bpm,
        "meter": meter_label,
        "body_entry": {
            "time": section.anchor,
            "key": section.key,
            "progression": section.progression,
            "score": section.score,
            "energy_score": section.energy_score,
            "loop_score": section.loop_score,
            "downbeat_score": section.downbeat_score,
            "anchor_score": section.anchor_score,
        },
        "sections": sections,
        "beat_grid": str(beat_grid),
        "notes": [
            "Bars before body_entry should be treated as intro/free time unless a later section says otherwise.",
            "If a long low-energy gap appears inside the song, run this tool per segment or use the reported grid as a new local section anchor.",
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Body entry: {section.anchor:.3f}s, meter={meter_label}, key={section.key}, progression={section.progression}, score={section.score:.3f}")
    print(f"Wrote: {beat_grid}")
    print(f"Wrote: {report_path}")

    if args.run_analysis:
        analysis_base = output_base.with_name(output_base.name + "_base")
        analysis_json = run_analysis(args.audio, analysis_base, beat_grid, section, args.external_chords, meter_label)
        refined_json = output_base.with_name(output_base.name + "_analysis.json")
        apply_sections(analysis_json, refined_json, sections)
        print(f"Wrote: {refined_json}")
        print(f"Wrote: {refined_json.with_name(refined_json.stem.replace('_analysis', '') + '_studio.html')}")


if __name__ == "__main__":
    main()
