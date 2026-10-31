#!/usr/bin/env python
"""Detect local key segments from an existing Chord Studio analysis."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from chord_studio import (
    MAJOR_ROMAN_BY_SEMITONE,
    MINOR_ROMAN_BY_SEMITONE,
    chord_to_roman,
    chord_to_roman_display,
    estimate_mode_from_key,
    note_name_for_pc,
    parse_chord_name,
    write_html,
)


MAJOR_SCALE = {
    0: ("I", ""),
    2: ("II", "m"),
    4: ("III", "m"),
    5: ("IV", ""),
    7: ("V", ""),
    9: ("VI", "m"),
    11: ("VII", "dim"),
}
MINOR_SCALE = {
    0: ("I", "m"),
    2: ("II", "dim"),
    3: ("III", ""),
    5: ("IV", "m"),
    7: ("V", ""),
    8: ("VI", ""),
    10: ("VII", ""),
}
COMMON_MAJOR_PAIRS = {
    ("I", "V"),
    ("V", "VI"),
    ("VI", "IV"),
    ("IV", "I"),
    ("IV", "V"),
    ("V", "I"),
    ("II", "V"),
    ("III", "VI"),
}
COMMON_MAJOR_LOOPS = {
    ("I", "V", "VI", "IV"),
    ("VI", "IV", "I", "V"),
    ("IV", "I", "V", "VI"),
    ("IV", "V", "VI", "IV"),
    ("I", "VI", "IV", "V"),
}
ENTRY_UNSTABLE_MAJOR = {"V", "VII", "bVII", "#IV"}
TONIC_FUNCTION_MAJOR = {"I", "VI"}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_MAJOR_BY_PC = {
    0: "C",
    1: "Db",
    2: "D",
    3: "Eb",
    4: "E",
    5: "F",
    6: "Gb",
    7: "G",
    8: "Ab",
    9: "A",
    10: "Bb",
    11: "B",
}


@dataclass(frozen=True)
class BarChord:
    bar: int
    start: float
    end: float
    chord: str
    weight: float


@dataclass(frozen=True)
class KeyWindow:
    start_bar: int
    end_bar: int
    key: str
    score: float
    margin: float


def key_candidates() -> list[str]:
    return [*NOTE_NAMES, *[f"{name}m" for name in NOTE_NAMES]]


def functional_label(chord: str, key: str) -> str:
    roman = chord_to_roman(chord, key)
    if roman == "N":
        return "N"
    text = roman.split("/", 1)[0]
    for degree in ("bVII", "#VII", "VII", "bVI", "#VI", "VI", "#IV", "IV", "bIII", "#III", "III", "bII", "II", "V", "I"):
        if text.startswith(degree):
            return degree
    return text


def chord_family(chord: str) -> str:
    _, suffix = parse_chord_name(chord)
    if suffix.startswith("m") and not suffix.startswith("maj"):
        return "m"
    if suffix.startswith("dim") or "dim" in suffix:
        return "dim"
    return ""


def chord_fit(chord: str, key: str) -> float:
    root, _ = parse_chord_name(chord)
    key_root, key_suffix = parse_chord_name(key)
    if root is None or key_root is None:
        return -0.05
    interval = (root - key_root) % 12
    family = chord_family(chord)
    scale = MINOR_SCALE if key_suffix == "m" else MAJOR_SCALE
    expected = scale.get(interval)
    if expected is None:
        return -0.45
    degree, expected_family = expected
    if expected_family == family:
        if key_suffix != "m" and degree in {"I", "IV", "V"}:
            return 1.25
        if key_suffix != "m" and degree == "VI":
            return 1.05
        return 0.95
    if degree == "V" and family == "":
        return 1.0
    if expected_family == "" and family == "m":
        return 0.20
    if expected_family == "m" and family == "":
        return -0.35
    return -0.20


def span_chord(span: dict) -> str:
    return str(span.get("chord_display") or span.get("chord") or "N").split("/", 1)[0]


def bar_chords(payload: dict) -> list[BarChord]:
    rows: list[BarChord] = []
    for bar in payload.get("bars", []):
        bar_number = int(bar.get("bar", 0))
        for span in bar.get("spans", []):
            chord = span_chord(span)
            if chord == "N":
                continue
            start = float(span.get("start", bar.get("start", 0.0)))
            end = float(span.get("end", bar.get("end", start)))
            rows.append(BarChord(bar_number, start, end, chord, max(0.1, end - start)))
    return rows


def score_key(rows: list[BarChord], key: str) -> float:
    if not rows:
        return -999.0
    total_weight = sum(row.weight for row in rows)
    score = sum(chord_fit(row.chord, key) * row.weight for row in rows) / max(total_weight, 1e-6)
    seq: list[str] = []
    previous = None
    for row in rows:
        label = functional_label(row.chord, key)
        if label != "N" and label != previous:
            seq.append(label)
            previous = label
    if len(seq) >= 2:
        pair_total = len(seq) - 1
        pair_hits = sum(1 for a, b in zip(seq, seq[1:]) if (a, b) in COMMON_MAJOR_PAIRS)
        score += 0.20 * pair_hits / pair_total
        awkward = sum(1 for label in seq if label in {"bII", "#IV", "bVI", "#VI", "#VII"})
        score -= 0.22 * awkward / max(len(seq), 1)
        if not key.endswith("m"):
            first = seq[0]
            if first in ENTRY_UNSTABLE_MAJOR:
                score -= 0.34
            elif first in TONIC_FUNCTION_MAJOR:
                score += 0.18
            if seq[:2] in (["V", "VI"], ["V", "VII"]):
                score -= 0.18
            if seq[:3] in (["V", "VI", "VII"], ["V", "VI", "V"]):
                score -= 0.20
    if len(seq) >= 4:
        four_total = len(seq) - 3
        loop_hits = sum(1 for item in zip(seq, seq[1:], seq[2:], seq[3:]) if item in COMMON_MAJOR_LOOPS)
        score += 0.38 * loop_hits / four_total
    if not key.endswith("m") and seq:
        weights_by_label: dict[str, float] = {}
        for row in rows:
            label = functional_label(row.chord, key)
            if label != "N":
                weights_by_label[label] = weights_by_label.get(label, 0.0) + row.weight
        if weights_by_label:
            strongest = max(weights_by_label, key=weights_by_label.get)
            if strongest == "I":
                score += 0.22
            elif strongest == "V":
                score -= 0.18
    ending = [functional_label(row.chord, key) for row in rows[-4:]]
    if "I" in ending:
        score += 0.08
    if key.endswith("m"):
        score -= 0.12
    return score


def best_key(rows: list[BarChord], forced_initial_key: str | None = None, prefer_key: str | None = None) -> tuple[str, float, float]:
    scores = all_key_scores(rows)
    if forced_initial_key:
        forced = normalize_key(forced_initial_key)
        scores = [(forced, score_key(rows, forced))]
    if prefer_key:
        preferred = normalize_key(prefer_key)
        scores = [(key, score + (0.05 if key == preferred else 0.0)) for key, score in scores]
    scores.sort(key=lambda item: item[1], reverse=True)
    margin = scores[0][1] - scores[1][1] if len(scores) > 1 else 999.0
    return scores[0][0], scores[0][1], margin


def all_key_scores(rows: list[BarChord]) -> list[tuple[str, float]]:
    return [(key, score_key(rows, key)) for key in key_candidates()]


def normalize_key(value: str) -> str:
    return value.replace("♯", "#").replace("♭", "b").strip()


def display_key(value: str, notation: str = "auto") -> str:
    key = normalize_key(value)
    root, suffix = parse_chord_name(key)
    if root is None or suffix == "m" or notation == "sharps":
        return key
    if notation == "flats" or key in {"G#", "D#", "A#"}:
        return FLAT_MAJOR_BY_PC[root]
    return key


def key_root_pc(key: str) -> int | None:
    root, _ = parse_chord_name(key)
    return root


def key_suffix(key: str) -> str:
    _, suffix = parse_chord_name(key)
    return suffix


def key_transition_score(previous: str, current: str) -> float:
    if previous == current:
        return 0.24
    previous_root = key_root_pc(previous)
    current_root = key_root_pc(current)
    if previous_root is None or current_root is None:
        return -0.18
    if previous_root == current_root and key_suffix(previous) == key_suffix(current):
        return 0.20
    diff = (current_root - previous_root) % 12
    if key_suffix(previous) != key_suffix(current):
        if diff in {3, 9}:
            return -0.02
        return -0.24
    if diff == 1:
        return 0.04
    if diff in {5, 7}:
        return -0.04
    if diff in {3, 4, 8, 9}:
        return -0.12
    return -0.24


def window_rows(rows: list[BarChord], start: int, end: int) -> list[BarChord]:
    return [row for row in rows if start <= row.bar <= end]


def build_window_ranges(rows: list[BarChord], window_bars: int, step_bars: int) -> list[tuple[int, int]]:
    if not rows:
        return []
    min_bar = min(row.bar for row in rows)
    max_bar = max(row.bar for row in rows)
    ranges = []
    for start in range(min_bar, max_bar + 1, step_bars):
        end = min(max_bar, start + window_bars - 1)
        ranges.append((start, end))
        if end >= max_bar:
            break
    return ranges


def window_track(
    rows: list[BarChord],
    window_bars: int,
    step_bars: int,
    forced_initial_key: str | None,
) -> list[KeyWindow]:
    if not rows:
        return []
    min_bar = min(row.bar for row in rows)
    max_bar = max(row.bar for row in rows)
    windows = []
    initial_cutoff = min(max_bar, min_bar + max(window_bars, 16) - 1)
    initial_rows = window_rows(rows, min_bar, initial_cutoff)
    initial_key = normalize_key(forced_initial_key) if forced_initial_key else best_key(initial_rows)[0]
    for start, end in build_window_ranges(rows, window_bars, step_bars):
        rows_in_window = window_rows(rows, start, end)
        prefer = initial_key if start <= min_bar + window_bars else None
        key, score, margin = best_key(rows_in_window, forced_initial_key if start == min_bar and forced_initial_key else None, prefer)
        windows.append(KeyWindow(start, end, key, score, margin))
    return windows


def viterbi_window_track(
    rows: list[BarChord],
    window_bars: int,
    step_bars: int,
    forced_initial_key: str | None,
    prefer_initial_bars: int = 16,
) -> list[KeyWindow]:
    ranges = build_window_ranges(rows, window_bars, step_bars)
    if not ranges:
        return []
    keys = key_candidates()
    min_bar = min(row.bar for row in rows)
    initial_cutoff = min(max(row.bar for row in rows), min_bar + max(window_bars, prefer_initial_bars) - 1)
    initial_rows = window_rows(rows, min_bar, initial_cutoff)
    initial_key = normalize_key(forced_initial_key) if forced_initial_key else best_key(initial_rows)[0]
    emissions: list[dict[str, float]] = []
    local_margins: list[float] = []
    for start, end in ranges:
        rows_in_window = window_rows(rows, start, end)
        raw = dict(all_key_scores(rows_in_window))
        if forced_initial_key and start == ranges[0][0]:
            raw = {initial_key: raw.get(initial_key, score_key(rows_in_window, initial_key))}
        elif start <= min_bar + window_bars:
            raw[initial_key] = raw.get(initial_key, score_key(rows_in_window, initial_key)) + 0.05
        ordered = sorted(raw.items(), key=lambda item: item[1], reverse=True)
        local_margins.append(ordered[0][1] - ordered[1][1] if len(ordered) > 1 else 999.0)
        emissions.append(raw)

    dp: list[dict[str, float]] = []
    back: list[dict[str, str | None]] = []
    first_scores = emissions[0]
    dp.append({key: score for key, score in first_scores.items()})
    back.append({key: None for key in first_scores})
    for index in range(1, len(emissions)):
        current_scores: dict[str, float] = {}
        current_back: dict[str, str | None] = {}
        for current, emission in emissions[index].items():
            best_previous = None
            best_score = -999999.0
            for previous, previous_score in dp[index - 1].items():
                score = previous_score + emission + key_transition_score(previous, current)
                if score > best_score:
                    best_score = score
                    best_previous = previous
            current_scores[current] = best_score
            current_back[current] = best_previous
        dp.append(current_scores)
        back.append(current_back)

    last_key = max(dp[-1], key=dp[-1].get)
    path = [last_key]
    for index in range(len(dp) - 1, 0, -1):
        previous = back[index][path[-1]]
        path.append(previous or path[-1])
    path.reverse()

    windows = []
    for (start, end), key, score_map, margin in zip(ranges, path, emissions, local_margins):
        windows.append(KeyWindow(start, end, key, score_map.get(key, -999.0), margin))
    return windows


def stable_segments(
    windows: list[KeyWindow],
    window_bars: int,
    min_segment_bars: int,
    min_margin: float,
    base_key: str | None,
) -> list[dict[str, object]]:
    if not windows:
        return []
    current_key = normalize_key(base_key) if base_key else windows[0].key
    segments: list[dict[str, object]] = [{"start_bar": windows[0].start_bar, "end_bar": None, "key": current_key}]
    pending_key = None
    pending_start = None
    pending_count = 0
    for window in windows[1:]:
        if window.key == current_key or window.margin < min_margin:
            pending_key = None
            pending_start = None
            pending_count = 0
            continue
        if window.key != pending_key:
            pending_key = window.key
            pending_start = window.start_bar
            pending_count = 1
        else:
            pending_count += 1
        if pending_key and pending_start is not None and pending_count * max(1, windows[1].start_bar - windows[0].start_bar) >= min_segment_bars:
            switch_bar = pending_start + max(1, window_bars // 2)
            if switch_bar - int(segments[-1]["start_bar"]) < min_segment_bars:
                continue
            segments[-1]["end_bar"] = switch_bar - 1
            segments.append({"start_bar": switch_bar, "end_bar": None, "key": pending_key})
            current_key = pending_key
            pending_key = None
            pending_start = None
            pending_count = 0
    return segments


def respell_chord(chord: str, key: str) -> str:
    if chord == "N":
        return chord
    if "/" in chord:
        head, bass = chord.split("/", 1)
        return f"{respell_chord(head, key)}/{respell_chord(bass, key)}"
    root, suffix = parse_chord_name(chord)
    if root is None:
        return chord
    return f"{note_name_for_pc(root, key)}{suffix}"


def relabel_item(item: dict, key: str) -> None:
    chord = span_chord(item)
    display_chord = respell_chord(chord, key)
    item["chord"] = display_chord
    item["chord_display"] = display_chord
    item["local_key"] = key
    item["roman"] = chord_to_roman(display_chord, key)
    item["roman_display"] = chord_to_roman_display(
        display_chord,
        key,
        str(item.get("bass_root") or "N"),
        float(item.get("bass_confidence") or 0.0),
        float(item.get("confidence") or 1.0),
    )


def promote_transition_dominants(output: dict, segments: list[dict[str, object]]) -> None:
    if len(segments) < 2:
        return
    for previous, current in zip(segments, segments[1:]):
        previous_key = str(previous["key"])
        current_key = str(current["key"])
        previous_root = key_root_pc(previous_key)
        current_root = key_root_pc(current_key)
        if previous_root is None or current_root is None:
            continue
        if (current_root - previous_root) % 12 != 1:
            continue
        switch_bar = int(current["start_bar"])
        targets = {
            switch_bar - 2: (previous_root + 4) % 12,
            switch_bar - 1: (previous_root + 5) % 12,
        }
        for bar in output.get("bars", []):
            bar_number = int(bar.get("bar", 0))
            target_root = targets.get(bar_number)
            if target_root is None:
                continue
            for collection in (bar.get("beats", []), bar.get("spans", [])):
                for item in collection:
                    chord = span_chord(item)
                    root, suffix = parse_chord_name(chord)
                    if root != target_root or "7" in suffix:
                        continue
                    chord7 = f"{note_name_for_pc(root, previous_key)}7"
                    item["chord"] = chord7
                    item["chord_display"] = chord7
                    item["local_key"] = previous_key
                    item["roman"] = chord_to_roman(chord7, previous_key)
                    item["roman_display"] = chord_to_roman_display(chord7, previous_key)
                    item["dominant_seventh_inferred"] = True


def apply_segments(payload: dict, segments: list[dict[str, object]], notation: str = "auto") -> dict:
    output = deepcopy(payload)
    if not segments:
        return output
    segments = [{**segment, "key": display_key(str(segment["key"]), notation)} for segment in segments]
    output["modulations"] = []
    for index, segment in enumerate(segments):
        key = str(segment["key"])
        start_bar = int(segment["start_bar"])
        end_bar = segment.get("end_bar")
        output["modulations"].append(
            {
                "start_bar": start_bar,
                "end_bar": end_bar,
                "key": key,
                "mode": estimate_mode_from_key(key),
                "progression": "",
                "unit": "bar",
                "phase": 0,
                "reason": "sliding-window chord-key analysis" if index else "initial key segment",
            }
        )
    for bar in output.get("bars", []):
        bar_number = int(bar.get("bar", 0))
        active = next(
            (
                segment
                for segment in reversed(segments)
                if bar_number >= int(segment["start_bar"])
                and (segment.get("end_bar") is None or bar_number <= int(segment["end_bar"]))
            ),
            segments[0],
        )
        key = str(active["key"])
        for beat in bar.get("beats", []):
            relabel_item(beat, key)
        for span in bar.get("spans", []):
            relabel_item(span, key)
    promote_transition_dominants(output, segments)
    output["events"] = sorted(
        [
            {**deepcopy(span), "bar": int(bar.get("bar", span.get("bar", 0)))}
            for bar in output.get("bars", [])
            for span in bar.get("spans", [])
        ],
        key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))),
    )
    labels = [f"{item['key']} {item['mode']}" for item in output["modulations"]]
    output["key"] = " -> ".join(dict.fromkeys(labels))
    output["mode"] = "modulating" if len(output["modulations"]) > 1 else output["modulations"][0]["mode"]
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_json", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--initial-key", help="Trust this key for the first stable section.")
    parser.add_argument("--window-bars", type=int, default=8)
    parser.add_argument("--step-bars", type=int, default=2)
    parser.add_argument("--min-segment-bars", type=int, default=8)
    parser.add_argument("--min-margin", type=float, default=0.10)
    parser.add_argument("--decoder", choices=["viterbi", "stable"], default="viterbi")
    parser.add_argument("--notation", choices=["auto", "sharps", "flats"], default="auto")
    parser.add_argument("--html", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.analysis_json.read_text(encoding="utf-8"))
    rows = bar_chords(payload)
    if args.decoder == "stable":
        windows = window_track(rows, args.window_bars, args.step_bars, args.initial_key)
    else:
        windows = viterbi_window_track(rows, args.window_bars, args.step_bars, args.initial_key)
    segments = stable_segments(windows, args.window_bars, args.min_segment_bars, args.min_margin, args.initial_key)
    output = apply_segments(payload, segments, args.notation)
    output["key_timeline"] = {
        "method": "sliding-window chord-key analysis",
        "window_bars": args.window_bars,
        "step_bars": args.step_bars,
        "min_segment_bars": args.min_segment_bars,
        "min_margin": args.min_margin,
        "decoder": args.decoder,
        "windows": [window.__dict__ for window in windows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output}")
    print("segments:")
    for segment in segments:
        print(f"  bar {segment['start_bar']}-{segment.get('end_bar') or ''}: {display_key(str(segment['key']), args.notation)}")
    if args.html:
        html_path = args.output.with_name(args.output.stem.replace("_analysis", "") + "_studio.html")
        write_html(html_path, output)
        print(f"Wrote: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
