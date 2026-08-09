#!/usr/bin/env python
"""Fast chord timeline from downbeats + external chord recognizer output."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chord_studio import (
    BeatInterval,
    ExternalChordHint,
    HarmonicInterval,
    build_beat_intervals,
    chord_display_name,
    chord_to_roman,
    chord_to_roman_display,
    chord_quality_family,
    diatonic_profile,
    display_bpm_for_meter,
    estimate_mode_from_key,
    format_timestamp,
    harmonic_intervals_from_beats,
    meter_from_downbeat_rows,
    note_name_for_pc,
    parse_chord_name,
    parse_meter,
    read_beat_grid_rows,
    read_downbeat_rows,
    read_external_chord_hints,
    write_html,
)


COMMON_TRANSITIONS = {
    ("I", "V"),
    ("I", "VIm"),
    ("I", "IV"),
    ("V", "I"),
    ("V", "VIm"),
    ("VIm", "IV"),
    ("VIm", "V"),
    ("IV", "V"),
    ("IV", "I"),
    ("IV", "VIm"),
    ("IIm", "V"),
    ("IIIm", "VIm"),
    ("bVII", "IV"),
}


@dataclass(frozen=True)
class AutoSplitCandidate:
    bar_cell: object
    sub_cells: list[object]
    strict: bool
    relaxed: bool


@dataclass(frozen=True)
class EdgeFillCandidate:
    start: float
    end: float
    chord: str
    confidence: float


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def hint_scores_for_cell(
    cell_start: float,
    cell_end: float,
    hints: list[ExternalChordHint],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    duration = max(cell_end - cell_start, 1e-9)
    for hint in hints:
        amount = overlap(cell_start, cell_end, hint.start, hint.end)
        if amount <= 0:
            continue
        scores[hint.chord] = scores.get(hint.chord, 0.0) + (amount / duration) * hint.confidence
    if not scores:
        return {"N": 0.0}
    return scores


def scale_hints(hints: list[ExternalChordHint], weight: float) -> list[ExternalChordHint]:
    if abs(weight - 1.0) <= 1e-9:
        return hints
    return [
        ExternalChordHint(item.start, item.end, item.chord, float(item.confidence) * weight)
        for item in hints
    ]


def base_evidence_rows(base: dict[str, object]) -> list[dict[str, object]]:
    if base.get("analysis_model") == "external-fusion":
        return []
    rows: list[dict[str, object]] = []
    for bar in base.get("bars", []):
        if not isinstance(bar, dict):
            continue
        items = [("beat", item) for item in bar.get("beats") or []]
        items += [("span", item) for item in bar.get("spans") or []]
        for source_type, item in items:
            if not isinstance(item, dict):
                continue
            try:
                rows.append(
                    {
                        "source_type": source_type,
                        "start": float(item["start"]),
                        "end": float(item["end"]),
                        "chord": str(item.get("chord") or "N"),
                        "confidence": float(item.get("confidence") or 0.0),
                        "bass_root": str(item.get("bass_root") or "N"),
                        "bass_confidence": float(item.get("bass_confidence") or 0.0),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(rows, key=lambda item: float(item["start"]))


def local_display_bpm(local_pulse_bpm: float, meter_label: str) -> float:
    if meter_label == "6/8" and local_pulse_bpm > 80:
        return local_pulse_bpm / 3.0
    return local_pulse_bpm


def base_scores_for_cell(
    cell_start: float,
    cell_end: float,
    rows: list[dict[str, object]],
    weight: float,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    if weight <= 0:
        return scores
    duration = max(cell_end - cell_start, 1e-9)
    for row in rows:
        chord = str(row.get("chord") or "N")
        if chord == "N":
            continue
        amount = overlap(cell_start, cell_end, float(row["start"]), float(row["end"]))
        if amount <= 0:
            continue
        confidence = float(row.get("confidence") or 0.0)
        scores[chord] = scores.get(chord, 0.0) + weight * (amount / duration) * confidence
    return scores


def bass_vote_for_cell(
    cell_start: float,
    cell_end: float,
    rows: list[dict[str, object]],
) -> tuple[int | None, float]:
    votes: dict[int, float] = {}
    duration = max(cell_end - cell_start, 1e-9)
    for row in rows:
        if row.get("source_type") != "beat":
            continue
        bass_root, _ = parse_chord_name(str(row.get("bass_root") or "N"))
        if bass_root is None:
            continue
        amount = overlap(cell_start, cell_end, float(row["start"]), float(row["end"]))
        if amount <= 0:
            continue
        confidence = float(row.get("bass_confidence") or 0.0)
        votes[bass_root] = votes.get(bass_root, 0.0) + confidence * (amount / duration)
    if not votes:
        return None, 0.0
    root, score = max(votes.items(), key=lambda item: item[1])
    return root, float(np.clip(score, 0.0, 1.0))


def seventh_color_vote_for_cell(
    cell_start: float,
    cell_end: float,
    rows: list[dict[str, object]],
) -> tuple[str | None, float]:
    votes: dict[str, float] = {}
    duration = max(cell_end - cell_start, 1e-9)
    for row in rows:
        chord = str(row.get("chord") or "N")
        root, suffix = parse_chord_name(chord)
        family = chord_quality_family(suffix)
        if root is None or suffix not in {"7", "m7"} or family not in {"m", "dom"}:
            continue
        amount = overlap(cell_start, cell_end, float(row["start"]), float(row["end"]))
        if amount <= 0:
            continue
        confidence = float(row.get("confidence") or 0.0)
        votes[chord] = votes.get(chord, 0.0) + confidence * (amount / duration)
    if not votes:
        return None, 0.0
    chord, score = max(votes.items(), key=lambda item: item[1])
    return chord, float(np.clip(score, 0.0, 1.0))


def merge_score_rows(*rows: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for row in rows:
        for chord, score in row.items():
            merged[chord] = merged.get(chord, 0.0) + float(score)
    return merged


def top_score_label(scores: dict[str, float]) -> tuple[str, float]:
    if not scores:
        return "N", 0.0
    return max(scores.items(), key=lambda item: item[1])


def second_score(scores: dict[str, float], top_label: str) -> float:
    values = [float(score) for label, score in scores.items() if label != top_label]
    return max(values) if values else 0.0


def root_family(label: str) -> tuple[int | None, str]:
    root, suffix = parse_chord_name(label)
    return root, chord_quality_family(suffix)


def compatible_candidate(left: str, right: str) -> bool:
    left_root, left_family = root_family(left)
    right_root, right_family = root_family(right)
    if left_root is None or right_root is None:
        return left == right
    if left_root != right_root:
        return False
    if left_family == right_family:
        return True
    return {left_family, right_family} <= {"", "dom"}


def gated_extra_scores_for_cell(
    cell_start: float,
    cell_end: float,
    primary_hints: list[ExternalChordHint],
    extra_hints: list[ExternalChordHint],
    weight: float,
    mode: str,
    primary_strong: float = 0.78,
    primary_margin: float = 0.18,
    extra_min_score: float = 0.58,
) -> dict[str, float]:
    raw_extra = hint_scores_for_cell(cell_start, cell_end, extra_hints)
    if not raw_extra or weight <= 0:
        return {}
    if mode == "off":
        return {label: float(score) * weight for label, score in raw_extra.items()}

    primary = hint_scores_for_cell(cell_start, cell_end, primary_hints)
    primary_label, primary_score = top_score_label(primary)
    primary_second = second_score(primary, primary_label)
    primary_clear = primary_score >= primary_strong and (primary_score - primary_second) >= primary_margin
    extra_label, extra_score = top_score_label(raw_extra)

    gated: dict[str, float] = {}
    for label, score in raw_extra.items():
        if label == "N":
            continue
        multiplier = 0.0
        if compatible_candidate(label, primary_label):
            multiplier = weight
        elif not primary_clear and extra_score >= extra_min_score and label == extra_label:
            multiplier = weight * 0.75
        elif mode == "soft" and label == extra_label:
            multiplier = weight * 0.25
        if multiplier > 0:
            gated[label] = float(score) * multiplier
    return gated


def source_agreement(
    primary_hints: list[ExternalChordHint],
    extra_hints: list[ExternalChordHint],
    frame: float = 0.25,
) -> float:
    if not primary_hints or not extra_hints:
        return 0.0
    end = max([item.end for item in primary_hints] + [item.end for item in extra_hints])
    if end <= 0:
        return 0.0
    hits = 0
    total = 0
    primary_index = 0
    extra_index = 0
    time = 0.0
    while time < end:
        while primary_index + 1 < len(primary_hints) and time >= primary_hints[primary_index].end:
            primary_index += 1
        while extra_index + 1 < len(extra_hints) and time >= extra_hints[extra_index].end:
            extra_index += 1
        primary = primary_hints[primary_index] if primary_hints[primary_index].start <= time < primary_hints[primary_index].end else None
        extra = extra_hints[extra_index] if extra_hints[extra_index].start <= time < extra_hints[extra_index].end else None
        if primary is not None and extra is not None and primary.chord != "N" and extra.chord != "N":
            total += 1
            if compatible_candidate(primary.chord, extra.chord):
                hits += 1
        time += frame
    return hits / max(total, 1)


def edge_fill_candidates(
    intervals: list[BeatInterval],
    duration: float,
    primary_hints: list[ExternalChordHint],
    extra_hints: list[ExternalChordHint],
    extra_weight: float,
    extra_gating: str,
    min_duration: float,
    min_score: float,
    mode: str = "summary",
) -> list[EdgeFillCandidate]:
    if not intervals or min_duration <= 0:
        return []
    ranges = []
    first_start = min(item.start for item in intervals)
    last_end = max(item.end for item in intervals)
    if first_start >= min_duration:
        ranges.append((0.0, first_start))
    if duration - last_end >= min_duration:
        ranges.append((last_end, duration))
    candidates: list[EdgeFillCandidate] = []
    for start, end in ranges:
        if mode == "segments":
            for hint in primary_hints:
                amount = overlap(start, end, hint.start, hint.end)
                if amount < min_duration or hint.chord == "N" or hint.confidence < min_score:
                    continue
                candidates.append(
                    EdgeFillCandidate(
                        max(start, hint.start),
                        min(end, hint.end),
                        hint.chord,
                        hint.confidence,
                    )
                )
            continue
        scores = merge_score_rows(
            hint_scores_for_cell(start, end, primary_hints),
            gated_extra_scores_for_cell(
                start,
                end,
                primary_hints,
                extra_hints,
                extra_weight,
                extra_gating,
            ),
        )
        chord, score = top_score_label(scores)
        if chord != "N" and score >= min_score:
            candidates.append(EdgeFillCandidate(start, end, chord, score))
    merged: list[EdgeFillCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (item.start, item.end, item.chord)):
        if (
            merged
            and merged[-1].chord == candidate.chord
            and abs(merged[-1].end - candidate.start) < 1e-6
        ):
            previous = merged[-1]
            merged[-1] = EdgeFillCandidate(
                previous.start,
                candidate.end,
                previous.chord,
                max(previous.confidence, candidate.confidence),
            )
        else:
            merged.append(candidate)
    return merged


def should_split_bar(
    bar_scores: dict[str, float],
    sub_scores: list[dict[str, float]],
    min_sub_score: float = 0.56,
    min_gain: float = 0.10,
) -> bool:
    if len(sub_scores) < 2:
        return False
    bar_label, bar_score = top_score_label(bar_scores)
    bar_root, _ = root_family(bar_label)
    sub_tops = [top_score_label(scores) for scores in sub_scores]
    sub_roots = [root_family(label)[0] for label, score in sub_tops if label != "N" and score >= min_sub_score]
    if len(sub_roots) < 2:
        return False
    if len(set(sub_roots)) < 2:
        return False
    average_sub_score = float(np.mean([score for _, score in sub_tops]))
    if average_sub_score < bar_score + min_gain:
        return False
    if bar_root is not None and bar_score >= 0.84:
        matching = sum(1 for root in sub_roots if root == bar_root)
        if matching >= len(sub_roots) - 1:
            return False
    return True


def relaxed_split_candidate(
    bar_scores: dict[str, float],
    sub_scores: list[dict[str, float]],
    min_sub_score: float = 0.48,
    min_gain: float = -0.04,
) -> bool:
    if len(sub_scores) < 2:
        return False
    _, bar_score = top_score_label(bar_scores)
    sub_tops = [top_score_label(scores) for scores in sub_scores]
    confident = [
        root_family(label)[0]
        for label, score in sub_tops
        if label != "N" and score >= min_sub_score
    ]
    if len(confident) < 2 or len(set(confident)) < 2:
        return False
    average_sub_score = float(np.mean([score for _, score in sub_tops]))
    return average_sub_score >= bar_score + min_gain


def window_supported_split(index: int, candidates: list[AutoSplitCandidate], window: int, required: int) -> bool:
    if window <= 1:
        return False
    half = max(1, window // 2)
    start = max(0, index - half)
    end = min(len(candidates), index + half + 1)
    nearby = candidates[start:end]
    return sum(1 for item in nearby if item.relaxed) >= required


def auto_harmonic_intervals(
    intervals: list[BeatInterval],
    beats_per_bar: int,
    hints: list[ExternalChordHint],
    base_rows: list[dict[str, object]],
    base_weight: float,
    min_sub_score: float = 0.56,
    min_gain: float = 0.10,
    segment_window: int = 0,
    segment_required: int = 3,
) -> list[object]:
    bar_cells = harmonic_intervals_from_beats(intervals, "bar", beats_per_bar)
    split_unit = "compound-half" if beats_per_bar == 6 else "half-bar"
    if beats_per_bar not in {4, 6}:
        return bar_cells
    sub_cells = harmonic_intervals_from_beats(intervals, split_unit, beats_per_bar)
    candidates: list[AutoSplitCandidate] = []
    for bar_cell in bar_cells:
        subs = [cell for cell in sub_cells if cell.bar == bar_cell.bar]
        if len(subs) < 2:
            candidates.append(AutoSplitCandidate(bar_cell, [], False, False))
            continue
        bar_scores = merge_score_rows(
            hint_scores_for_cell(bar_cell.start, bar_cell.end, hints),
            base_scores_for_cell(bar_cell.start, bar_cell.end, base_rows, base_weight),
        )
        sub_scores = [
            merge_score_rows(
                hint_scores_for_cell(cell.start, cell.end, hints),
                base_scores_for_cell(cell.start, cell.end, base_rows, base_weight),
            )
            for cell in subs
        ]
        candidates.append(
            AutoSplitCandidate(
                bar_cell,
                subs,
                strict=should_split_bar(
                    bar_scores,
                    sub_scores,
                    min_sub_score=min_sub_score,
                    min_gain=min_gain,
                ),
                relaxed=relaxed_split_candidate(bar_scores, sub_scores),
            )
        )
    selected: list[object] = []
    for index, candidate in enumerate(candidates):
        split = candidate.strict or (
            candidate.relaxed
            and window_supported_split(index, candidates, segment_window, segment_required)
        )
        selected.extend(candidate.sub_cells if split and candidate.sub_cells else [candidate.bar_cell])
    return selected


def inherit_seventh_colors(
    chords: list[str],
    seventh_votes: list[tuple[str | None, float]],
    threshold: float,
) -> list[str]:
    refined = list(chords)
    for index, chord in enumerate(chords):
        root, suffix = parse_chord_name(chord)
        voted_chord, score = seventh_votes[index]
        voted_root, voted_suffix = parse_chord_name(voted_chord or "N")
        if root is None or voted_root is None or root != voted_root:
            continue
        family = chord_quality_family(suffix)
        voted_family = chord_quality_family(voted_suffix)
        if family not in {"", "m", "dom"} or voted_family not in {"m", "dom"}:
            continue
        if family == "" and voted_family == "dom":
            pass
        elif family != voted_family:
            continue
        if score >= threshold:
            refined[index] = voted_chord or chord
    return refined


def roman_function_label(label: str) -> str:
    text = label.split("/", 1)[0]
    if text == "N":
        return "N"
    degrees = ["bVII", "#VII", "VII", "bVI", "#VI", "VI", "#IV", "IV", "bIII", "#III", "III", "bII", "II", "V", "I"]
    for degree in degrees:
        if not text.startswith(degree):
            continue
        suffix = text[len(degree) :]
        return f"{degree}m" if suffix.startswith("m") else degree
    return text


def diatonic_shadow_chord(chord: str, key: str) -> str | None:
    root, suffix = parse_chord_name(chord)
    if root is None:
        return None
    profile = diatonic_profile(key)
    expected_quality = profile.get(root)
    if expected_quality not in {"", "m"}:
        return None
    family = chord_quality_family(suffix)
    if family == expected_quality:
        return None
    if family not in {"", "m", "dom"}:
        return None
    seventh = "7" if "7" in suffix else ""
    return f"{note_name_for_pc(root, key)}{expected_quality}{seventh}"


def add_diatonic_shadow_scores(
    cell_scores: list[dict[str, float]],
    key: str,
    shadow_weight: float = 0.0,
) -> list[dict[str, float]]:
    expanded: list[dict[str, float]] = []
    for row in cell_scores:
        new_row = dict(row)
        for chord, score in row.items():
            shadow = diatonic_shadow_chord(chord, key)
            if shadow:
                new_row[shadow] = max(new_row.get(shadow, -1.0), float(score) * shadow_weight)
        expanded.append(new_row)
    return expanded


def smooth_chords(
    cell_scores: list[dict[str, float]],
    key: str,
    bass_votes: list[tuple[int | None, float]] | None = None,
    progression_scores: dict[str, float] | None = None,
    progression_strength: float = 0.045,
    stay_bonus: float = 0.085,
    common_bonus: float = 0.095,
    uncommon_penalty: float = 0.035,
    diatonic_bonus: float = 0.0,
    bass_root_bonus: float = 0.075,
    diatonic_shadow_weight: float = 0.0,
) -> list[str]:
    cell_scores = add_diatonic_shadow_scores(cell_scores, key, shadow_weight=diatonic_shadow_weight)
    labels = sorted({label for scores in cell_scores for label in scores} | {"N"})
    label_to_index = {label: index for index, label in enumerate(labels)}
    roman = {
        label: roman_function_label(chord_to_roman_display(label, key, "N", 0.0, 1.0))
        for label in labels
    }
    profile = diatonic_profile(key)
    scores = np.full((len(cell_scores), len(labels)), -0.35, dtype=np.float32)
    for time_index, row in enumerate(cell_scores):
        for label, score in row.items():
            scores[time_index, label_to_index[label]] = float(score)
    for label, index in label_to_index.items():
        root, suffix = parse_chord_name(label)
        if root is not None and profile.get(root) == chord_quality_family(suffix):
            scores[:, index] += diatonic_bonus
    if bass_votes:
        for time_index, (bass_pc, bass_confidence) in enumerate(bass_votes):
            if bass_pc is None or bass_confidence < 0.45:
                continue
            for label, index in label_to_index.items():
                root, _ = parse_chord_name(label)
                if root == bass_pc:
                    scores[time_index, index] += bass_root_bonus * bass_confidence

    dp = np.empty_like(scores)
    back = np.zeros_like(scores, dtype=np.int32)
    dp[0] = scores[0]
    for time_index in range(1, scores.shape[0]):
        for curr_index, curr_label in enumerate(labels):
            best_score = -1e9
            best_prev = 0
            curr_roman = roman[curr_label]
            for prev_index, prev_label in enumerate(labels):
                prev_roman = roman[prev_label]
                transition = 0.0
                if curr_label == prev_label:
                    transition += stay_bonus
                if progression_scores:
                    transition += progression_strength * progression_scores.get(f"{prev_roman}->{curr_roman}", 0.0)
                if (prev_roman, curr_roman) in COMMON_TRANSITIONS:
                    transition += common_bonus
                if prev_label != "N" and curr_label != "N" and curr_label != prev_label:
                    transition -= uncommon_penalty
                value = float(dp[time_index - 1, prev_index]) + transition
                if value > best_score:
                    best_score = value
                    best_prev = prev_index
            dp[time_index, curr_index] = scores[time_index, curr_index] + best_score
            back[time_index, curr_index] = best_prev

    path = np.empty(scores.shape[0], dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))
    for time_index in range(scores.shape[0] - 2, -1, -1):
        path[time_index] = back[time_index + 1, path[time_index + 1]]
    return [labels[int(index)] for index in path]


def parse_phrase_lengths(text: str) -> list[int]:
    values: list[int] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            continue
        if value > 1 and value not in values:
            values.append(value)
    return values


def repeat_consensus_refine(
    chords: list[str],
    cell_scores: list[dict[str, float]],
    cells: list[object],
    phrase_lengths: list[int],
    min_support: int = 3,
    min_share: float = 0.60,
    max_score_gap: float = 0.08,
    max_current_score: float = 0.86,
) -> list[str]:
    refined = list(chords)
    for phrase_bars in phrase_lengths:
        groups: dict[tuple[int, int, int], list[int]] = {}
        for index, cell in enumerate(cells):
            key = ((int(cell.bar) - 1) % phrase_bars, int(cell.beat_start), int(cell.beat_end))
            groups.setdefault(key, []).append(index)
        next_refined = list(refined)
        for indexes in groups.values():
            if len(indexes) < min_support + 1:
                continue
            for index in indexes:
                support: dict[str, float] = {}
                count: dict[str, int] = {}
                for other in indexes:
                    if other == index:
                        continue
                    label = refined[other]
                    if label == "N":
                        continue
                    confidence = max(cell_scores[other].values()) if cell_scores[other] else 0.0
                    support[label] = support.get(label, 0.0) + confidence
                    count[label] = count.get(label, 0) + 1
                if not support:
                    continue
                consensus, consensus_support = max(support.items(), key=lambda item: item[1])
                if consensus == refined[index] or count.get(consensus, 0) < min_support:
                    continue
                total_count = sum(count.values())
                if total_count <= 0 or count[consensus] / total_count < min_share:
                    continue
                scores = cell_scores[index]
                if not scores or consensus not in scores:
                    continue
                current_score = float(scores.get(refined[index], 0.0))
                consensus_score = float(scores[consensus])
                if current_score > max_current_score:
                    continue
                if consensus_score + max_score_gap < current_score:
                    continue
                next_refined[index] = consensus
        refined = next_refined
    return refined


def span_unit(beat_start: int, beat_end: int, beats_per_bar: int) -> str:
    if beat_start == 1 and beat_end >= beats_per_bar:
        return "bar"
    if beats_per_bar == 6 and beat_end - beat_start + 1 == 3:
        return "compound-half"
    if beat_end - beat_start + 1 == max(1, beats_per_bar // 2):
        return "half-bar"
    return "beat"


def make_payload(
    base: dict[str, object],
    intervals: list[BeatInterval],
    cells: list[object],
    chords: list[str],
    cell_scores: list[dict[str, float]],
    bass_votes: list[tuple[int | None, float]],
    key: str,
    meter_label: str,
    beats_per_bar: int,
    beat_period: float,
    harmonic_unit: str,
    edge_fills: list[EdgeFillCandidate] | None = None,
) -> dict[str, object]:
    cell_rows = []
    for cell, chord, scores, bass_vote in zip(cells, chords, cell_scores, bass_votes):
        confidence = max(scores.values()) if scores else 0.0
        bass_pc, bass_confidence = bass_vote
        bass_root = note_name_for_pc(bass_pc, key) if bass_pc is not None else "N"
        cell_rows.append(
            {
                "bar": cell.bar,
                "unit": span_unit(cell.beat_start, cell.beat_end, beats_per_bar),
                "beat_start": cell.beat_start,
                "beat_end": cell.beat_end,
                "start": cell.start,
                "end": cell.end,
                "chord": chord,
                "chord_display": chord_display_name(chord, bass_root, bass_confidence, confidence),
                "roman": chord_to_roman(chord, key),
                "roman_display": chord_to_roman_display(chord, key, bass_root, bass_confidence, confidence),
                "bass_root": bass_root,
                "bass_confidence": bass_confidence,
                "confidence": confidence,
            }
        )

    bars = []
    events = []
    for bar_number in sorted({item.bar for item in intervals}):
        bar_intervals = [item for item in intervals if item.bar == bar_number]
        bar_cells = [item for item in cell_rows if item["bar"] == bar_number]
        durations = [item.end - item.start for item in bar_intervals if item.end > item.start]
        local_pulse_bpm = 60.0 / float(np.median(durations)) if durations else 0.0
        local_bpm = local_display_bpm(local_pulse_bpm, meter_label)
        beats = []
        for beat in bar_intervals:
            covering = next(
                (
                    cell
                    for cell in bar_cells
                    if cell["beat_start"] <= beat.beat <= cell["beat_end"]
                ),
                bar_cells[-1] if bar_cells else None,
            )
            if covering is None:
                chord = "N"
                confidence = 0.0
                roman = "N"
                bass_root = "N"
                bass_confidence = 0.0
            else:
                chord = str(covering["chord"])
                confidence = float(covering["confidence"])
                roman = str(covering["roman_display"])
                bass_root = str(covering["bass_root"])
                bass_confidence = float(covering["bass_confidence"])
            beats.append(
                {
                    "beat": beat.beat,
                    "start": beat.start,
                    "end": beat.end,
                    "chord": chord,
                    "chord_display": chord_display_name(chord, bass_root, bass_confidence, confidence),
                    "roman": chord_to_roman(chord, key),
                    "roman_display": roman,
                    "bass_root": bass_root,
                    "bass_confidence": bass_confidence,
                    "confidence": confidence,
                }
            )
        bars.append(
            {
                "bar": bar_number,
                "start": bar_intervals[0].start,
                "end": bar_intervals[-1].end,
                "local_bpm": local_bpm,
                "local_pulse_bpm": local_pulse_bpm,
                "beats": beats,
                "spans": bar_cells,
            }
        )
        events.extend(bar_cells)

    if edge_fills:
        last_bar = max([item.bar for item in intervals], default=0)
        for index, fill in enumerate(edge_fills, start=1):
            bass_root = "N"
            bass_confidence = 0.0
            is_leading_edge = bool(bars) and fill.end <= float(bars[0]["start"])
            bar_number = 0 if is_leading_edge else last_bar + index
            events.append(
                {
                    "bar": bar_number,
                    "unit": "edge",
                    "beat_start": 0,
                    "beat_end": 0,
                    "start": fill.start,
                    "end": fill.end,
                    "chord": fill.chord,
                    "chord_display": chord_display_name(fill.chord, bass_root, bass_confidence, fill.confidence),
                    "roman": chord_to_roman(fill.chord, key),
                    "roman_display": chord_to_roman_display(fill.chord, key, bass_root, bass_confidence, fill.confidence),
                    "bass_root": bass_root,
                    "bass_confidence": bass_confidence,
                    "confidence": fill.confidence,
                }
            )
    events.sort(key=lambda item: (float(item["start"]), float(item["end"])))

    source = str(base.get("source") or "")
    audio_file = str(base.get("audio_file") or Path(source).name)
    return {
        **base,
        "source": source,
        "audio_file": audio_file,
        "key": key,
        "mode": estimate_mode_from_key(key),
        "meter": meter_label,
        "meter_confidence": 0.95,
        "beat_period": beat_period,
        "beats_per_bar": beats_per_bar,
        "beat_grid": "madmom_downbeat_external_fusion",
        "beat_source": "madmom",
        "analysis_model": "external-fusion",
        "tempo_variant": "madmom-downbeat-external",
        "harmonic_unit": harmonic_unit,
        "display": base.get("display") or "roman",
        "tempo_segments": tempo_segments_from_bars(bars),
        "bars": bars,
        "events": events,
    }


def tempo_segments_from_bars(bars: list[dict[str, object]], tolerance: float = 3.0) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    values: list[float] = []
    for bar in bars:
        bpm = float(bar.get("local_bpm") or 0.0)
        if bpm <= 0:
            continue
        if current is None:
            current = {
                "bar_start": int(bar["bar"]),
                "bar_end": int(bar["bar"]),
                "start": float(bar["start"]),
                "end": float(bar["end"]),
            }
            values = [bpm]
            continue
        median = float(np.median(values))
        if abs(bpm - median) <= tolerance:
            current["bar_end"] = int(bar["bar"])
            current["end"] = float(bar["end"])
            values.append(bpm)
            continue
        current["bpm"] = float(np.median(values))
        current["min_bpm"] = float(min(values))
        current["max_bpm"] = float(max(values))
        segments.append(current)
        current = {
            "bar_start": int(bar["bar"]),
            "bar_end": int(bar["bar"]),
            "start": float(bar["start"]),
            "end": float(bar["end"]),
        }
        values = [bpm]
    if current is not None:
        current["bpm"] = float(np.median(values))
        current["min_bpm"] = float(min(values))
        current["max_bpm"] = float(max(values))
        segments.append(current)
    return segments


def read_progression_scores(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    scores = raw.get("pair_scores") if isinstance(raw, dict) else None
    if not isinstance(scores, dict):
        raise ValueError(f"Progression prior does not contain pair_scores: {path}")
    return {str(key): float(value) for key, value in scores.items()}


def write_txt(path: Path, payload: dict[str, object]) -> None:
    lines = [
        f"Source: {payload.get('source')}",
        f"BPM: {payload.get('bpm')}",
        f"Key: {payload.get('key')} {payload.get('mode')}",
        f"Meter: {payload.get('meter')}",
        "",
    ]
    for event in payload.get("events", []):
        beat_label = (
            str(event["beat_start"])
            if event["beat_start"] == event["beat_end"]
            else f"{event['beat_start']}-{event['beat_end']}"
        )
        lines.append(
            f"{int(event['bar']):03d} beat {beat_label} "
            f"{format_timestamp(float(event['start']))}-{format_timestamp(float(event['end']))} "
            f"{event['roman_display']} / {event['chord_display']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def beat_rows_from_base_analysis(base: dict[str, object]) -> list[tuple[float, int]]:
    rows: list[tuple[float, int]] = []
    for bar in base.get("bars", []):
        if not isinstance(bar, dict):
            continue
        for beat in bar.get("beats", []):
            if not isinstance(beat, dict):
                continue
            try:
                rows.append((float(beat["start"]), int(beat.get("beat", 1))))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(rows, key=lambda item: item[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast downbeat-aligned external chord fusion.")
    parser.add_argument("base_analysis", type=Path)
    parser.add_argument("downbeats", type=Path)
    parser.add_argument("external_chords", type=Path)
    parser.add_argument(
        "--base-evidence-analysis",
        type=Path,
        help="Optional non-fused analysis JSON used only for local bass/template evidence.",
    )
    parser.add_argument(
        "--external-chords-extra",
        type=Path,
        action="append",
        default=None,
        help="Additional external chord .lab/.csv/.json files to ensemble with the primary source.",
    )
    parser.add_argument(
        "--external-chords-extra-weight",
        type=float,
        default=1.0,
        help="Confidence multiplier for --external-chords-extra sources.",
    )
    parser.add_argument(
        "--extra-gating",
        choices=["off", "conservative", "soft"],
        default="off",
        help="Gate additional external sources per harmonic cell instead of blindly adding them.",
    )
    parser.add_argument(
        "--extra-gating-primary-strong",
        type=float,
        default=0.78,
        help="Primary-source cell score treated as strong for --extra-gating.",
    )
    parser.add_argument(
        "--extra-gating-primary-margin",
        type=float,
        default=0.18,
        help="Required primary top-vs-second margin for --extra-gating.",
    )
    parser.add_argument(
        "--extra-gating-min-score",
        type=float,
        default=0.58,
        help="Minimum extra-source top score allowed to override an unclear primary source.",
    )
    parser.add_argument(
        "--extra-gating-min-track-agreement",
        type=float,
        default=0.0,
        help="Disable extra sources when whole-track agreement with the primary source is below this value.",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--key", default=None)
    parser.add_argument("--meter", default="6/8")
    parser.add_argument(
        "--beat-grid-from-base",
        action="store_true",
        help="Use beat times already stored in base_analysis instead of reading the downbeats argument.",
    )
    parser.add_argument(
        "--harmonic-unit",
        choices=["bar", "half-bar", "compound-half", "beat", "auto"],
        default="bar",
    )
    parser.add_argument(
        "--base-weight",
        type=float,
        default=0.0,
        help="Weak weight for the local template chord evidence stored in base_analysis.",
    )
    parser.add_argument(
        "--bass-root-weight",
        type=float,
        default=0.075,
        help="Bonus for candidates whose root matches the local bass-root estimate.",
    )
    parser.add_argument(
        "--dominant-seventh-threshold",
        type=float,
        default=0.45,
        help="Inherit same-root seventh color from the local template model above this score.",
    )
    parser.add_argument(
        "--progression-prior",
        type=Path,
        help="JSON generated by progression_stats.py with pair_scores.",
    )
    parser.add_argument(
        "--progression-prior-strength",
        type=float,
        default=0.045,
        help="Weight for learned Roman-numeral transition prior.",
    )
    parser.add_argument(
        "--auto-min-sub-score",
        type=float,
        default=0.56,
        help="Auto harmonic-unit split threshold for each sub-cell top score.",
    )
    parser.add_argument(
        "--auto-min-gain",
        type=float,
        default=0.10,
        help="Auto harmonic-unit split threshold for average sub-cell confidence gain.",
    )
    parser.add_argument(
        "--auto-segment-window",
        type=int,
        default=0,
        help="Enable segment-level half-bar splitting with this many neighboring bars. 0 disables it.",
    )
    parser.add_argument(
        "--auto-segment-required",
        type=int,
        default=3,
        help="Relaxed split candidates required inside --auto-segment-window.",
    )
    parser.add_argument(
        "--diatonic-shadow-weight",
        type=float,
        default=0.0,
        help="Strength for adding same-root diatonic-quality shadow candidates.",
    )
    parser.add_argument(
        "--diatonic-bonus",
        type=float,
        default=0.0,
        help="Bonus for candidates whose quality matches the global key profile.",
    )
    parser.add_argument(
        "--edge-fill",
        action="store_true",
        help="Fill leading/trailing regions outside the beat grid when external chords are confident.",
    )
    parser.add_argument(
        "--edge-fill-min-duration",
        type=float,
        default=0.50,
        help="Minimum leading/trailing uncovered duration to fill.",
    )
    parser.add_argument(
        "--edge-fill-min-score",
        type=float,
        default=0.70,
        help="Minimum external chord score for --edge-fill.",
    )
    parser.add_argument(
        "--edge-fill-mode",
        choices=["summary", "segments"],
        default="summary",
        help="summary fills one averaged edge chord; segments keeps confident external edge segments.",
    )
    parser.add_argument(
        "--repeat-consensus-refine",
        action="store_true",
        help="Use repeated phrase positions to fix low-confidence outlier cells.",
    )
    parser.add_argument("--repeat-phrase-bars", default="4,8")
    parser.add_argument("--repeat-min-support", type=int, default=3)
    parser.add_argument("--repeat-min-share", type=float, default=0.60)
    parser.add_argument("--repeat-max-score-gap", type=float, default=0.08)
    parser.add_argument("--repeat-max-current-score", type=float, default=0.86)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = json.loads(args.base_analysis.read_text(encoding="utf-8"))
    evidence_base = (
        json.loads(args.base_evidence_analysis.read_text(encoding="utf-8"))
        if args.base_evidence_analysis
        else base
    )
    key = args.key or str(base.get("key") or "C")
    duration = float(base.get("duration") or 0.0)
    if args.beat_grid_from_base:
        beat_rows = beat_rows_from_base_analysis(base)
    else:
        try:
            beat_rows = read_downbeat_rows(args.downbeats)
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            beat_rows = read_beat_grid_rows(args.downbeats)
    meter = meter_from_downbeat_rows(beat_rows, duration, parse_meter(args.meter))
    if meter is None:
        raise SystemExit("Downbeat grid is not usable")
    primary_hints = read_external_chord_hints(args.external_chords, "sharps")
    extra_hints: list[ExternalChordHint] = []
    for extra_path in args.external_chords_extra or []:
        extra_hints.extend(read_external_chord_hints(extra_path, "sharps"))
    hints = list(primary_hints)
    if args.extra_gating == "off":
        hints.extend(scale_hints(extra_hints, args.external_chords_extra_weight))
    elif (
        extra_hints
        and args.extra_gating_min_track_agreement > 0
        and source_agreement(primary_hints, extra_hints) < args.extra_gating_min_track_agreement
    ):
        extra_hints = []
    hints.sort(key=lambda item: (item.start, item.end, item.chord))
    primary_hints.sort(key=lambda item: (item.start, item.end, item.chord))
    extra_hints.sort(key=lambda item: (item.start, item.end, item.chord))
    base_rows = base_evidence_rows(evidence_base)
    intervals = [item for item in build_beat_intervals(meter, duration) if item.bar >= 1]
    if args.harmonic_unit == "auto":
        cells = auto_harmonic_intervals(
            intervals,
            meter.beats_per_bar,
            hints,
            base_rows,
            args.base_weight,
            min_sub_score=args.auto_min_sub_score,
            min_gain=args.auto_min_gain,
            segment_window=args.auto_segment_window,
            segment_required=args.auto_segment_required,
        )
    else:
        cells = harmonic_intervals_from_beats(intervals, args.harmonic_unit, meter.beats_per_bar)
    cell_scores = [
        merge_score_rows(
            hint_scores_for_cell(cell.start, cell.end, primary_hints),
            gated_extra_scores_for_cell(
                cell.start,
                cell.end,
                primary_hints,
                extra_hints,
                args.external_chords_extra_weight,
                args.extra_gating,
                primary_strong=args.extra_gating_primary_strong,
                primary_margin=args.extra_gating_primary_margin,
                extra_min_score=args.extra_gating_min_score,
            ),
            base_scores_for_cell(cell.start, cell.end, base_rows, args.base_weight),
        )
        for cell in cells
    ]
    bass_votes = [bass_vote_for_cell(cell.start, cell.end, base_rows) for cell in cells]
    seventh_votes = [seventh_color_vote_for_cell(cell.start, cell.end, base_rows) for cell in cells]
    progression_scores = read_progression_scores(args.progression_prior)
    chords = smooth_chords(
        cell_scores,
        key,
        bass_votes,
        progression_scores=progression_scores,
        progression_strength=args.progression_prior_strength,
        bass_root_bonus=args.bass_root_weight,
        diatonic_bonus=args.diatonic_bonus,
        diatonic_shadow_weight=args.diatonic_shadow_weight,
    )
    chords = inherit_seventh_colors(chords, seventh_votes, args.dominant_seventh_threshold)
    if args.repeat_consensus_refine:
        chords = repeat_consensus_refine(
            chords,
            cell_scores,
            cells,
            parse_phrase_lengths(args.repeat_phrase_bars),
            min_support=args.repeat_min_support,
            min_share=args.repeat_min_share,
            max_score_gap=args.repeat_max_score_gap,
            max_current_score=args.repeat_max_current_score,
        )
    edge_fills = (
        edge_fill_candidates(
            intervals,
            duration,
            primary_hints,
            extra_hints,
            args.external_chords_extra_weight,
            args.extra_gating,
            args.edge_fill_min_duration,
            args.edge_fill_min_score,
            mode=args.edge_fill_mode,
        )
        if args.edge_fill
        else []
    )
    payload = make_payload(
        base,
        intervals,
        cells,
        chords,
        cell_scores,
        bass_votes,
        key,
        meter.meter_label,
        meter.beats_per_bar,
        meter.beat_period,
        args.harmonic_unit,
        edge_fills=edge_fills,
    )
    payload["bpm"] = display_bpm_for_meter(meter)
    payload["pulse_bpm"] = meter.bpm
    json_path = args.output.with_name(args.output.name + "_analysis.json")
    html_path = args.output.with_name(args.output.name + "_studio.html")
    txt_path = args.output.with_name(args.output.name + "_bar_chords.txt")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(html_path, payload)
    write_txt(txt_path, payload)
    print(f"Wrote: {json_path}")
    print(f"Wrote: {html_path}")
    print(f"Wrote: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
