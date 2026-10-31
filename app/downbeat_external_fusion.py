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
    normalize_external_chord_label,
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
class PathProposal:
    bar: int
    unit: str
    cells: list[object]
    cell_scores: list[dict[str, float]]
    bar_score: float
    avg_top_score: float
    avg_margin: float
    strong_root_count: int
    root_switches: int
    distinct_roots: int
    score: float
    chosen: bool = False


@dataclass(frozen=True)
class EdgeFillCandidate:
    start: float
    end: float
    chord: str
    confidence: float


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def source_label(label: str) -> str:
    """Normalize explicit bass syntax without rewriting ordinary chord labels."""
    if label.strip().upper() == "X":
        return "X"
    if "/" in label:
        return normalize_external_chord_label(label, "sharps")
    return label


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
        label = source_label(hint.chord)
        scores[label] = scores.get(label, 0.0) + (amount / duration) * hint.confidence
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
        chord = source_label(str(row.get("chord") or "N"))
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
        chord = source_label(str(row.get("chord") or "N"))
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
            label = source_label(chord)
            merged[label] = merged.get(label, 0.0) + float(score)
    return merged


def top_score_label(scores: dict[str, float]) -> tuple[str, float]:
    if not scores:
        return "N", 0.0
    return max(scores.items(), key=lambda item: item[1])


def second_score(scores: dict[str, float], top_label: str) -> float:
    values = [float(score) for label, score in scores.items() if label != top_label]
    return max(values) if values else 0.0


def root_family(label: str) -> tuple[int | None, str]:
    root, suffix = parse_chord_name(source_label(label))
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
        if primary is not None and extra is not None and source_label(primary.chord) not in {"N", "X"} and source_label(extra.chord) not in {"N", "X"}:
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
    include_no_chord: bool = False,
) -> list[EdgeFillCandidate]:
    """Retain observed edge segments; optional N never fills a source gap.

    ``include_no_chord`` applies to segment preservation. Legacy summary mode
    still produces only chord estimates, rather than inventing a summarized N.
    """
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
                if amount < min_duration or (hint.chord == "N" and not include_no_chord) or hint.confidence < min_score:
                    continue
                candidates.append(
                    EdgeFillCandidate(
                        max(start, hint.start),
                        min(end, hint.end),
                        source_label(hint.chord),
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
    sub_roots = [root_family(label)[0] for label, score in sub_tops if root_family(label)[0] is not None and score >= min_sub_score]
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
        if root_family(label)[0] is not None and score >= min_sub_score
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


def path_unit_candidates(beats_per_bar: int) -> list[str]:
    units = ["bar"]
    if beats_per_bar == 6:
        units.append("compound-half")
    elif beats_per_bar == 4:
        units.append("half-bar")
    if beats_per_bar in {4, 6}:
        units.append("beat")
    return units


def cells_for_unit(intervals: list[BeatInterval], unit: str, beats_per_bar: int) -> list[object]:
    if unit == "bar":
        return harmonic_intervals_from_beats(intervals, "bar", beats_per_bar)
    if unit == "beat":
        return harmonic_intervals_from_beats(intervals, "beat", beats_per_bar)
    if unit in {"half-bar", "compound-half"}:
        return harmonic_intervals_from_beats(intervals, unit, beats_per_bar)
    return harmonic_intervals_from_beats(intervals, "bar", beats_per_bar)


def external_harmonic_intervals(
    intervals: list[BeatInterval], hints: list[ExternalChordHint], snap_tolerance: float = 0.0,
) -> list[HarmonicInterval]:
    """Preserve external chord changes, including offbeat/quality-only changes.

    Bar lines remain display boundaries; they do not move acoustic boundaries.
    A chord gap is retained as its own cell instead of stretching adjacent labels.
    """
    cells: list[HarmonicInterval] = []
    boundaries = sorted({float(value) for hint in hints for value in (hint.start, hint.end)
                         if np.isfinite(value)})
    if snap_tolerance > 0 and intervals:
        beat_times = np.asarray(sorted({value for beat in intervals for value in (beat.start, beat.end)}))
        snapped = []
        for time in boundaries:
            index = int(np.argmin(np.abs(beat_times - time)))
            snapped.append(float(beat_times[index]) if abs(float(beat_times[index]) - time) <= snap_tolerance else time)
        boundaries = sorted(set(snapped))
    for bar in sorted({beat.bar for beat in intervals}):
        beats = sorted((beat for beat in intervals if beat.bar == bar), key=lambda beat: beat.start)
        start, end = beats[0].start, beats[-1].end
        inner = [time for time in boundaries if start + 1e-7 < time < end - 1e-7]
        cuts = [start, *inner, end]
        for left, right in zip(cuts, cuts[1:]):
            if right - left <= 1e-7:
                continue
            covering = [beat for beat in beats if overlap(left, right, beat.start, beat.end) > 1e-7]
            if covering:
                cells.append(HarmonicInterval(bar, covering[0].beat, covering[-1].beat, left, right))
    return cells


def path_proposal_metrics(
    cells: list[object],
    hints: list[ExternalChordHint],
    base_rows: list[dict[str, object]],
    base_weight: float,
    min_sub_score: float,
) -> tuple[list[dict[str, float]], float, float, float, int, int, int, float]:
    scores = [
        merge_score_rows(
            hint_scores_for_cell(cell.start, cell.end, hints),
            base_scores_for_cell(cell.start, cell.end, base_rows, base_weight),
        )
        for cell in cells
    ]
    top_pairs = [top_score_label(row) for row in scores]
    top_scores = [float(score) for _, score in top_pairs]
    margins = [float(score - second_score(scores[index], label)) for index, (label, score) in enumerate(top_pairs)]
    strong_roots = []
    for label, score in top_pairs:
        if label == "N" or score < min_sub_score:
            continue
        strong_roots.append(root_family(label)[0])
    distinct_roots = len({root for root in strong_roots if root is not None})
    root_switches = sum(
        1 for left, right in zip(strong_roots, strong_roots[1:]) if left is not None and right is not None and left != right
    )
    avg_top_score = float(np.mean(top_scores)) if top_scores else 0.0
    avg_margin = float(np.mean(margins)) if margins else 0.0
    strong_root_count = len([root for root in strong_roots if root is not None])
    coverage = strong_root_count / max(len(cells), 1)
    path_score = (
        avg_top_score
        + 0.12 * root_switches
        + 0.05 * distinct_roots
        + 0.04 * avg_margin
        - 0.03 * max(0, len(cells) - 1)
        + 0.02 * coverage
    )
    bar_score = avg_top_score if cells else 0.0
    return scores, bar_score, avg_top_score, avg_margin, strong_root_count, root_switches, distinct_roots, path_score


def propose_segment_paths(
    intervals: list[BeatInterval],
    beats_per_bar: int,
    hints: list[ExternalChordHint],
    base_rows: list[dict[str, object]],
    base_weight: float,
    min_sub_score: float = 0.56,
) -> tuple[list[object], list[PathProposal]]:
    proposals: list[PathProposal] = []
    selected_cells: list[object] = []
    allowed_units = path_unit_candidates(beats_per_bar)
    bars = sorted({item.bar for item in intervals})
    for bar_number in bars:
        bar_intervals = [item for item in intervals if item.bar == bar_number]
        unit_candidates: list[tuple[str, list[object]]] = []
        for unit in allowed_units:
            cells = [cell for cell in cells_for_unit(bar_intervals, unit, beats_per_bar) if cell.bar == bar_number]
            if cells:
                unit_candidates.append((unit, cells))
        if not unit_candidates:
            continue
        bar_choice: PathProposal | None = None
        for unit, cells in unit_candidates:
            scores, bar_score, avg_top_score, avg_margin, strong_root_count, root_switches, distinct_roots, path_score = path_proposal_metrics(
                cells,
                hints,
                base_rows,
                base_weight,
                min_sub_score,
            )
            proposal = PathProposal(
                bar=bar_number,
                unit=unit,
                cells=cells,
                cell_scores=scores,
                bar_score=bar_score,
                avg_top_score=avg_top_score,
                avg_margin=avg_margin,
                strong_root_count=strong_root_count,
                root_switches=root_switches,
                distinct_roots=distinct_roots,
                score=path_score,
            )
            if bar_choice is None or proposal.score > bar_choice.score:
                bar_choice = proposal
        assert bar_choice is not None
        proposals.append(bar_choice)
        selected_cells.extend(bar_choice.cells)
    return selected_cells, proposals


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
) -> tuple[list[object], list[PathProposal]]:
    if beats_per_bar not in {4, 6}:
        cells = harmonic_intervals_from_beats(intervals, "bar", beats_per_bar)
        proposals = []
        for cell in cells:
            scores = merge_score_rows(
                hint_scores_for_cell(cell.start, cell.end, hints),
                base_scores_for_cell(cell.start, cell.end, base_rows, base_weight),
            )
            top_label, top_score = top_score_label(scores)
            proposals.append(
                PathProposal(
                    bar=cell.bar,
                    unit="bar",
                    cells=[cell],
                    cell_scores=[scores],
                    bar_score=top_score,
                    avg_top_score=top_score,
                    avg_margin=top_score - second_score(scores, top_label),
                    strong_root_count=1 if root_family(top_label)[0] is not None else 0,
                    root_switches=0,
                    distinct_roots=1 if root_family(top_label)[0] is not None else 0,
                    score=top_score,
                    chosen=True,
                )
            )
        return cells, proposals

    path_cells, proposals = propose_segment_paths(
        intervals,
        beats_per_bar,
        hints,
        base_rows,
        base_weight,
        min_sub_score=min_sub_score,
    )
    if not path_cells:
        return harmonic_intervals_from_beats(intervals, "bar", beats_per_bar), proposals
    return path_cells, proposals


def inherit_seventh_colors(
    chords: list[str],
    seventh_votes: list[tuple[str | None, float]],
    threshold: float,
) -> list[str]:
    refined = list(chords)
    for index, chord in enumerate(chords):
        chord = source_label(chord)
        root, suffix = parse_chord_name(chord)
        voted_chord, score = seventh_votes[index]
        voted_chord = source_label(voted_chord) if voted_chord else voted_chord
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
            # A quality vote may add a seventh, never replace the primary
            # source's explicit inversion/pedal with a template bass choice.
            voted_body = (voted_chord or chord).split("/", 1)[0]
            refined[index] = voted_body + ("/" + chord.split("/", 1)[1] if "/" in chord else "")
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


def roman_extended_label(label: str) -> str:
    text = label.split("/", 1)[0]
    if text == "N":
        return "N"
    degrees = ["bVII", "#VII", "VII", "bVI", "#VI", "VI", "#IV", "IV", "bIII", "#III", "III", "bII", "II", "V", "I"]
    for degree in degrees:
        if not text.startswith(degree):
            continue
        suffix = text[len(degree) :]
        minor = "m" if suffix.startswith("m") else ""
        extension_suffix = suffix[1:] if minor else suffix
        extension = ""
        for marker in ("maj7", "m7b5", "m7", "7", "dim", "aug", "sus4", "sus2", "add9", "9", "6"):
            if marker in extension_suffix:
                extension = marker
                break
        return f"{degree}{minor}{extension}"
    return text


def progression_prior_uses_extensions(progression_scores: dict[str, float] | None) -> bool:
    if not progression_scores:
        return False
    markers = ("7", "add", "sus", "dim", "aug", "6", "9")
    return any(any(marker in key for marker in markers) for key in progression_scores)


def diatonic_shadow_chord(chord: str, key: str) -> str | None:
    chord = source_label(chord)
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
    explicit_bass = "/" + chord.split("/", 1)[1] if "/" in chord else ""
    return f"{note_name_for_pc(root, key)}{expected_quality}{seventh}{explicit_bass}"


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
    if not cell_scores:
        return []
    cell_scores = add_diatonic_shadow_scores([merge_score_rows(row) for row in cell_scores], key, shadow_weight=diatonic_shadow_weight)
    labels = sorted({label for scores in cell_scores for label in scores} | {"N"})
    label_to_index = {label: index for index, label in enumerate(labels)}
    use_extended_prior = progression_prior_uses_extensions(progression_scores)
    roman_common = {
        label: roman_function_label(chord_to_roman_display(label, key, "N", 0.0, 1.0))
        for label in labels
    }
    roman_prior = {
        label: (
            roman_extended_label(chord_to_roman_display(label, key, "N", 0.0, 1.0))
            if use_extended_prior
            else roman_common[label]
        )
        for label in labels
    }
    profile = diatonic_profile(key)
    scores = np.full((len(cell_scores), len(labels)), -0.35, dtype=np.float32)
    for time_index, row in enumerate(cell_scores):
        for label, score in row.items():
            scores[time_index, label_to_index[label]] = float(score)
    if "X" in label_to_index:
        unknown_index = label_to_index["X"]
        for time_index, row in enumerate(cell_scores):
            if "X" not in row:
                # Unknown is a local source observation, not a harmonic state
                # that the progression model can introduce into other cells.
                scores[time_index, unknown_index] = -np.inf
            elif not any(label != "X" and score > 0 for label, score in row.items()):
                # Preserve explicit unknown-only evidence even with low source
                # confidence; absence of a chord cannot establish silence N.
                observed_score = scores[time_index, unknown_index]
                scores[time_index, :] = -np.inf
                scores[time_index, unknown_index] = observed_score
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
            curr_common = roman_common[curr_label]
            curr_prior = roman_prior[curr_label]
            for prev_index, prev_label in enumerate(labels):
                prev_common = roman_common[prev_label]
                prev_prior = roman_prior[prev_label]
                transition = 0.0
                if curr_label == prev_label and curr_label != "X":
                    transition += stay_bonus
                if progression_scores and "X" not in {prev_label, curr_label}:
                    transition += progression_strength * progression_scores.get(f"{prev_prior}->{curr_prior}", 0.0)
                if (prev_common, curr_common) in COMMON_TRANSITIONS:
                    transition += common_bonus
                if prev_label not in {"N", "X"} and curr_label not in {"N", "X"} and curr_label != prev_label:
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
                if refined[index] == "X":
                    continue
                support: dict[str, float] = {}
                count: dict[str, int] = {}
                for other in indexes:
                    if other == index:
                        continue
                    label = refined[other]
                    if label in {"N", "X"}:
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
    path_proposals: list[PathProposal] | None = None,
) -> dict[str, object]:
    cell_rows = []
    for cell, chord, scores, bass_vote in zip(cells, chords, cell_scores, bass_votes):
        chord = source_label(chord)
        scores = merge_score_rows(scores)
        # The selected path may differ from the emission winner. Reporting the
        # winner's confidence for another chord makes incorrect overrides look certain.
        confidence = float(np.clip(scores.get(chord, 0.0), 0.0, 1.0))
        bass_pc, bass_confidence = bass_vote
        bass_root = note_name_for_pc(bass_pc, key) if bass_pc is not None else "N"
        cell_rows.append(
            {
                "bar": cell.bar,
                "unit": "external" if harmonic_unit == "external" else span_unit(cell.beat_start, cell.beat_end, beats_per_bar),
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
            covering = max(bar_cells,
                           key=lambda cell: overlap(beat.start, beat.end, float(cell["start"]), float(cell["end"])),
                           default=None)
            if covering is not None and overlap(beat.start, beat.end, float(covering["start"]), float(covering["end"])) <= 0:
                covering = None
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
        grid_start = min(item.start for item in intervals)
        grid_end = max(item.end for item in intervals)
        edge_groups: dict[int, list[dict[str, object]]] = {}
        for fill in edge_fills:
            chord = source_label(fill.chord)
            bass_root = "N"
            bass_confidence = 0.0
            is_leading_edge = bool(bars) and fill.end <= float(bars[0]["start"])
            bar_number = 0 if is_leading_edge else last_bar + 1
            edge_event = {
                    "bar": bar_number,
                    "unit": "edge",
                    "beat_start": 0,
                    "beat_end": 0,
                    "start": fill.start,
                    "end": fill.end,
                    "chord": chord,
                    "chord_display": chord_display_name(chord, bass_root, bass_confidence, fill.confidence),
                    "roman": chord_to_roman(chord, key),
                    "roman_display": chord_to_roman_display(chord, key, bass_root, bass_confidence, fill.confidence),
                    "bass_root": bass_root,
                    "bass_confidence": bass_confidence,
                    "confidence": fill.confidence,
                    "grid_status": "uncovered",
                    "source": "external_chord_edge",
                }
            events.append(edge_event)
            edge_groups.setdefault(bar_number, []).append(edge_event)
        # Edge audio is real evidence but has no observed pulse. Give it a
        # matching unplaced display region without inventing beats or bars.
        for bar_number, spans in edge_groups.items():
            leading = bar_number == 0
            bars.append({"bar": bar_number, "start": 0.0 if leading else grid_end,
                         "end": grid_start if leading else float(base.get("duration") or max(fill.end for fill in edge_fills)),
                         "beats": [], "spans": sorted(spans, key=lambda item: float(item["start"])),
                         "grid_status": "uncovered", "grid_reason": "outside_constructed_beat_intervals",
                         "beat_source": "uncovered", "local_bpm": 0.0, "local_pulse_bpm": 0.0})
        bars.sort(key=lambda item: (float(item["start"]), int(item["bar"])))
    proposal_rows = []
    if path_proposals:
        for proposal in path_proposals:
            proposal_rows.append(
                {
                    "bar": proposal.bar,
                    "unit": proposal.unit,
                    "score": round(float(proposal.score), 6),
                    "bar_score": round(float(proposal.bar_score), 6),
                    "avg_top_score": round(float(proposal.avg_top_score), 6),
                    "avg_margin": round(float(proposal.avg_margin), 6),
                    "strong_root_count": int(proposal.strong_root_count),
                    "root_switches": int(proposal.root_switches),
                    "distinct_roots": int(proposal.distinct_roots),
                    "chosen": bool(proposal.chosen),
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
        "meter_confidence": float(base.get("meter_confidence") or 0.0),
        "beat_period": beat_period,
        "beats_per_bar": beats_per_bar,
        "beat_grid": "external_chord_fusion",
        "beat_source": str(base.get("beat_source") or "external"),
        "analysis_model": "external-fusion",
        "confidence_kind": "weighted_temporal_support",
        "confidence_note": "Weighted temporal support for the selected label; external LABs without scores use unit weight. Not a calibrated model correctness probability.",
        "tempo_variant": "provided-grid-external-chords",
        "harmonic_unit": harmonic_unit,
        "display": base.get("display") or "roman",
        "tempo_segments": tempo_segments_from_bars(bars),
        "bars": bars,
        "events": events,
        "path_proposals": proposal_rows,
    }


def apply_bass_display_policy(payload: dict, infer: bool) -> dict:
    """Preserve explicit source inversions; optionally suppress inferred bass display.

    Bass evidence and canonical chord/time fields are retained for review. This
    changes the display used by export/scoring, never the decoded chord path.
    """
    payload["bass_display_policy"] = "legacy_inferred_slash" if infer else "explicit_source_only"
    if infer:
        return payload

    def update(item: dict) -> None:
        chord = source_label(str(item.get("chord") or "N"))
        old_display = str(item.get("chord_display") or chord)
        if "/" not in chord and "/" in old_display:
            item["inferred_bass_candidate"] = {
                "chord_display": old_display, "roman_display": item.get("roman_display"),
                "bass_root": item.get("bass_root"), "bass_confidence": item.get("bass_confidence"),
                "source": "template_bass_display", "score_kind": "uncalibrated_evidence_score",
            }
        item["chord_display"] = chord
        item["roman_display"] = chord_to_roman(chord, str(item.get("local_key") or payload.get("key") or "C"))
        for span in item.get("spans", []):
            update(span)

    for event in payload.get("events", []):
        update(event)
    for bar in payload.get("bars", []):
        for item in [*bar.get("spans", []), *bar.get("beats", [])]:
            update(item)
    return payload


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
    parser.add_argument("--no-html", action="store_true", help="Skip workbench rendering for reproducible batch evaluation.")
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
    parser.add_argument("--beat-provenance", choices=["external", "reference", "madmom", "librosa"], default="external",
                        help="Origin of the supplied beat grid, recorded in output metadata.")
    parser.add_argument(
        "--beat-grid-from-base",
        action="store_true",
        help="Use beat times already stored in base_analysis instead of reading the downbeats argument.",
    )
    parser.add_argument(
        "--harmonic-unit",
        choices=["bar", "half-bar", "compound-half", "beat", "auto", "external"],
        default="bar",
    )
    parser.add_argument(
        "--base-weight",
        type=float,
        default=0.0,
        help="Weak weight for the local template chord evidence stored in base_analysis.",
    )
    parser.add_argument("--boundary-snap-tolerance", type=float, default=0.0,
                        help="For external cells only: snap source boundaries within this many seconds of a beat; retain offbeat changes.")
    parser.add_argument(
        "--bass-root-weight",
        type=float,
        default=0.0,
        help="Bonus for candidates whose root matches the local bass-root estimate.",
    )
    parser.add_argument("--infer-slash-bass", action=argparse.BooleanOptionalAction, default=False,
                        help="Opt into legacy uncalibrated template-bass inversions; default displays explicit source slash chords only.")
    parser.add_argument(
        "--dominant-seventh-threshold",
        type=float,
        default=1.01,
        help="Experimental template seventh inheritance threshold. Default 1.01 disables it; legacy 0.45 over-added sevenths in held-out evaluation.",
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retain source chord segments outside the observed beat grid; --no-edge-fill reproduces legacy clipping.",
    )
    parser.add_argument(
        "--edge-fill-min-duration",
        type=float,
        default=0.001,
        help="Minimum leading/trailing uncovered duration to fill.",
    )
    parser.add_argument(
        "--edge-fill-include-no-chord", action=argparse.BooleanOptionalAction, default=True,
        help="Keep explicit source N segments in edge segment mode; disabling reproduces legacy missing-N coverage.",
    )
    parser.add_argument(
        "--edge-fill-min-score",
        type=float,
        default=0.0,
        help="Minimum external chord score for --edge-fill.",
    )
    parser.add_argument(
        "--edge-fill-mode",
        choices=["summary", "segments"],
        default="segments",
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
    if not np.isfinite(args.dominant_seventh_threshold) or args.dominant_seventh_threshold < 0:
        raise SystemExit("--dominant-seventh-threshold must be finite and nonnegative")
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
        base = {**base, "beat_source": args.beat_provenance, "meter_confidence": 0.0}
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
    if args.harmonic_unit == "external":
        cells = external_harmonic_intervals(intervals, primary_hints, args.boundary_snap_tolerance)
        path_proposals = []
    elif args.harmonic_unit == "auto":
        cells, path_proposals = auto_harmonic_intervals(
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
        path_proposals = []
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
            include_no_chord=args.edge_fill_include_no_chord,
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
        path_proposals=path_proposals,
    )
    payload["bpm"] = display_bpm_for_meter(meter)
    apply_bass_display_policy(payload, args.infer_slash_bass)
    payload["pulse_bpm"] = meter.bpm
    payload["fusion_provenance"] = {
        "beat_source": payload["beat_source"],
        "key_source": "provided" if args.key else "base_analysis",
        "primary_chords": str(args.external_chords),
        "progression_prior": str(args.progression_prior) if args.progression_prior else None,
        "boundary_snap_tolerance": args.boundary_snap_tolerance,
        "dominant_seventh_threshold": args.dominant_seventh_threshold,
        "template_seventh_inheritance_enabled": args.dominant_seventh_threshold <= 1.0,
        "template_seventh_vote_method": "legacy_beat_and_span_sum_uncalibrated",
        "bass_root_weight": args.bass_root_weight,
        "infer_slash_bass": args.infer_slash_bass,
        "base_weight": args.base_weight,
        "progression_prior_strength": args.progression_prior_strength,
        "edge_fill": args.edge_fill,
        "edge_fill_mode": args.edge_fill_mode,
        "edge_fill_include_no_chord": args.edge_fill_include_no_chord,
        "edge_fill_min_duration": args.edge_fill_min_duration,
        "edge_fill_min_score": args.edge_fill_min_score,
    }
    json_path = args.output.with_name(args.output.name + "_analysis.json")
    html_path = args.output.with_name(args.output.name + "_studio.html")
    txt_path = args.output.with_name(args.output.name + "_bar_chords.txt")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_html:
        write_html(html_path, payload)
    write_txt(txt_path, payload)
    print(f"Wrote: {json_path}")
    if not args.no_html:
        print(f"Wrote: {html_path}")
    print(f"Wrote: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
