#!/usr/bin/env python
"""Detect local key segments from a candidate table instead of top-1 chords."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from detect_key_timeline import (
    KeyWindow,
    apply_segments,
    build_window_ranges,
    chord_fit,
    display_key,
    key_candidates,
    key_transition_score,
    normalize_key,
    stable_segments,
)
from train_candidate_reranker import resolve_path


@dataclass(frozen=True)
class CandidateChord:
    bar: int
    start: float
    end: float
    chord: str
    source: str
    score: float
    confidence: float
    rank: float

    @property
    def weight(self) -> float:
        return max(0.1, self.end - self.start)


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def source_allowed(source: str, include: list[str], exclude: list[str]) -> bool:
    if include and not any(token in source for token in include):
        return False
    if exclude and any(token in source for token in exclude):
        return False
    return True


def load_candidates(
    path: Path,
    song_id: str,
    include_source: list[str],
    exclude_source: list[str],
    max_rank: float,
) -> list[CandidateChord]:
    rows: list[CandidateChord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if song_id and str(row.get("song_id") or "") != song_id:
                continue
            source = str(row.get("candidate_source") or "")
            if not source_allowed(source, include_source, exclude_source):
                continue
            rank = parse_float(row.get("candidate_rank"), 999.0)
            if rank > max_rank:
                continue
            chord = str(row.get("candidate_chord") or "N").split("/", 1)[0]
            if chord == "N":
                continue
            start = parse_float(row.get("reference_start") or row.get("start"))
            end = parse_float(row.get("reference_end") or row.get("end"))
            rows.append(
                CandidateChord(
                    bar=int(parse_float(row.get("bar"), 0.0)),
                    start=start,
                    end=end,
                    chord=chord,
                    source=source,
                    score=parse_float(row.get("candidate_score"), parse_float(row.get("candidate_confidence"), 0.0)),
                    confidence=parse_float(row.get("candidate_confidence"), 0.0),
                    rank=rank,
                )
            )
    return sorted(rows, key=lambda item: (item.bar, item.start, item.end, item.rank, -item.score))


def group_candidates(rows: list[CandidateChord]) -> list[list[CandidateChord]]:
    groups: dict[tuple[int, float, float], list[CandidateChord]] = {}
    for row in rows:
        groups.setdefault((row.bar, round(row.start, 3), round(row.end, 3)), []).append(row)
    return [groups[key] for key in sorted(groups)]


def acoustic_bonus(candidate: CandidateChord, score_weight: float, rank_weight: float) -> float:
    score = candidate.score if math.isfinite(candidate.score) else 0.0
    confidence = candidate.confidence if math.isfinite(candidate.confidence) else 0.0
    rank_penalty = 0.0 if candidate.rank >= 900 else rank_weight * max(candidate.rank - 1.0, 0.0)
    return score_weight * max(score, confidence) - rank_penalty


def score_group_for_key(group: list[CandidateChord], key: str, score_weight: float, rank_weight: float) -> float:
    return max(chord_fit(candidate.chord, key) + acoustic_bonus(candidate, score_weight, rank_weight) for candidate in group)


def functional_sequence_for_key(groups: list[list[CandidateChord]], key: str, score_weight: float, rank_weight: float) -> list[str]:
    from detect_key_timeline import functional_label

    labels: list[str] = []
    previous = None
    for group in groups:
        best = max(group, key=lambda item: chord_fit(item.chord, key) + acoustic_bonus(item, score_weight, rank_weight))
        label = functional_label(best.chord, key)
        if label != "N" and label != previous:
            labels.append(label)
            previous = label
    return labels


def score_key_from_groups(groups: list[list[CandidateChord]], key: str, score_weight: float, rank_weight: float) -> float:
    if not groups:
        return -999.0
    total_weight = sum(max(candidate.weight for candidate in group) for group in groups)
    base = sum(
        score_group_for_key(group, key, score_weight, rank_weight) * max(candidate.weight for candidate in group)
        for group in groups
    ) / max(total_weight, 1e-6)

    from detect_key_timeline import COMMON_MAJOR_LOOPS, COMMON_MAJOR_PAIRS, ENTRY_UNSTABLE_MAJOR, TONIC_FUNCTION_MAJOR

    seq = functional_sequence_for_key(groups, key, score_weight, rank_weight)
    if len(seq) >= 2:
        pair_total = len(seq) - 1
        pair_hits = sum(1 for a, b in zip(seq, seq[1:]) if (a, b) in COMMON_MAJOR_PAIRS)
        base += 0.20 * pair_hits / pair_total
        awkward = sum(1 for label in seq if label in {"bII", "#IV", "bVI", "#VI", "#VII"})
        base -= 0.22 * awkward / max(len(seq), 1)
        if not key.endswith("m"):
            if seq[0] in ENTRY_UNSTABLE_MAJOR:
                base -= 0.34
            elif seq[0] in TONIC_FUNCTION_MAJOR:
                base += 0.18
    if len(seq) >= 4:
        four_total = len(seq) - 3
        loop_hits = sum(1 for item in zip(seq, seq[1:], seq[2:], seq[3:]) if item in COMMON_MAJOR_LOOPS)
        base += 0.38 * loop_hits / four_total
    if key.endswith("m"):
        base -= 0.12
    return base


def window_groups(groups: list[list[CandidateChord]], start: int, end: int) -> list[list[CandidateChord]]:
    return [group for group in groups if start <= group[0].bar <= end]


def candidate_windows(
    groups: list[list[CandidateChord]],
    window_bars: int,
    step_bars: int,
    forced_initial_key: str | None,
    score_weight: float,
    rank_weight: float,
) -> list[KeyWindow]:
    if not groups:
        return []
    bars = [group[0].bar for group in groups]
    ranges = build_window_ranges(
        [
            type("Row", (), {"bar": bar})()  # minimal object for existing helper
            for bar in sorted(set(bars))
        ],
        window_bars,
        step_bars,
    )
    keys = key_candidates()
    min_bar = min(bars)
    initial_cutoff = min(max(bars), min_bar + max(window_bars, 16) - 1)
    initial_groups = window_groups(groups, min_bar, initial_cutoff)
    initial_scores = [(key, score_key_from_groups(initial_groups, key, score_weight, rank_weight)) for key in keys]
    initial_scores.sort(key=lambda item: item[1], reverse=True)
    initial_key = normalize_key(forced_initial_key) if forced_initial_key else initial_scores[0][0]

    emissions: list[dict[str, float]] = []
    local_margins: list[float] = []
    for start, end in ranges:
        current_groups = window_groups(groups, start, end)
        raw = {key: score_key_from_groups(current_groups, key, score_weight, rank_weight) for key in keys}
        if forced_initial_key and start == ranges[0][0]:
            raw = {initial_key: raw.get(initial_key, -999.0)}
        elif start <= min_bar + window_bars:
            raw[initial_key] = raw.get(initial_key, -999.0) + 0.05
        ordered = sorted(raw.items(), key=lambda item: item[1], reverse=True)
        local_margins.append(ordered[0][1] - ordered[1][1] if len(ordered) > 1 else 999.0)
        emissions.append(raw)

    dp: list[dict[str, float]] = []
    back: list[dict[str, str | None]] = []
    dp.append({key: score for key, score in emissions[0].items()})
    back.append({key: None for key in emissions[0]})
    for index in range(1, len(emissions)):
        current_scores: dict[str, float] = {}
        current_back: dict[str, str | None] = {}
        for current, emission in emissions[index].items():
            previous_key = None
            best_score = -999999.0
            for previous, previous_score in dp[index - 1].items():
                score = previous_score + emission + key_transition_score(previous, current)
                if score > best_score:
                    best_score = score
                    previous_key = previous
            current_scores[current] = best_score
            current_back[current] = previous_key
        dp.append(current_scores)
        back.append(current_back)

    last_key = max(dp[-1], key=dp[-1].get)
    path = [last_key]
    for index in range(len(dp) - 1, 0, -1):
        previous = back[index][path[-1]]
        path.append(previous or path[-1])
    path.reverse()
    return [
        KeyWindow(start, end, key, emissions[index].get(key, -999.0), local_margins[index])
        for index, ((start, end), key) in enumerate(zip(ranges, path))
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_json", type=Path, help="Analysis to relabel after detecting key segments.")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--include-source", action="append", default=[])
    parser.add_argument("--exclude-source", action="append", default=["corrected_grid"])
    parser.add_argument("--max-rank", type=float, default=8.0)
    parser.add_argument("--score-weight", type=float, default=0.20)
    parser.add_argument("--rank-weight", type=float, default=0.03)
    parser.add_argument("--initial-key")
    parser.add_argument("--window-bars", type=int, default=8)
    parser.add_argument("--step-bars", type=int, default=2)
    parser.add_argument("--min-segment-bars", type=int, default=8)
    parser.add_argument("--min-margin", type=float, default=0.10)
    parser.add_argument("--notation", choices=["auto", "sharps", "flats"], default="auto")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--html", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(resolve_path(args.analysis_json).read_text(encoding="utf-8"))
    rows = load_candidates(
        resolve_path(args.candidate_csv),
        args.song_id,
        args.include_source,
        args.exclude_source,
        args.max_rank,
    )
    groups = group_candidates(rows)
    windows = candidate_windows(
        groups,
        args.window_bars,
        args.step_bars,
        args.initial_key,
        args.score_weight,
        args.rank_weight,
    )
    segments = stable_segments(windows, args.window_bars, args.min_segment_bars, args.min_margin, args.initial_key)
    output = apply_segments(payload, segments, args.notation)
    output["key_timeline"] = {
        "method": "candidate-table sliding-window key analysis",
        "candidate_csv": str(args.candidate_csv),
        "song_id": args.song_id,
        "include_source": args.include_source,
        "exclude_source": args.exclude_source,
        "max_rank": args.max_rank,
        "score_weight": args.score_weight,
        "rank_weight": args.rank_weight,
        "window_bars": args.window_bars,
        "step_bars": args.step_bars,
        "min_segment_bars": args.min_segment_bars,
        "min_margin": args.min_margin,
        "windows": [window.__dict__ for window in windows],
    }
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {output_path}")
    print("segments:")
    for segment in segments:
        print(f"  bar {segment['start_bar']}-{segment.get('end_bar') or ''}: {display_key(str(segment['key']), args.notation)}")
    if args.html:
        from chord_studio import write_html

        html_path = output_path.with_name(output_path.stem.replace("_analysis", "") + "_studio.html")
        write_html(html_path, output)
        print(f"Wrote: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
