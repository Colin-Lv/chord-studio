#!/usr/bin/env python
"""Add bar-level harmonic phrase prior features to candidate reranker rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from progression_stats import roman_function_label


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_prior(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_roman(value: object) -> str:
    text = str(value or "N").strip()
    return roman_function_label(text) if text else "N"


def pattern_score_map(prior: dict[str, Any]) -> dict[tuple[int, int, tuple[str, ...]], float]:
    rows: list[dict[str, Any]] = []
    rows.extend(prior.get("bar_phrase_ngrams", []))
    ngrams = prior.get("ngrams", {})
    if isinstance(ngrams, dict):
        for size_text, size_rows in ngrams.items():
            try:
                size = int(size_text)
            except (TypeError, ValueError):
                continue
            if not isinstance(size_rows, list):
                continue
            for row in size_rows:
                if not isinstance(row, dict):
                    continue
                rows.append(
                    {
                        "slots_per_bar": row.get("slots_per_bar", 1),
                        "bars": row.get("bars", size),
                        "pattern": row.get("pattern", []),
                        "count": row.get("count", 0.0),
                    }
                )
    totals: dict[tuple[int, int], float] = defaultdict(float)
    counts: dict[tuple[int, int, tuple[str, ...]], float] = {}
    for row in rows:
        try:
            slots = int(row.get("slots_per_bar") or 0)
            bars = int(row.get("bars") or 0)
            count = float(row.get("count") or 0.0)
        except (TypeError, ValueError):
            continue
        pattern = tuple(clean_roman(label) for label in row.get("pattern", []))
        if slots <= 0 or bars <= 0 or not pattern or count <= 0:
            continue
        key = (slots, bars, pattern)
        counts[key] = max(counts.get(key, 0.0), count)
        totals[(slots, bars)] += count
    scores: dict[tuple[int, int, tuple[str, ...]], float] = {}
    for (slots, bars, pattern), count in counts.items():
        total = totals[(slots, bars)] or 1.0
        scores[(slots, bars, pattern)] = count / total
    return scores


def center(row: dict[str, str]) -> float:
    try:
        return (float(row.get("start") or 0.0) + float(row.get("end") or 0.0)) / 2.0
    except ValueError:
        return 0.0


def interval_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        str(row.get("song_id") or ""),
        str(row.get("bar") or ""),
        str(row.get("reference_start") or row.get("start") or ""),
        str(row.get("reference_end") or row.get("end") or ""),
    )


def candidate_rank(row: dict[str, str]) -> float:
    try:
        return float(row.get("candidate_rank") or 9999.0)
    except ValueError:
        source = str(row.get("candidate_source") or "")
        if "_rank" in source:
            try:
                return float(source.rsplit("_rank", 1)[1])
            except ValueError:
                pass
        return 9999.0


def candidate_confidence(row: dict[str, str]) -> float:
    for key in ("candidate_confidence", "candidate_score"):
        try:
            return float(row.get(key) or 0.0)
        except ValueError:
            continue
    return 0.0


def selected_context(rows: list[dict[str, str]], target_column: str, context_source: str) -> dict[tuple[str, int], str]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[interval_key(row)].append(row)
    context: dict[tuple[str, int], str] = {}
    for group_rows in grouped.values():
        try:
            bar = int(float(group_rows[0].get("bar") or 0))
        except ValueError:
            continue
        song_id = str(group_rows[0].get("song_id") or "")
        if context_source == "oracle":
            positives = [row for row in group_rows if str(row.get(target_column) or "0") == "1"]
            chosen = positives[0] if positives else min(group_rows, key=candidate_rank)
        elif context_source == "confidence":
            chosen = max(group_rows, key=candidate_confidence)
        else:
            chosen = min(group_rows, key=candidate_rank)
        label = clean_roman(chosen.get("candidate_roman"))
        if label != "N":
            context[(song_id, bar)] = label
    return context


def best_phrase_scores(
    candidate_label: str,
    context: dict[tuple[str, int], str],
    song_id: str,
    bar: int,
    phrase_scores: dict[tuple[int, int, tuple[str, ...]], float],
) -> dict[str, float]:
    result = {
        "phrase_1bar_forward_score": 0.0,
        "phrase_1bar_backward_score": 0.0,
        "phrase_2bar_score": 0.0,
        "phrase_4bar_score": 0.0,
        "phrase_8bar_score": 0.0,
        "phrase_completion_score": 0.0,
        "phrase_any_score": 0.0,
    }
    candidate = clean_roman(candidate_label)
    if candidate == "N" or bar <= 0:
        return result
    for (slots, bars, pattern), score in phrase_scores.items():
        if slots != 1:
            continue
        length = len(pattern)
        if bars == 1 and length == 1 and pattern[0] == candidate:
            result["phrase_1bar_forward_score"] = max(result["phrase_1bar_forward_score"], score)
            result["phrase_1bar_backward_score"] = max(result["phrase_1bar_backward_score"], score)
        if length <= 1:
            continue
        for offset, label in enumerate(pattern):
            if label != candidate:
                continue
            start_bar = bar - offset
            known = 0
            possible = True
            for index, expected in enumerate(pattern):
                if index == offset:
                    continue
                actual = context.get((song_id, start_bar + index))
                if actual is None:
                    continue
                known += 1
                if actual != expected:
                    possible = False
                    break
            if not possible:
                continue
            weighted = score * (known / max(1, length - 1))
            if bars <= 2:
                result["phrase_2bar_score"] = max(result["phrase_2bar_score"], weighted)
            if bars <= 4:
                result["phrase_4bar_score"] = max(result["phrase_4bar_score"], weighted)
            if bars <= 8:
                result["phrase_8bar_score"] = max(result["phrase_8bar_score"], weighted)
            if offset == length - 1 and known >= max(1, length - 2):
                result["phrase_completion_score"] = max(result["phrase_completion_score"], weighted)
    result["phrase_any_score"] = max(result.values())
    return result


def augment(
    rows: list[dict[str, str]],
    prior: dict[str, Any],
    target_column: str,
    context_source: str,
) -> list[dict[str, str]]:
    phrase_scores = pattern_score_map(prior)
    context = selected_context(rows, target_column, context_source)
    output = []
    for row in rows:
        new_row = dict(row)
        try:
            bar = int(float(row.get("bar") or 0))
        except ValueError:
            bar = 0
        scores = best_phrase_scores(
            clean_roman(row.get("candidate_roman")),
            context,
            str(row.get("song_id") or ""),
            bar,
            phrase_scores,
        )
        for key, value in scores.items():
            new_row[key] = f"{value:.8f}"
        output.append(new_row)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--target-column", default="target_exact_chord")
    parser.add_argument(
        "--context-source",
        choices=["rank1", "confidence", "oracle"],
        default="rank1",
        help="How to build surrounding phrase context. Use oracle only for diagnostics, never for deployable training.",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    prior = load_prior(resolve_path(args.prior))
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    rows = augment(rows, prior, args.target_column, args.context_source)
    for column in [
        "phrase_1bar_forward_score",
        "phrase_1bar_backward_score",
        "phrase_2bar_score",
        "phrase_4bar_score",
        "phrase_8bar_score",
        "phrase_completion_score",
        "phrase_any_score",
    ]:
        if column not in fieldnames:
            fieldnames.append(column)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} phrase-augmented row(s): {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
