#!/usr/bin/env python
"""Select chord candidates with sequence-level harmonic phrase decoding."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_candidate_reranker import group_key, load_model
from progression_stats import roman_function_label
from train_candidate_reranker import filter_evaluable_rows, load_rows, resolve_path


REPO_ROOT = Path(__file__).resolve().parent.parent


def clean_label(value: object) -> str:
    text = str(value or "N").strip()
    return roman_function_label(text) if text else "N"


def load_prior(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-5), 1.0 - 1e-5)
    return math.log(probability / (1.0 - probability))


def phrase_score_maps(prior: dict[str, Any]) -> tuple[dict[str, float], dict[tuple[str, ...], float]]:
    pair_scores = {str(key): float(value) for key, value in prior.get("pair_scores", {}).items()}
    rows: list[dict[str, Any]] = []
    rows.extend(prior.get("bar_phrase_ngrams", []))
    ngrams = prior.get("ngrams", {})
    if isinstance(ngrams, dict):
        for size_rows in ngrams.values():
            if isinstance(size_rows, list):
                rows.extend(row for row in size_rows if isinstance(row, dict))
    max_count = max((float(row.get("count") or 0.0) for row in rows), default=1.0)
    ngram_scores: dict[tuple[str, ...], float] = {}
    for row in rows:
        pattern = tuple(clean_label(item) for item in row.get("pattern", []))
        count = float(row.get("count") or 0.0)
        if len(pattern) < 3 or "N" in pattern or count <= 0:
            continue
        # Log scaling keeps common phrases useful without making them hard rules.
        score = math.log1p(count) / math.log1p(max_count)
        ngram_scores[pattern] = max(ngram_scores.get(pattern, 0.0), score)
    return pair_scores, ngram_scores


def best_ngram_bonus(history: tuple[str, ...], ngram_scores: dict[tuple[str, ...], float], max_order: int) -> float:
    if len(history) < 3:
        return 0.0
    best = 0.0
    tail = history[-max_order:]
    for length in range(3, len(tail) + 1):
        pattern = tail[-length:]
        best = max(best, ngram_scores.get(pattern, 0.0))
    return best


def row_dict(row: Any, score: float, correct: int) -> dict[str, object]:
    return {
        "song_id": str(row.song_id),
        "bar": int(row.bar),
        "reference_start": f"{float(row.reference_start):.6f}",
        "reference_end": f"{float(row.reference_end):.6f}",
        "candidate_source": str(row.candidate_source),
        "candidate_chord": str(row.candidate_chord),
        "candidate_roman": str(row.candidate_roman),
        "reference_chord": str(row.reference_chord),
        "reference_roman": str(row.reference_roman),
        "sequence_score": f"{score:.6f}",
        "correct": correct,
    }


def decode_song(
    groups: list[Any],
    target: str,
    pair_scores: dict[str, float],
    ngram_scores: dict[tuple[str, ...], float],
    base_weight: float,
    pair_weight: float,
    ngram_weight: float,
    max_order: int,
) -> tuple[int, int, int, list[dict[str, object]]]:
    total = 0
    correct = 0
    oracle = 0
    selected_rows: list[dict[str, object]] = []
    previous_states: list[tuple[float, tuple[str, ...], int, Any]] = []
    backrefs: list[list[int]] = []
    state_rows: list[list[Any]] = []
    state_scores: list[list[float]] = []

    for group_index, group in enumerate(groups):
        candidates = [row for _, row in group.iterrows()]
        rows_for_group: list[Any] = []
        scores_for_group: list[float] = []
        back_for_group: list[int] = []
        for row in candidates:
            label = clean_label(row.candidate_roman)
            base = base_weight * logit(float(row.reranker_score))
            if group_index == 0 or not previous_states:
                score = base
                history = (label,)
                back = -1
            else:
                best_score = -1e18
                best_history: tuple[str, ...] = (label,)
                best_back = 0
                for prev_index, (prev_score, prev_history, _, prev_row) in enumerate(previous_states):
                    prev_label = clean_label(prev_row.candidate_roman)
                    transition = pair_weight * pair_scores.get(f"{prev_label}->{label}", 0.0)
                    history = (*prev_history, label)[-max_order:]
                    transition += ngram_weight * best_ngram_bonus(history, ngram_scores, max_order)
                    value = prev_score + base + transition
                    if value > best_score:
                        best_score = value
                        best_history = history
                        best_back = prev_index
                score = best_score
                history = best_history
                back = best_back
            rows_for_group.append(row)
            scores_for_group.append(score)
            back_for_group.append(back)
        state_rows.append(rows_for_group)
        state_scores.append(scores_for_group)
        backrefs.append(back_for_group)
        previous_states = [
            (scores_for_group[index], ((*previous_states[back_for_group[index]][1], clean_label(rows_for_group[index].candidate_roman))[-max_order:] if back_for_group[index] >= 0 and previous_states else (clean_label(rows_for_group[index].candidate_roman),)), index, rows_for_group[index])
            for index in range(len(rows_for_group))
        ]

    if not state_rows:
        return total, correct, oracle, selected_rows
    index = int(np.argmax(state_scores[-1]))
    path_indices = [0] * len(state_rows)
    for group_index in range(len(state_rows) - 1, -1, -1):
        path_indices[group_index] = index
        index = backrefs[group_index][index]
        if index < 0:
            index = 0

    for group_index, row_index in enumerate(path_indices):
        group_rows = state_rows[group_index]
        row = group_rows[row_index]
        total += 1
        is_correct = int(getattr(row, target))
        correct += is_correct
        oracle += int(max(int(getattr(candidate, target)) for candidate in group_rows))
        selected_rows.append(row_dict(row, state_scores[group_index][row_index], is_correct))
    return total, correct, oracle, selected_rows


def evaluate(
    model_path: Path,
    candidate_csv: Path,
    prior_path: Path | None,
    target: str | None,
    base_weight: float,
    pair_weight: float,
    ngram_weight: float,
    max_order: int,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    pipeline, model_target = load_model(resolve_path(model_path))
    target = target or model_target
    df = load_rows(resolve_path(candidate_csv))
    df = filter_evaluable_rows(df, target)
    df["reranker_score"] = pipeline.predict_proba(df)[:, 1]
    df["group_key"] = [group_key(row) for _, row in df.iterrows()]
    prior = load_prior(prior_path)
    pair_scores, ngram_scores = phrase_score_maps(prior)
    total = correct = oracle = 0
    selected_rows: list[dict[str, object]] = []
    by_song: dict[str, list[int]] = {}
    oracle_by_song: dict[str, list[int]] = {}
    for song_id, song_df in df.sort_values(["song_id", "start", "end"]).groupby("song_id", sort=True):
        groups = [
            group.sort_values("reranker_score", ascending=False)
            for _, group in song_df.groupby("group_key", sort=False)
        ]
        song_total, song_correct, song_oracle, song_selected = decode_song(
            groups,
            target,
            pair_scores,
            ngram_scores,
            base_weight,
            pair_weight,
            ngram_weight,
            max_order,
        )
        total += song_total
        correct += song_correct
        oracle += song_oracle
        selected_rows.extend(song_selected)
        by_song[str(song_id)] = [int(row["correct"]) for row in song_selected]
        oracle_by_song[str(song_id)] = [1] * song_oracle + [0] * max(0, song_total - song_oracle)
    report = {
        "target": target,
        "groups": total,
        "sequence_accuracy": round(correct / total, 4) if total else 0.0,
        "oracle_accuracy": round(oracle / total, 4) if total else 0.0,
        "base_weight": base_weight,
        "pair_weight": pair_weight,
        "ngram_weight": ngram_weight,
        "max_order": max_order,
        "model": str(resolve_path(model_path)),
        "candidate_csv": str(resolve_path(candidate_csv)),
        "prior": "" if prior_path is None else str(resolve_path(prior_path)),
        "by_song": {
            song: {
                "groups": len(values),
                "sequence_accuracy": round(sum(values) / len(values), 4) if values else 0.0,
                "oracle_accuracy": round(sum(oracle_by_song.get(song, [])) / len(oracle_by_song.get(song, [])), 4)
                if oracle_by_song.get(song)
                else 0.0,
            }
            for song, values in sorted(by_song.items())
        },
    }
    return report, selected_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--pair-weight", type=float, default=0.05)
    parser.add_argument("--ngram-weight", type=float, default=0.03)
    parser.add_argument("--max-order", type=int, default=8)
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "candidate_sequence_decoder_report.json")
    parser.add_argument("--selected", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "candidate_sequence_decoder_selected.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report, selected_rows = evaluate(
        args.model,
        args.candidate_csv,
        args.prior,
        args.target,
        args.base_weight,
        args.pair_weight,
        args.ngram_weight,
        args.max_order,
    )
    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    selected_path = resolve_path(args.selected)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    with selected_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "song_id",
            "bar",
            "reference_start",
            "reference_end",
            "candidate_source",
            "candidate_chord",
            "candidate_roman",
            "reference_chord",
            "reference_roman",
            "sequence_score",
            "correct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote: {selected_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
