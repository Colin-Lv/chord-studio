#!/usr/bin/env python
"""Evaluate a trained candidate reranker as a selector over reference-aligned candidates."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import warnings
from pathlib import Path

import pandas as pd

from train_candidate_reranker import FEATURE_SCHEMA, filter_evaluable_rows, load_rows, resolve_path


REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_CHOICES = ["target_exact_chord", "target_same_root", "target_same_roman_root", "target_same_chord_family"]


def load_model(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("feature_schema") != FEATURE_SCHEMA:
        warnings.warn("Legacy candidate model uses reference-group feature semantics; retrain with the current song-held-out pipeline before deployment.",
                      RuntimeWarning, stacklevel=2)
    return payload["pipeline"], payload.get("target", "target_same_roman_root")


def group_key(row: pd.Series) -> tuple[str, float, float]:
    return (
        str(row.song_id),
        round(float(row.reference_start), 6),
        round(float(row.reference_end), 6),
    )


def evaluate(df: pd.DataFrame, scores, target: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    df = df.copy()
    df["reranker_score"] = scores
    df["group_key"] = [group_key(row) for _, row in df.iterrows()]
    selected_rows: list[dict[str, object]] = []
    total = 0
    correct = 0
    oracle_correct = 0
    by_song: dict[str, list[int]] = {}
    oracle_by_song: dict[str, list[int]] = {}
    duration_total = duration_correct = duration_oracle = 0.0
    for _, group in df.groupby("group_key"):
        if group.empty:
            continue
        best = group.sort_values("reranker_score", ascending=False).iloc[0]
        oracle = int(group[target].astype(int).max())
        total += 1
        is_correct = int(best[target])
        correct += is_correct
        oracle_correct += oracle
        duration = max(0.0, float(best.reference_end) - float(best.reference_start))
        duration_total += duration
        duration_correct += duration * is_correct
        duration_oracle += duration * oracle
        by_song.setdefault(str(best.song_id), []).append(is_correct)
        oracle_by_song.setdefault(str(best.song_id), []).append(oracle)
        selected_rows.append(
            {
                "song_id": best.song_id,
                "bar": int(best.bar),
                "reference_start": f"{float(best.reference_start):.6f}",
                "reference_end": f"{float(best.reference_end):.6f}",
                "candidate_source": best.candidate_source,
                "candidate_chord": best.candidate_chord,
                "candidate_roman": best.candidate_roman,
                "reference_chord": best.reference_chord,
                "reference_roman": best.reference_roman,
                "candidate_score": best.get("candidate_score", ""),
                "candidate_confidence": best.get("candidate_confidence", ""),
                "candidate_rank": best.get("candidate_rank", ""),
                "local_key": best.get("local_key", ""),
                "reranker_score": f"{float(best.reranker_score):.6f}",
                "correct": is_correct,
            }
        )
    report = {
        "target": target,
        "evaluation_scope": "reference-aligned candidate selection; not end-to-end recognition or segmentation",
        "grouping": "song_id, reference_start, reference_end; predicted bar number never duplicates a reference segment",
        "groups": total,
        "selector_accuracy": round(correct / total, 4) if total else 0.0,
        "oracle_accuracy": round(oracle_correct / total, 4) if total else 0.0,
        "macro_song_selector_accuracy": round(sum(sum(values) / len(values) for values in by_song.values()) / len(by_song), 4) if by_song else 0.0,
        "reference_group_seconds": duration_total,
        "duration_weighted_selector_accuracy": round(duration_correct / duration_total, 4) if duration_total else 0.0,
        "duration_weighted_oracle_accuracy": round(duration_oracle / duration_total, 4) if duration_total else 0.0,
        "by_song": {
            song: {
                "groups": len(values),
                "selector_accuracy": round(sum(values) / len(values), 4) if values else 0.0,
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
    parser.add_argument("--target", choices=TARGET_CHOICES)
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "candidate_reranker_selector_report.json")
    parser.add_argument("--selected", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "candidate_reranker_selected_rows.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = resolve_path(args.model)
    candidate_csv = resolve_path(args.candidate_csv)
    pipeline, model_target = load_model(model_path)
    target = args.target or model_target
    df = load_rows(candidate_csv)
    df = filter_evaluable_rows(df, target)
    scores = pipeline.predict_proba(df)[:, 1]
    report, selected_rows = evaluate(df, scores, target)
    report["model"] = str(model_path)
    report["candidate_csv"] = str(candidate_csv)
    with model_path.open("rb") as handle:
        metadata = pickle.load(handle)
    training_songs = metadata.get("training_song_ids")
    if training_songs is None:
        report["training_overlap_status"] = "unknown: legacy model has no training-song provenance"
    else:
        overlap = sorted(set(training_songs) & set(df["song_id"].astype(str)))
        report["training_overlap_songs"] = overlap
        report["training_overlap_status"] = "in-sample or mixed; not held-out" if overlap else "song-disjoint"

    args.report = resolve_path(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.selected = resolve_path(args.selected)
    args.selected.parent.mkdir(parents=True, exist_ok=True)
    with args.selected.open("w", newline="", encoding="utf-8-sig") as handle:
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
            "candidate_score",
            "candidate_confidence",
            "candidate_rank",
            "local_key",
            "reranker_score",
            "correct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote: {args.selected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
