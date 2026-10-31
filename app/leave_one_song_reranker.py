#!/usr/bin/env python
"""Leave-one-song-out selector evaluation for candidate rerankers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_candidate_reranker import TARGET_CHOICES, evaluate
from train_candidate_reranker import (build_pipeline, filter_evaluable_rows, load_rows,
                                     positive_class_scores, resolve_path, sample_weights)


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument(
        "--target",
        choices=TARGET_CHOICES,
        default="target_same_roman_root",
    )
    parser.add_argument("--model", choices=["logistic", "random_forest"], default="random_forest")
    parser.add_argument("--beatles-weight", type=float, default=1.0)
    parser.add_argument("--user-weight", type=float, default=1.0)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "reports" / "candidate_reranker_leave_one_song.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_csv = resolve_path(args.candidate_csv)
    df = load_rows(candidate_csv)
    df = filter_evaluable_rows(df, args.target)
    if df["song_id"].nunique() < 2:
        raise ValueError("Leave-one-song-out validation requires at least two songs.")
    by_song: dict[str, object] = {}
    totals = []
    correct = []
    oracle = []
    for song in sorted(str(item) for item in df["song_id"].dropna().unique()):
        train = df[df["song_id"].astype(str) != song].copy()
        test = df[df["song_id"].astype(str) == song].copy()
        pipeline, _, _ = build_pipeline(train, args.model)
        train_targets = train[args.target].astype(int).to_numpy()
        if len(np.unique(train_targets)) == 1:
            scores = np.full(len(test), float(train_targets[0]))
        else:
            pipeline.fit(train, train_targets, model__sample_weight=sample_weights(train, args.beatles_weight, args.user_weight))
            scores = positive_class_scores(pipeline, test)
        report, _ = evaluate(test, scores, args.target)
        by_song[song] = report
        totals.append(int(report["groups"]))
        correct.append(float(report["selector_accuracy"]) * int(report["groups"]))
        oracle.append(float(report["oracle_accuracy"]) * int(report["groups"]))
    total_groups = int(np.sum(totals))
    output = {
        "candidate_csv": str(candidate_csv),
        "target": args.target,
        "model": args.model,
        "evaluation_protocol": "leave-one-song-out; reference-aligned candidate diagnostic, not end-to-end",
        "sample_weights": {"beatles": args.beatles_weight, "user": args.user_weight},
        "groups": total_groups,
        "selector_accuracy": round(float(np.sum(correct) / total_groups), 4) if total_groups else 0.0,
        "oracle_accuracy": round(float(np.sum(oracle) / total_groups), 4) if total_groups else 0.0,
        "by_song": by_song,
    }
    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
