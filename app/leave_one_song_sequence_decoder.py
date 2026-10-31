#!/usr/bin/env python
"""Leave-one-song-out evaluation for the sequence phrase decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from decode_candidate_sequence import decode_song, load_prior, phrase_score_maps
from evaluate_candidate_reranker import TARGET_CHOICES, group_key
from train_candidate_reranker import build_pipeline, filter_evaluable_rows, load_rows, resolve_path


REPO_ROOT = Path(__file__).resolve().parent.parent


def song_groups(df: Any) -> list[Any]:
    df = df.copy()
    df["group_key"] = [group_key(row) for _, row in df.iterrows()]
    return [
        group.sort_values("reranker_score", ascending=False)
        for _, group in df.sort_values(["song_id", "start", "end"]).groupby("group_key", sort=False)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--target", choices=TARGET_CHOICES, default="target_exact_chord")
    parser.add_argument("--model", choices=["logistic", "random_forest"], default="random_forest")
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--pair-weight", type=float, default=0.08)
    parser.add_argument("--ngram-weight", type=float, default=0.05)
    parser.add_argument("--max-order", type=int, default=8)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "reports" / "candidate_sequence_decoder_loso_report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = load_rows(resolve_path(args.candidate_csv))
    df = filter_evaluable_rows(df, args.target)
    pair_scores, ngram_scores = phrase_score_maps(load_prior(args.prior))
    by_song: dict[str, Any] = {}
    totals = []
    correct = []
    oracle = []
    for song in sorted(str(item) for item in df["song_id"].dropna().unique()):
        train = df[df["song_id"].astype(str) != song].copy()
        test = df[df["song_id"].astype(str) == song].copy()
        pipeline, _, _ = build_pipeline(train, args.model)
        pipeline.fit(train, train[args.target].astype(int).to_numpy())
        test["reranker_score"] = pipeline.predict_proba(test)[:, 1]
        song_total, song_correct, song_oracle, _ = decode_song(
            song_groups(test),
            args.target,
            pair_scores,
            ngram_scores,
            args.base_weight,
            args.pair_weight,
            args.ngram_weight,
            args.max_order,
        )
        totals.append(song_total)
        correct.append(song_correct)
        oracle.append(song_oracle)
        by_song[song] = {
            "groups": song_total,
            "sequence_accuracy": round(song_correct / song_total, 4) if song_total else 0.0,
            "oracle_accuracy": round(song_oracle / song_total, 4) if song_total else 0.0,
        }
    total_groups = int(np.sum(totals))
    output = {
        "candidate_csv": str(resolve_path(args.candidate_csv)),
        "prior": str(resolve_path(args.prior)),
        "target": args.target,
        "model": args.model,
        "groups": total_groups,
        "sequence_accuracy": round(float(np.sum(correct) / total_groups), 4) if total_groups else 0.0,
        "oracle_accuracy": round(float(np.sum(oracle) / total_groups), 4) if total_groups else 0.0,
        "base_weight": args.base_weight,
        "pair_weight": args.pair_weight,
        "ngram_weight": args.ngram_weight,
        "max_order": args.max_order,
        "by_song": by_song,
    }
    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
