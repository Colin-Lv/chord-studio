#!/usr/bin/env python
"""Grid-search sequence decoder weights without repeatedly loading the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from decode_candidate_sequence import decode_song, load_prior, phrase_score_maps
from evaluate_candidate_reranker import group_key, load_model
from train_candidate_reranker import filter_evaluable_rows, load_rows, resolve_path


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def prepared_groups(model_path: Path, candidate_csv: Path, target: str | None) -> Any:
    pipeline, model_target = load_model(resolve_path(model_path))
    target = target or model_target
    df = load_rows(resolve_path(candidate_csv))
    df = filter_evaluable_rows(df, target)
    df["reranker_score"] = pipeline.predict_proba(df)[:, 1]
    df["group_key"] = [group_key(row) for _, row in df.iterrows()]
    songs = {}
    for song_id, song_df in df.sort_values(["song_id", "start", "end"]).groupby("song_id", sort=True):
        songs[str(song_id)] = [
            group.sort_values("reranker_score", ascending=False)
            for _, group in song_df.groupby("group_key", sort=False)
        ]
    return target, songs


def run_once(
    songs: dict[str, list[Any]],
    target: str,
    pair_scores: dict[str, float],
    ngram_scores: dict[tuple[str, ...], float],
    base_weight: float,
    pair_weight: float,
    ngram_weight: float,
    max_order: int,
) -> dict[str, Any]:
    total = correct = oracle = 0
    by_song = {}
    for song_id, groups in songs.items():
        song_total, song_correct, song_oracle, _ = decode_song(
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
        by_song[song_id] = {
            "groups": song_total,
            "sequence_accuracy": round(song_correct / song_total, 4) if song_total else 0.0,
            "oracle_accuracy": round(song_oracle / song_total, 4) if song_total else 0.0,
        }
    return {
        "target": target,
        "groups": total,
        "sequence_accuracy": round(correct / total, 4) if total else 0.0,
        "oracle_accuracy": round(oracle / total, 4) if total else 0.0,
        "base_weight": base_weight,
        "pair_weight": pair_weight,
        "ngram_weight": ngram_weight,
        "max_order": max_order,
        "by_song": dict(sorted(by_song.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--target")
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--pair-weights", default="0,0.005,0.01,0.02,0.04,0.06,0.08,0.10")
    parser.add_argument("--ngram-weights", default="0,0.0025,0.005,0.01,0.02,0.03,0.05,0.08")
    parser.add_argument("--max-order", type=int, default=8)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "reports" / "candidate_sequence_decoder_grid_report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target, songs = prepared_groups(args.model, args.candidate_csv, args.target)
    pair_scores, ngram_scores = phrase_score_maps(load_prior(args.prior))
    results = []
    for pair_weight in parse_float_list(args.pair_weights):
        for ngram_weight in parse_float_list(args.ngram_weights):
            results.append(
                run_once(
                    songs,
                    target,
                    pair_scores,
                    ngram_scores,
                    args.base_weight,
                    pair_weight,
                    ngram_weight,
                    args.max_order,
                )
            )
    results.sort(
        key=lambda row: (
            row["sequence_accuracy"],
            np.mean([item["sequence_accuracy"] for item in row["by_song"].values()]),
        ),
        reverse=True,
    )
    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in results[:10]:
        print(
            f"acc={row['sequence_accuracy']:.4f} pair={row['pair_weight']} "
            f"ngram={row['ngram_weight']} oracle={row['oracle_accuracy']:.4f}"
        )
    print(f"Wrote: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
