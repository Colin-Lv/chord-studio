#!/usr/bin/env python
"""Train/evaluate a pairwise groupwise selector over chord candidates."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from train_candidate_reranker import build_pipeline, filter_evaluable_rows, group_columns, load_rows, resolve_path


REPO_ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["target_exact_chord", "target_same_root", "target_same_roman_root", "target_same_chord_family"]


def group_ids(df: pd.DataFrame) -> pd.Series:
    columns = group_columns(df)
    return df[columns].apply(lambda row: "|".join(str(value) for value in row), axis=1)


def bar_groups(df: pd.DataFrame) -> np.ndarray:
    return (df["song_id"].astype(str) + ":" + df["bar"].fillna(-1).astype(int).astype(str)).to_numpy()


def pairwise_training_matrix(x, df: pd.DataFrame, target: str, max_pairs_per_group: int) -> tuple[object, np.ndarray]:
    rows = []
    labels: list[int] = []
    rng = np.random.default_rng(42)
    row_positions = {index: position for position, index in enumerate(df.index)}
    for _, group in df.groupby(group_ids(df), sort=False):
        positives = group[group[target].astype(int) == 1]
        negatives = group[group[target].astype(int) == 0]
        if positives.empty or negatives.empty:
            continue
        pairs: list[tuple[int, int]] = []
        for pos_index in positives.index:
            for neg_index in negatives.index:
                pairs.append((row_positions[pos_index], row_positions[neg_index]))
        if max_pairs_per_group > 0 and len(pairs) > max_pairs_per_group:
            chosen = rng.choice(len(pairs), size=max_pairs_per_group, replace=False)
            pairs = [pairs[int(index)] for index in chosen]
        for pos, neg in pairs:
            rows.append(sparse.csr_matrix(x[pos : pos + 1] - x[neg : neg + 1]))
            labels.append(1)
            rows.append(sparse.csr_matrix(x[neg : neg + 1] - x[pos : pos + 1]))
            labels.append(0)
    if not rows:
        raise SystemExit("No pairwise training examples could be built.")
    return sparse.vstack(rows), np.asarray(labels, dtype=np.int8)


def pairwise_scores(model: LogisticRegression, x_group) -> np.ndarray:
    size = x_group.shape[0]
    if size == 1:
        return np.array([1.0], dtype=np.float32)
    scores = np.zeros(size, dtype=np.float64)
    for i in range(size):
        wins = []
        for j in range(size):
            if i == j:
                continue
            diff = sparse.csr_matrix(x_group[i : i + 1] - x_group[j : j + 1])
            wins.append(float(model.predict_proba(diff)[0, 1]))
        scores[i] = float(np.mean(wins)) if wins else 0.0
    return scores


def evaluate_selector(
    df: pd.DataFrame,
    scores: np.ndarray,
    target: str,
    blend_base_score: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    scored = df.copy()
    scored["groupwise_score"] = scores
    scored["selector_score"] = scored["groupwise_score"]
    if blend_base_score > 0:
        base = pd.to_numeric(scored.get("candidate_score", 0.0), errors="coerce").fillna(0.0)
        scored["_base_score_norm"] = base.groupby(group_ids(scored)).transform(
            lambda values: (values - values.min()) / (values.max() - values.min() + 1e-9)
        )
        scored["selector_score"] = scored["groupwise_score"] * (1.0 - blend_base_score) + scored["_base_score_norm"] * blend_base_score
    total = 0
    correct = 0
    oracle = 0
    selected_rows: list[dict[str, object]] = []
    by_song: dict[str, list[int]] = {}
    oracle_by_song: dict[str, list[int]] = {}
    for _, group in scored.groupby(group_ids(scored), sort=False):
        if group.empty:
            continue
        best = group.sort_values(["selector_score", "candidate_score"], ascending=False).iloc[0]
        is_correct = int(best[target])
        has_oracle = int(group[target].astype(int).max())
        total += 1
        correct += is_correct
        oracle += has_oracle
        song = str(best.song_id)
        by_song.setdefault(song, []).append(is_correct)
        oracle_by_song.setdefault(song, []).append(has_oracle)
        selected_rows.append(
            {
                "song_id": song,
                "bar": int(best.bar),
                "reference_start": f"{float(best.reference_start):.6f}",
                "reference_end": f"{float(best.reference_end):.6f}",
                "candidate_source": best.candidate_source,
                "candidate_chord": best.candidate_chord,
                "candidate_roman": best.candidate_roman,
                "reference_chord": best.reference_chord,
                "reference_roman": best.reference_roman,
                "groupwise_score": f"{float(best.groupwise_score):.6f}",
                "selector_score": f"{float(best.selector_score):.6f}",
                "correct": is_correct,
            }
        )
    return (
        {
            "target": target,
            "groups": total,
            "selector_accuracy": round(correct / total, 4) if total else 0.0,
            "oracle_accuracy": round(oracle / total, 4) if total else 0.0,
            "by_song": {
                song: {
                    "groups": len(values),
                    "selector_accuracy": round(sum(values) / len(values), 4) if values else 0.0,
                    "oracle_accuracy": round(sum(oracle_by_song[song]) / len(oracle_by_song[song]), 4)
                    if oracle_by_song.get(song)
                    else 0.0,
                }
                for song, values in sorted(by_song.items())
            },
        },
        selected_rows,
    )


def cross_validated_scores(
    df: pd.DataFrame,
    target: str,
    model: LogisticRegression,
    max_pairs_per_group: int,
    cv_group: str,
) -> np.ndarray:
    groups = df["song_id"].astype(str).to_numpy() if cv_group == "song" else bar_groups(df)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        return np.full(len(df), 0.0)
    splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
    output = np.zeros(len(df), dtype=np.float64)
    for train_index, test_index in splitter.split(df, df[target].astype(int), groups):
        train_df = df.iloc[train_index].copy()
        test_df = df.iloc[test_index].copy()
        pipeline, _, _ = build_pipeline(train_df, "logistic")
        preprocessor = clone(pipeline.named_steps["features"])
        x_train = preprocessor.fit_transform(train_df)
        x_test = preprocessor.transform(test_df)
        pair_x, pair_y = pairwise_training_matrix(x_train, train_df, target, max_pairs_per_group)
        fold_model = clone(model)
        fold_model.fit(pair_x, pair_y)
        local_scores = np.zeros(len(test_df), dtype=np.float64)
        for _, group in test_df.groupby(group_ids(test_df), sort=False):
            positions = [test_df.index.get_loc(index) for index in group.index]
            group_scores = pairwise_scores(fold_model, x_test[positions])
            for position, score in zip(positions, group_scores):
                local_scores[position] = score
        output[test_index] = local_scores
    return output


def train_final(df: pd.DataFrame, target: str, model: LogisticRegression, max_pairs_per_group: int) -> dict[str, object]:
    pipeline, numeric, categorical = build_pipeline(df, "logistic")
    preprocessor = pipeline.named_steps["features"]
    x = preprocessor.fit_transform(df)
    pair_x, pair_y = pairwise_training_matrix(x, df, target, max_pairs_per_group)
    model.fit(pair_x, pair_y)
    return {
        "preprocessor": preprocessor,
        "model": model,
        "target": target,
        "numeric": numeric,
        "categorical": categorical,
        "kind": "pairwise_groupwise_selector",
    }


def write_selected(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "groupwise_score",
        "selector_score",
        "correct",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--target", choices=TARGETS, default="target_exact_chord")
    parser.add_argument("--max-pairs-per-group", type=int, default=64)
    parser.add_argument("--blend-base-score", type=float, default=0.0)
    parser.add_argument("--cv-group", choices=["bar", "song"], default="bar")
    parser.add_argument("-o", "--output", type=Path, default=REPO_ROOT / "benchmarks" / "models" / "groupwise_selector.pkl")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "groupwise_selector_report.json")
    parser.add_argument("--selected", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "groupwise_selector_selected.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_csv = resolve_path(args.candidate_csv)
    df = load_rows(candidate_csv)
    df = filter_evaluable_rows(df, args.target)
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    scores = cross_validated_scores(df, args.target, model, args.max_pairs_per_group, args.cv_group)
    cv_report, selected_rows = evaluate_selector(df, scores, args.target, args.blend_base_score)

    payload = train_final(df, args.target, model, args.max_pairs_per_group)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(payload, handle)

    report = {
        "candidate_csv": str(candidate_csv),
        "rows": int(len(df)),
        "target": args.target,
        "model": str(output),
        "max_pairs_per_group": args.max_pairs_per_group,
        "blend_base_score": args.blend_base_score,
        "cv_group": args.cv_group,
        "cross_validated_selector": cv_report,
    }
    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_selected(resolve_path(args.selected), selected_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
