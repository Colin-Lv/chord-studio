#!/usr/bin/env python
"""Train/evaluate a segment path acceptor from proposal rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REPO_ROOT = Path(__file__).resolve().parent.parent


CATEGORICAL_COLUMNS = ["path_name"]
NUMERIC_COLUMNS = [
    "path_score",
    "gain_over_current_score",
    "boundary_score",
    "span_count",
    "boundary_count",
    "avg_confidence",
    "avg_margin",
    "root_change_count",
    "distinct_root_count",
    "contains_n",
    "meaningful_root_change",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df.get(column), errors="coerce").fillna(0.0)
    for column in CATEGORICAL_COLUMNS:
        df[column] = df.get(column, "").fillna("").astype(str)
    return df


def make_model(random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_COLUMNS,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])


def predict_scores(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS])[:, 1]
    return model.decision_function(df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS])


def selector_metrics(df: pd.DataFrame, scores: np.ndarray, best_target: str) -> dict[str, Any]:
    work = df.copy()
    work["_score"] = scores
    groups = list(work.groupby(["song_id", "bar"], sort=False))
    selected_correct = 0
    current_correct = 0
    oracle_correct = 0
    for _, group in groups:
        selected = group.sort_values("_score", ascending=False).iloc[0]
        selected_correct += int(selected[best_target])
        current = group[group["path_name"] == "current"]
        if not current.empty:
            current_correct += int(current.iloc[0][best_target])
        oracle_correct += int(group[best_target].astype(int).max())
    total = max(len(groups), 1)
    return {
        "bar_groups": len(groups),
        "selector_accuracy": selected_correct / total,
        "current_accuracy": current_correct / total,
        "oracle_accuracy": oracle_correct / total,
    }


def gated_selector_metrics(df: pd.DataFrame, scores: np.ndarray, best_target: str) -> list[dict[str, Any]]:
    work = df.copy()
    work["_score"] = scores
    groups = list(work.groupby(["song_id", "bar"], sort=False))
    rows: list[dict[str, Any]] = []
    for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
        correct = 0
        changed = 0
        for _, group in groups:
            current = group[group["path_name"] == "current"]
            current_row = current.iloc[0] if not current.empty else group.iloc[0]
            alternatives = group[group["path_name"] != "current"]
            selected = current_row
            if not alternatives.empty:
                candidate = alternatives.sort_values("_score", ascending=False).iloc[0]
                if float(candidate["_score"]) >= threshold:
                    selected = candidate
                    changed += 1
            correct += int(selected[best_target])
        total = max(len(groups), 1)
        rows.append(
            {
                "threshold": threshold,
                "selector_accuracy": correct / total,
                "changed_bars": changed,
                "change_rate": changed / total,
            }
        )
    return rows


def evaluate_cv(df: pd.DataFrame, target: str, best_target: str, random_state: int) -> dict[str, Any]:
    groups = df["song_id"].astype(str).to_numpy()
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        return {"skipped": "need at least two songs for GroupKFold"}
    splits = min(4, len(unique_groups))
    splitter = GroupKFold(n_splits=splits)
    oof = np.zeros(len(df), dtype=np.float32)
    fold_reports = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(df, df[target].astype(int), groups), start=1):
        model = make_model(random_state + fold)
        model.fit(df.iloc[train_idx][NUMERIC_COLUMNS + CATEGORICAL_COLUMNS], df.iloc[train_idx][target].astype(int))
        scores = predict_scores(model, df.iloc[test_idx])
        oof[test_idx] = scores
        fold_reports.append(
            {
                "fold": fold,
                "test_songs": sorted(set(df.iloc[test_idx]["song_id"].astype(str))),
                **selector_metrics(df.iloc[test_idx], scores, best_target),
            }
        )
    y = df[target].astype(int).to_numpy()
    report: dict[str, Any] = {
        "folds": fold_reports,
        "selector": selector_metrics(df, oof, best_target),
        "gated_selector": gated_selector_metrics(df, oof, best_target),
    }
    if len(set(y)) >= 2:
        report["roc_auc"] = float(roc_auc_score(y, oof))
        report["average_precision"] = float(average_precision_score(y, oof))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows_csv", type=Path)
    parser.add_argument("--target", choices=["target_accept_roman_root", "target_accept_exact"], default="target_accept_roman_root")
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows_path = resolve_path(args.rows_csv)
    df = load_rows(rows_path)
    best_target = "target_best_roman_root" if args.target.endswith("roman_root") else "target_best_exact"
    report = {
        "rows": int(len(df)),
        "songs": sorted(set(df["song_id"].astype(str))),
        "target": args.target,
        "positive_rows": int(df[args.target].astype(int).sum()),
        "best_target": best_target,
        "cv": evaluate_cv(df, args.target, best_target, args.random_state),
    }
    model = make_model(args.random_state)
    model.fit(df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS], df[args.target].astype(int))
    model_output = resolve_path(args.model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "numeric_columns": NUMERIC_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "target": args.target,
            "best_target": best_target,
        },
        model_output,
    )
    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote model: {model_output}")
    print(f"Wrote report: {report_path}")
    print(json.dumps(report["cv"].get("selector", {}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
