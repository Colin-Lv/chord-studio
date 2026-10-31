#!/usr/bin/env python
"""Evaluate threshold gates for long-tail specialist candidate subsets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from train_candidate_reranker import resolve_path


DEFAULT_FEATURES = [
    "candidate_score",
    "candidate_confidence",
    "audio_root_energy",
    "audio_bass_root_energy",
    "audio_flat7_energy",
    "audio_9_energy",
    "audio_4_energy",
    "audio_flat7_over_triad",
    "audio_9_over_triad",
    "audio_dom7_vs_maj7",
    "harm_next_acoustic_matches_resolution",
    "harm_sus9_next_acoustic_matches_resolution",
    "harm_sus9_is_plain_v_to_i",
]


def to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column not in df:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    if "target_exact_chord" not in df:
        df["target_exact_chord"] = 0
    df["target_exact_chord"] = pd.to_numeric(df["target_exact_chord"], errors="coerce").fillna(0).astype(int)
    return df


def metrics(mask: pd.Series, y: pd.Series) -> dict[str, float | int]:
    selected = int(mask.sum())
    positives = int(y.sum())
    true_positive = int((mask & y.eq(1)).sum())
    false_positive = selected - true_positive
    false_negative = positives - true_positive
    precision = true_positive / selected if selected else 0.0
    recall = true_positive / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "selected": selected,
        "positives": positives,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def parse_condition(value: str) -> tuple[str, str, float]:
    for operator in (">=", "<=", ">", "<", "=="):
        if operator in value:
            name, raw = value.split(operator, 1)
            return name.strip(), operator, float(raw.strip())
    raise ValueError(f"Condition must look like feature>=value: {value}")


def condition_mask(df: pd.DataFrame, condition: tuple[str, str, float]) -> pd.Series:
    column, operator, value = condition
    series = pd.to_numeric(df.get(column, 0.0), errors="coerce").fillna(0.0)
    if operator == ">=":
        return series >= value
    if operator == ">":
        return series > value
    if operator == "<=":
        return series <= value
    if operator == "<":
        return series < value
    if operator == "==":
        return series == value
    raise ValueError(operator)


def threshold_rows(df: pd.DataFrame, feature: str, thresholds: list[float], base_mask: pd.Series) -> list[dict[str, object]]:
    rows = []
    y = df["target_exact_chord"]
    values = pd.to_numeric(df.get(feature, 0.0), errors="coerce").fillna(0.0)
    for threshold in thresholds:
        row = {
            "feature": feature,
            "operator": ">=",
            "threshold": round(float(threshold), 6),
            **metrics(base_mask & values.ge(threshold), y),
        }
        rows.append(row)
    return rows


def combo_rows(
    df: pd.DataFrame,
    features: list[str],
    thresholds_by_feature: dict[str, list[float]],
    base_mask: pd.Series,
    max_combos: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    y = df["target_exact_chord"]
    numeric = {feature: pd.to_numeric(df.get(feature, 0.0), errors="coerce").fillna(0.0) for feature in features}
    combos: list[tuple[str, float]] = []
    for feature in features:
        for threshold in thresholds_by_feature.get(feature, []):
            combos.append((feature, threshold))
    seen: set[tuple[tuple[str, float], ...]] = set()
    for index, first in enumerate(combos):
        for second in combos[index + 1 :]:
            if first[0] == second[0]:
                continue
            key = tuple(sorted((first, second)))
            if key in seen:
                continue
            seen.add(key)
            mask = base_mask & numeric[first[0]].ge(first[1]) & numeric[second[0]].ge(second[1])
            item = {
                "feature": f"{first[0]} & {second[0]}",
                "operator": ">=",
                "threshold": f"{first[1]:.6g} & {second[1]:.6g}",
                **metrics(mask, y),
            }
            rows.append(item)
            if len(rows) >= max_combos:
                break
        if len(rows) >= max_combos:
            break
    return rows


def parse_thresholds(value: str | None, df: pd.DataFrame, feature: str) -> list[float]:
    if value:
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    values = pd.to_numeric(df.get(feature, 0.0), errors="coerce").dropna()
    if values.empty:
        return [0.0]
    quantiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    return sorted({round(float(values.quantile(q)), 6) for q in quantiles})


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--thresholds")
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--include-combos", action="store_true")
    parser.add_argument("--max-combos", type=int, default=10000)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features = args.feature or DEFAULT_FEATURES
    conditions = [parse_condition(item) for item in args.condition]
    condition_columns = [item[0] for item in conditions]
    df = pd.read_csv(resolve_path(args.candidate_csv), encoding="utf-8-sig", low_memory=False)
    df = to_numeric(df, sorted(set(features + condition_columns)))
    base_mask = pd.Series(True, index=df.index)
    for condition in conditions:
        base_mask &= condition_mask(df, condition)

    rows: list[dict[str, object]] = []
    thresholds_by_feature: dict[str, list[float]] = {}
    for feature in features:
        thresholds = parse_thresholds(args.thresholds, df[base_mask], feature)
        thresholds_by_feature[feature] = thresholds
        rows.extend(threshold_rows(df, feature, thresholds, base_mask))
    if args.include_combos:
        rows.extend(combo_rows(df, features, thresholds_by_feature, base_mask, args.max_combos))
    output = resolve_path(args.output)
    write_csv(output, rows)
    best_precision = sorted(rows, key=lambda row: (row["precision"], row["true_positive"], -row["false_positive"]), reverse=True)[:10]
    best_f1 = sorted(rows, key=lambda row: (row["f1"], row["precision"], row["true_positive"]), reverse=True)[:10]
    report = {
        "candidate_csv": str(resolve_path(args.candidate_csv)),
        "rows": int(len(df)),
        "base_selected": int(base_mask.sum()),
        "positives": int(df["target_exact_chord"].sum()),
        "conditions": args.condition,
        "output": str(output),
        "best_precision": best_precision,
        "best_f1": best_f1,
    }
    if args.report:
        report_path = resolve_path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
