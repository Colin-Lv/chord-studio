#!/usr/bin/env python
"""Diagnose candidate proposal layers by recall gain and noise.

This script answers a more actionable question than a single oracle score:
which proposal source/layer adds correct candidates, and how much noise does it
introduce?  It reports per-layer positive yield, standalone oracle, and oracle
gain over a configurable baseline layer such as acoustic_topn.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from train_candidate_reranker import resolve_path


TARGETS = ["target_exact_chord", "target_same_root", "target_same_roman_root", "target_same_chord_family"]


def proposal_type(source: object) -> str:
    text = str(source or "").lower()
    if "half_dim" in text or "m7b5" in text:
        return "half_dim"
    if "functional" in text:
        return "functional_dom7"
    if "sus9" in text or "9sus" in text:
        return "sus9"
    if "dom7" in text:
        return "dominant7"
    if "scale_degree" in text:
        return "scale_degree"
    if "borrowed" in text:
        return "borrowed"
    if "slash" in text or "bass" in text:
        return "slash_bass"
    if "boundary" in text:
        return "boundary"
    if "topn" in text:
        return "acoustic_topn"
    if "chordino" in text or "madmom" in text or "external" in text:
        return "external_acr"
    if "key_timeline" in text or "key_v" in text:
        return "key_timeline"
    if "base" in text or "regularized" in text or "fixedgrid" in text or "refined" in text or "current" in text:
        return "base_analysis"
    return "other"


def group_key(df: pd.DataFrame) -> pd.Series:
    columns = ["song_id", "bar", "reference_start", "reference_end"]
    parts = []
    for column in columns:
        if column not in df:
            parts.append(pd.Series("", index=df.index))
            continue
        selected = df[column]
        if isinstance(selected, pd.DataFrame):
            selected = selected.iloc[:, 0]
        parts.append(selected.fillna("").astype(str))
    keyed = pd.DataFrame({str(index): series.to_numpy() for index, series in enumerate(parts)}, index=df.index)
    return pd.Series(["|".join(row) for row in keyed.to_numpy(dtype=str)], index=df.index)


def normalize_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for target in TARGETS:
        if target in df:
            df[target] = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)
    df["_proposal_type"] = df["candidate_source"].map(proposal_type)
    df["_group_key"] = group_key(df)
    return df


def oracle_for(df: pd.DataFrame, target: str) -> tuple[int, int, float]:
    if df.empty:
        return 0, 0, 0.0
    oracle = df.groupby("_group_key", sort=False)[target].max().astype(int)
    correct = int(oracle.sum())
    total = int(len(oracle))
    return total, correct, round(correct / total, 4) if total else 0.0


def oracle_hits_over_groups(df: pd.DataFrame, target: str, all_group_keys: pd.Index) -> tuple[int, int, float]:
    if len(all_group_keys) == 0:
        return 0, 0, 0.0
    if df.empty:
        return int(len(all_group_keys)), 0, 0.0
    oracle = df.groupby("_group_key", sort=False)[target].max().astype(int)
    aligned = oracle.reindex(all_group_keys, fill_value=0).astype(int)
    correct = int(aligned.sum())
    total = int(len(all_group_keys))
    return total, correct, round(correct / total, 4) if total else 0.0


def source_metrics(df: pd.DataFrame, target: str, group_cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for values, group in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        total_groups, oracle_hits, oracle = oracle_for(group, target)
        positive_rows = int(group[target].sum())
        row_count = int(len(group))
        touched_groups = int(group["_group_key"].nunique())
        rows.append(
            {
                **{column: value for column, value in zip(group_cols, values)},
                "rows": row_count,
                "groups_touched": touched_groups,
                "positive_rows": positive_rows,
                "positive_yield": round(positive_rows / row_count, 4) if row_count else 0.0,
                "standalone_oracle_hits": oracle_hits,
                "standalone_oracle": oracle,
                "avg_candidates_per_group": round(row_count / touched_groups, 2) if touched_groups else 0.0,
                "noise_rows": row_count - positive_rows,
            }
        )
    return rows


def layer_gain_rows(df: pd.DataFrame, target: str, baseline: set[str]) -> list[dict[str, Any]]:
    all_group_keys = pd.Index(df["_group_key"].drop_duplicates())
    all_groups = int(len(all_group_keys))
    baseline_df = df[df["_proposal_type"].isin(baseline)].copy()
    _, baseline_hits, baseline_oracle = oracle_hits_over_groups(baseline_df, target, all_group_keys)
    rows = []
    for layer, _ in sorted(df.groupby("_proposal_type")):
        layer_df = df[df["_proposal_type"].eq(layer)].copy()
        combined = pd.concat([baseline_df, layer_df], ignore_index=True)
        _, layer_hits, layer_oracle = oracle_hits_over_groups(layer_df, target, all_group_keys)
        _, combined_hits, combined_oracle = oracle_hits_over_groups(combined, target, all_group_keys)
        added = combined_hits - baseline_hits
        layer_positive_rows = int(layer_df[target].sum())
        rows.append(
            {
                "proposal_type": layer,
                "baseline_layers": ",".join(sorted(baseline)),
                "total_groups": all_groups,
                "baseline_oracle_hits": baseline_hits,
                "baseline_oracle": baseline_oracle,
                "standalone_oracle_hits": layer_hits,
                "standalone_oracle": layer_oracle,
                "combined_oracle_hits": combined_hits,
                "combined_oracle": combined_oracle,
                "oracle_gain_hits": added,
                "oracle_gain": round(combined_oracle - baseline_oracle, 4),
                "rows": int(len(layer_df)),
                "positive_rows": layer_positive_rows,
                "positive_yield": round(layer_positive_rows / len(layer_df), 4) if len(layer_df) else 0.0,
                "gain_per_100_rows": round(100 * added / len(layer_df), 4) if len(layer_df) else 0.0,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--target", choices=TARGETS, default="target_exact_chord")
    parser.add_argument("--baseline-layer", action="append", default=["acoustic_topn"])
    parser.add_argument("-o", "--output-prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_csv = resolve_path(args.candidate_csv)
    df = pd.read_csv(candidate_csv, encoding="utf-8-sig", low_memory=False)
    df = normalize_targets(df)
    target = args.target
    output_prefix = resolve_path(args.output_prefix)

    by_layer = source_metrics(df, target, ["_proposal_type"])
    by_source = source_metrics(df, target, ["_proposal_type", "candidate_source"])
    gains = layer_gain_rows(df, target, set(args.baseline_layer))

    layer_path = output_prefix.with_name(output_prefix.name + "_by_layer.csv")
    source_path = output_prefix.with_name(output_prefix.name + "_by_source.csv")
    gain_path = output_prefix.with_name(output_prefix.name + "_oracle_gain.csv")
    write_csv(layer_path, by_layer)
    write_csv(source_path, by_source)
    write_csv(gain_path, gains)

    _, total_hits, total_oracle = oracle_for(df, target)
    report = {
        "candidate_csv": str(candidate_csv),
        "target": target,
        "rows": int(len(df)),
        "groups": int(df["_group_key"].nunique()),
        "overall_oracle_hits": total_hits,
        "overall_oracle": total_oracle,
        "by_layer": str(layer_path),
        "by_source": str(source_path),
        "oracle_gain": str(gain_path),
    }
    if args.report:
        report_path = resolve_path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
