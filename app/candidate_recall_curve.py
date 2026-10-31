#!/usr/bin/env python
"""Measure candidate-pool oracle recall by source layers and top-k size."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from train_candidate_reranker import resolve_path


REPO_ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["target_exact_chord", "target_same_root", "target_same_roman_root", "target_same_chord_family"]


def group_key(df: pd.DataFrame) -> pd.Series:
    columns = ["song_id", "bar", "reference_start", "reference_end"]
    values = []
    for column in columns:
        selected = df[column]
        if isinstance(selected, pd.DataFrame):
            selected = selected.iloc[:, 0]
        values.append(selected.fillna("").astype(str))
    keyed = pd.DataFrame({str(index): series.to_numpy() for index, series in enumerate(values)}, index=df.index)
    return pd.Series(["|".join(row) for row in keyed.to_numpy(dtype=str)], index=df.index)


def source_layer(source: object) -> str:
    text = str(source or "").lower()
    if "half_dim" in text or "m7b5" in text:
        return "half_dim"
    if text.startswith("functional"):
        return "functional"
    if "topn" in text:
        return "acoustic_topn"
    if "dom7" in text:
        return "dominant7"
    if "chordino" in text or "madmom" in text or "external" in text:
        return "external_acr"
    return "other"


def candidate_rank(row: pd.Series, layer_order: dict[str, int]) -> tuple[int, float, int]:
    layer = source_layer(row.get("candidate_source"))
    layer_rank = layer_order.get(layer, 999)
    raw_rank = pd.to_numeric(row.get("candidate_rank"), errors="coerce")
    if pd.isna(raw_rank):
        raw_rank = 999
    raw_score = pd.to_numeric(row.get("candidate_score"), errors="coerce")
    score = 0.0 if pd.isna(raw_score) else float(raw_score)
    return (layer_rank, -score, int(raw_rank))


def limited_candidates(df: pd.DataFrame, top_k: int, layers: set[str] | None, layer_order: dict[str, int]) -> pd.DataFrame:
    work = df.copy()
    work["_layer"] = work["candidate_source"].map(source_layer)
    if layers is not None:
        work = work[work["_layer"].isin(layers)].copy()
    if top_k <= 0:
        return work
    selected = []
    work["_group_key"] = group_key(work)
    for _, group in work.groupby("_group_key", sort=False):
        ordered = group.copy()
        ordered["_sort_tuple"] = [candidate_rank(row, layer_order) for _, row in ordered.iterrows()]
        ordered = ordered.sort_values("_sort_tuple").head(top_k)
        selected.append(ordered)
    if not selected:
        return work.iloc[0:0]
    return pd.concat(selected, ignore_index=True)


def subset_mask(df: pd.DataFrame, subset: str) -> pd.Series:
    roman = df.get("reference_roman", pd.Series("", index=df.index)).fillna("").astype(str)
    chord = df.get("reference_chord", pd.Series("", index=df.index)).fillna("").astype(str)
    if subset == "all":
        return pd.Series(True, index=df.index)
    if subset == "seventh":
        return chord.str.contains("7", regex=False)
    if subset == "III7":
        return roman.eq("III7")
    if subset == "functional_dom7":
        return roman.isin(["III7", "VI7", "II7", "V7", "I7", "IV7", "bVII7"])
    if subset == "within_bar":
        starts_raw = df["beat_start"] if "beat_start" in df else pd.Series(1, index=df.index)
        ends_raw = df["beat_end"] if "beat_end" in df else pd.Series(1, index=df.index)
        starts = pd.to_numeric(starts_raw, errors="coerce").fillna(1)
        ends = pd.to_numeric(ends_raw, errors="coerce").fillna(1)
        return starts.ne(1) | ends.lt(4)
    raise ValueError(f"Unknown subset: {subset}")


def oracle_for(df: pd.DataFrame, target: str, subset: str) -> tuple[int, float, float]:
    if df.empty:
        return 0, 0.0, 0.0
    mask = subset_mask(df, subset)
    scoped = df[mask].copy()
    if scoped.empty:
        return 0, 0.0, 0.0
    scoped["_group_key"] = group_key(scoped)
    groups = scoped.groupby("_group_key", sort=False)
    oracle = groups[target].max().astype(int)
    sizes = groups.size()
    return int(len(oracle)), round(float(oracle.mean()), 4), round(float(sizes.mean()), 2)


def run(df: pd.DataFrame, target: str, layer_specs: list[tuple[str, set[str] | None]], top_ks: list[int]) -> list[dict[str, object]]:
    layer_order = {
        "acoustic_topn": 0,
        "functional": 1,
        "dominant7": 2,
        "half_dim": 3,
        "external_acr": 3,
        "other": 9,
    }
    rows: list[dict[str, object]] = []
    subsets = ["all", "seventh", "III7", "functional_dom7", "within_bar"]
    for layer_name, layers in layer_specs:
        for top_k in top_ks:
            limited = limited_candidates(df, top_k, layers, layer_order)
            for subset in subsets:
                groups, oracle, avg_candidates = oracle_for(limited, target, subset)
                rows.append(
                    {
                        "layer_set": layer_name,
                        "top_k": top_k if top_k > 0 else "all",
                        "subset": subset,
                        "groups": groups,
                        "oracle": oracle,
                        "avg_candidates_per_group": avg_candidates,
                    }
                )
    return rows


def parse_layer_spec(value: str) -> tuple[str, set[str] | None]:
    name, _, raw_layers = value.partition(":")
    name = name.strip()
    if not raw_layers.strip() or raw_layers.strip().lower() == "all":
        return name, None
    return name, {item.strip() for item in raw_layers.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--target", choices=TARGETS, default="target_exact_chord")
    parser.add_argument("--top-k", default="1,2,4,8,12,16,0", help="Comma-separated; 0 means all candidates.")
    parser.add_argument(
        "--layer-set",
        action="append",
        default=None,
        help="name:layer,layer. Known layers include acoustic_topn,functional,dominant7,external_acr,other.",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(resolve_path(args.candidate_csv), encoding="utf-8-sig", low_memory=False)
    for column in TARGETS:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
    top_ks = [int(item.strip()) for item in args.top_k.split(",") if item.strip()]
    layer_set_values = args.layer_set or [
        "acoustic:acoustic_topn",
        "functional:functional",
        "acoustic+functional:acoustic_topn,functional",
        "all:all",
    ]
    layer_specs = [parse_layer_spec(item) for item in layer_set_values]
    rows = run(df, args.target, layer_specs, top_ks)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "candidate_csv": str(resolve_path(args.candidate_csv)),
        "target": args.target,
        "rows": len(df),
        "output": str(output),
    }
    if args.summary:
        resolve_path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
