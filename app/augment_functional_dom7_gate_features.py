#!/usr/bin/env python
"""Add gate-oriented context features for functional dominant seventh candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from augment_harmonic_context_features import SECONDARY_DOMINANT_TARGET, roman_root
from train_candidate_reranker import resolve_path


def group_id(df: pd.DataFrame) -> pd.Series:
    parts = []
    for column in ("song_id", "bar", "reference_start", "reference_end"):
        selected = df[column] if column in df else pd.Series("", index=df.index)
        if isinstance(selected, pd.DataFrame):
            selected = selected.iloc[:, 0]
        parts.append(selected.fillna("").astype(str))
    keyed = pd.DataFrame({str(i): s.to_numpy() for i, s in enumerate(parts)}, index=df.index)
    return pd.Series(["|".join(row) for row in keyed.to_numpy(dtype=str)], index=df.index)


def top_acoustic_root(group: pd.DataFrame) -> str:
    topn = group[group["candidate_source"].fillna("").astype(str).str.contains("topn_rank1", regex=False)]
    row = topn.iloc[0] if not topn.empty else group.iloc[0]
    return roman_root(row.get("candidate_roman") or row.get("candidate_roman_root") or "")


def build_contexts(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    groups = []
    ids = group_id(df)
    for key, group in df.groupby(ids, sort=False):
        first = group.iloc[0]
        groups.append(
            {
                "key": key,
                "song_id": str(first.get("song_id", "")),
                "start": float(first.get("reference_start", first.get("start", 0.0)) or 0.0),
                "bar": int(float(first.get("bar", 0) or 0)),
                "reference_root": roman_root(first.get("reference_roman") or ""),
                "acoustic_root": top_acoustic_root(group),
            }
        )
    groups = sorted(groups, key=lambda item: (item["song_id"], item["start"]))
    contexts: dict[str, dict[str, str]] = {}
    for index, item in enumerate(groups):
        prev_item = groups[index - 1] if index > 0 and groups[index - 1]["song_id"] == item["song_id"] else None
        next_item = groups[index + 1] if index + 1 < len(groups) and groups[index + 1]["song_id"] == item["song_id"] else None
        contexts[item["key"]] = {
            "prev_reference_root": "" if prev_item is None else prev_item["reference_root"],
            "next_reference_root": "" if next_item is None else next_item["reference_root"],
            "prev_acoustic_root": "" if prev_item is None else prev_item["acoustic_root"],
            "next_acoustic_root": "" if next_item is None else next_item["acoustic_root"],
        }
    return contexts


def feature_names() -> list[str]:
    return [
        "gate_dom7_expected_resolution",
        "gate_dom7_next_reference_matches",
        "gate_dom7_next_acoustic_matches",
        "gate_dom7_prev_reference_is_tonic_or_predominant",
        "gate_dom7_next_reference_is_predominant",
        "gate_dom7_candidate_is_vi7",
    ]


def add_features(row: dict[str, str], context: dict[str, str]) -> None:
    roman = str(row.get("candidate_roman") or row.get("candidate_roman_root") or "")
    root = roman_root(roman)
    expected = SECONDARY_DOMINANT_TARGET.get(roman) or SECONDARY_DOMINANT_TARGET.get(root + "7") or ""
    next_ref = context.get("next_reference_root", "")
    next_acoustic = context.get("next_acoustic_root", "")
    prev_ref = context.get("prev_reference_root", "")
    row["gate_dom7_expected_resolution"] = expected
    row["gate_dom7_next_reference_matches"] = str(int(bool(expected) and next_ref == expected))
    row["gate_dom7_next_acoustic_matches"] = str(int(bool(expected) and next_acoustic == expected))
    row["gate_dom7_prev_reference_is_tonic_or_predominant"] = str(int(prev_ref in {"I", "IV", "VIm", "IIm"}))
    row["gate_dom7_next_reference_is_predominant"] = str(int(next_ref in {"IV", "IIm"}))
    row["gate_dom7_candidate_is_vi7"] = str(int(roman == "VI7" or root + "7" == "VI7"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = resolve_path(args.candidate_csv)
    df = pd.read_csv(input_csv, encoding="utf-8-sig", low_memory=False)
    contexts = build_contexts(df)
    ids = group_id(df)
    rows = df.fillna("").astype(str).to_dict("records")
    fieldnames = list(df.columns)
    for name in feature_names():
        if name not in fieldnames:
            fieldnames.append(name)
    for row, key in zip(rows, ids):
        add_features(row, contexts.get(key, {}))
        for field in fieldnames:
            row.setdefault(field, "")
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} row(s): {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
