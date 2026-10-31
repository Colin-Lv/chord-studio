#!/usr/bin/env python
"""Evaluate candidate oracle coverage by complex-harmony category."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from build_complex_harmony_challenge import BORROWED_ROOTS, EXTENSION_MARKERS, FUNCTIONAL_DOM7, SECONDARY_DOM7, roman_root
from train_candidate_reranker import resolve_path


REPO_ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["target_exact_chord", "target_same_root", "target_same_roman_root", "target_same_chord_family"]


def group_key(row: pd.Series) -> tuple[str, int, float, float]:
    return (
        str(row.get("song_id", "")),
        int(float(row.get("bar", 0) or 0)),
        round(float(row.get("reference_start", row.get("start", 0.0)) or 0.0), 3),
        round(float(row.get("reference_end", row.get("end", 0.0)) or 0.0), 3),
    )


def tags_for_reference(row: pd.Series) -> set[str]:
    chord = str(row.get("reference_chord") or "N")
    roman = str(row.get("reference_roman") or "N")
    root = roman_root(roman)
    dom7_token = roman if roman in FUNCTIONAL_DOM7 else root + "7"
    tags = {"all"}
    if chord != "N":
        low = chord.lower()
        if "/" in chord:
            tags.add("slash")
        if any(marker in low for marker in EXTENSION_MARKERS):
            tags.add("extension")
    if roman.endswith("7") and dom7_token in FUNCTIONAL_DOM7:
        tags.add("functional_dom7")
    if roman.endswith("7") and dom7_token in SECONDARY_DOM7:
        tags.add("secondary_dominant")
    if "/" in roman and roman.startswith("V"):
        tags.update({"functional_dom7", "secondary_dominant"})
    if root in BORROWED_ROOTS or roman in BORROWED_ROOTS or "IVm" in roman:
        tags.add("borrowed")
    if re.search(r"b(?:II|III|VI|VII)", roman):
        tags.add("borrowed")
    if "beat_start" in row.index and "beat_end" in row.index:
        try:
            beat_start = float(row.get("beat_start") or 1)
            beat_end = float(row.get("beat_end") or 4)
            if beat_start != 1 or beat_end < 4:
                tags.add("within_bar")
        except (TypeError, ValueError):
            pass
    return tags


def evaluate(df: pd.DataFrame, target: str) -> list[dict[str, Any]]:
    for column in TARGETS:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
    df["_group_key"] = [group_key(row) for _, row in df.iterrows()]
    grouped_rows: dict[tuple[str, int, float, float], pd.DataFrame] = {
        key: group for key, group in df.groupby("_group_key", sort=False)
    }
    buckets: dict[str, list[int]] = defaultdict(list)
    sizes: dict[str, list[int]] = defaultdict(list)
    for _, group in grouped_rows.items():
        first = group.iloc[0]
        oracle = int(group[target].max())
        tags = tags_for_reference(first)
        for tag in tags:
            buckets[tag].append(oracle)
            sizes[tag].append(len(group))
    rows = []
    for tag, values in sorted(buckets.items()):
        rows.append(
            {
                "tag": tag,
                "groups": len(values),
                "oracle": round(sum(values) / len(values), 4) if values else 0.0,
                "avg_candidates": round(sum(sizes[tag]) / len(sizes[tag]), 2) if sizes[tag] else 0.0,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--target", choices=TARGETS, default="target_exact_chord")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_csv = resolve_path(args.candidate_csv)
    df = pd.read_csv(candidate_csv, encoding="utf-8-sig", low_memory=False)
    rows = evaluate(df, args.target)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "candidate_csv": str(candidate_csv),
        "target": args.target,
        "rows": rows,
    }
    if args.report:
        report_path = resolve_path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
