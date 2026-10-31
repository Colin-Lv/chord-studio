#!/usr/bin/env python
"""Add functional-resolution and local context features to candidate rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from chord_studio import note_name_for_pc, parse_chord_name


REPO_ROOT = Path(__file__).resolve().parent.parent
MAJOR_ROMAN_TO_PC = {
    "I": 0,
    "bII": 1,
    "II": 2,
    "bIII": 3,
    "III": 4,
    "IV": 5,
    "#IV": 6,
    "V": 7,
    "bVI": 8,
    "VI": 9,
    "bVII": 10,
    "VII": 11,
}
SECONDARY_DOMINANT_TARGET = {
    "II7": "V",
    "III7": "VI",
    "VI7": "II",
    "VII7": "III",
    "V7": "I",
    "I7": "IV",
    "IV7": "bVII",
    "bVII7": "bIII",
    "bVI7": "VI",
}
SUS9_TARGET = {
    "I9sus4": "IV",
    "II9sus4": "V",
    "III9sus4": "VI",
    "IV9sus4": "bVII",
    "V9sus4": "I",
    "VI9sus4": "II",
    "VII9sus4": "III",
    "bVII9sus4": "bIII",
    "bVI9sus4": "bII",
}
HALF_DIM_TARGET = {
    "IIm7b5": "V",
    "#IVm7b5": "V",
    "bVIm7b5": "VI",
    "VIIm7b5": "I",
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_knowledge_base(path: Path | None) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    if path is None:
        return {}, {}
    payload = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    counts = {
        str(item.get("item", "")): float(item.get("prob", 0.0) or 0.0)
        for item in payload.get("functional_dom7_counts", []) or []
    }
    resolutions: dict[str, dict[str, float]] = {}
    for roman, rows in (payload.get("functional_dom7_resolution", {}) or {}).items():
        resolutions[str(roman)] = {
            str(item.get("item", "")): float(item.get("prob", 0.0) or 0.0) for item in rows
        }
    return counts, resolutions


def group_columns(df: pd.DataFrame) -> list[str]:
    return ["song_id", "bar", "reference_start", "reference_end"]


def group_id(df: pd.DataFrame) -> pd.Series:
    values = []
    for column in group_columns(df):
        selected = df[column]
        if isinstance(selected, pd.DataFrame):
            selected = selected.iloc[:, 0]
        values.append(selected.fillna("").astype(str))
    keyed = pd.DataFrame({str(index): series.to_numpy() for index, series in enumerate(values)}, index=df.index)
    return pd.Series(["|".join(row) for row in keyed.to_numpy(dtype=str)], index=df.index)


def roman_root(value: object) -> str:
    text = str(value or "")
    for suffix in ("9sus4", "7sus4", "maj7", "m7b5", "mMaj7", "m7", "dim7", "add9", "9", "7", "m", "dim", "aug", "sus4", "sus2", "sus"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def chord_root(value: object) -> str:
    root, _ = parse_chord_name(str(value or "N"))
    return "" if root is None else str(root)


def next_group_lookup(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    groups = []
    for key, group in df.groupby(group_id(df), sort=False):
        first = group.iloc[0]
        acoustic = group[group["candidate_source"].fillna("").astype(str).str.contains("topn_rank1", regex=False)]
        acoustic_top1 = acoustic.iloc[0] if not acoustic.empty else group.iloc[0]
        groups.append(
            {
                "key": key,
                "song_id": str(first.song_id),
                "start": float(first.reference_start),
                "bar": int(first.bar),
                "reference_roman_root": roman_root(first.reference_roman),
                "reference_chord_root": chord_root(first.reference_chord),
                "acoustic_roman_root": roman_root(acoustic_top1.candidate_roman),
                "acoustic_chord_root": chord_root(acoustic_top1.candidate_chord),
            }
        )
    groups = sorted(groups, key=lambda item: (item["song_id"], item["start"]))
    lookup: dict[str, dict[str, str]] = {}
    for index, item in enumerate(groups):
        next_item = groups[index + 1] if index + 1 < len(groups) and groups[index + 1]["song_id"] == item["song_id"] else None
        prev_item = groups[index - 1] if index > 0 and groups[index - 1]["song_id"] == item["song_id"] else None
        lookup[item["key"]] = {
            "next_reference_roman_root": "" if next_item is None else next_item["reference_roman_root"],
            "next_reference_chord_root": "" if next_item is None else next_item["reference_chord_root"],
            "next_acoustic_roman_root": "" if next_item is None else next_item["acoustic_roman_root"],
            "next_acoustic_chord_root": "" if next_item is None else next_item["acoustic_chord_root"],
            "prev_reference_roman_root": "" if prev_item is None else prev_item["reference_roman_root"],
            "prev_acoustic_roman_root": "" if prev_item is None else prev_item["acoustic_roman_root"],
        }
    return lookup


def feature_names() -> list[str]:
    return [
        "harm_candidate_is_functional_dom7",
        "harm_candidate_expected_resolution",
        "harm_next_acoustic_matches_resolution",
        "harm_next_reference_matches_resolution",
        "harm_prev_acoustic_same_root",
        "harm_next_acoustic_same_root",
        "harm_candidate_is_chromatic_to_key",
        "harm_kb_functional_dom7_prob",
        "harm_kb_next_acoustic_resolution_prob",
        "harm_kb_next_reference_resolution_prob",
        "harm_kb_supports_next_acoustic",
        "harm_candidate_is_sus9",
        "harm_sus9_expected_resolution",
        "harm_sus9_next_acoustic_matches_resolution",
        "harm_sus9_next_reference_matches_resolution",
        "harm_sus9_is_plain_v_to_i",
        "harm_candidate_is_halfdim",
        "harm_halfdim_expected_resolution",
        "harm_halfdim_next_acoustic_matches_resolution",
        "harm_halfdim_next_reference_matches_resolution",
        "harm_halfdim_prev_acoustic_common_prep",
    ]


def add_features(
    row: dict[str, str],
    context: dict[str, str],
    kb_counts: dict[str, float],
    kb_resolutions: dict[str, dict[str, float]],
) -> None:
    roman = str(row.get("candidate_roman") or row.get("candidate_roman_root") or "")
    root_roman = roman_root(roman)
    expected = SECONDARY_DOMINANT_TARGET.get(roman)
    if expected is None and roman.endswith("7"):
        expected = SECONDARY_DOMINANT_TARGET.get(root_roman + "7")
    kb_roman = roman if roman.endswith("7") else root_roman + "7"
    kb_prob = kb_counts.get(kb_roman, 0.0)
    next_acoustic_prob = kb_resolutions.get(kb_roman, {}).get(context.get("next_acoustic_roman_root", ""), 0.0)
    next_reference_prob = kb_resolutions.get(kb_roman, {}).get(context.get("next_reference_roman_root", ""), 0.0)
    candidate_root, _ = parse_chord_name(row.get("candidate_chord") or "N")
    key_root, _ = parse_chord_name(row.get("local_key") or "C")
    chromatic = 0
    if candidate_root is not None and key_root is not None:
        offset = (candidate_root - key_root) % 12
        chromatic = int(offset not in {0, 2, 4, 5, 7, 9, 11})
    row["harm_candidate_is_functional_dom7"] = str(int(expected is not None))
    row["harm_candidate_expected_resolution"] = expected or ""
    row["harm_next_acoustic_matches_resolution"] = str(int(bool(expected) and context.get("next_acoustic_roman_root") == expected))
    row["harm_next_reference_matches_resolution"] = str(int(bool(expected) and context.get("next_reference_roman_root") == expected))
    row["harm_prev_acoustic_same_root"] = str(int(context.get("prev_acoustic_roman_root") == root_roman))
    row["harm_next_acoustic_same_root"] = str(int(context.get("next_acoustic_roman_root") == root_roman))
    row["harm_candidate_is_chromatic_to_key"] = str(chromatic)
    row["harm_kb_functional_dom7_prob"] = f"{kb_prob:.6f}"
    row["harm_kb_next_acoustic_resolution_prob"] = f"{next_acoustic_prob:.6f}"
    row["harm_kb_next_reference_resolution_prob"] = f"{next_reference_prob:.6f}"
    row["harm_kb_supports_next_acoustic"] = str(int(next_acoustic_prob > 0.0))

    sus9_expected = SUS9_TARGET.get(roman)
    if sus9_expected is None and roman.endswith("9sus4"):
        sus9_expected = SUS9_TARGET.get(root_roman + "9sus4")
    row["harm_candidate_is_sus9"] = str(int(sus9_expected is not None))
    row["harm_sus9_expected_resolution"] = sus9_expected or ""
    row["harm_sus9_next_acoustic_matches_resolution"] = str(
        int(bool(sus9_expected) and context.get("next_acoustic_roman_root") == sus9_expected)
    )
    row["harm_sus9_next_reference_matches_resolution"] = str(
        int(bool(sus9_expected) and context.get("next_reference_roman_root") == sus9_expected)
    )
    row["harm_sus9_is_plain_v_to_i"] = str(
        int(roman == "V9sus4" and context.get("next_acoustic_roman_root") == "I")
    )

    halfdim_expected = HALF_DIM_TARGET.get(roman)
    if halfdim_expected is None and roman.endswith("m7b5"):
        halfdim_expected = HALF_DIM_TARGET.get(root_roman + "m7b5")
    prev_root = context.get("prev_acoustic_roman_root", "")
    common_prep = {
        "bVIm7b5": {"I", "IIIm", "V"},
        "#IVm7b5": {"IV", "I", "V"},
        "VIIm7b5": {"V", "I"},
        "IIm7b5": {"IV", "I"},
    }.get(roman, set())
    row["harm_candidate_is_halfdim"] = str(int(halfdim_expected is not None))
    row["harm_halfdim_expected_resolution"] = halfdim_expected or ""
    row["harm_halfdim_next_acoustic_matches_resolution"] = str(
        int(bool(halfdim_expected) and context.get("next_acoustic_roman_root") == halfdim_expected)
    )
    row["harm_halfdim_next_reference_matches_resolution"] = str(
        int(bool(halfdim_expected) and context.get("next_reference_roman_root") == halfdim_expected)
    )
    row["harm_halfdim_prev_acoustic_common_prep"] = str(int(bool(common_prep) and prev_root in common_prep))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--knowledge-base", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = resolve_path(args.candidate_csv)
    df = pd.read_csv(input_csv, encoding="utf-8-sig", low_memory=False)
    contexts = next_group_lookup(df)
    kb_counts, kb_resolutions = load_knowledge_base(args.knowledge_base)
    rows = df.fillna("").astype(str).to_dict("records")
    fieldnames = list(df.columns)
    for name in feature_names():
        if name not in fieldnames:
            fieldnames.append(name)
    for row in rows:
        key = "|".join(str(row.get(column, "")) for column in group_columns(df))
        add_features(row, contexts.get(key, {}), kb_counts, kb_resolutions)
        for name in fieldnames:
            row.setdefault(name, "")
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
