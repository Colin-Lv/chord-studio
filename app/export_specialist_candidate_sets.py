#!/usr/bin/env python
"""Export long-tail chord candidate subsets for specialist training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from train_candidate_reranker import resolve_path


SPECIALISTS = {
    "functional_dom7": lambda df: df["candidate_source"].fillna("").astype(str).str.contains("functional", case=False)
    & df["candidate_roman"].fillna("").astype(str).str.endswith("7"),
    "half_dim": lambda df: df["candidate_source"].fillna("").astype(str).str.contains("half_dim|m7b5", case=False, regex=True)
    | df["candidate_chord"].fillna("").astype(str).str.contains("m7b5", case=False, regex=False),
    "sus9": lambda df: df["candidate_source"].fillna("").astype(str).str.contains("sus9|9sus", case=False, regex=True)
    | df["candidate_chord"].fillna("").astype(str).str.contains("9sus4", case=False, regex=False),
}


KEEP_COLUMNS = [
    "song_id",
    "bar",
    "reference_start",
    "reference_end",
    "start",
    "end",
    "local_key",
    "candidate_source",
    "candidate_chord",
    "candidate_roman",
    "candidate_roman_root",
    "candidate_chord_family",
    "candidate_score",
    "candidate_confidence",
    "candidate_rank",
    "reference_chord",
    "reference_roman",
    "reference_roman_root",
    "reference_chord_family",
    "target_exact_chord",
    "target_same_root",
    "target_same_roman_root",
    "target_same_chord_family",
    "audio_root_energy",
    "audio_third_energy",
    "audio_fifth_energy",
    "audio_triad_mean",
    "audio_flat7_energy",
    "audio_maj7_energy",
    "audio_9_energy",
    "audio_4_energy",
    "audio_flat7_over_triad",
    "audio_9_over_triad",
    "audio_dom7_vs_maj7",
    "audio_bass_root_energy",
    "audio_candidate_has_7",
    "audio_candidate_has_9",
    "audio_candidate_has_sus4",
    "audio_candidate_has_m7b5",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def target_rate(df: pd.DataFrame, target: str) -> float:
    if target not in df or df.empty:
        return 0.0
    values = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)
    return round(float(values.mean()), 4)


def main() -> int:
    args = parse_args()
    df = pd.read_csv(resolve_path(args.candidate_csv), encoding="utf-8-sig", low_memory=False)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "candidate_csv": str(resolve_path(args.candidate_csv)),
        "specialists": {},
    }
    for name, mask_fn in SPECIALISTS.items():
        subset = df[mask_fn(df)].copy()
        columns = [column for column in KEEP_COLUMNS if column in subset.columns]
        subset = subset[columns]
        prefix = args.prefix
        if prefix and not prefix.endswith(("_", "-", ".")):
            prefix += "_"
        output = output_dir / f"{prefix}{name}_candidates.csv"
        subset.to_csv(output, index=False, encoding="utf-8-sig")
        summary["specialists"][name] = {
            "rows": int(len(subset)),
            "songs": int(subset["song_id"].nunique()) if "song_id" in subset else 0,
            "exact_positive_rate": target_rate(subset, "target_exact_chord"),
            "roman_root_positive_rate": target_rate(subset, "target_same_roman_root"),
            "output": str(output),
        }

    prefix = args.prefix
    if prefix and not prefix.endswith(("_", "-", ".")):
        prefix += "_"
    summary_path = resolve_path(args.summary) if args.summary else output_dir / f"{prefix}specialist_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
