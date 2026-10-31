#!/usr/bin/env python
"""Evaluate a two-stage selector: choose function/root first, then chord spelling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_candidate_reranker import TARGET_CHOICES, group_key, load_model
from train_candidate_reranker import filter_evaluable_rows, load_rows, resolve_path


REPO_ROOT = Path(__file__).resolve().parent.parent


def evaluate_two_stage(
    df: pd.DataFrame,
    root_scores,
    selection_scores,
    root_target: str,
    selection_target: str,
    report_target: str,
    root_column: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    df = df.copy()
    df["root_score"] = root_scores
    df["selection_score"] = selection_scores
    df["group_key"] = [group_key(row) for _, row in df.iterrows()]
    total = 0
    exact_correct = 0
    root_correct = 0
    exact_oracle = 0
    root_oracle = 0
    by_song: dict[str, list[int]] = {}
    selected_rows: list[dict[str, object]] = []
    for _, group in df.groupby("group_key"):
        if group.empty:
            continue
        root_best = group.sort_values("root_score", ascending=False).iloc[0]
        root_value = getattr(root_best, root_column)
        narrowed = group[group[root_column].astype(str) == str(root_value)]
        if narrowed.empty:
            narrowed = group
        exact_best = narrowed.sort_values("selection_score", ascending=False).iloc[0]
        total += 1
        root_is_correct = int(root_best[root_target])
        exact_is_correct = int(exact_best[report_target])
        root_correct += root_is_correct
        exact_correct += exact_is_correct
        root_oracle += int(group[root_target].astype(int).max())
        exact_oracle += int(group[report_target].astype(int).max())
        by_song.setdefault(str(exact_best.song_id), []).append(exact_is_correct)
        selected_rows.append(
            {
                "song_id": str(exact_best.song_id),
                "bar": int(exact_best.bar),
                "reference_start": f"{float(exact_best.reference_start):.6f}",
                "reference_end": f"{float(exact_best.reference_end):.6f}",
                "root_choice": str(root_value),
                "candidate_source": str(exact_best.candidate_source),
                "candidate_chord": str(exact_best.candidate_chord),
                "candidate_roman": str(exact_best.candidate_roman),
                "reference_chord": str(exact_best.reference_chord),
                "reference_roman": str(exact_best.reference_roman),
                "root_score": f"{float(root_best.root_score):.6f}",
                "selection_score": f"{float(exact_best.selection_score):.6f}",
                "root_correct": root_is_correct,
                "selection_correct": int(exact_best[selection_target]),
                "exact_correct": exact_is_correct,
                "narrowed_candidates": int(len(narrowed)),
            }
        )
    report = {
        "groups": total,
        "root_target": root_target,
        "selection_target": selection_target,
        "report_target": report_target,
        "root_column": root_column,
        "root_accuracy": round(root_correct / total, 4) if total else 0.0,
        "root_oracle": round(root_oracle / total, 4) if total else 0.0,
        "report_accuracy": round(exact_correct / total, 4) if total else 0.0,
        "report_oracle": round(exact_oracle / total, 4) if total else 0.0,
        "by_song": {
            song: {
                "groups": len(values),
                "report_accuracy": round(sum(values) / len(values), 4) if values else 0.0,
            }
            for song, values in sorted(by_song.items())
        },
    }
    return report, selected_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_model", type=Path)
    parser.add_argument("exact_model", type=Path)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--root-target", choices=["target_same_root", "target_same_roman_root"], default="target_same_roman_root")
    parser.add_argument("--selection-target", choices=TARGET_CHOICES, default="target_exact_chord")
    parser.add_argument("--report-target", choices=TARGET_CHOICES, default="target_exact_chord")
    parser.add_argument("--root-column", choices=["candidate_roman_root", "candidate_chord_root"], default="candidate_roman_root")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "two_stage_selector_report.json")
    parser.add_argument("--selected", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_model_path = resolve_path(args.root_model)
    exact_model_path = resolve_path(args.exact_model)
    candidate_csv = resolve_path(args.candidate_csv)
    root_model, _ = load_model(root_model_path)
    exact_model, _ = load_model(exact_model_path)
    df = load_rows(candidate_csv)
    df = filter_evaluable_rows(df, args.report_target)
    root_scores = root_model.predict_proba(df)[:, 1]
    selection_scores = exact_model.predict_proba(df)[:, 1]
    report, selected_rows = evaluate_two_stage(
        df,
        root_scores,
        selection_scores,
        args.root_target,
        args.selection_target,
        args.report_target,
        args.root_column,
    )
    report.update(
        {
            "root_model": str(root_model_path),
            "exact_model": str(exact_model_path),
            "candidate_csv": str(candidate_csv),
        }
    )
    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.selected:
        selected_path = resolve_path(args.selected)
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(selected_rows).to_csv(selected_path, index=False, encoding="utf-8-sig")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
