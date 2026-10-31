#!/usr/bin/env python
"""Evaluate simple routing policies between two candidate rerankers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_candidate_reranker import TARGET_CHOICES, group_key, load_model
from train_candidate_reranker import domain_for_song, load_rows, resolve_path


REPO_ROOT = Path(__file__).resolve().parent.parent


def group_choice(group: pd.DataFrame, scores: np.ndarray, target: str) -> dict[str, object]:
    local = group.copy()
    local["score"] = scores
    ordered = local.sort_values("score", ascending=False)
    best = ordered.iloc[0]
    top = float(ordered.iloc[0].score)
    second = float(ordered.iloc[1].score) if len(ordered) > 1 else 0.0
    return {
        "source": str(best.candidate_source),
        "chord": str(best.candidate_chord),
        "roman": str(best.candidate_roman),
        "score": top,
        "margin": top - second,
        "correct": int(best[target]),
    }


def choose_policy(policy: str, row: pd.Series, choice_a: dict[str, object], choice_b: dict[str, object]) -> str:
    if policy == "model_a":
        return "a"
    if policy == "model_b":
        return "b"
    if policy == "domain":
        return "a" if domain_for_song(row.song_id) == "beatles" else "b"
    if policy == "score":
        return "a" if float(choice_a["score"]) >= float(choice_b["score"]) else "b"
    if policy == "margin":
        return "a" if float(choice_a["margin"]) >= float(choice_b["margin"]) else "b"
    if policy == "oracle":
        if int(choice_a["correct"]) and not int(choice_b["correct"]):
            return "a"
        if int(choice_b["correct"]) and not int(choice_a["correct"]):
            return "b"
        return "a" if float(choice_a["score"]) >= float(choice_b["score"]) else "b"
    raise ValueError(f"Unknown policy: {policy}")


def evaluate_router(
    df: pd.DataFrame,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    target: str,
    policy: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    df = df.copy()
    df["score_a"] = scores_a
    df["score_b"] = scores_b
    df["group_key"] = [group_key(row) for _, row in df.iterrows()]
    total = 0
    correct = 0
    oracle_correct = 0
    disagreements = 0
    routed_a = 0
    routed_b = 0
    by_song: dict[str, list[int]] = {}
    selected_rows: list[dict[str, object]] = []
    for _, group in df.groupby("group_key"):
        if group.empty:
            continue
        choice_a = group_choice(group, group["score_a"].to_numpy(), target)
        choice_b = group_choice(group, group["score_b"].to_numpy(), target)
        row = group.iloc[0]
        chosen = choose_policy(policy, row, choice_a, choice_b)
        choice = choice_a if chosen == "a" else choice_b
        total += 1
        routed_a += int(chosen == "a")
        routed_b += int(chosen == "b")
        disagreements += int(choice_a["chord"] != choice_b["chord"])
        is_correct = int(choice["correct"])
        correct += is_correct
        oracle_correct += int(group[target].astype(int).max())
        by_song.setdefault(str(row.song_id), []).append(is_correct)
        selected_rows.append(
            {
                "song_id": str(row.song_id),
                "bar": int(row.bar),
                "reference_start": f"{float(row.reference_start):.6f}",
                "reference_end": f"{float(row.reference_end):.6f}",
                "chosen_model": chosen,
                "chosen_chord": choice["chord"],
                "chosen_roman": choice["roman"],
                "choice_a_chord": choice_a["chord"],
                "choice_b_chord": choice_b["chord"],
                "choice_a_score": f"{float(choice_a['score']):.6f}",
                "choice_b_score": f"{float(choice_b['score']):.6f}",
                "choice_a_margin": f"{float(choice_a['margin']):.6f}",
                "choice_b_margin": f"{float(choice_b['margin']):.6f}",
                "reference_chord": str(row.reference_chord),
                "reference_roman": str(row.reference_roman),
                "correct": is_correct,
            }
        )
    report = {
        "target": target,
        "policy": policy,
        "groups": total,
        "selector_accuracy": round(correct / total, 4) if total else 0.0,
        "oracle_accuracy": round(oracle_correct / total, 4) if total else 0.0,
        "disagreement_rate": round(disagreements / total, 4) if total else 0.0,
        "routed_to_model_a": routed_a,
        "routed_to_model_b": routed_b,
        "by_song": {
            song: {
                "groups": len(values),
                "selector_accuracy": round(sum(values) / len(values), 4) if values else 0.0,
            }
            for song, values in sorted(by_song.items())
        },
    }
    return report, selected_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_a", type=Path)
    parser.add_argument("model_b", type=Path)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--target", choices=TARGET_CHOICES)
    parser.add_argument(
        "--policy",
        choices=["model_a", "model_b", "domain", "score", "margin", "oracle"],
        default="domain",
    )
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "candidate_router_report.json")
    parser.add_argument("--selected", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_a_path = resolve_path(args.model_a)
    model_b_path = resolve_path(args.model_b)
    candidate_csv = resolve_path(args.candidate_csv)
    model_a, target_a = load_model(model_a_path)
    model_b, target_b = load_model(model_b_path)
    target = args.target or target_a or target_b
    df = load_rows(candidate_csv)
    scores_a = model_a.predict_proba(df)[:, 1]
    scores_b = model_b.predict_proba(df)[:, 1]
    report, selected_rows = evaluate_router(df, scores_a, scores_b, target, args.policy)
    report.update(
        {
            "model_a": str(model_a_path),
            "model_b": str(model_b_path),
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
