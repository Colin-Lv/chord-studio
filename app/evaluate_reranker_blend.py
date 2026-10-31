#!/usr/bin/env python
"""Evaluate a weighted score blend of two trained candidate rerankers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_candidate_reranker import TARGET_CHOICES, evaluate, load_model
from train_candidate_reranker import load_rows, resolve_path


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_a", type=Path)
    parser.add_argument("model_b", type=Path)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--target", choices=TARGET_CHOICES)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for model_a; model_b receives 1-alpha.")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "candidate_blend_report.json")
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
    alpha = max(0.0, min(1.0, args.alpha))
    scores = alpha * scores_a + (1.0 - alpha) * scores_b
    report, _ = evaluate(df, scores, target)
    report.update(
        {
            "model_a": str(model_a_path),
            "model_b": str(model_b_path),
            "candidate_csv": str(candidate_csv),
            "alpha": alpha,
            "target": target,
        }
    )
    args.report = resolve_path(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
