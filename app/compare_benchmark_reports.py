#!/usr/bin/env python
"""Compare two benchmark JSON reports item by item."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(row: dict[str, Any], name: str) -> float | None:
    value = row.get(name)
    if value in {"", None}:
        return None
    return float(value)


def by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("id", "")): row for row in report.get("items", [])}


def compare(base: dict[str, Any], experiment: dict[str, Any], primary_metric: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_rows = by_id(base)
    experiment_rows = by_id(experiment)
    common_ids = sorted(set(base_rows).intersection(experiment_rows))
    rows = []
    for row_id in common_ids:
        base_row = base_rows[row_id]
        experiment_row = experiment_rows[row_id]
        output: dict[str, Any] = {
            "id": row_id,
            "artist": experiment_row.get("artist") or base_row.get("artist", ""),
            "title": experiment_row.get("title") or base_row.get("title", ""),
        }
        for name in ("root_accuracy", "triad_accuracy", "beat_f1", "downbeat_f1"):
            before = metric(base_row, name)
            after = metric(experiment_row, name)
            output[f"base_{name}"] = "" if before is None else before
            output[f"experiment_{name}"] = "" if after is None else after
            output[f"delta_{name}"] = "" if before is None or after is None else after - before
        rows.append(output)

    deltas = [float(row[f"delta_{primary_metric}"]) for row in rows if row[f"delta_{primary_metric}"] != ""]
    summary = {
        "items": len(rows),
        "metric": primary_metric,
        "mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "improved": sum(1 for value in deltas if value > 1e-9),
        "declined": sum(1 for value in deltas if value < -1e-9),
        "unchanged": sum(1 for value in deltas if abs(value) <= 1e-9),
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "artist",
        "title",
        "base_root_accuracy",
        "experiment_root_accuracy",
        "delta_root_accuracy",
        "base_triad_accuracy",
        "experiment_triad_accuracy",
        "delta_triad_accuracy",
        "base_beat_f1",
        "experiment_beat_f1",
        "delta_beat_f1",
        "base_downbeat_f1",
        "experiment_downbeat_f1",
        "delta_downbeat_f1",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_report", type=Path)
    parser.add_argument("experiment_report", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--metric", default="triad_accuracy")
    parser.add_argument("--limit", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, rows = compare(load_report(args.base_report), load_report(args.experiment_report), args.metric)
    if args.output:
        write_csv(args.output, rows)
        print(f"Wrote: {args.output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    delta_field = f"delta_{args.metric}"
    ranked = [row for row in rows if row[delta_field] != ""]
    ranked.sort(key=lambda row: float(row[delta_field]))
    print("worst:")
    for row in ranked[: args.limit]:
        print(f"{float(row[delta_field]):+.4f} {row['id']}")
    print("best:")
    for row in ranked[-args.limit:][::-1]:
        print(f"{float(row[delta_field]):+.4f} {row['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
