#!/usr/bin/env python
"""Audit whether candidate rows can be applied to an analysis grid."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from train_candidate_reranker import resolve_path


def span_key(bar: int, start: float, end: float) -> tuple[int, float, float]:
    return (int(bar), round(float(start), 3), round(float(end), 3))


def interval_contains(outer: tuple[int, float, float], inner: tuple[int, float, float], tolerance: float) -> bool:
    return outer[0] == inner[0] and outer[1] <= inner[1] + tolerance and outer[2] + tolerance >= inner[2]


def overlap_seconds(left: tuple[int, float, float], right: tuple[int, float, float]) -> float:
    if left[0] != right[0]:
        return 0.0
    return max(0.0, min(left[2], right[2]) - max(left[1], right[1]))


def iter_analysis_spans(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bar in payload.get("bars") or []:
        bar_number = int(bar.get("bar") or bar.get("number") or 0)
        for span in bar.get("spans") or []:
            start = float(span.get("start") or 0.0)
            end = float(span.get("end") or 0.0)
            rows.append(
                {
                    "bar": bar_number,
                    "start": start,
                    "end": end,
                    "duration": max(0.0, end - start),
                    "chord": span.get("chord_display") or span.get("chord") or "N",
                    "roman": span.get("roman_display") or span.get("roman") or "N",
                    "local_key": span.get("local_key") or bar.get("local_key") or "",
                }
            )
    return rows


def load_candidate_keys(path: Path, song_id: str) -> list[tuple[int, float, float]]:
    keys: set[tuple[int, float, float]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if song_id and str(row.get("song_id") or "") != song_id:
                continue
            start = float(row.get("reference_start") or row.get("start") or 0.0)
            end = float(row.get("reference_end") or row.get("end") or 0.0)
            keys.add(span_key(int(float(row.get("bar") or 0)), start, end))
    return sorted(keys)


def classify_span(
    span: dict[str, Any],
    candidate_keys: list[tuple[int, float, float]],
    tolerance: float,
) -> tuple[str, tuple[int, float, float] | None, float]:
    key = span_key(int(span["bar"]), float(span["start"]), float(span["end"]))
    if key in candidate_keys:
        return "exact", key, float(span["duration"])
    containing = [candidate for candidate in candidate_keys if interval_contains(candidate, key, tolerance)]
    if containing:
        best = min(containing, key=lambda candidate: candidate[2] - candidate[1])
        return "candidate_contains_span", best, overlap_seconds(key, best)
    contained = [candidate for candidate in candidate_keys if interval_contains(key, candidate, tolerance)]
    if contained:
        best = max(contained, key=lambda candidate: overlap_seconds(key, candidate))
        return "span_contains_candidate", best, overlap_seconds(key, best)
    overlapping = [candidate for candidate in candidate_keys if overlap_seconds(key, candidate) > tolerance]
    if overlapping:
        best = max(overlapping, key=lambda candidate: overlap_seconds(key, candidate))
        return "partial_overlap", best, overlap_seconds(key, best)
    return "unmatched", None, 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--song-id", default="")
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(resolve_path(args.analysis).read_text(encoding="utf-8"))
    spans = iter_analysis_spans(payload)
    candidate_keys = load_candidate_keys(resolve_path(args.candidate_csv), args.song_id)
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    duration_by_status: dict[str, float] = {}
    for span in spans:
        status, candidate_key, overlap = classify_span(span, candidate_keys, args.tolerance)
        counts[status] = counts.get(status, 0) + 1
        duration_by_status[status] = duration_by_status.get(status, 0.0) + float(span["duration"])
        rows.append(
            {
                **span,
                "status": status,
                "matched_bar": "" if candidate_key is None else candidate_key[0],
                "matched_start": "" if candidate_key is None else f"{candidate_key[1]:.3f}",
                "matched_end": "" if candidate_key is None else f"{candidate_key[2]:.3f}",
                "matched_overlap": f"{overlap:.6f}",
            }
        )

    total = max(len(spans), 1)
    report = {
        "analysis": str(resolve_path(args.analysis)),
        "candidate_csv": str(resolve_path(args.candidate_csv)),
        "song_id": args.song_id,
        "spans": len(spans),
        "candidate_intervals": len(candidate_keys),
        "counts": counts,
        "rates": {key: round(value / total, 6) for key, value in counts.items()},
        "duration_by_status": {key: round(value, 6) for key, value in duration_by_status.items()},
    }
    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output:
        output = resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "bar",
            "start",
            "end",
            "duration",
            "chord",
            "roman",
            "local_key",
            "status",
            "matched_bar",
            "matched_start",
            "matched_end",
            "matched_overlap",
        ]
        with output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
