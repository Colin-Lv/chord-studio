#!/usr/bin/env python
"""Summarize whether downloaded POP909 audio appears to match the annotations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {str(row.get("id") or ""): row for row in csv.DictReader(handle)}


def read_manifest_status(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {str(row.get("id") or ""): str(row.get("alignment_status") or "") for row in csv.DictReader(handle)}


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key) or default)
    except ValueError:
        return default


def classify(duration_delta: float, best_shift: float, best_triad: float, accepted_status: str, strict_status: str) -> tuple[str, str]:
    abs_delta = abs(duration_delta)
    abs_shift = abs(best_shift)
    if strict_status == "accepted" or (best_triad >= 0.60 and abs_delta <= 2.0 and abs_shift <= 2.0):
        return "strong_match", "high chord agreement with small duration/offset difference"
    if accepted_status == "accepted" or best_triad >= 0.50:
        if abs_delta > 8.0:
            return "usable_with_caution", "chord agreement is acceptable, but duration differs a lot"
        if abs_shift > 6.0:
            return "usable_with_offset", "same-song evidence, but intro/outro offset is large"
        return "aligned_match", "usable after reference time shift"
    if best_triad >= 0.40 and abs_delta <= 3.0:
        return "needs_listen", "duration is close but chord agreement is borderline"
    if best_triad >= 0.40:
        return "likely_different_edit", "some chord overlap, but duration/arrangement likely differs"
    if abs_delta > 8.0:
        return "likely_mismatch", "low chord agreement and large duration difference"
    return "reject_for_training", "low chord agreement; do not train on this without manual verification"


def make_rows(
    duration_rows: dict[str, dict[str, str]],
    shift_rows: dict[str, dict[str, str]],
    transpose_rows: dict[str, dict[str, str]],
    accepted: dict[str, str],
    strict: dict[str, str],
    rejected: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    ids = sorted(set(duration_rows) | set(shift_rows) | set(transpose_rows))
    for row_id in ids:
        duration = duration_rows.get(row_id, {})
        shift = shift_rows.get(row_id, {})
        transpose = transpose_rows.get(row_id, {})
        duration_delta = number(duration, "duration_delta")
        best_shift = number(shift, "best_time_shift")
        best_root = number(shift, "best_root")
        best_triad = number(shift, "best_triad")
        base_triad = number(shift, "base_triad")
        transpose_shift = number(transpose, "best_shift")
        transpose_triad = number(transpose, "best_triad")
        accepted_status = accepted.get(row_id, "")
        strict_status = strict.get(row_id, "")
        rejected_status = rejected.get(row_id, "")
        status, reason = classify(duration_delta, best_shift, best_triad, accepted_status, strict_status)
        rows.append(
            {
                "id": row_id,
                "title": duration.get("title") or shift.get("title") or transpose.get("title") or "",
                "artist": duration.get("artist") or transpose.get("artist") or "",
                "status": status,
                "reason": reason,
                "duration_delta": f"{duration_delta:.3f}",
                "best_time_shift": f"{best_shift:.3f}",
                "base_triad": f"{base_triad:.3f}",
                "best_root": f"{best_root:.3f}",
                "best_triad": f"{best_triad:.3f}",
                "transpose_shift": f"{transpose_shift:.0f}",
                "transpose_best_triad": f"{transpose_triad:.3f}",
                "accepted_manifest": accepted_status,
                "strict_manifest": strict_status,
                "rejected_manifest": rejected_status,
                "audio": duration.get("audio", ""),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "title",
        "artist",
        "status",
        "reason",
        "duration_delta",
        "best_time_shift",
        "base_triad",
        "best_root",
        "best_triad",
        "transpose_shift",
        "transpose_best_triad",
        "accepted_manifest",
        "strict_manifest",
        "rejected_manifest",
        "audio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-check", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "pop909_real_001_034_duration_check.csv")
    parser.add_argument("--shift-scan", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "pop909_real_001_034_time_shift_scan.csv")
    parser.add_argument("--transpose-scan", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "pop909_real_001_034_transpose_scan.csv")
    parser.add_argument("--accepted", type=Path, default=REPO_ROOT / "benchmarks" / "manifest.pop909_real_001_034_aligned_accepted.csv")
    parser.add_argument("--strict", type=Path, default=REPO_ROOT / "benchmarks" / "manifest.pop909_real_001_034_aligned_strict.csv")
    parser.add_argument("--rejected", type=Path, default=REPO_ROOT / "benchmarks" / "manifest.pop909_real_001_034_aligned_rejected.csv")
    parser.add_argument("-o", "--output", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "pop909_real_001_034_version_audit.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = make_rows(
        read_csv_by_id(resolve_path(args.duration_check)),
        read_csv_by_id(resolve_path(args.shift_scan)),
        read_csv_by_id(resolve_path(args.transpose_scan)),
        read_manifest_status(resolve_path(args.accepted)),
        read_manifest_status(resolve_path(args.strict)),
        read_manifest_status(resolve_path(args.rejected)),
    )
    write_csv(resolve_path(args.output), rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    print(f"Wrote: {resolve_path(args.output)}")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")
    print("\nTop manual-listen candidates:")
    priority = {"needs_listen", "likely_different_edit", "usable_with_caution", "usable_with_offset"}
    for row in [item for item in rows if item["status"] in priority][:12]:
        print(f"{row['id']} {row['title']} - {row['artist']}: {row['status']}, triad={row['best_triad']}, delta={row['duration_delta']}s, shift={row['best_time_shift']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
