#!/usr/bin/env python
"""Export human-edited chord spans from Chord Studio analysis JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def iter_analysis_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    for path in paths:
        if path.is_dir():
            result.extend(sorted(path.rglob("*_analysis.json")))
        elif path.exists():
            result.append(path)
    return result


def edited_reason(payload: dict[str, Any], bar_number: int) -> str:
    for item in payload.get("bar_overrides", []):
        if int(item.get("start_bar", -1)) <= bar_number <= int(item.get("end_bar", -1)):
            return str(item.get("reason") or "bar_override")
    if payload.get("progression_override"):
        return "progression_override"
    if payload.get("modulations"):
        return "progression_section"
    return "manual_edit"


def rows_from_analysis(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for bar in payload.get("bars", []):
        bar_number = int(bar.get("bar", 0))
        for span in bar.get("spans", []):
            if not span.get("edited"):
                continue
            rows.append(
                {
                    "analysis": str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path),
                    "source": payload.get("source", ""),
                    "title": payload.get("title", ""),
                    "bar": bar_number,
                    "beat_start": span.get("beat_start", ""),
                    "beat_end": span.get("beat_end", ""),
                    "start": f"{float(span.get('start', 0.0)):.6f}",
                    "end": f"{float(span.get('end', 0.0)):.6f}",
                    "key": span.get("local_key") or payload.get("key", ""),
                    "meter": payload.get("meter", ""),
                    "harmonic_unit": span.get("unit") or payload.get("harmonic_unit", ""),
                    "chord": span.get("chord", ""),
                    "roman": span.get("roman", ""),
                    "bass_root": span.get("bass_root", ""),
                    "reason": edited_reason(payload, bar_number),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Analysis JSON files or directories to scan.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "reports" / "user_correction_examples.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = iter_analysis_paths([resolve_path(path) for path in args.paths])
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(rows_from_analysis(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "analysis",
        "source",
        "title",
        "bar",
        "beat_start",
        "beat_end",
        "start",
        "end",
        "key",
        "meter",
        "harmonic_unit",
        "chord",
        "roman",
        "bass_root",
        "reason",
    ]
    with args.output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} correction example(s): {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
