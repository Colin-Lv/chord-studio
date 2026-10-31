#!/usr/bin/env python
"""Copy a benchmark manifest while redirecting prediction paths to an output directory."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def resolve_path(base: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else base / path


def portable_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("prediction_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = args.manifest.parent.resolve()
    prediction_dir = args.prediction_dir.resolve()
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    output_base = args.output.parent.resolve()
    for row in rows:
        for field in ("audio", "chord_reference", "beat_reference"):
            source = resolve_path(base, row.get(field, ""))
            if source is not None:
                row[field] = portable_path(source, output_base)
        row_id = row.get("id", "")
        prediction = prediction_dir / f"{safe_id(row_id)}_analysis.json"
        row["prediction"] = portable_path(prediction, output_base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
