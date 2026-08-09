#!/usr/bin/env python
"""Create a deterministic subset manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small deterministic benchmark subset.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("-n", "--count", type=int, required=True)
    parser.add_argument("--require-chord", action="store_true")
    parser.add_argument("--require-beat", action="store_true")
    parser.add_argument("--require-key", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    filtered = []
    for row in rows:
        if args.require_chord and not row.get("chord_reference"):
            continue
        if args.require_beat and not row.get("beat_reference"):
            continue
        if args.require_key and not row.get("key"):
            continue
        filtered.append(row)

    selected = filtered[: args.count]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    print(f"Selected: {len(selected)} / {len(filtered)}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
