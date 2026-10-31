#!/usr/bin/env python
"""Split a benchmark manifest into deterministic train/eval manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def resolve_path(base: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else base / path


def has_existing_path(row: dict[str, str], base: Path, field: str) -> bool:
    path = resolve_path(base, row.get(field, ""))
    return path is not None and path.exists()


def row_key(row: dict[str, str], seed: str) -> str:
    text = "|".join(
        [
            seed,
            row.get("dataset", ""),
            row.get("artist", ""),
            row.get("title", ""),
            row.get("id", ""),
        ]
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def filtered_rows(rows: list[dict[str, str]], base: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    output = []
    for row in rows:
        if args.dataset and row.get("dataset") not in set(args.dataset):
            continue
        if args.require_audio and not has_existing_path(row, base, "audio"):
            continue
        if args.require_chord and not has_existing_path(row, base, "chord_reference"):
            continue
        if args.require_beat and not has_existing_path(row, base, "beat_reference"):
            continue
        if args.require_prediction and not has_existing_path(row, base, "prediction"):
            continue
        if args.require_key and not row.get("key"):
            continue
        output.append(dict(row))
    return output


def set_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    output = []
    for row in rows:
        item = dict(row)
        item["split"] = split
        output.append(item)
    return output


def write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", default="chord-studio-v1")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--require-audio", action="store_true")
    parser.add_argument("--require-chord", action="store_true")
    parser.add_argument("--require-beat", action="store_true")
    parser.add_argument("--require-key", action="store_true")
    parser.add_argument("--require-prediction", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise SystemExit("--train-ratio must be between 0 and 1.")

    manifest = args.manifest.resolve()
    base = manifest.parent
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if "split" not in fieldnames:
        fieldnames.insert(min(2, len(fieldnames)), "split")

    usable = filtered_rows(rows, base, args)
    usable.sort(key=lambda row: row_key(row, args.seed))
    train_count = max(1, min(len(usable) - 1, round(len(usable) * args.train_ratio))) if len(usable) > 1 else len(usable)
    train_rows = set_split(usable[:train_count], "train")
    eval_rows = set_split(usable[train_count:], "eval")

    write_manifest(args.train_output, train_rows, fieldnames)
    write_manifest(args.eval_output, eval_rows, fieldnames)
    print(f"source rows: {len(rows)}")
    print(f"usable rows: {len(usable)}")
    print(f"train rows: {len(train_rows)} -> {args.train_output}")
    print(f"eval rows: {len(eval_rows)} -> {args.eval_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
