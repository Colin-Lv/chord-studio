#!/usr/bin/env python
"""Fill missing candidate local_key values from benchmark manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_key_map(manifests: list[Path]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for manifest in manifests:
        with resolve_path(manifest).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                song_id = row.get("id", "")
                key = row.get("key", "")
                if song_id and key:
                    mapping[song_id] = key
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    key_by_song = load_key_map(args.manifest)
    input_csv = resolve_path(args.candidate_csv)
    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "local_key" not in fieldnames:
        fieldnames.append("local_key")
    filled = 0
    missing = set()
    for row in rows:
        song_id = row.get("song_id", "")
        key = key_by_song.get(song_id, "")
        if not key:
            missing.add(song_id)
            continue
        if not str(row.get("local_key") or "").strip():
            row["local_key"] = key
            filled += 1
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} row(s), filled {filled} local_key value(s): {output}")
    if missing:
        print(f"Missing manifest key for {len(missing)} song(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
