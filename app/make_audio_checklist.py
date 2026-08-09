#!/usr/bin/env python
"""Create human-friendly audio acquisition checklists from benchmark manifests."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def clean_name(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip()


def beatles_album_track(item_id: str) -> tuple[str, str]:
    marker = "_The_Beatles_"
    if marker not in item_id:
        return "", clean_name(item_id)
    tail = item_id.split(marker, 1)[1]
    match = re.search(r"_(\d\d(?:CD\d)?_-_.+)$", tail)
    if not match:
        return "", clean_name(tail)
    album = clean_name(tail[: match.start()])
    track = clean_name(match.group(1))
    return album, track


def line_for(index: int, row: dict[str, str]) -> str:
    item_id = row.get("id", "")
    dataset = row.get("dataset", "")
    if dataset == "isophonics":
        album, track = beatles_album_track(item_id)
        key = row.get("key", "")
        return (
            f"- [ ] {index:03d}. The Beatles - {track} | "
            f"album: {album} | id: `{item_id}` | key: {key}"
        )
    if dataset == "ballroom":
        return f"- [ ] {index:03d}. Ballroom - {row.get('title', '')} | id: `{item_id}`"
    artist = row.get("artist", "")
    title = row.get("title", "")
    return f"- [ ] {index:03d}. {artist} - {title} | id: `{item_id}`"


def write_checklist(manifest: Path, output: Path) -> None:
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8-sig", newline="")))
    lines = [f"# Audio Checklist: {manifest.name}", ""]
    lines.extend(line_for(index, row) for index, row in enumerate(rows, 1))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an audio checklist markdown file.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    write_checklist(args.manifest, args.output)
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
