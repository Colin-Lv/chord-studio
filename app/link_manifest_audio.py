#!/usr/bin/env python
"""Link user-provided audio files to rows in a benchmark manifest."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma"}


def clean_text(value: str) -> str:
    return re.sub(r"[\W_]+", "", str(value).casefold())


def tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[\W_]+", str(value).casefold()) if token]


def relative_or_absolute(base: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def score_candidate(row: dict[str, str], path: Path) -> int:
    stem_clean = clean_text(path.stem)
    stem_tokens = set(tokens(path.stem))
    row_id = row.get("id", "")
    title = row.get("title", "")
    artist = row.get("artist", "")
    score = 0
    if row_id and clean_text(row_id) in stem_clean:
        score += 120
    if title and clean_text(title) in stem_clean:
        score += 90
    if artist and clean_text(artist) in stem_clean:
        score += 40
    for token in tokens(title):
        if len(token) >= 3 and token in stem_tokens:
            score += 8
    for token in tokens(artist):
        if len(token) >= 3 and token in stem_tokens:
            score += 4
    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("audio_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--min-score", type=int, default=60)
    parser.add_argument("--allow-reuse", action="store_true", help="Allow one audio file to be linked to multiple rows.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    base = manifest.parent
    audio_root = args.audio_dir.resolve()
    audio_files = [
        path
        for path in audio_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in AUDIO_EXTENSIONS
    ]
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if "audio_match_score" not in fieldnames:
        fieldnames.append("audio_match_score")

    linked = 0
    used: set[Path] = set()
    for row in rows:
        best_path: Path | None = None
        best_score = -1
        for candidate in audio_files:
            if candidate in used and not args.allow_reuse:
                continue
            score = score_candidate(row, candidate)
            if score > best_score:
                best_path = candidate
                best_score = score
        row["audio_match_score"] = str(max(best_score, 0))
        if best_path is not None and best_score >= args.min_score:
            row["audio"] = relative_or_absolute(base, best_path)
            notes = row.get("notes", "").strip()
            row["notes"] = (notes + "; " if notes else "") + "linked to user-provided audio"
            used.add(best_path)
            linked += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"audio files: {len(audio_files)}")
    print(f"linked: {linked} / {len(rows)}")
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
