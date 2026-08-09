#!/usr/bin/env python3
"""Link user-provided POP909 audio files into a benchmark manifest."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma"}


def clean_text(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.casefold())


def song_number(row_id: str) -> str:
    match = re.search(r"(\d{1,3})$", row_id)
    return f"{int(match.group(1)):03d}" if match else ""


def read_index(path: Path | None) -> dict[str, tuple[str, str]]:
    if not path or not path.exists():
        return {}
    result: dict[str, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_id = (row.get("song_id") or row.get("id") or "").strip()
            if not raw_id:
                continue
            number = f"{int(float(raw_id)):03d}"
            result[number] = ((row.get("name") or "").strip(), (row.get("artist") or "").strip())
    return result


def relative_or_absolute(base: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def score_candidate(number: str, title: str, artist: str, path: Path) -> int:
    stem = clean_text(path.stem)
    score = 0
    if re.search(rf"(^|[^\d])0*{int(number)}([^\d]|$)", path.stem):
        score += 100
    if clean_text(f"pop909{number}") in stem:
        score += 100
    if clean_text(title) and clean_text(title) in stem:
        score += 40
    if clean_text(artist) and clean_text(artist) in stem:
        score += 15
    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("audio_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=Path("benchmarks/raw/pop909/POP909/index_POP909.csv"))
    parser.add_argument("--min-score", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = args.manifest.resolve().parent
    audio_root = args.audio_dir.resolve()
    audio_files = [
        path
        for path in audio_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in AUDIO_EXTENSIONS
    ]
    index = read_index(args.index)

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    linked = 0
    unmatched: list[str] = []
    used: set[Path] = set()
    for row in rows:
        number = song_number(row.get("id", "")) or song_number(row.get("title", ""))
        title, artist = index.get(number, ("", ""))
        best_path: Path | None = None
        best_score = -1
        for candidate in audio_files:
            if candidate in used:
                continue
            score = score_candidate(number, title, artist, candidate)
            if score > best_score:
                best_path = candidate
                best_score = score
        if best_path and best_score >= args.min_score:
            row["audio"] = relative_or_absolute(base, best_path)
            row["artist"] = artist or row.get("artist", "")
            row["title"] = title or row.get("title", "")
            notes = row.get("notes", "").strip()
            row["notes"] = (notes + "; " if notes else "") + "linked to user-provided real audio"
            used.add(best_path)
            linked += 1
        else:
            unmatched.append(row.get("id", ""))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"audio files: {len(audio_files)}")
    print(f"linked: {linked} / {len(rows)}")
    print(f"unmatched: {len(unmatched)}")
    if unmatched[:20]:
        print("first unmatched:", ", ".join(unmatched[:20]))
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
