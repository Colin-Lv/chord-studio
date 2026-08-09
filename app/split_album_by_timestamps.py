#!/usr/bin/env python3
"""Split one continuous album audio file into tracks from a timestamp CSV."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


def parse_time(value: str) -> float:
    parts = value.strip().split(":")
    if not parts or any(part == "" for part in parts):
        raise ValueError(f"empty timestamp: {value!r}")
    nums = [float(part) for part in parts]
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        minutes, seconds = nums
        return minutes * 60 + seconds
    if len(nums) == 3:
        hours, minutes, seconds = nums
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"unsupported timestamp: {value!r}")


def format_seconds(seconds: float) -> str:
    return f"{max(0.0, seconds):.3f}"


def safe_name(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value.rstrip(". ") or "track"


def find_ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def read_segments(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"track", "title", "start"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(sorted(missing))}")
    rows.sort(key=lambda row: parse_time(row["start"]))
    return rows


def split_album(
    audio: Path,
    segments_csv: Path,
    out_dir: Path,
    extension: str,
    overwrite: bool,
) -> list[Path]:
    rows = read_segments(segments_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    outputs: list[Path] = []

    for idx, row in enumerate(rows):
        start = parse_time(row["start"])
        end = parse_time(rows[idx + 1]["start"]) if idx + 1 < len(rows) else None
        track = row["track"].zfill(2)
        title = row["title"].strip()
        artist = row.get("artist", "").strip()
        album = row.get("album", "").strip()
        out_path = out_dir / f"{track} - {safe_name(title)}.{extension.lstrip('.')}"

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-ss",
            format_seconds(start),
            "-i",
            str(audio),
        ]
        if end is not None:
            command += ["-to", format_seconds(end - start)]
        command += ["-vn"]
        if extension.lower().lstrip(".") == "wav":
            command += ["-acodec", "pcm_s16le", "-ar", "44100"]
        else:
            command += ["-c:a", "libmp3lame", "-q:a", "2"]
        if title:
            command += ["-metadata", f"title={title}"]
        if artist:
            command += ["-metadata", f"artist={artist}"]
        if album:
            command += ["-metadata", f"album={album}"]
        command.append(str(out_path))

        subprocess.run(command, check=True)
        outputs.append(out_path)
        print(f"{track}: {title} -> {out_path}")

    return outputs


def normalize_title(value: str) -> str:
    value = value.lower()
    value = value.replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def manifest_relative(path: Path, manifest: Path) -> str:
    try:
        return str(path.resolve().relative_to(manifest.resolve().parent))
    except ValueError:
        return str(path)


def update_manifest(manifest: Path, outputs: list[Path], album_token: str) -> None:
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    if not rows:
        raise SystemExit(f"{manifest} has no rows")

    by_track = {path.name.split(" ", 1)[0]: path for path in outputs}
    by_title = {normalize_title(path.stem.split(" - ", 1)[-1]): path for path in outputs}
    updated = 0
    for row in rows:
        row_id = row.get("id", "")
        if album_token not in row_id:
            continue
        track_match = re.search(rf"{re.escape(album_token)}_(\d{{2}})_-", row_id)
        if track_match and track_match.group(1) in by_track:
            row["audio"] = manifest_relative(by_track[track_match.group(1)], manifest)
            updated += 1
            continue

        # Fall back to title matching when a manifest does not encode track numbers.
        for key, path in by_title.items():
            if key and (key in normalize_title(row_id) or normalize_title(row_id).endswith(key)):
                row["audio"] = manifest_relative(path, manifest)
                updated += 1
                break

    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"updated {updated} manifest audio paths in {manifest}")


def album_token_from_segments(segments: Path) -> str | None:
    rows = read_segments(segments)
    albums = {row.get("album", "").strip() for row in rows if row.get("album", "").strip()}
    if len(albums) != 1:
        return None
    album = next(iter(albums))
    token = safe_name(album).replace(" ", "_").replace("'", "")
    return f"{token}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Local legally obtained album audio file.")
    parser.add_argument("--segments", type=Path, required=True, help="CSV with track,title,start columns.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for split track audio.")
    parser.add_argument("--format", default="mp3", choices=["mp3", "wav"], help="Output audio format.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing split files.")
    parser.add_argument("--update-manifest", type=Path, help="Optional benchmark manifest to fill audio paths.")
    parser.add_argument("--album-token", help="Manifest id token for this album, e.g. 03_-_A_Hard_Day_s_Night.")
    args = parser.parse_args()

    if not args.audio.exists():
        raise SystemExit(f"audio file not found: {args.audio}")
    outputs = split_album(args.audio, args.segments, args.out_dir, args.format, args.overwrite)
    if args.update_manifest:
        album_token = args.album_token or album_token_from_segments(args.segments)
        if not album_token:
            raise SystemExit("--album-token is required when it cannot be inferred from the segments CSV")
        update_manifest(args.update_manifest, outputs, album_token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
