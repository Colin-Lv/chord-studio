#!/usr/bin/env python3
"""Import POP909 annotations into the local benchmark layout."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from chord_studio import normalize_external_chord_label, normalize_key_label


def parse_key(path: Path) -> str:
    if not path.exists():
        return ""
    best_duration = -1.0
    best_key = ""
    for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            start = float(parts[0])
            end = float(parts[1])
        except ValueError:
            continue
        key = normalize_key_label(parts[2].replace(":maj", "").replace(":min", "m"))
        if key and end - start > best_duration:
            best_duration = end - start
            best_key = key
    return best_key


def convert_chords(source: Path, target: Path) -> int:
    count = 0
    rows: list[str] = []
    for line in source.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            start = float(parts[0])
            end = float(parts[1])
        except ValueError:
            continue
        if end <= start:
            continue
        chord = normalize_external_chord_label(parts[2], "sharps")
        rows.append(f"{start:.6f} {end:.6f} {chord}")
        count += 1
    target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return count


def convert_beats(source: Path, target: Path) -> int:
    count = 0
    rows: list[str] = []
    for line in source.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            time = float(parts[0])
            beat = int(float(parts[1]))
        except ValueError:
            continue
        rows.append(f"{time:.6f} {beat}")
        count += 1
    target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return count


def infer_meter(beat_file: Path) -> str:
    labels: list[int] = []
    if not beat_file.exists():
        return ""
    for line in beat_file.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            labels.append(int(float(parts[1])))
        except ValueError:
            continue
    if not labels:
        return ""
    beats_per_bar = max(labels)
    if beats_per_bar in {2, 3, 4, 5, 7, 8}:
        return f"{beats_per_bar}/4"
    if beats_per_bar in {6, 9, 12}:
        return f"{beats_per_bar}/8"
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pop909_dir", type=Path, help="Directory containing POP909 song folders.")
    parser.add_argument(
        "-o",
        "--output-manifest",
        type=Path,
        default=Path("benchmarks/manifest.pop909.csv"),
    )
    parser.add_argument(
        "--import-dir",
        type=Path,
        default=Path("benchmarks/imported/pop909"),
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.pop909_dir.resolve()
    import_root = args.import_dir.resolve()
    manifest = args.output_manifest.resolve()
    import_root.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    song_dirs = [path for path in sorted(source_root.iterdir()) if path.is_dir() and path.name.isdigit()]
    if args.limit:
        song_dirs = song_dirs[: args.limit]
    for song_dir in song_dirs:
        song_id = song_dir.name
        chord_source = song_dir / "chord_audio.txt"
        beat_source = song_dir / "beat_audio.txt"
        key_source = song_dir / "key_audio.txt"
        if not chord_source.exists() or not beat_source.exists():
            continue
        item_id = f"pop909_{song_id}"
        item_dir = import_root / item_id
        item_dir.mkdir(parents=True, exist_ok=True)
        chord_target = item_dir / "chords.lab"
        beat_target = item_dir / "beats.beats"
        chord_count = convert_chords(chord_source, chord_target)
        beat_count = convert_beats(beat_source, beat_target)
        key = parse_key(key_source)
        meter = infer_meter(beat_source)
        rows.append(
            {
                "id": item_id,
                "dataset": "pop909",
                "split": "train",
                "artist": "POP909",
                "title": song_id,
                "audio": "",
                "prediction": f"predictions/{item_id}_analysis.json",
                "chord_reference": str(chord_target.relative_to(manifest.parent)),
                "beat_reference": str(beat_target.relative_to(manifest.parent)),
                "key": key,
                "meter": meter,
                "notes": f"POP909 symbolic/audio-aligned annotations; chords={chord_count}; beats={beat_count}; no audio bundled",
            }
        )

    fieldnames = [
        "id",
        "dataset",
        "split",
        "artist",
        "title",
        "audio",
        "prediction",
        "chord_reference",
        "beat_reference",
        "key",
        "meter",
        "notes",
    ]
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {manifest}")
    print(f"items: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
