#!/usr/bin/env python3
"""Generate Chordino chord labs for a benchmark manifest using Sonic Annotator."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from generate_madmom_chords_from_manifest import output_path_for_row


DEFAULT_TRANSFORM_ID = "vamp:nnls-chroma:chordino:simplechord"


def resolve_path(base: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else base / path


def has_transform(sonic_annotator: Path, transform_id: str) -> bool:
    try:
        result = subprocess.run(
            [str(sonic_annotator), "--list"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return transform_id.lower() in result.stdout.lower()


def audio_duration(path: Path) -> float:
    try:
        import librosa

        return float(librosa.get_duration(path=str(path)))
    except Exception:
        return 0.0


def chordino_points_to_interval_lab(text: str, duration: float) -> str:
    points: list[tuple[float, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        try:
            start = float(parts[0])
        except ValueError:
            continue
        label = parts[1].strip().strip('"').strip("'") or "N"
        points.append((start, label))
    points.sort(key=lambda item: item[0])
    rows: list[tuple[float, float, str]] = []
    for index, (start, label) in enumerate(points):
        end = points[index + 1][0] if index + 1 < len(points) else duration
        if end <= start:
            continue
        if rows and rows[-1][2] == label and abs(rows[-1][1] - start) <= 1e-3:
            rows[-1] = (rows[-1][0], end, label)
        else:
            rows.append((start, end, label))
    return "".join(f"{start:.3f} {end:.3f} {label}\n" for start, end, label in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--sonic-annotator",
        type=Path,
        default=Path("tools/sonic-annotator-1.7-win64/sonic-annotator-win64/sonic-annotator.exe"),
    )
    parser.add_argument("--transform-id", default=DEFAULT_TRANSFORM_ID)
    parser.add_argument("--id-contains")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    base = manifest.parent
    sonic_annotator = args.sonic_annotator.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sonic_annotator.exists():
        print(f"Sonic Annotator not found: {sonic_annotator}", file=sys.stderr)
        return 2
    if not has_transform(sonic_annotator, args.transform_id):
        print(
            "Chordino transform is not available. Install the Vamp Plugin Pack / NNLS Chroma plugin, "
            f"then confirm `{sonic_annotator} --list` contains `{args.transform_id}`.",
            file=sys.stderr,
        )
        return 2

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected: list[dict[str, str]] = []
    for row in rows:
        row_id = row.get("id", "")
        if args.id_contains and args.id_contains not in row_id:
            continue
        audio = resolve_path(base, row.get("audio", ""))
        output = output_path_for_row(output_dir, row_id)
        if not audio or not audio.exists():
            continue
        if output.exists() and not args.force:
            continue
        selected.append(row)
        if args.limit and len(selected) >= args.limit:
            break

    print(f"selected rows: {len(selected)}")
    failures = 0
    for index, row in enumerate(selected, start=1):
        row_id = row.get("id", f"row-{index}")
        audio = resolve_path(base, row.get("audio", ""))
        output = output_path_for_row(output_dir, row_id)
        print(f"[{index}/{len(selected)}] {row_id}")
        try:
            result = subprocess.run(
                [
                    str(sonic_annotator),
                    "-q",
                    "-d",
                    args.transform_id,
                    "-w",
                    "lab",
                    "--lab-stdout",
                    str(audio),
                ],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            duration = audio_duration(audio) if audio else 0.0
            output.write_text(chordino_points_to_interval_lab(result.stdout, duration), encoding="utf-8")
        except (OSError, subprocess.CalledProcessError) as exc:
            failures += 1
            print(f"FAILED {row_id}: {exc}", file=sys.stderr)
    print(f"done: {len(selected) - failures} ok, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
