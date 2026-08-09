#!/usr/bin/env python3
"""Generate labs with the ISMIR 2019 large-vocabulary chord model."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from chord_studio import note_name_for_pc, parse_chord_name
from generate_madmom_chords_from_manifest import output_path_for_row


DEFAULT_MODEL_DIR = Path(
    "tools/models/large-vocabulary-chord-recognition-20260809081849/"
    "ISMIR2019-Large-Vocabulary-Chord-Recognition-master"
)


def resolve_path(base: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else base / path


def transpose_label(label: str, semitones: int) -> str:
    root, suffix = parse_chord_name(label)
    if root is None:
        return "N"
    return f"{note_name_for_pc((root + semitones) % 12, 'C')}{suffix}"


def transpose_lab(text: str, semitones: int) -> str:
    if semitones % 12 == 0:
        return text
    rows = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        label = transpose_label(parts[2], semitones)
        rows.append(f"{parts[0]} {parts[1]} {label}")
    return "\n".join(rows) + ("\n" if rows else "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--chord-dict", default="submission")
    parser.add_argument("--transpose", type=int, default=0)
    parser.add_argument("--id-contains")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    base = manifest.parent
    model_dir = args.model_dir.resolve()
    script = model_dir / "chord_recognition.py"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not script.exists():
        print(f"Model script not found: {script}", file=sys.stderr)
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
        temporary = output.with_suffix(".raw.lab")
        print(f"[{index}/{len(selected)}] {row_id}")
        try:
            subprocess.run(
                [args.python, str(script), str(audio), str(temporary), args.chord_dict],
                cwd=str(model_dir),
                check=True,
            )
            text = temporary.read_text(encoding="utf-8")
            output.write_text(transpose_lab(text, args.transpose), encoding="utf-8")
            temporary.unlink(missing_ok=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            failures += 1
            print(f"FAILED {row_id}: {exc}", file=sys.stderr)
    print(f"done: {len(selected) - failures} ok, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
