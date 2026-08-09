#!/usr/bin/env python3
"""Generate madmom chord .lab files for manifest rows with audio."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from madmom_bridge import DEFAULT_MADMOM_PYTHON


def resolve_path(base: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else base / path


def output_path_for_row(output_dir: Path, row_id: str) -> Path:
    safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in row_id)
    return output_dir / f"{safe_id}.lab"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("benchmarks/external_chords/madmom"))
    parser.add_argument("--id-contains")
    parser.add_argument("--dataset")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--madmom-python", type=Path, default=DEFAULT_MADMOM_PYTHON)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    base = manifest.parent
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bridge = Path(__file__).resolve().parent / "madmom_bridge.py"

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected: list[tuple[dict[str, str], Path, Path]] = []
    for row in rows:
        row_id = row.get("id", "")
        if args.dataset and row.get("dataset") != args.dataset:
            continue
        if args.id_contains and args.id_contains not in row_id:
            continue
        audio = resolve_path(base, row.get("audio", ""))
        if not audio or not audio.exists():
            continue
        output = output_path_for_row(output_dir, row_id)
        if output.exists() and not args.force:
            continue
        selected.append((row, audio, output))
        if args.limit and len(selected) >= args.limit:
            break

    print(f"selected rows: {len(selected)}")
    failures = 0
    for index, (row, audio, output) in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {row.get('id', '')}")
        command = [
            args.python,
            str(bridge),
            str(audio),
            "-o",
            str(output),
            "--madmom-python",
            str(args.madmom_python),
        ]
        try:
            subprocess.run(command, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            failures += 1
            print(f"FAILED {row.get('id', '')}: {exc}", file=sys.stderr)
    print(f"done: {len(selected) - failures} ok, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
