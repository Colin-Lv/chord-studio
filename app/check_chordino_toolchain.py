#!/usr/bin/env python3
"""Check whether Sonic Annotator and Chordino are available."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_TRANSFORM_ID = "vamp:nnls-chroma:chordino:simplechord"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sonic-annotator",
        type=Path,
        default=Path("tools/sonic-annotator-1.7-win64/sonic-annotator-win64/sonic-annotator.exe"),
    )
    parser.add_argument("--transform-id", default=DEFAULT_TRANSFORM_ID)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sonic_annotator = args.sonic_annotator.resolve()
    if not sonic_annotator.exists():
        print(f"missing sonic-annotator: {sonic_annotator}")
        return 2

    version = subprocess.run(
        [str(sonic_annotator), "--version"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print(f"sonic-annotator: {sonic_annotator}")
    print(f"version: {(version.stdout or version.stderr).strip()}")

    listing = subprocess.run(
        [str(sonic_annotator), "--list"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if listing.returncode != 0:
        print("could not list Vamp transforms")
        print(listing.stderr.strip())
        return 2
    matches = [
        line.strip()
        for line in listing.stdout.splitlines()
        if "chord" in line.lower() or "nnls" in line.lower()
    ]
    if matches:
        print("matching transforms:")
        for line in matches:
            print(f"  {line}")
    else:
        print("matching transforms: none")

    if args.transform_id.lower() not in listing.stdout.lower():
        print(f"missing required transform: {args.transform_id}")
        return 1
    print(f"ready: {args.transform_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
