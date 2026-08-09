#!/usr/bin/env python3
"""Render POP909 MIDI files to simple synthesized audio and update a manifest."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf


def safe_id(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def resolve_path(base: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else base / path


def render_midi(source: Path, target: Path, sample_rate: int, normalize_peak: float) -> None:
    midi = pretty_midi.PrettyMIDI(str(source))
    audio = midi.synthesize(fs=sample_rate)
    if audio.size == 0:
        raise ValueError(f"empty render: {source}")
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-9:
        audio = audio * (normalize_peak / peak)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, audio, sample_rate)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("pop909_dir", type=Path)
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("benchmarks/audio/pop909_synth"),
    )
    parser.add_argument(
        "-o",
        "--output-manifest",
        type=Path,
        default=Path("benchmarks/manifest.pop909_synth.csv"),
    )
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--normalize-peak", type=float, default=0.85)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--id-contains")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--copy-midi", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    manifest_base = manifest.parent
    pop909_dir = args.pop909_dir.resolve()
    audio_dir = args.audio_dir.resolve()
    rows = load_manifest(manifest)
    if not rows:
        print("manifest has no rows", file=sys.stderr)
        return 1
    fieldnames = list(rows[0].keys())
    selected = []
    for row in rows:
        row_id = row.get("id", "")
        if args.id_contains and args.id_contains not in row_id:
            continue
        selected.append(row)
        if args.limit and len(selected) >= args.limit:
            break

    done = 0
    failed = 0
    for index, row in enumerate(selected, start=1):
        row_id = row.get("id", "")
        song_number = row_id.replace("pop909_", "")
        midi_path = pop909_dir / song_number / f"{song_number}.mid"
        audio_path = audio_dir / f"{song_number}.wav"
        print(f"[{index}/{len(selected)}] {row_id}")
        if not midi_path.exists():
            failed += 1
            print(f"missing MIDI: {midi_path}", file=sys.stderr)
            continue
        try:
            if args.force or not audio_path.exists():
                render_midi(midi_path, audio_path, args.sample_rate, args.normalize_peak)
            if args.copy_midi:
                shutil.copy2(midi_path, audio_dir / f"{song_number}.mid")
            row["audio"] = str(audio_path.relative_to(manifest_base))
            row["notes"] = (row.get("notes", "") + "; synthesized from POP909 MIDI").strip("; ")
            done += 1
        except Exception as exc:
            failed += 1
            print(f"FAILED {row_id}: {exc}", file=sys.stderr)

    write_manifest(args.output_manifest.resolve(), rows, fieldnames)
    print(f"Wrote: {args.output_manifest}")
    print(f"rendered/linked: {done}")
    print(f"failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
