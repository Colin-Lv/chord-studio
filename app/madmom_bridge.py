#!/usr/bin/env python
"""Run madmom chord recognition through an isolated Python environment.

The main Chord Studio environment decodes the source audio, writes a temporary
44.1 kHz numpy array, and asks a separate madmom environment to run CNN+CRF
chord recognition. The output is a .lab file that can be passed to
chord_studio.py with --external-chords.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from chord_recognizer import decode_audio


DEFAULT_MADMOM_PYTHON = (
    Path.home() / ".codex" / "acr_envs" / "madmom39" / "Scripts" / "python.exe"
)
WORKSPACE_TMP = Path(__file__).resolve().parent.parent / "tmp" / "madmom_bridge"

MADMOM_HELPER = r"""
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor
from madmom.processors import SequentialProcessor

audio_array = np.load(sys.argv[1])
output_path = Path(sys.argv[2])
output_path.parent.mkdir(parents=True, exist_ok=True)

processor = SequentialProcessor([CNNChordFeatureProcessor(), CRFChordRecognitionProcessor()])
segments = processor(audio_array)

with output_path.open("w", encoding="utf-8") as handle:
    for start, end, label in segments:
        handle.write(f"{float(start):.3f} {float(end):.3f} {label}\n")

print(f"Wrote: {output_path}")
print(f"Chord segments: {len(segments)}")
"""


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a madmom chord .lab file for Chord Studio fusion."
    )
    parser.add_argument("audio", type=Path, help="Input audio file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .lab path. Defaults to <audio>_madmom_chords.lab.",
    )
    parser.add_argument(
        "--madmom-python",
        type=Path,
        default=DEFAULT_MADMOM_PYTHON,
        help="Python executable where madmom is installed.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate sent to madmom. Keep 44100 unless you know otherwise.",
    )
    parser.add_argument("--offset", type=float, default=0.0, help="Decode offset in seconds.")
    parser.add_argument("--duration", type=float, default=None, help="Decode duration in seconds.")
    return parser.parse_args(argv)


def default_output_path(audio_path: Path) -> Path:
    return audio_path.with_name(f"{audio_path.stem}_madmom_chords.lab")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    audio_path = args.audio.resolve()
    output_path = (
        args.output.resolve() if args.output is not None else default_output_path(audio_path).resolve()
    )
    madmom_python = args.madmom_python.resolve()

    if not audio_path.exists():
        print(f"Input file not found: {audio_path}", file=sys.stderr)
        return 2
    if not madmom_python.exists():
        print(f"madmom Python not found: {madmom_python}", file=sys.stderr)
        return 2
    if args.sample_rate <= 0:
        print("--sample-rate must be greater than 0.", file=sys.stderr)
        return 2

    print(f"Decoding for madmom: {audio_path.name}")
    y = decode_audio(
        audio_path,
        sample_rate=args.sample_rate,
        offset=args.offset,
        duration=args.duration,
    ).astype(np.float32)
    if y.size == 0:
        print("Decoded audio is empty.", file=sys.stderr)
        return 2

    WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
    npy_path = WORKSPACE_TMP / f"{output_path.stem}_audio_44100.npy"
    np.save(npy_path, y)
    command = [str(madmom_python), "-c", MADMOM_HELPER, str(npy_path), str(output_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    try:
        npy_path.unlink()
    except OSError:
        pass
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
        return result.returncode
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
