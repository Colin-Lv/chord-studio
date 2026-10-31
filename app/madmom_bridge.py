#!/usr/bin/env python
"""Run madmom chord recognition through an isolated Python environment.

The main Chord Studio environment decodes the source audio, writes a temporary
44.1 kHz numpy array, and asks a separate madmom environment to run CNN+CRF
chord recognition. The output is a .lab file that can be passed to
chord_studio.py with --external-chords.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from chord_recognizer import decode_audio


DEFAULT_MADMOM_PYTHON = (
    Path.home() / ".codex" / "acr_envs" / "madmom39" / "Scripts" / "python.exe"
)
WORKSPACE_TMP = Path(__file__).resolve().parent.parent / "tmp" / "madmom_bridge"


def provenance_path(lab: Path) -> Path:
    return lab.with_name(lab.name + ".provenance.json")


def file_identity(path: Path) -> dict:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OSError(f"File changed while hashing: {path}")
    return {"path": str(path.resolve()), "bytes": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": digest.hexdigest()}


def cache_matches_audio(lab: Path, audio: Path) -> tuple[bool, str]:
    """Only auto-reuse a complete, full-source cache with matching contents."""
    try:
        record = json.loads(provenance_path(lab).read_text(encoding="utf-8"))
        if record.get("schema") != "madmom_chord_cache_v1" or record.get("offset") != 0 or record.get("requested_duration") is not None:
            return False, "cache is not a documented complete-source analysis"
        if record.get("sample_rate") != 44100:
            return False, "cache sample rate is incompatible"
        if record.get("audio", {}).get("sha256") != file_identity(audio)["sha256"]:
            return False, "audio content changed"
        if record.get("lab", {}).get("sha256") != file_identity(lab)["sha256"]:
            return False, "LAB content changed"
        return True, "audio and LAB SHA256 match"
    except (OSError, ValueError, TypeError, AttributeError):
        return False, "cache provenance is absent or invalid"

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
        help="Fixed at 44100: the pretrained frontend assumes this rate for ndarray input.",
    )
    parser.add_argument("--offset", type=float, default=0.0, help="Decode offset in seconds.")
    parser.add_argument("--duration", type=float, default=None, help="Decode duration in seconds.")
    return parser.parse_args(argv)


def default_output_path(audio_path: Path) -> Path:
    return audio_path.with_name(f"{audio_path.stem}_madmom_chords.lab")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.sample_rate != 44100:
        print("--sample-rate must be 44100 for the pretrained madmom ndarray frontend.", file=sys.stderr)
        return 2
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

    print(f"Decoding for madmom: {audio_path.name}")
    source_identity = file_identity(audio_path)
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
    npy_path = pending_lab = pending_record = None
    try:
        with tempfile.NamedTemporaryFile(suffix="_audio_44100.npy", dir=WORKSPACE_TMP, delete=False) as handle:
            npy_path = Path(handle.name)
            np.save(handle, y)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".lab", prefix=".madmom_", dir=output_path.parent, delete=False) as handle:
            pending_lab = Path(handle.name)
        command = [str(madmom_python), "-c", MADMOM_HELPER, str(npy_path), str(pending_lab)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
        if result.returncode != 0:
            return result.returncode
        if file_identity(audio_path) != source_identity:
            print("Audio changed during decoding/inference; output was not published. Retry with a stable source.", file=sys.stderr)
            return 2
        lab_identity = file_identity(pending_lab)
        pending_lab.replace(output_path)
        lab_identity["path"] = str(output_path)
        record = {"schema": "madmom_chord_cache_v1", "audio": source_identity,
                  "lab": lab_identity, "sample_rate": args.sample_rate,
                  "offset": args.offset, "requested_duration": args.duration, "duration": len(y) / args.sample_rate,
                  "timestamp_origin": "decoded_window_start", "frontend": "madmom CNNChordFeatureProcessor + CRFChordRecognitionProcessor",
                  "runtime_python": str(madmom_python)}
        with tempfile.NamedTemporaryFile(suffix=".json", prefix=".madmom_", dir=output_path.parent,
                                         mode="w", encoding="utf-8", delete=False) as handle:
            pending_record = Path(handle.name)
            json.dump(record, handle, ensure_ascii=False, indent=2)
        pending_record.replace(provenance_path(output_path))
        print(f"Wrote: {output_path}")
    finally:
        for temporary in (npy_path, pending_lab, pending_record):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
