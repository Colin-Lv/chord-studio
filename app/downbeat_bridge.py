#!/usr/bin/env python
"""Run madmom beat/downbeat tracking through the isolated madmom environment."""

from __future__ import annotations

import os
import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_MADMOM_PYTHON = (
    Path.home() / ".codex" / "acr_envs" / "madmom39" / "Scripts" / "python.exe"
)


MADMOM_DOWNBEAT_HELPER = r"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

audio_path = sys.argv[1]
beats_per_bar = [int(value) for value in sys.argv[2].split(",") if value]
fps = int(sys.argv[3])

activations = RNNDownBeatProcessor()(audio_path)
if len(sys.argv) > 4 and sys.argv[4]:
    import numpy as np
    # Export before decoding so diagnostics can inspect evidence independently.
    with open(sys.argv[4], "wb") as handle:
        np.save(handle, activations)
processor = DBNDownBeatTrackingProcessor(beats_per_bar=beats_per_bar, fps=fps)
beats = processor(activations)

print(
    json.dumps(
        {"beats": [{"time": float(row[0]), "beat": int(row[1])} for row in beats]},
        ensure_ascii=False,
    )
)
"""


def run_madmom_downbeats(
    audio_path: Path,
    beats_per_bar: list[int],
    madmom_python: Path = DEFAULT_MADMOM_PYTHON,
    fps: int = 100,
    # madmom's own LSTM runs at roughly 0.6x realtime here, so a four-minute song
    # needs ~180 s and any longer track needs more; the old 180 s cap turned a
    # slow-but-correct DBN pass into a user-visible "拍点估计失败".
    timeout: int = 900,
    activations_output: Path | None = None,
) -> list[tuple[float, int]]:
    # The pretrained RNN's spectrogram front-end is fixed at 100 fps. Changing
    # only the DBN rate changes returned seconds, rather than resampling it.
    if fps != 100:
        raise ValueError("The pretrained madmom downbeat RNN outputs 100 fps; fps must be 100")
    if not madmom_python.exists():
        raise FileNotFoundError(f"madmom Python not found: {madmom_python}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Input audio not found: {audio_path}")
    if not beats_per_bar:
        raise ValueError("beats_per_bar must not be empty")

    command = [
        str(madmom_python),
        "-c",
        MADMOM_DOWNBEAT_HELPER,
        str(audio_path),
        ",".join(str(value) for value in beats_per_bar),
        str(fps),
    ]
    if activations_output is not None:
        activations_output = activations_output.resolve()
        activations_output.parent.mkdir(parents=True, exist_ok=True)
        command.append(str(activations_output))
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "madmom downbeat tracking failed"
        raise RuntimeError(message)
    rows = json.loads(result.stdout.strip()).get("beats", [])

    return [
        (float(row["time"]), int(row["beat"]))
        for row in rows
        if float(row.get("time", -1)) >= 0 and int(row.get("beat", 0)) > 0
    ]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate madmom beat/downbeat JSON.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--beats-per-bar", default="3,4,6")
    parser.add_argument("--madmom-python", type=Path, default=DEFAULT_MADMOM_PYTHON)
    parser.add_argument("--fps", type=int, default=100)
    parser.add_argument("--activations-output", type=Path, help="Optional [frames,2] NumPy output for independent metrical diagnostics")
    parser.add_argument("--timeout", type=int, default=900, help="Seconds allowed for the madmom pass (roughly 0.6x realtime)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    beats_per_bar = [int(value) for value in args.beats_per_bar.split(",") if value.strip()]
    rows = run_madmom_downbeats(args.audio.resolve(), beats_per_bar, args.madmom_python, args.fps,
                              activations_output=args.activations_output, timeout=args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([{"time": time, "beat": beat} for time, beat in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote: {args.output}")
    print(f"Downbeats: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
