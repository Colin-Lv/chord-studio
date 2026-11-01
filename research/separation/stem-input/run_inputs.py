"""Run the frozen v90 frontend on each input variant, then decode with the app's full decoder.

Same weights, same beats (the song's beat grid is not a stem property), everything else
recomputed from the substituted audio.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BACKEND = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend")
sys.path.insert(0, str(BACKEND))

from app.web.fusion_engine import full_decode_segments  # noqa: E402
from app.web.storage import read_json  # noqa: E402

WORK = Path(r"D:\ChordStudio\research\separation\stem-input")
INPUTS = WORK / "inputs"
FRONT = WORK / "front"
MIX_RUN = Path(r"D:\ChordStudio\data\projects\aa5a41797bb343e590abcf4d8b27ae54\runs\75ad217baca0484c83fa23f7f0f75442")
CACHE = Path(r"D:\ChordStudio\data\audio-cache")
PY = sys.executable


def run_frontend(audio: Path, out: Path) -> Path:
    if (out / "posterior.npz").is_file():
        return out
    if out.exists():
        raise SystemExit(f"clean {out}")
    command = [PY, "-I", "-B", str(BACKEND / "inference_windowless.py"), "--audio", str(audio),
               "--output-dir", str(out), "--cache-dir", str(CACHE), "--max-seconds", "600",
               "--memory-budget-mib", "1500"]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=str(BACKEND), capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if result.returncode:
        print(result.stdout[-1200:], result.stderr[-1200:])
        raise SystemExit(f"frontend failed for {audio.name}")
    print(f"  frontend {audio.name}: {elapsed:.1f}s")
    return out


def decode_condition(name: str, folder: Path) -> list:
    prediction = read_json(folder / "prediction.json")
    cache = Path(prediction["cache_directory"])
    posterior = np.load(folder / "posterior.npz")["chord"]
    cells = read_json(cache / "cells.json")
    deep = np.load(cache / "deep.npz")
    notes = np.load(cache / "notes.npz")
    beats_rows = read_json(MIX_RUN / "beats.json")
    from app.web.full_decode import decode as decode_full
    rows = decode_full(
        posterior=posterior,
        starts=np.array([cell["start"] for cell in cells], dtype=float),
        ends=np.array([cell["end"] for cell in cells], dtype=float),
        chroma=deep["chroma"], act=notes["activations"], note_times=notes["times"],
        beats=np.array([float(row["time"]) for row in beats_rows], dtype=float),
        beat_pos=np.array([int(row.get("beat", 0)) for row in beats_rows], dtype=int))
    out = [dict(start=float(a), end=float(b), label=str(c)) for a, b, c in rows]
    (WORK / f"{name}.lab").write_text(
        "".join(f"{row['start']:.17g} {row['end']:.17g} {row['label']}\n" for row in out),
        encoding="utf-8", newline="\n")
    return out


def main():
    FRONT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for audio in sorted(INPUTS.glob("*.wav")):
        name = audio.stem
        print(f"== {name}")
        folder = run_frontend(audio, FRONT / name)
        rows = decode_condition(name, folder)
        summary[name] = dict(segments=len(rows), seconds=round(rows[-1]["end"], 1))
        print(f"  segments {len(rows)}")
    (WORK / "decode-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                              encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
