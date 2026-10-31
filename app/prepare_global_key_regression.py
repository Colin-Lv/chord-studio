"""Recompute neutral key evidence from audio and cached acoustic beat grids.

This prepares known regression data, never trains/tunes fusion. Grid JSON must
state beat_source=madmom; chord labels and keys in that JSON are not passed to
the new neutral run. CNN records follow <id>_key.json beside --key-report.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evaluate_global_key_fusion import signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-report", type=Path, required=True)
    parser.add_argument("--grid-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    grids = {}
    for path in args.grid_dir.glob("*_base_grid_analysis.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("beat_source") == "madmom":
            grids[str(Path(data["source"]).resolve()).casefold()] = (path, data)
    rows, runs = [], []
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    for i, song in enumerate(json.loads(args.key_report.read_text(encoding="utf-8"))["songs"]):
        audio = Path(song["source"]).resolve()
        source_grid, grid = grids[str(audio).casefold()]
        beats = [{"time":b["start"], "beat":b["beat"]} for bar in grid["bars"] for b in bar["beats"]]
        folder = args.output / f"{i:03d}"
        folder.mkdir(exist_ok=True)
        beat_path = folder / "acoustic_beats.json"
        beat_path.write_text(json.dumps(beats), encoding="utf-8")
        command = [sys.executable, str(Path(__file__).parent / "chord_studio.py"), str(audio),
                   "-o", str(folder / "neutral"), "--downbeats", str(beat_path), "--feature-mode", "fast",
                   "--analysis-model", "template", "--mode", "seventh", "--all-key-candidates", "--no-html", "--no-structure"]
        start = time.perf_counter()
        with (folder / "neutral.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        runs.append(dict(id=song["id"], command=command, elapsed_seconds=time.perf_counter()-start,
                         source_grid=signature(source_grid), beats=signature(beat_path), audio=signature(audio)))
        rows.append(dict(id=song["id"], audio=str(audio), reference=song["reference"],
                         cnn=str((args.key_report.parent / (song["id"]+"_key.json")).resolve()),
                         raw=str((args.raw_dir / (song["id"]+".lab")).resolve()),
                         neutral=str((folder / "neutral_analysis.json").resolve())))
        print(f"Neutral {i+1}: {song['id']}", flush=True)
    with (args.output / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "preparation.json").write_text(json.dumps(dict(runs=runs, acoustic_inference=True,
        new_frontend="studio fast chroma + bass + functional_joint", reused_models="CNN and raw LAB and acoustic beats",
        training=False, reference_used_in_inference=False), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
