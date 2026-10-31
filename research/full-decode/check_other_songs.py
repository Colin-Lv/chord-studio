"""Sanity-check the shipped full decoder on the other two production songs (no golden needed).

Prints segment counts and the harmonic key implied by each source, so the second tier can be
compared with the shipped one on songs it was never tuned on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend")
sys.path.insert(0, str(BACKEND / "app" / "web"))
sys.path.insert(0, str(BACKEND / "app"))

PROJECTS = {
    "虚拟": "aa5a41797bb343e590abcf4d8b27ae54",
    "坏女孩": "0bc0304e01ed40d8baac6b76575b25b6",
    "沧海一粟": "e18b69d0edef4c5e91f6594520a816cb",
}


def latest_run(project):
    runs = sorted((Path(r"D:\ChordStudio\data\projects") / project / "runs").iterdir(),
                  key=lambda p: p.stat().st_mtime)
    for run in reversed(runs):
        if (run / "retained10" / "posterior.npz").is_file() and (run / "beats.json").is_file():
            cells = Path(json.loads((run / "retained10/prediction.json").read_text(
                encoding="utf-8"))["cache_directory"])
            if all((cells / name).is_file() for name in ("cells.json", "deep.npz", "notes.npz")):
                return run
    return None


def main():
    import full_decode as FD
    from key_from_chords import infer as infer_key, key_name, parse_rows

    for song, project in PROJECTS.items():
        run = latest_run(project)
        if run is None:
            print(f"{song}: no usable run")
            continue
        pred = json.loads((run / "retained10/prediction.json").read_text(encoding="utf-8"))
        cache = Path(pred["cache_directory"])
        post = np.load(run / "retained10/posterior.npz")
        cells = json.loads((cache / "cells.json").read_text(encoding="utf-8"))
        deep = np.load(cache / "deep.npz")
        notes = np.load(cache / "notes.npz")
        beats = json.loads((run / "beats.json").read_text(encoding="utf-8"))
        rows = FD.decode(
            posterior=post["chord"],
            starts=np.array([c["start"] for c in cells], dtype=float),
            ends=np.array([c["end"] for c in cells], dtype=float),
            chroma=deep["chroma"], act=notes["activations"], note_times=notes["times"],
            beats=np.array([float(b["time"]) for b in beats]),
            beat_pos=np.array([int(b.get("beat", 0)) for b in beats]))
        FD.write_lab(run / "retained10_full.lab", rows)
        counts = {}
        keys = {}
        for name in ("retained10_stable", "retained10_span", "retained10_full"):
            path = run / f"{name}.lab"
            if not path.is_file():
                continue
            table = FD.read_lab(path)
            counts[name] = len(table)
            parsed, _ = parse_rows([dict(start=a, end=b, label=label) for a, b, label in table])
            if len(parsed) >= 3:
                ev = infer_key(parsed)
                keys[name] = key_name(ev["pred"][0], ev["pred"][1])
            else:
                keys[name] = None
        key_json = run / "key.json"
        acoustic = json.loads(key_json.read_text(encoding="utf-8")).get("key") if key_json.is_file() else None
        print(f"{song} (run {run.name[:8]}, audio {pred['audio_seconds']:.0f}s)")
        print(f"  segments {counts}")
        print(f"  harmonic key {keys}   acoustic key {acoustic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
