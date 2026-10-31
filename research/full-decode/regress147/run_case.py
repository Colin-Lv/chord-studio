"""One 147-case regression case, through the exact production modules.

Stages mirrored from web/retained_worker.execute():
  inference_windowless  -> frozen retained10 prediction + audio cache
  key_bridge            -> madmom CNN key
  downbeat_bridge       -> beats.json
  stabilize_lab         -> retained10_stable
  span_pool.pool_lab    -> retained10_span
  key_from_chords + key_decision -> final key under each chord source

Usage:  python run_case.py --case 1007941
Writes <out>/<case>/result.json and is resumable.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BACKEND = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend")
APP = BACKEND / "app"
WEB = APP / "web"
import os as _os
CACHE = Path(_os.environ.get("REGRESS_CACHE", r"D:\ChordStudio\data\audio-cache"))
HERE = Path(__file__).resolve().parent
for p in (str(APP), str(WEB)):
    if p not in sys.path:
        sys.path.insert(0, p)

NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def key_id(label):
    if not label:
        return None
    body = str(label).strip()
    mode = "minor" if body.lower().endswith("minor") or (
        body.endswith("m") and not body.endswith("maj")) else "major"
    root = body.replace(" Major", "").replace(" minor", "").replace(" major", "").strip()
    if root.endswith("m"):
        root = root[:-1]
    if not root or root[0] not in NOTE_PC:
        return None
    pc = NOTE_PC[root[0]]
    for ch in root[1:]:
        pc += 1 if ch == "#" else -1
    return [pc % 12, mode]


def run(command, timeout=1800):
    proc = subprocess.run([str(c) for c in command], cwd=str(BACKEND), capture_output=True,
                          timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW
                          if os.name == "nt" else 0)
    if proc.returncode:
        name = Path(str(command[2])).name if len(command) > 2 else str(command[0])
        raise RuntimeError(f"{name} failed: {proc.stderr.decode('utf-8', errors='replace')[-1500:]}")
    return proc


def key_stages(rows):
    from key_from_chords import infer as infer_key, key_name, parse_rows
    from key_decision import guard_456

    parsed, _ = parse_rows([dict(start=r["start"], end=r["end"], label=r["label"]) for r in rows])
    if len(parsed) < 3:
        return dict(error="too few chords", chords=len(parsed)), []
    evidence = infer_key(parsed)
    pc, mode = evidence["pred"]
    raw_key = key_name(pc, mode)
    guarded, guard_info = guard_456(parsed, raw_key)
    return dict(chord_key=guarded, raw_key=raw_key, mode=mode,
                margin=round(float(evidence["pred_pair"]["margin"]), 4),
                chords=len(parsed), guard=guard_info), parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    manifest = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    entry = manifest[args.case]
    audio = Path(entry["audio"])
    out = HERE / "cases" / args.case
    result_path = out / "result.json"
    if result_path.exists() and not args.force:
        print("skip (done)", args.case)
        return
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run_dir = out / "run"
    run_dir.mkdir(exist_ok=True)
    pred_dir = run_dir / "retained10"
    python = sys.executable

    if not (pred_dir / "prediction.json").exists():
        # the inference entry insists on creating its own output directory
        if pred_dir.exists():
            shutil.rmtree(pred_dir)
        run([python, "-I", "-B", str(BACKEND / "inference_windowless.py"),
             "--audio", audio, "--output-dir", pred_dir, "--cache-dir", CACHE,
             "--max-seconds", "600", "--memory-budget-mib", "512"], timeout=2400)
    prediction = json.loads((pred_dir / "prediction.json").read_text(encoding="utf-8"))
    cache = Path(prediction["cache_directory"])
    shutil.copyfile(pred_dir / "prediction.lab", run_dir / "retained10.lab")

    key_path = run_dir / "key.json"
    if not key_path.exists():
        run([python, "-B", str(APP / "key_bridge.py"), audio, "-o", key_path], timeout=900)
    beats_path = run_dir / "beats.json"
    if not beats_path.exists():
        run([python, "-B", str(APP / "downbeat_bridge.py"), audio, "-o", beats_path,
             "--timeout", "900"], timeout=1200)
    beats = json.loads(beats_path.read_text(encoding="utf-8")) if beats_path.exists() else []
    key = json.loads(key_path.read_text(encoding="utf-8")) if key_path.exists() else None
    downbeats = [float(r["time"]) for r in beats if int(r.get("beat", 0)) == 1] if beats else []

    from source_rows import lab_rows
    from stabilize_timeline import stabilize_lab
    from span_pool import pool_lab
    import numpy as np

    stable_path = run_dir / "retained10_stable.lab"
    span_path = run_dir / "retained10_span.lab"
    for path in (stable_path, span_path):
        if path.exists():
            path.unlink()
    cells = cache / "cells.json"
    stabilize_lab(lab_path=run_dir / "retained10.lab",
                  posterior_path=pred_dir / "posterior.npz",
                  cells_path=cells, output_path=stable_path, downbeats=downbeats)
    chroma = None
    deep = cache / "deep.npz"
    if deep.is_file():
        chroma = np.load(deep)["chroma"]
    pool_lab(stable_path, span_path, beats, chroma=chroma)

    from key_decision import decide
    final = {}
    for name, path in (("stable", stable_path), ("span", span_path)):
        rows = lab_rows(path)
        ev, parsed = key_stages(rows)
        item = dict(rows=len(rows), evidence=ev)
        if key and parsed:
            d = decide(key.get("key"), ev["chord_key"], ev["margin"], parsed)
            item["decision"] = d
            item["final_key"] = d.get("chord_key") if d.get("use_chord") else key.get("key")
        elif key:
            # chord evidence unavailable (e.g. one pooled segment): the published key stays
            # the acoustic key, exactly as the worker leaves it.
            item["final_key"] = key.get("key")
            item["fallback"] = "acoustic key (chord evidence unavailable)"
        final[name] = item

    ref = entry.get("reference")
    result = dict(case=args.case, audio=str(audio), reference=ref, reference_id=key_id(ref),
                  madmom_key=(key or {}).get("key"), madmom_key_id=key_id((key or {}).get("key")),
                  duration=prediction["audio_seconds"], beats=len(beats),
                  downbeats=len(downbeats), stable=final["stable"], span=final["span"],
                  elapsed_seconds=round(time.time() - t0, 1))
    result_path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{args.case}: stable {final['stable']['rows']} rows key {final['stable'].get('final_key')} | "
          f"span {final['span']['rows']} rows key {final['span'].get('final_key')} | ref {ref} | "
          f"{result['elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    main()

