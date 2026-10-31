"""Recompute the shipped full decoder from the backend module and score it on the golden.

This is the acceptance script: it imports the module the app actually runs
(app/web/full_decode.py) plus the model the app actually loads (app/web/models), rebuilds the
timeline for a real run, writes `<run>/retained10_full.lab` and prints the metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\ChordStudio")
BACKEND = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend")
sys.path.insert(0, str(BACKEND / "app" / "web"))
sys.path.insert(0, str(BACKEND / "app"))

GOLDEN = ROOT / "research/key-v3/chord_seg/reference_golden.lab"


def load_lab(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            a, b, label = line.split(maxsplit=2)
            rows.append((float(a), float(b), label.strip()))
    return rows


def frame_acc(pred, ref, step=0.02):
    end = min(pred[-1][1], ref[-1][1])
    grid = np.arange(ref[0][0], end, step)

    def at(table, t):
        for row in table:
            if row[0] <= t < row[1]:
                return row[2]
        return "N"

    exact = family = 0
    for t in grid:
        p, q = at(pred, t), at(ref, t)
        exact += p == q
        pr, _, pq = p.partition(":")
        qr, _, qq = q.partition(":")
        family += (pr == qr) and (pq[:3] == qq[:3])
    return exact / len(grid), family / len(grid)


def boundary_f1(pred, truth, tol=0.06):
    pb = [a for a, _, _ in pred[1:]]
    tb = [a for a, _, _ in truth[1:]]
    hp = sum(1 for p in pb if any(abs(p - t) <= tol for t in tb))
    ht = sum(1 for t in tb if any(abs(p - t) <= tol for p in pb))
    prec, rec = hp / max(1, len(pb)), ht / max(1, len(tb))
    return prec, rec, 2 * prec * rec / max(1e-9, prec + rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    runs = sorted((ROOT / "data/projects/aa5a41797bb343e590abcf4d8b27ae54/runs").iterdir(),
                  key=lambda p: p.stat().st_mtime)
    run = Path(args.run) if args.run else runs[-1]
    prediction = json.loads((run / "retained10/prediction.json").read_text(encoding="utf-8"))
    cache = Path(prediction["cache_directory"])
    post = np.load(run / "retained10/posterior.npz")
    cells = json.loads((cache / "cells.json").read_text(encoding="utf-8"))
    deep = np.load(cache / "deep.npz")
    notes = np.load(cache / "notes.npz")
    beats_raw = json.loads((run / "beats.json").read_text(encoding="utf-8"))

    import full_decode as FD

    rows = FD.decode(
        posterior=post["chord"],
        starts=np.array([c["start"] for c in cells], dtype=float),
        ends=np.array([c["end"] for c in cells], dtype=float),
        chroma=deep["chroma"],
        act=notes["activations"],
        note_times=notes["times"],
        beats=np.array([float(b["time"]) for b in beats_raw], dtype=float),
        beat_pos=np.array([int(b.get("beat", 0)) for b in beats_raw], dtype=int),
    )
    out = run / "retained10_full.lab"
    FD.write_lab(out, rows)
    ref = load_lab(GOLDEN)
    ex, fam = frame_acc(rows, ref)
    p, r, f1 = boundary_f1(rows, ref)
    print(f"run {run.name}")
    print(f"full decoder -> {out.name}")
    print(f"  segments {len(rows):3d} (truth {len(ref)})  exact {ex:.3f}  family {fam:.3f}")
    print(f"  boundary P {p:.3f}  R {r:.3f}  F1 {f1:.3f}")
    ok_exact = ex >= 0.50
    ok_f1 = f1 >= 0.90
    print(f"  exact >= 0.50 : {'PASS' if ok_exact else 'FAIL'}")
    print(f"  F1    >= 0.90 : {'PASS' if ok_f1 else 'FAIL'}")

    # production wiring: the full decoder is the published default, bar pooling stays
    # available as a switchable source, and the stage records that it ran.
    analysis_path = run / "analysis.json"
    wired = dict(source_published=False, default_is_full=None, span_available=None,
                 provenance_applied=None, panel_rows=None)
    if analysis_path.is_file():
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        ids = [s["id"] for s in analysis.get("sources", [])]
        wired["source_published"] = "retained10_full" in ids
        wired["default_is_full"] = analysis.get("default_source") == "retained10_full"
        wired["span_available"] = "retained10_span" in ids
        prov = (analysis.get("provenance") or {}).get("full_decode") or {}
        wired["provenance_applied"] = bool(prov.get("applied"))
        row = next((s for s in analysis["sources"] if s["id"] == "retained10_full"), None)
        wired["panel_rows"] = len(row["rows"]) if row else None
    print(f"  published as a source          : {'PASS' if wired['source_published'] else 'FAIL'}")
    print(f"  default source is the decoder  : "
          f"{'PASS' if wired['default_is_full'] else 'FAIL'}")
    print(f"  bar pooling still selectable   : "
          f"{'PASS' if wired['span_available'] else 'FAIL'}")
    print(f"  stage recorded applied         : "
          f"{'PASS' if wired['provenance_applied'] else 'FAIL'}")
    ok_wire = all(v is True for k, v in wired.items() if k != "panel_rows")
    print("  overall:", "PASS" if (ok_exact and ok_f1 and ok_wire) else "FAIL")
    (Path(__file__).resolve().parent / "verify_prod_full.json").write_text(
        json.dumps(dict(run=run.name, lab=str(out), segments=len(rows), exact=ex, family=fam,
                        precision=p, recall=r, f1=f1, pass_exact=ok_exact, pass_f1=ok_f1,
                        wiring=wired, pass_wiring=ok_wire),
                   indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
