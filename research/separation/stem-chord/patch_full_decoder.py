"""Decode the fused posterior with the app's own full decoder (same as retained10_full)."""
from __future__ import annotations

from pathlib import Path

ENGINE = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\fusion_engine.py")
text = ENGINE.read_text(encoding="utf-8")

HELPER = '''def full_decode_segments(run_dir: Path, probability) -> list[dict]:
    """Run the app's full decoder on an arbitrary posterior (bar grid + notes from the mix run)."""
    from .full_decode import decode as decode_full

    prediction = read_json(run_dir / "retained10" / "prediction.json")
    cache = Path(prediction["cache_directory"])
    cells = read_json(cache / "cells.json")
    deep = __import__("numpy").load(cache / "deep.npz")
    notes = __import__("numpy").load(cache / "notes.npz")
    beats_rows = read_json(run_dir / "beats.json")
    count = min(len(probability), len(cells))
    rows = decode_full(
        posterior=probability[:count],
        starts=__import__("numpy").array([cell["start"] for cell in cells[:count]], dtype=float),
        ends=__import__("numpy").array([cell["end"] for cell in cells[:count]], dtype=float),
        chroma=deep["chroma"],
        act=notes["activations"], note_times=notes["times"],
        beats=__import__("numpy").array([float(row["time"]) for row in beats_rows], dtype=float),
        beat_pos=__import__("numpy").array([int(row.get("beat", 0)) for row in beats_rows], dtype=int))
    out = []
    for index, (start, end, label) in enumerate(rows):
        out.append(dict(id=index, start=round(float(start), 3), end=round(float(end), 3),
                        start_text=f"{float(start):.6f}", end_text=f"{float(end):.6f}",
                        label=str(label), candidates=[]))
    return out


def apply_to_analysis('''
OLD = '''def apply_to_analysis('''
assert text.count(OLD) == 1
text = text.replace(OLD, HELPER, 1)

OLD_CALL = '''    duration = duration_of(run_dir)
    cells = cells_for(len(probability), duration)
    segments = segments_from(labels_from(probability, nprob), cells)'''
NEW_CALL = '''    duration = duration_of(run_dir)
    cells = cells_for(len(probability), duration)
    try:
        segments = full_decode_segments(run_dir, probability)
        decoder = "full"
    except Exception:                                  # noqa: BLE001
        segments = segments_from(labels_from(probability, nprob), cells)
        decoder = "cell"'''
assert text.count(OLD_CALL) == 1
text = text.replace(OLD_CALL, NEW_CALL)

OLD_META = '''    meta = dict(created_at=now(), model=model, lambda_=lam, beta=beta,'''
NEW_META = '''    meta = dict(created_at=now(), model=model, lambda_=lam, beta=beta, decoder=decoder,'''
assert text.count(OLD_META) == 1
text = text.replace(OLD_META, NEW_META)
ENGINE.write_text(text, encoding="utf-8")
print("fusion_engine: fused posterior goes through the full decoder")
