"""Label the manifest package per model instead of always saying demucs."""
from __future__ import annotations

from pathlib import Path

WORKER = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\stems_worker.py")
text = WORKER.read_text(encoding="utf-8")
OLD = 'package="demucs 4.1.0",'
NEW = 'package=("audio-separator 0.48" if model == "bs_roformer" else "demucs 4.1.0"),'
assert text.count(OLD) == 1, text.count(OLD)
WORKER.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("package label fixed")
