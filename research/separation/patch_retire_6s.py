"""Retire htdemucs_6s (dominated by bs_roformer_sw) and fix the package label."""
from __future__ import annotations

from pathlib import Path

WORKER = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\stems_worker.py")
text = WORKER.read_text(encoding="utf-8")

OLD = '    "htdemucs_6s": dict(id="htdemucs_6s", label="六轨", note="额外分出吉他/钢琴，整体质量弱于四轨模型", stems=6),\n'
assert text.count(OLD) == 1, text.count(OLD)
text = text.replace(OLD, "")

OLD2 = 'package=("audio-separator 0.48" if model == "bs_roformer" else "demucs 4.1.0"),'
NEW2 = 'package=("audio-separator 0.48" if model.startswith("bs_roformer") else "demucs 4.1.0"),'
assert text.count(OLD2) == 1
text = text.replace(OLD2, NEW2)
WORKER.write_text(text, encoding="utf-8")
print("retired htdemucs_6s; package label fixed")
