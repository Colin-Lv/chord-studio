"""Show the selected checkpoint in the separation stage label."""
from __future__ import annotations

from pathlib import Path

WORKER = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\stems_worker.py")
text = WORKER.read_text(encoding="utf-8")
OLD = '''    stage("separate", "正在分离音轨（htdemucs，4 轨）", separate)'''
NEW = '''    stage("separate", f"正在分离音轨（{MODELS[model]['label']} · 4 轨）", separate)'''
assert text.count(OLD) == 1
WORKER.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("stage label now includes the model")
