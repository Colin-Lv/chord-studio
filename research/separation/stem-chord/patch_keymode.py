"""Allow the pipeline job to request the continuous key mode (it runs a retained10 analysis)."""
from __future__ import annotations

from pathlib import Path

JOBS = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\jobs.py")
text = JOBS.read_text(encoding="utf-8")
OLD = 'if profile != "retained10" and profile != "stems" and key_mode == "continuous":'
NEW = 'if profile not in {"retained10", "stems", "pipeline"} and key_mode == "continuous":'
assert text.count(OLD) == 1, text.count(OLD)
JOBS.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("key_mode guard allows pipeline")
