"""Pipeline progress: dynamic step numbering, and the final label survives job completion."""
from __future__ import annotations

from pathlib import Path

BACKEND = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web")

worker = BACKEND / "pipeline_worker.py"
text = worker.read_text(encoding="utf-8")
OLD = '''STAGE_LABELS = {
    "analysis": "1/5 和声分析",
    "stems": "2/5 音轨分离",
    "vocal_parts": "3/5 人声细分",
    "guitar_parts": "4/5 吉他细分",
    "drums_parts": "4/5 鼓组细分",
    "fusion": "5/5 分轨融合和声",
}'''
NEW = '''STAGE_LABELS = {
    "analysis": "和声分析",
    "stems": "音轨分离",
    "vocal_parts": "人声细分",
    "guitar_parts": "吉他细分",
    "drums_parts": "鼓组细分",
    "fusion": "分轨融合和声",
}'''
assert text.count(OLD) == 1
text = text.replace(OLD, NEW)

OLD_LABEL = '''        label = STAGE_LABELS.get(key, key)'''
NEW_LABEL = '''        label = f"{index}/{len(steps)} {STAGE_LABELS.get(key, key)}"'''
assert text.count(OLD_LABEL) == 1
text = text.replace(OLD_LABEL, NEW_LABEL)
worker.write_text(text, encoding="utf-8")
print("pipeline_worker: dynamic step numbering")

jobs = BACKEND / "jobs.py"
text = jobs.read_text(encoding="utf-8")
OLD_DONE = '''                else:
                    project.update(current_run=job_id)
                    job.update(status="completed", stage="complete", stage_label="分析完成",
                               finished_at=now())'''
NEW_DONE = '''                else:
                    project.update(current_run=job_id)
                    label = "完整流程完成" if job.get("profile") == "pipeline" else "分析完成"
                    job.update(status="completed", stage="complete", stage_label=label,
                               finished_at=now())'''
assert text.count(OLD_DONE) == 1, text.count(OLD_DONE)
jobs.write_text(text.replace(OLD_DONE, NEW_DONE), encoding="utf-8")
print("jobs.py: pipeline completion label")
