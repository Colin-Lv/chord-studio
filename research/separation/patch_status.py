"""Show separation as its own project status instead of '分析中'."""
from __future__ import annotations

from pathlib import Path

BACKEND = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web")
JOBS = BACKEND / "jobs.py"
text = JOBS.read_text(encoding="utf-8")
OLD = """        with self.lock:
            project = self.store.project(project_id)
            project["status"] = "running"
            self.store.save_project(project)
"""
NEW = """        with self.lock:
            project = self.store.project(project_id)
            project["status"] = "stems" if job.get("profile") == "stems" else "running"
            self.store.save_project(project)
"""
assert text.count(OLD) == 1, text.count(OLD)
JOBS.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("jobs.py: stems status")

APP = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\frontend\app.js")
text = APP.read_text(encoding="utf-8")
OLD_NAMES = 'statusNames = {ready:"待分析",queued:"排队中",running:"分析中",completed:"已完成",failed:"失败，可重试",interrupted:"已中断"};'
NEW_NAMES = 'statusNames = {ready:"待分析",queued:"排队中",running:"分析中",stems:"分轨中",completed:"已完成",failed:"失败，可重试",interrupted:"已中断"};'
assert text.count(OLD_NAMES) == 1, text.count(OLD_NAMES)
APP.write_text(text.replace(OLD_NAMES, NEW_NAMES), encoding="utf-8")
print("app.js: stems status label")
