"""Wire the stems job into the worker dispatch, the job queue and the HTTP API."""
from __future__ import annotations

from pathlib import Path

BACKEND = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web")

# ---------------------------------------------------------------- worker.py dispatch
worker = BACKEND / "worker.py"
text = worker.read_text(encoding="utf-8")
OLD_MAIN = """    store = Store(args.store)
    try:
        execute(store, args.job)
"""
NEW_MAIN = """    store = Store(args.store)
    try:
        job = read_json(store.job_path(args.job))
        if job.get("profile") == "stems":
            from .stems_worker import execute as execute_stems
            execute_stems(store, args.job)
        else:
            execute(store, args.job)
"""
assert text.count(OLD_MAIN) == 1, text.count(OLD_MAIN)
worker.write_text(text.replace(OLD_MAIN, NEW_MAIN), encoding="utf-8")
print("worker.py: stems dispatch added")

# ---------------------------------------------------------------- jobs.py
jobs = BACKEND / "jobs.py"
text = jobs.read_text(encoding="utf-8")

OLD_SUBMIT = """    def submit(self, project_id, profile, download=False, key_mode="madmom"):
        if key_mode not in {"madmom", "fused", "continuous"}:
            raise ValueError("未知主调分析模式")
        if profile == "retained10" and key_mode == "fused":
            raise ValueError("retained10 的主调支持 CNN 或连续评分（实验）；分歧裁决仅适用于旧模型。")
        if profile != "retained10" and key_mode == "continuous":
            raise ValueError("连续评分主调（实验）仅适用于当前冻结模型。")
        if profile not in PROFILES:
            raise ValueError("未知模型选项")
"""
NEW_SUBMIT = """    def submit(self, project_id, profile, download=False, key_mode="madmom",
               force=False, model=None):
        if profile != "stems" and key_mode not in {"madmom", "fused", "continuous"}:
            raise ValueError("未知主调分析模式")
        if profile == "retained10" and key_mode == "fused":
            raise ValueError("retained10 的主调支持 CNN 或连续评分（实验）；分歧裁决仅适用于旧模型。")
        if profile != "retained10" and profile != "stems" and key_mode == "continuous":
            raise ValueError("连续评分主调（实验）仅适用于当前冻结模型。")
        if profile not in PROFILES and profile != "stems":
            raise ValueError("未知模型选项")
"""
assert text.count(OLD_SUBMIT) == 1, text.count(OLD_SUBMIT)
text = text.replace(OLD_SUBMIT, NEW_SUBMIT)

OLD_JOB = """            job = dict(id=uuid4().hex, project_id=project_id, profile=profile, download=download, key_mode=key_mode,
                       status="queued", stage="queued", stage_label="等待前一项任务完成", stages=[],
                       created_at=now(), error=None)
"""
NEW_JOB = """            job = dict(id=uuid4().hex, project_id=project_id, profile=profile, download=download, key_mode=key_mode,
                       force=bool(force), model=model,
                       status="queued", stage="queued", stage_label="等待前一项任务完成", stages=[],
                       created_at=now(), error=None)
"""
assert text.count(OLD_JOB) == 1, text.count(OLD_JOB)
text = text.replace(OLD_JOB, NEW_JOB)

OLD_DONE = """            with self.lock:
                project = self.store.project(project_id)
                project.update(current_run=job_id, active_job=None, status="completed")
                self.store.save_project(project)
                job = read_json(path)
                job.update(status="completed", stage="complete", stage_label="分析完成", finished_at=now())
                atomic_json(path, job)
"""
NEW_DONE = """            with self.lock:
                project = self.store.project(project_id)
                job = read_json(path)
                project.update(active_job=None, status="completed")
                if job.get("profile") == "stems":
                    # Separation produces a side asset: it must never replace the analysis run.
                    project.update(stems=dict(job_id=job_id, updated_at=now()))
                    job.update(status="completed", stage="complete", stage_label="分轨完成",
                               finished_at=now())
                else:
                    project.update(current_run=job_id)
                    job.update(status="completed", stage="complete", stage_label="分析完成",
                               finished_at=now())
                self.store.save_project(project)
                atomic_json(path, job)
"""
assert text.count(OLD_DONE) == 1, text.count(OLD_DONE)
jobs.write_text(text.replace(OLD_DONE, NEW_DONE), encoding="utf-8")
print("jobs.py: stems profile wired")
