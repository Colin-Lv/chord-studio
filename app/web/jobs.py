"""Serial process queue: keep HTTP responsive and acoustic runtimes isolated."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from .storage import ROOT, PROFILES, atomic_json, now, read_json


class Jobs:
    def __init__(self, store, model_python=None):
        self.store = store
        self.store_lock = (store.root / ".server.lock").open("a+b")
        self.store_lock.seek(0, 2)
        if not self.store_lock.tell():
            self.store_lock.write(b"0")
            self.store_lock.flush()
        self.store_lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.store_lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.store_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.store_lock.close()
            raise RuntimeError("此项目目录已有工作台服务运行，请使用已打开的服务地址。") from None
        self.python = model_python or os.environ.get("HARMONY_MODEL_PYTHON") or getattr(sys, "_base_executable", sys.executable)
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="harmony-job")
        # A restart never reports a disappeared worker as still progressing.
        for path in (store.root / "jobs").glob("*.json"):
            job = read_json(path)
            if job["status"] in {"queued", "running"}:
                job.update(status="interrupted", error="服务重启中断了此任务，请重新分析。", finished_at=now())
                atomic_json(path, job)
                project = store.project(job["project_id"])
                project.update(active_job=None, status="interrupted")
                store.save_project(project)

    def submit(self, project_id, profile, download=False, key_mode="madmom"):
        if key_mode not in {"madmom", "fused", "continuous"}:
            raise ValueError("未知主调分析模式")
        if profile == "retained10" and key_mode == "fused":
            raise ValueError("retained10 的主调支持 CNN 或连续评分（实验）；分歧裁决仅适用于旧模型。")
        if profile != "retained10" and key_mode == "continuous":
            raise ValueError("连续评分主调（实验）仅适用于当前冻结模型。")
        if profile not in PROFILES:
            raise ValueError("未知模型选项")
        with self.lock:
            project = self.store.project(project_id)
            if project.get("active_job"):
                return read_json(self.store.job_path(project["active_job"]))
            job = dict(id=uuid4().hex, project_id=project_id, profile=profile, download=download, key_mode=key_mode,
                       status="queued", stage="queued", stage_label="等待前一项任务完成", stages=[],
                       created_at=now(), error=None)
            atomic_json(self.store.job_path(job["id"]), job)
            project.update(active_job=job["id"], status="queued")
            self.store.save_project(project)
            self.executor.submit(self._execute, job["id"])
            return job

    def _execute(self, job_id):
        path = self.store.job_path(job_id)
        job = read_json(path)
        project_id = job["project_id"]
        log_path = path.with_suffix(".log")
        command = [self.python, "-B", "-m", "app.web.worker", "--store", str(self.store.root), "--job", job_id]
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env.setdefault("HARMONY_DOWNLOAD_PYTHON", sys.executable)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "2"
        job.update(status="running", stage_label="正在启动分析进程", started_at=now(), command=command)
        atomic_json(path, job)
        with self.lock:
            project = self.store.project(project_id)
            project["status"] = "running"
            self.store.save_project(project)
        try:
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=7200, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            with self.lock:
                project = self.store.project(project_id)
                project.update(current_run=job_id, active_job=None, status="completed")
                self.store.save_project(project)
                job = read_json(path)
                job.update(status="completed", stage="complete", stage_label="分析完成", finished_at=now())
                atomic_json(path, job)
        except Exception as exc:
            with self.lock:
                job = read_json(path)
                job.update(status="failed", error=job.get("error") or f"{type(exc).__name__}: {exc}", finished_at=now())
                atomic_json(path, job)
                project = self.store.project(project_id)
                project.update(active_job=None, status="failed")
                self.store.save_project(project)

    def shutdown(self):
        try:
            self.executor.shutdown(wait=True)
        finally:
            self.store_lock.close()
