"""Serial process queue: keep HTTP responsive and acoustic runtimes isolated."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from .storage import ROOT, PROFILES, atomic_json, now, read_json
from .job_process import JobProcess

ACTIVE = {"queued", "running", "cancelling"}


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
        self.processes = {}
        self.cancel_requested = set()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="harmony-job")
        # A restart never reports a disappeared worker as still progressing.
        for path in (store.root / "jobs").glob("*.json"):
            job = read_json(path)
            if job["status"] in ACTIVE:
                job.update(status="interrupted", error="服务重启中断了此任务，请重新分析。", finished_at=now())
                atomic_json(path, job)
                project = store.project(job["project_id"])
                project.update(active_job=None, status=self.saved_status(project))
                store.save_project(project)

    @staticmethod
    def saved_status(project):
        return "completed" if any(project.get(k) for k in
            ("current_run", "stems", "piano_transcription", "song_reconstruction", "reconstruction")) else "ready"

    def get(self, job_id):
        with self.lock:
            job = read_json(self.store.job_path(job_id))
            if job_id in self.cancel_requested and job['status'] not in {'cancelled', 'completed'}:
                job.update(status='cancelling', stage_label='正在停止任务并释放资源')
            return job

    def list(self):
        with self.lock:
            rows = [self.get(p.stem) for p in (self.store.root/'jobs').glob('*.json')]
            active = sorted((r for r in rows if r['status'] in ACTIVE), key=lambda r:r.get('created_at',''))
            recent = sorted((r for r in rows if r['status'] not in ACTIVE),
                            key=lambda r:r.get('finished_at') or r.get('created_at',''), reverse=True)[:12]
            position = 0
            for row in active + recent:
                if row['status'] == 'queued':position += 1
                row['queue_position'] = position if row['status'] == 'queued' else None
                try:row['project_title'] = self.store.project(row['project_id'])['title']
                except FileNotFoundError:row['project_title'] = '已移除的歌曲'
            return dict(active=active, recent=recent)

    def _finish_cancel(self, job_id):
        job = read_json(self.store.job_path(job_id))
        job.update(status='cancelled',stage='cancelled',stage_label='已取消',error=None,finished_at=now())
        atomic_json(self.store.job_path(job_id),job)
        project = self.store.project(job['project_id'])
        if project.get('active_job') == job_id:
            project.update(active_job=None,status=self.saved_status(project))
            self.store.save_project(project)
        return job

    def cancel(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job['status'] not in ACTIVE:return job
            if job['status'] == 'queued':return self._finish_cancel(job_id)
            process = self.processes.get(job_id)
            if process:process.terminate()
            self.cancel_requested.add(job_id)
            job.update(status='cancelling',stage_label='正在停止任务并释放资源',cancel_requested_at=now())
            atomic_json(self.store.job_path(job_id),job)
            return job

    def submit(self, project_id, profile, download=False, key_mode="madmom",
               force=False, model=None, guitar_mode=None, pipeline=None, split_guitar=False):
        if profile != "stems" and key_mode not in {"madmom", "fused", "continuous"}:
            raise ValueError("未知主调分析模式")
        if profile == "retained10" and key_mode == "fused":
            raise ValueError("retained10 的主调支持 CNN 或连续评分（实验）；分歧裁决仅适用于旧模型。")
        if profile not in {"retained10", "stems", "pipeline"} and key_mode == "continuous":
            raise ValueError("连续评分主调（实验）仅适用于当前冻结模型。")
        if profile not in PROFILES and profile not in {"stems", "drums_parts", "vocal_parts",
                                                       "guitar_parts", "pipeline", "fusion", "piano", "reconstruction"}:
            raise ValueError("未知模型选项")
        with self.lock:
            project = self.store.project(project_id)
            if project.get("active_job"):
                previous = self.get(project["active_job"])
                if previous['status'] in ACTIVE:return previous
            job = dict(id=uuid4().hex, project_id=project_id, profile=profile, download=download, key_mode=key_mode,
                       force=bool(force), model=model, guitar_mode=guitar_mode,
                       split_guitar=bool(split_guitar), requested_profile=profile,
                       pipeline=pipeline,
                       status="queued", stage="queued", stage_label="等待前一项任务完成", stages=[],
                       created_at=now(), error=None)
            atomic_json(self.store.job_path(job["id"]), job)
            project.update(active_job=job["id"], status="queued")
            self.store.save_project(project)
            self.executor.submit(self._execute, job["id"])
            return job

    def _execute(self, job_id):
        path = self.store.job_path(job_id)
        with self.lock:
            job = read_json(path)
            if job['status'] != 'queued':return
        project_id = job["project_id"]
        log_path = path.with_suffix(".log")
        command = [self.python, "-B", "-m", "app.web.worker", "--store", str(self.store.root), "--job", job_id]
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env.setdefault("HARMONY_DOWNLOAD_PYTHON", sys.executable)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "2"
        try:
            with self.lock:
                if read_json(path)['status'] != 'queued':return
                job.update(status="running", stage_label="正在启动分析进程", started_at=now(), command=command)
                atomic_json(path, job)
                project = self.store.project(project_id)
                project["status"] = ("stems" if job.get("profile") in {"stems", "drums_parts",
                                                                        "vocal_parts", "guitar_parts", "fusion", "piano", "reconstruction"}
                                     else "running")
                self.store.save_project(project)
            with log_path.open("w", encoding="utf-8") as log:
                with self.lock:
                    if job_id in self.cancel_requested:
                        self._finish_cancel(job_id); return
                    process = JobProcess(command,cwd=ROOT,env=env,log=log,gate=path.with_suffix('.go'))
                    self.processes[job_id] = process
                try:
                    code = process.wait(timeout=7200)
                    if code:raise subprocess.CalledProcessError(code,command)
                finally:
                    # Closing the job also cleans any leftover child processes before
                    # another queued task may acquire GPU/cache resources.
                    with self.lock:
                        process.close()
                        self.processes.pop(job_id,None)
            with self.lock:
                if job_id in self.cancel_requested:
                    self._finish_cancel(job_id); return
                project = self.store.project(project_id)
                job = read_json(path)
                project.update(active_job=None, status="completed")
                if job.get("profile") in {"stems", "drums_parts", "vocal_parts", "guitar_parts", "fusion", "piano", "reconstruction"}:
                    # Separation produces a side asset: it must never replace the analysis run.
                    if job.get("profile") in {"drums_parts", "vocal_parts", "guitar_parts", "piano", "reconstruction"}:
                        project[job["profile"]] = dict(job_id=job_id, updated_at=now())
                    else:
                        # The worker already persisted per-model stem identities.
                        project["stems_job"] = dict(job_id=job_id, updated_at=now())
                    job.update(status="completed", stage="complete",
                               stage_label={"stems":"分轨完成","vocal_parts":"人声细分完成","drums_parts":"鼓组细分完成","guitar_parts":"吉他细分完成","fusion":"分轨融合完成","piano":"钢琴 MIDI 已生成","reconstruction":"整曲重构已生成"}[job['profile']],
                               finished_at=now())
                else:
                    project.update(current_run=job_id)
                    label = "完整流程完成" if job.get("profile") == "pipeline" else "分析完成"
                    job.update(status="completed", stage="complete", stage_label=label,
                               finished_at=now())
                self.store.save_project(project)
                atomic_json(path, job)
        except Exception as exc:
            with self.lock:
                if job_id in self.cancel_requested:
                    self._finish_cancel(job_id); return
                job = read_json(path)
                job.update(status="failed", error=job.get("error") or f"{type(exc).__name__}: {exc}", finished_at=now())
                atomic_json(path, job)
                project = self.store.project(project_id)
                # A failed transcription does not invalidate an existing
                # harmony analysis or a previously saved piano generation.
                status = ("completed" if project.get("current_run") or project.get("piano_transcription") or project.get("stems") else "ready") if job.get("profile") in {"piano", "reconstruction", "stems"} else "failed"
                project.update(active_job=None, status=status)
                self.store.save_project(project)
        finally:
            with self.lock:self.cancel_requested.discard(job_id)

    def shutdown(self):
        try:
            self.executor.shutdown(wait=True)
        finally:
            self.store_lock.close()
