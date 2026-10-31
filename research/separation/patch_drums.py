"""Wire the drum-parts job: dispatch, queue, API routes and file serving."""
from __future__ import annotations

from pathlib import Path

BACKEND = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web")

# --- worker dispatch
worker = BACKEND / "worker.py"
text = worker.read_text(encoding="utf-8")
OLD = """        if job.get("profile") == "stems":
            from .stems_worker import execute as execute_stems
            execute_stems(store, args.job)
        else:
            execute(store, args.job)
"""
NEW = """        if job.get("profile") == "stems":
            from .stems_worker import execute as execute_stems
            execute_stems(store, args.job)
        elif job.get("profile") == "drums_parts":
            from .drums_worker import execute as execute_drums
            execute_drums(store, args.job)
        else:
            execute(store, args.job)
"""
assert text.count(OLD) == 1
worker.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("worker.py: drums dispatch")

# --- jobs queue
jobs = BACKEND / "jobs.py"
text = jobs.read_text(encoding="utf-8")
OLD_ALLOW = """        if profile not in PROFILES and profile != "stems":
            raise ValueError("未知模型选项")"""
NEW_ALLOW = """        if profile not in PROFILES and profile not in {"stems", "drums_parts"}:
            raise ValueError("未知模型选项")"""
assert text.count(OLD_ALLOW) == 1
text = text.replace(OLD_ALLOW, NEW_ALLOW)

OLD_STATUS = """            project["status"] = "stems" if job.get("profile") == "stems" else "running\""""
NEW_STATUS = """            project["status"] = ("stems" if job.get("profile") in {"stems", "drums_parts"}
                                 else "running")"""
assert text.count(OLD_STATUS) == 1, text.count(OLD_STATUS)
text = text.replace(OLD_STATUS, NEW_STATUS)

OLD_DONE = """                if job.get("profile") == "stems":
                    # Separation produces a side asset: it must never replace the analysis run.
                    project.update(stems=dict(job_id=job_id, updated_at=now()))
                    job.update(status="completed", stage="complete", stage_label="分轨完成",
                               finished_at=now())"""
NEW_DONE = """                if job.get("profile") in {"stems", "drums_parts"}:
                    # Separation produces a side asset: it must never replace the analysis run.
                    if job.get("profile") == "drums_parts":
                        project.update(drum_parts=dict(job_id=job_id, updated_at=now()))
                    else:
                        project.update(stems=dict(job_id=job_id, updated_at=now()))
                    job.update(status="completed", stage="complete",
                               stage_label="分轨完成" if job.get("profile") == "stems" else "鼓组细分完成",
                               finished_at=now())"""
assert text.count(OLD_DONE) == 1
jobs.write_text(text.replace(OLD_DONE, NEW_DONE), encoding="utf-8")
print("jobs.py: drums profile")

# --- server routes
server = BACKEND / "server.py"
text = server.read_text(encoding="utf-8")
OLD_ROUTE = """    def _stems_sha(project_id: str, project: dict) -> str:"""
NEW_ROUTE = '''    @application.post("/api/projects/{project_id}/drums-parts", status_code=202)
    def drums_parts(project_id: str, body: SeparateRequest | None = None):
        from .stems_worker import MODELS
        body = body or SeparateRequest()
        model = body.model if body.model in MODELS else "htdemucs"
        return application.state.jobs.submit(project_id, "drums_parts", key_mode="madmom",
                                             force=body.force, model=model)

    @application.get("/api/projects/{project_id}/stems/drums-parts")
    def drums_parts_manifest(project_id: str, model: str | None = None):
        from .drums_worker import parts_manifest
        from .stems_worker import available_models
        project = store.project(project_id)
        sha = _stems_sha(project_id, project)
        ready = [item["id"] for item in available_models(sha) if item["ready"]]
        model = model or ("htdemucs" if "htdemucs" in ready else (ready[0] if ready else "htdemucs"))
        path = parts_manifest(sha, model)
        if not path.is_file():
            raise HTTPException(404, "尚未细分鼓组")
        manifest = read_json(path)
        manifest["urls"] = {track["id"]: f"/api/projects/{project_id}/stems/drums_parts/{track['file']}"
                            for track in manifest.get("tracks", [])}
        return manifest

    def _stems_sha(project_id: str, project: dict) -> str:'''
assert text.count(OLD_ROUTE) == 1
text = text.replace(OLD_ROUTE, NEW_ROUTE)

OLD_AUDIO = '''        if not re.fullmatch(r"(wav/)?[a-z_]+\\.(mp3|wav)", name):
            raise HTTPException(404, "未知文件")
        project = store.project(project_id)
        sha = _stems_sha(project_id, project)
        if model is None:
            ready = [item["id"] for item in available_models(sha) if item["ready"]]
            model = "htdemucs" if "htdemucs" in ready else (ready[0] if ready else MODEL_NAME)
        path = stems_dir(sha, model) / name
'''
NEW_AUDIO = '''        if not re.fullmatch(r"(drums_parts/)?(wav/)?[a-z_]+\\.(mp3|wav)", name):
            raise HTTPException(404, "未知文件")
        project = store.project(project_id)
        sha = _stems_sha(project_id, project)
        if model is None:
            ready = [item["id"] for item in available_models(sha) if item["ready"]]
            model = "htdemucs" if "htdemucs" in ready else (ready[0] if ready else MODEL_NAME)
        path = stems_dir(sha, model) / name
        if name.startswith("drums_parts/"):
            path = stems_dir(sha, model) / name
'''
assert text.count(OLD_AUDIO) == 1
server.write_text(text.replace(OLD_AUDIO, NEW_AUDIO), encoding="utf-8")
print("server.py: drum parts routes")
