"""Expose the two checkpoints through the stems API."""
from __future__ import annotations

from pathlib import Path

SERVER = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\server.py")
text = SERVER.read_text(encoding="utf-8")

OLD_SEP = '''    @application.post("/api/projects/{project_id}/separate", status_code=202)
    def separate(project_id: str, body: SeparateRequest | None = None):
        body = body or SeparateRequest()
        project = store.project(project_id)
        return application.state.jobs.submit(
            project_id, "stems",
            download=not project.get("audio") and project.get("input_type") == "bilibili",
            force=body.force, model=body.model)
'''
NEW_SEP = '''    @application.post("/api/projects/{project_id}/separate", status_code=202)
    def separate(project_id: str, body: SeparateRequest | None = None):
        from .stems_worker import MODELS
        body = body or SeparateRequest()
        if body.model not in MODELS:
            raise HTTPException(400, "未知的分轨模型")
        project = store.project(project_id)
        return application.state.jobs.submit(
            project_id, "stems",
            download=not project.get("audio") and project.get("input_type") == "bilibili",
            force=body.force, model=body.model)
'''
assert text.count(OLD_SEP) == 1
text = text.replace(OLD_SEP, NEW_SEP)

OLD_GET = '''    @application.get("/api/projects/{project_id}/stems")
    def stems(project_id: str):
        from .stems_worker import stems_dir
        project = store.project(project_id)
        if not project.get("audio"):
            raise HTTPException(404, "项目还没有音频")
        audio = store.project_dir(project_id) / project["audio"]
        if not audio.is_file():
            raise HTTPException(404, "音频文件不存在，请重新导入")
        path = stems_dir(signature(audio)["sha256"]) / "manifest.json"
        if not path.is_file():
            raise HTTPException(404, "尚未生成分轨")
        manifest = read_json(path)
        manifest["audio_url"] = f"/api/projects/{project_id}/audio"
        manifest["urls"] = {track["id"]: f"/api/projects/{project_id}/stems/{track['file']}"
                            for track in manifest.get("tracks", [])}
        manifest["wav_urls"] = {track["id"]: f"/api/projects/{project_id}/stems/{track['wav']}"
                                for track in manifest.get("tracks", [])}
        return manifest
'''
NEW_GET = '''    def _stems_sha(project_id: str, project: dict) -> str:
        audio = store.project_dir(project_id) / project["audio"]
        if not audio.is_file():
            raise HTTPException(404, "音频文件不存在，请重新导入")
        return signature(audio)["sha256"]

    @application.get("/api/projects/{project_id}/stems")
    def stems(project_id: str, model: str | None = None):
        from .stems_worker import available_models, stems_dir
        project = store.project(project_id)
        if not project.get("audio"):
            raise HTTPException(404, "项目还没有音频")
        sha = _stems_sha(project_id, project)
        options = available_models(sha)
        ready = [item["id"] for item in options if item["ready"]]
        if model is None:
            model = "htdemucs" if "htdemucs" in ready else (ready[0] if ready else None)
        if model is None or model not in [item["id"] for item in options]:
            raise HTTPException(404, "尚未生成分轨")
        path = stems_dir(sha, model) / "manifest.json"
        if not path.is_file():
            raise HTTPException(404, "该模型尚未生成分轨")
        manifest = read_json(path)
        manifest["audio_url"] = f"/api/projects/{project_id}/audio"
        manifest["model_options"] = options
        manifest["model_key"] = model
        query = f"?model={model}"
        manifest["urls"] = {track["id"]: f"/api/projects/{project_id}/stems/{track['file']}{query}"
                            for track in manifest.get("tracks", [])}
        manifest["wav_urls"] = {track["id"]: f"/api/projects/{project_id}/stems/{track['wav']}{query}"
                                for track in manifest.get("tracks", [])}
        return manifest
'''
assert text.count(OLD_GET) == 1
text = text.replace(OLD_GET, NEW_GET)

OLD_AUDIO = '''    @application.get("/api/projects/{project_id}/stems/{name:path}")
    def stem_audio(project_id: str, name: str):
        from .stems_worker import stems_dir
        if not re.fullmatch(r"(wav/)?[a-z_]+\\.(mp3|wav)", name):
            raise HTTPException(404, "未知文件")
        project = store.project(project_id)
        audio = store.project_dir(project_id) / project["audio"]
        path = stems_dir(signature(audio)["sha256"]) / name
        if not path.is_file():
            raise HTTPException(404, "文件不存在")
'''
NEW_AUDIO = '''    @application.get("/api/projects/{project_id}/stems/{name:path}")
    def stem_audio(project_id: str, name: str, model: str | None = None):
        from .stems_worker import MODEL_NAME, available_models, stems_dir
        if not re.fullmatch(r"(wav/)?[a-z_]+\\.(mp3|wav)", name):
            raise HTTPException(404, "未知文件")
        project = store.project(project_id)
        sha = _stems_sha(project_id, project)
        if model is None:
            ready = [item["id"] for item in available_models(sha) if item["ready"]]
            model = "htdemucs" if "htdemucs" in ready else (ready[0] if ready else MODEL_NAME)
        path = stems_dir(sha, model) / name
        if not path.is_file():
            raise HTTPException(404, "文件不存在")
'''
assert text.count(OLD_AUDIO) == 1
text = text.replace(OLD_AUDIO, NEW_AUDIO)

SERVER.write_text(text, encoding="utf-8")
print("server.py: model-aware stems API")
