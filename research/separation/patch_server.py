"""Add the stems API and the /stems page route to server.py."""
from __future__ import annotations

from pathlib import Path

SERVER = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\server.py")
text = SERVER.read_text(encoding="utf-8")

# 1) request model
OLD_MODEL = """class LinkRequest(AnalyzeRequest):
"""
NEW_MODEL = """class SeparateRequest(BaseModel):
    force: bool = False
    model: str = "htdemucs"


class LinkRequest(AnalyzeRequest):
"""
assert text.count(OLD_MODEL) == 1
text = text.replace(OLD_MODEL, NEW_MODEL)

# 2) routes, right after the analyze endpoint
OLD_ROUTE = """    @application.get("/api/jobs/{job_id}")
    def job(job_id: str):
        return read_json(store.job_path(job_id))
"""
NEW_ROUTE = '''    @application.post("/api/projects/{project_id}/separate", status_code=202)
    def separate(project_id: str, body: SeparateRequest | None = None):
        body = body or SeparateRequest()
        project = store.project(project_id)
        return application.state.jobs.submit(
            project_id, "stems",
            download=not project.get("audio") and project.get("input_type") == "bilibili",
            force=body.force, model=body.model)

    @application.get("/api/projects/{project_id}/stems")
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

    @application.get("/api/projects/{project_id}/stems/{name:path}")
    def stem_audio(project_id: str, name: str):
        from .stems_worker import stems_dir
        if not re.fullmatch(r"(wav/)?[a-z_]+\.(mp3|wav)", name):
            raise HTTPException(404, "未知文件")
        project = store.project(project_id)
        audio = store.project_dir(project_id) / project["audio"]
        path = stems_dir(signature(audio)["sha256"]) / name
        if not path.is_file():
            raise HTTPException(404, "文件不存在")
        disposition = "attachment" if name.endswith(".wav") else "inline"
        return FileResponse(path, media_type="audio/mpeg" if name.endswith(".mp3") else "audio/wav",
                            content_disposition_type=disposition)
'''
assert text.count(OLD_ROUTE) == 1
text = text.replace(OLD_ROUTE, NEW_ROUTE)

# 3) page route next to the workbench page
OLD_PAGE = """    @application.get("/")
    def index():
        return FileResponse(ROOT / "frontend/index.html", headers={"Cache-Control": "no-cache"})
"""
NEW_PAGE = """    @application.get("/")
    def index():
        return FileResponse(ROOT / "frontend/index.html", headers={"Cache-Control": "no-cache"})

    @application.get("/stems")
    def stems_page():
        return FileResponse(ROOT / "frontend/stems.html", headers={"Cache-Control": "no-cache"})
"""
assert text.count(OLD_PAGE) == 1
SERVER.write_text(text.replace(OLD_PAGE, NEW_PAGE), encoding="utf-8")
print("server.py: stems endpoints added")
