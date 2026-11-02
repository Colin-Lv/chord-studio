"""Local upload, recognition, playback and correction API."""
from __future__ import annotations

import os
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .jobs import Jobs
from .storage import (ROOT, DEFAULT_STORE, AUDIO_EXTENSIONS, MAX_UPLOAD, PROFILES,
                      Store, atomic_json, atomic_text, bilibili_url, identifier, now, read_json, signature)


class AnalyzeRequest(BaseModel):
    profile: str = "retained10"
    # "continuous" publishes the experimental continuous-scoring key (24-candidate evidence);
    # it is not calibrated and is labelled as experimental in the UI and the result.
    key_mode: Literal["madmom", "fused", "continuous"] = "madmom"


class PipelineRequest(BaseModel):
    model: str = "strings_v2"
    vocal_parts: bool = False
    guitar_parts: bool = True
    drums_parts: bool = False
    guitar_mode: Literal["timbre", "duet"] = "timbre"
    harmony: bool = False
    harmony_method: Literal['stem_evidence_v2', 'legacy_fusion'] = 'stem_evidence_v2'
    harmony_default: bool = False
    force_stems: bool = False


class SeparateRequest(BaseModel):
    force: bool = False
    model: str = "strings_v2"
    guitar_parts: bool = True
    guitar_mode: Literal["timbre", "duet"] = "timbre"


class GuitarPartsRequest(SeparateRequest):
    model: str = "bs_roformer_sw"
    mode: Literal["timbre", "duet"] = "timbre"


class LinkRequest(AnalyzeRequest):
    url: str = Field(max_length=2000)


class Edit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(ge=0, strict=True)
    label: str = Field(min_length=1, max_length=160)


class CorrectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    source_id: str
    source_sha256: str
    revision: int = Field(ge=0, strict=True)
    key: str = ""
    key_origin: str = "automatic"
    display: str = "roman"
    edits: list[Edit] = Field(default_factory=list, max_length=20000)
    key_review_revision: int | None = Field(default=None, ge=0, strict=True)


class KeyReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    audio_sha256: str | None
    revision: int = Field(ge=0, strict=True)
    action: Literal["enable", "disable", "adopt", "custom", "restore"]
    key: str = Field(default="", max_length=8)


class KeyAnnotationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    audio_sha256: str
    revision: int = Field(ge=0, strict=True)
    action: Literal["lock", "reveal"]
    label: dict = Field(default_factory=dict)
    entry_context: dict | None = None
    prior_exposure: Literal["unknown", "seen", "not_seen"] = "unknown"
    based_on_revision: int | None = Field(default=None, ge=1, strict=True)


def notation_script():
    source = (ROOT / "app/studio_assets/studio.js").read_text(encoding="utf-8")
    constants = source[source.index("    const ROOT_TO_PC ="):source.index("    let events = []")]
    helpers = source[source.index("    function parseChord(label) {"):source.index("    function parseScoreText(text) {")]
    return ('"use strict";window.HarmonyNotation=(()=>{let currentKey="";function editingKey(){return currentKey;}\n'
            + constants + helpers
            + '\nreturn {setKey(value){currentKey=value;},parse:parseChord,parseInKey(label,key){const previous=currentKey;try{currentKey=key;return parseChord(label);}finally{currentKey=previous;}},parseEditable:parseEditableChord,roots:Object.keys(ROOT_TO_PC)};})();')


def create_app(store_root=None, manager_factory=Jobs):
    store = Store(store_root or os.environ.get("HARMONY_WEB_STORE") or DEFAULT_STORE)
    save_lock = threading.RLock()

    @asynccontextmanager
    async def lifespan(application):
        application.state.jobs = manager_factory(store)
        yield
        application.state.jobs.shutdown()

    application = FastAPI(title="Harmony Studio", lifespan=lifespan)
    application.state.store = store
    from .editor import register_editor
    register_editor(application, store, save_lock)
    from .score_api import register_scores
    register_scores(application, store, save_lock)
    from .piano_api import register_piano
    register_piano(application, store)
    from .reconstruction_api import register_reconstruction
    register_reconstruction(application, store)
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])

    @application.middleware("http")
    async def local_requests(request: Request, call_next):
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = request.headers.get("origin")
            if request.headers.get("x-harmony-client") != "studio" or (origin and urlsplit(origin).netloc != request.headers.get("host")):
                return JSONResponse({"detail": "请从本地工作台提交此操作"}, status_code=403)
            length = request.headers.get("content-length")
            if length and (not length.isdigit() or int(length) > MAX_UPLOAD + 1024 * 1024):
                return JSONResponse({"detail": "文件超过 512 MB 上限"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.url.path.startswith("/api/") and not request.url.path.endswith("/audio"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @application.exception_handler(FileNotFoundError)
    async def missing(request, exc):
        return JSONResponse({"detail": "项目、结果或文件尚不存在"}, status_code=404)

    @application.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @application.get("/api/config")
    def config():
        from .stems_worker import MODELS
        from app.runtime_paths import config as runtime_config
        return dict(profiles=[dict(id=k, name=v) for k, v in PROFILES.items()], max_upload_bytes=MAX_UPLOAD,
                    default_profile="retained10", desktop_version=runtime_config().get('application_version','1.0.0'), retained10_max_seconds=600,
                    stems_models=[dict(id=m["id"], label=m["label"], note=m.get("note", ""),
                                       stems=m.get("stems", 4)) for m in MODELS.values()],
                    editor_preview=os.environ.get("HARMONY_EDITOR_PREVIEW") == "1")

    @application.get("/api/projects")
    def projects():
        values = [read_json(path) for path in store.root.glob("*/project.json")]
        return sorted(values, key=lambda row: row["created_at"], reverse=True)

    @application.get("/api/projects/{project_id}")
    def project(project_id: str):
        return store.project(project_id)

    @application.post("/api/upload", status_code=201)
    async def upload(file: UploadFile = File(...)):
        filename = (file.filename or "audio").replace("\\", "/").rsplit("/", 1)[-1]
        extension = Path(filename).suffix.lower()
        if extension not in AUDIO_EXTENSIONS:
            raise HTTPException(400, "请选择 MP3、WAV、FLAC、M4A 等音频文件")
        project = store.create(filename, original_filename=filename, input_type="upload")
        folder = store.project_dir(project["id"])
        target = folder / ("audio" + extension)
        temporary = target.with_suffix(extension + ".upload")
        size = 0
        try:
            with temporary.open("xb") as stream:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, "文件超过 512 MB 上限")
                    stream.write(chunk)
            if size == 0:
                raise HTTPException(400, "音频文件为空")
            temporary.replace(target)
            project.update(audio=target.name, audio_identity=signature(target))
            store.save_project(project)
        except Exception:
            temporary.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            (folder / "project.json").unlink(missing_ok=True)
            folder.rmdir()
            raise
        finally:
            await file.close()
        return project

    @application.post("/api/bilibili", status_code=202)
    def import_link(body: LinkRequest):
        if os.environ.get("HARMONY_EDITOR_PREVIEW") == "1":
            raise HTTPException(403, "独立编辑预览不下载或运行声学模型，请使用演示项目")
        url = bilibili_url(body.url)
        if body.profile not in PROFILES:
            raise HTTPException(400, "未知模型选项")
        project = store.create("B 站音频 · 等待下载", input_type="bilibili", source_url=url)
        return application.state.jobs.submit(project["id"], body.profile, download=True, key_mode=body.key_mode)

    @application.post("/api/projects/{project_id}/analyze", status_code=202)
    def analyze(project_id: str, body: AnalyzeRequest):
        if os.environ.get("HARMONY_EDITOR_PREVIEW") == "1":
            raise HTTPException(403, "独立编辑预览不运行声学模型，请使用演示项目")
        project = store.project(project_id)
        return application.state.jobs.submit(project_id, body.profile, download=not project.get("audio") and project.get("input_type") == "bilibili", key_mode=body.key_mode)

    @application.post("/api/projects/{project_id}/separate", status_code=202)
    def separate(project_id: str, body: SeparateRequest | None = None):
        from .stems_worker import MODELS
        body = body or SeparateRequest()
        if body.model not in MODELS:
            raise HTTPException(400, "未知的分轨模型")
        project = store.project(project_id)
        return application.state.jobs.submit(
            project_id, "stems",
            download=not project.get("audio") and project.get("input_type") == "bilibili",
            force=body.force, model=body.model,guitar_mode=body.guitar_mode,
            split_guitar=body.guitar_parts and MODELS[body.model].get('stems',4)>=6)

    @application.post("/api/projects/{project_id}/pipeline", status_code=202)
    def pipeline(project_id: str, body: PipelineRequest):
        from .stems_worker import MODELS
        if body.model not in MODELS:
            raise HTTPException(400, "未知的分轨模型")
        spec = body.model_dump()
        spec['guitar_parts'] = body.guitar_parts and MODELS[body.model].get('stems',4)>=6
        return application.state.jobs.submit(project_id, "pipeline", key_mode="continuous",
                                             model=body.model, pipeline=spec,
                                             force=body.force_stems)

    @application.post("/api/projects/{project_id}/fusion", status_code=202)
    def fusion(project_id: str, body: SeparateRequest | None = None):
        from .stems_worker import MODELS
        body = body or SeparateRequest()
        if body.model not in MODELS:
            raise HTTPException(400, '未知的分轨模型')
        return application.state.jobs.submit(project_id, "fusion", key_mode="madmom",
                                             model=body.model)

    @application.get("/api/projects/{project_id}/stem-fusion")
    def stem_fusion(project_id: str):
        project = store.project(project_id)
        if not project.get("current_run"):
            raise HTTPException(404, "尚未完成分析")
        run_dir = store.run_dir(project_id, project["current_run"])
        result = read_json(run_dir / "analysis.json")
        source = next((item for item in result.get("sources", [])
                       if item.get("id") == "stem_fusion"), None)
        if source is None:
            raise HTTPException(404, "这次分析还没有分轨融合结果")
        return dict(source=source, meta=result.get("stem_fusion"),
                    run_id=project["current_run"])

    @application.post("/api/projects/{project_id}/drums-parts", status_code=202)
    def drums_parts(project_id: str, body: SeparateRequest | None = None):
        from .stems_worker import MODELS
        body = body or SeparateRequest()
        if body.model not in MODELS:
            raise HTTPException(400, "未知的分轨模型")
        model = body.model
        return application.state.jobs.submit(project_id, "drums_parts", key_mode="madmom",
                                             force=body.force, model=model)

    @application.get("/api/projects/{project_id}/stems/drums-parts")
    def drums_parts_manifest(project_id: str, model: str | None = None):
        from .drums_worker import parts_manifest
        from .stems_worker import MODELS, available_models
        project = store.project(project_id)
        sha = _stems_sha(project_id, project)
        ready = [item["id"] for item in available_models(sha) if item["ready"]]
        model = model or ("htdemucs" if "htdemucs" in ready else (ready[0] if ready else "htdemucs"))
        if model not in MODELS:
            raise HTTPException(404, "未知的分轨模型")
        path = parts_manifest(sha, model)
        if not path.is_file():
            raise HTTPException(404, "尚未细分鼓组")
        manifest = read_json(path)
        if manifest.get('parent_sha256'):
            from .stems_worker import stem_wav
            parent=stem_wav(sha,model,'drums')
            if not parent.is_file() or signature(parent)['sha256']!=manifest['parent_sha256']:
                raise HTTPException(404,'鼓组源轨已更新，请重新细分')
        manifest["urls"] = {track["id"]: f"/api/projects/{project_id}/stems/drums_parts/{track['file']}?model={model}"
                            for track in manifest.get("tracks", [])}
        manifest["wav_urls"] = {track["id"]: f"/api/projects/{project_id}/stems/drums_parts/{track['wav']}?model={model}"
                                for track in manifest.get("tracks", [])}
        return manifest

    @application.post("/api/projects/{project_id}/vocal-parts", status_code=202)
    def vocal_parts(project_id: str, body: SeparateRequest | None = None):
        from .stems_worker import MODELS
        body = body or SeparateRequest()
        if body.model not in MODELS:
            raise HTTPException(400, "未知的分轨模型")
        return application.state.jobs.submit(project_id, "vocal_parts", key_mode="madmom",
                                             force=body.force, model=body.model)

    @application.get("/api/projects/{project_id}/stems/vocal-parts")
    def vocal_parts_manifest(project_id: str, model: str = "htdemucs"):
        from .vocal_parts_worker import valid_manifest
        project = store.project(project_id)
        manifest = valid_manifest(_stems_sha(project_id, project), model)
        if not manifest:
            raise HTTPException(404, "尚未细分人声，或源人声轨已更新")
        for key, field in (("urls", "file"), ("wav_urls", "wav")):
            manifest[key] = {t["id"]: f"/api/projects/{project_id}/stems/{t[field]}?model={model}"
                             for t in manifest["tracks"]}
        return manifest

    @application.post("/api/projects/{project_id}/guitar-parts", status_code=202)
    def guitar_parts(project_id: str, body: GuitarPartsRequest):
        from .guitar_parts_worker import fingerprint
        from .stems_worker import MODELS, stems_dir, stem_wav
        if body.model not in MODELS:
            raise HTTPException(400, "未知的分轨模型")
        sha = _stems_sha(project_id, store.project(project_id))
        if not stem_wav(sha, body.model,"guitar").is_file():
            raise HTTPException(409, "请先生成并切换到六轨精细版，再细分其中的吉他轨")
        if not fingerprint(body.mode):
            raise HTTPException(503, "此吉他细分模型尚未准备完整")
        return application.state.jobs.submit(project_id, "guitar_parts", key_mode="madmom",
            force=body.force, model=body.model, guitar_mode=body.mode)

    def guitar_urls(project_id, manifest):
        model = manifest['model']
        for key,field in [('urls','file'),('wav_urls','wav')]:
            manifest[key] = {t['id']:f"/api/projects/{project_id}/stems/{t[field]}?model={model}" for t in manifest['tracks']}
        return manifest

    @application.get("/api/projects/{project_id}/stems/guitar-parts")
    def guitar_parts_manifest(project_id: str, model: str = "bs_roformer_sw", mode: Literal['timbre','duet'] = 'timbre'):
        from .guitar_parts_worker import valid_manifest
        manifest = valid_manifest(_stems_sha(project_id,store.project(project_id)),model,mode)
        if not manifest:
            raise HTTPException(404, "尚未细分吉他，或源吉他及模型已更新")
        return guitar_urls(project_id,manifest)

    @application.get("/api/projects/{project_id}/stems/guitar-parts-catalog")
    def guitar_parts_catalog(project_id: str):
        from .guitar_parts_worker import RECIPES,valid_manifest,fingerprint
        from .stems_worker import available_models,stems_dir,stem_wav
        project = store.project(project_id);sha = _stems_sha(project_id,project)
        results = []
        for model in available_models(sha):
            if not model['ready'] or not stem_wav(sha,model['id'],'guitar').is_file():continue
            for mode in RECIPES:
                manifest = valid_manifest(sha,model['id'],mode)
                if manifest:results.append(guitar_urls(project_id,manifest))
        return dict(results=results,active_job=project.get('active_job'),
            modes=[dict(id=mode,label=row['label'],note=row['note'],ready=bool(fingerprint(mode))) for mode,row in RECIPES.items()])

    def _stems_sha(project_id: str, project: dict) -> str:
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
            model = "strings_v2" if "strings_v2" in ready else ("htdemucs" if "htdemucs" in ready else (ready[0] if ready else None))
        if model is None or model not in [item["id"] for item in options]:
            raise HTTPException(404, "尚未生成分轨")
        path = stems_dir(sha, model) / "manifest.json"
        if not path.is_file():
            raise HTTPException(404, "该模型尚未生成分轨")
        manifest = read_json(path)
        manifest["audio_url"] = (f"/api/projects/{project_id}/stems/{manifest['reference_file']}?model={model}" if manifest.get("reference_file") else f"/api/projects/{project_id}/audio")
        manifest["model_options"] = options
        manifest["model_key"] = model
        query = f"?model={model}"
        manifest["urls"] = {track["id"]: f"/api/projects/{project_id}/stems/{track['file']}{query}"
                            for track in manifest.get("tracks", [])}
        manifest["wav_urls"] = {track["id"]: f"/api/projects/{project_id}/stems/{track['wav']}{query}"
                                for track in manifest.get("tracks", [])}
        return manifest

    @application.get("/api/projects/{project_id}/stems/{name:path}")
    def stem_audio(project_id: str, name: str, model: str | None = None):
        from .stems_worker import MODEL_NAME, MODELS, available_models, stems_dir
        if not re.fullmatch(r"(?:(?:drums_parts/)?(?:wav/)?|(?:vocal_parts|guitar_parts)/[a-f0-9]{32}/|generations/[a-f0-9]{32}/(?:wav/)?)[a-z_]+\.(mp3|wav)", name):
            raise HTTPException(404, "未知文件")
        project = store.project(project_id)
        sha = _stems_sha(project_id, project)
        if model is None:
            ready = [item["id"] for item in available_models(sha) if item["ready"]]
            model = "strings_v2" if "strings_v2" in ready else ("htdemucs" if "htdemucs" in ready else (ready[0] if ready else MODEL_NAME))
        if model not in MODELS:
            raise HTTPException(404, "未知的分轨模型")
        path = stems_dir(sha, model) / name
        if name.startswith("drums_parts/"):
            path = stems_dir(sha, model) / name
        if not path.is_file():
            raise HTTPException(404, "文件不存在")
        disposition = "attachment" if name.endswith(".wav") else "inline"
        return FileResponse(path, media_type="audio/mpeg" if name.endswith(".mp3") else "audio/wav",
                            content_disposition_type=disposition)

    @application.get("/api/jobs")
    def job_queue():
        return application.state.jobs.list()

    @application.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        return application.state.jobs.cancel(job_id)

    @application.get("/api/jobs/{job_id}")
    def job(job_id: str):
        manager = application.state.jobs
        return manager.get(job_id) if hasattr(manager, 'get') else read_json(store.job_path(job_id))

    @application.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
    def job_log(job_id: str):
        path = store.job_path(job_id).with_suffix(".log")
        if not path.exists():
            return "任务尚未启动。"
        # Bound responses while preserving complete diagnostic files on disk.
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 64000))
            return stream.read().decode("utf-8", errors="replace")

    @application.get("/api/projects/{project_id}/audio")
    def audio(project_id: str):
        project = store.project(project_id)
        if not project.get("audio"):
            raise HTTPException(404, "音频尚未下载完成")
        return FileResponse(store.project_dir(project_id) / project["audio"], content_disposition_type="inline")

    @application.get("/api/projects/{project_id}/result")
    def result(project_id: str):
        return store.result(project_id)

    @application.get("/api/projects/{project_id}/key-review")
    def key_review(project_id: str):
        from .key_review import state
        with save_lock:
            return state(store, project_id)

    @application.get("/api/projects/{project_id}/source-comparison")
    def source_comparison(project_id: str, left: str, right: str):
        from .source_comparison import comparison
        with save_lock:
            return comparison(store, project_id, left, right)

    @application.post("/api/projects/{project_id}/key-review")
    def save_key_review(project_id: str, body: KeyReviewRequest):
        from .key_review import update
        with save_lock:
            try:
                return update(store, project_id, body.model_dump())
            except ValueError as exc:
                if str(exc).startswith("CONFLICT:"):
                    raise HTTPException(409, str(exc)) from exc
                raise

    @application.get("/api/projects/{project_id}/key-annotation")
    def key_annotation(project_id: str):
        from .key_annotation import state
        with save_lock:
            return state(store, project_id)

    @application.post("/api/projects/{project_id}/key-annotation")
    def save_key_annotation(project_id: str, body: KeyAnnotationRequest):
        from .key_annotation import update
        with save_lock:
            try:
                return update(store, project_id, body.model_dump())
            except ValueError as exc:
                if str(exc).startswith("CONFLICT:"):
                    raise HTTPException(409, str(exc)) from exc
                raise

    @application.get("/api/projects/{project_id}/key-annotation/predictions")
    def annotation_predictions(project_id: str):
        from .key_annotation import predictions
        with save_lock:
            return predictions(store, project_id)

    def correction_location(project_id, run_id, source_id):
        identifier(run_id)
        if not re.fullmatch(r"[a-zA-Z0-9_]+", source_id):
            raise ValueError("无效的来源 ID")
        return store.run_dir(project_id, run_id) / "corrections" / source_id

    @application.get("/api/projects/{project_id}/corrections/{source_id}")
    def corrections(project_id: str, source_id: str):
        from .corrections import context_and_record
        with save_lock:
            return context_and_record(store, project_id, source_id)[-1]

    @application.post("/api/projects/{project_id}/corrections")
    def save_corrections(project_id: str, body: CorrectionRequest):
        from app.chord_semantics_v2 import parse_chord
        with save_lock:
            result = store.result(project_id)
            if body.key_review_revision is not None:
                from .key_review import state
                if body.key_review_revision != state(store, project_id)["revision"]:
                    raise HTTPException(409, "主调审阅已在另一窗口修改，请重新打开项目；和弦草稿仍保留。")
            from .corrections import context_and_record
            _, result, source, duration, audio_sha, previous = context_and_record(store, project_id, body.source_id)
            if result["run_id"] != body.run_id or source is None or source["sha256"] != body.source_sha256:
                raise HTTPException(409, "分析结果已变更，请重新打开项目后再编辑")
            directory = correction_location(project_id, body.run_id, body.source_id)
            path = directory / "corrections.json"
            if body.revision != previous["revision"]:
                raise HTTPException(409, "另一个窗口已经保存了修改。请导出当前草稿，再重新打开项目。")
            if body.display not in {"roman", "name"} or body.key_origin not in {"automatic", "manual", "unknown"}:
                raise ValueError("无效的显示设置")
            if body.key and not re.fullmatch(r"(?:C|C#|Db|D|D#|Eb|E|F|F#|Gb|G|G#|Ab|A|A#|Bb|B)m?", body.key):
                raise ValueError("无效的主调")
            edits = {}
            for edit in body.edits:
                semantic = parse_chord(edit.label)
                if (edit.index >= len(source["rows"]) or edit.index in edits or semantic.parse_error
                        or semantic.state == "missing" or any(c.isspace() for c in edit.label)):
                    raise ValueError("修改包含无效和弦、重复序号或不存在的区间")
                edits[edit.index] = edit.label
            record = body.model_dump()
            record.update(schema="harmony_web_corrections_v1", revision=body.revision + 1, saved_at=now(),
                          human_edited_unverified=True, timing_changed=False,
                          project_id=project_id, audio_sha256=audio_sha, duration=duration)
            # One committed JSON snapshot is authoritative; exports derive from it.
            # This avoids serving a partly updated JSON/LAB pair after a crash.
            atomic_json(path, record)
            text = corrected_lab(source, record)
            atomic_text(directory / "corrected.lab", text)
            return record

    def corrected_lab(source, record):
        edits = {item["index"]: item["label"] for item in record.get("edits", [])}
        return "".join(row["start_text"] + "\t" + row["end_text"] + "\t" + edits.get(i, row["label"]) + "\n"
                       for i, row in enumerate(source["rows"]))

    @application.get("/api/projects/{project_id}/export/{source_id}/{kind}")
    def export(project_id: str, source_id: str, kind: str):
        if kind == "original.lab":
            from .editor import context
            _, result, source, _, _ = context(store, project_id, source_id)
            return FileResponse(store.run_dir(project_id, result["run_id"]) / source["path"], filename="original.lab")
        from .corrections import context_and_record
        with save_lock:
            _, _, source, _, _, record = context_and_record(store, project_id, source_id)
        if kind == "corrected.lab":
            return Response(corrected_lab(source, record), media_type="text/plain; charset=utf-8",
                            headers={"Content-Disposition": 'attachment; filename="corrected.lab"'})
        if kind == "corrections.json":
            return JSONResponse(record, headers={"Content-Disposition": 'attachment; filename="corrections.json"'})
        raise HTTPException(404, "不存在的导出格式")

    @application.get("/assets/notation.js")
    def notation():
        return Response(notation_script(), media_type="text/javascript")

    application.mount("/assets", StaticFiles(directory=ROOT / "frontend"), name="assets")

    @application.get("/")
    def index():
        return FileResponse(ROOT / "frontend/index.html", headers={"Cache-Control": "no-cache"})

    @application.get("/stems")
    def stems_page():
        return FileResponse(ROOT / "frontend/stems.html", headers={"Cache-Control": "no-cache"})

    @application.get("/{page}")
    def named_page(page: str):
        path = ROOT / "frontend" / f"{page}.html"
        if not path.is_file():
            raise HTTPException(404, "没有这个页面")
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    return application


app = create_app()
