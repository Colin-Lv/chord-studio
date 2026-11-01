"""Wire the pipeline and fusion jobs into dispatch, the queue and the HTTP API."""
from __future__ import annotations

from pathlib import Path

BACKEND = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web")

# ---- worker dispatch
worker = BACKEND / "worker.py"
text = worker.read_text(encoding="utf-8")
OLD = """        elif job.get("profile") == "vocal_parts":"""
NEW = """        elif job.get("profile") == "pipeline":
            from .pipeline_worker import execute as execute_pipeline
            execute_pipeline(store, args.job)
        elif job.get("profile") == "fusion":
            from .fusion_worker import execute as execute_fusion
            execute_fusion(store, args.job)
        elif job.get("profile") == "vocal_parts":"""
assert text.count(OLD) == 1
worker.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("worker.py: pipeline + fusion dispatch")

# ---- jobs queue
jobs = BACKEND / "jobs.py"
text = jobs.read_text(encoding="utf-8")

OLD_SUBMIT = """    def submit(self, project_id, profile, download=False, key_mode="madmom",
               force=False, model=None, guitar_mode=None):"""
NEW_SUBMIT = """    def submit(self, project_id, profile, download=False, key_mode="madmom",
               force=False, model=None, guitar_mode=None, pipeline=None):"""
assert text.count(OLD_SUBMIT) == 1
text = text.replace(OLD_SUBMIT, NEW_SUBMIT)

OLD_ALLOW = """        if profile not in PROFILES and profile not in {"stems", "drums_parts", "vocal_parts", "guitar_parts"}:
            raise ValueError("未知模型选项")"""
NEW_ALLOW = """        if profile not in PROFILES and profile not in {"stems", "drums_parts", "vocal_parts",
                                                       "guitar_parts", "pipeline", "fusion"}:
            raise ValueError("未知模型选项")"""
assert text.count(OLD_ALLOW) == 1, text.count(OLD_ALLOW)
text = text.replace(OLD_ALLOW, NEW_ALLOW)

OLD_JOB = """                       force=bool(force), model=model, guitar_mode=guitar_mode,"""
NEW_JOB = """                       force=bool(force), model=model, guitar_mode=guitar_mode,
                       pipeline=pipeline,"""
assert text.count(OLD_JOB) == 1
text = text.replace(OLD_JOB, NEW_JOB)

OLD_STATUS = """            project["status"] = ("stems" if job.get("profile") in {"stems", "drums_parts", "vocal_parts", "guitar_parts"}
                                 else "running")"""
NEW_STATUS = """            project["status"] = ("stems" if job.get("profile") in {"stems", "drums_parts",
                                                                    "vocal_parts", "guitar_parts", "fusion"}
                                 else "running")"""
assert text.count(OLD_STATUS) == 1, text.count(OLD_STATUS)
text = text.replace(OLD_STATUS, NEW_STATUS)

OLD_DONE = """                if job.get("profile") in {"stems", "drums_parts", "vocal_parts", "guitar_parts"}:"""
NEW_DONE = """                if job.get("profile") in {"stems", "drums_parts", "vocal_parts", "guitar_parts", "fusion"}:"""
assert text.count(OLD_DONE) == 1
text = text.replace(OLD_DONE, NEW_DONE)

OLD_MAP = """                               stage_label={"stems":"分轨完成","vocal_parts":"人声细分完成","drums_parts":"鼓组细分完成","guitar_parts":"吉他细分完成"}[job['profile']],"""
NEW_MAP = """                               stage_label={"stems":"分轨完成","vocal_parts":"人声细分完成","drums_parts":"鼓组细分完成","guitar_parts":"吉他细分完成","fusion":"分轨融合完成"}[job['profile']],"""
assert text.count(OLD_MAP) == 1
text = text.replace(OLD_MAP, NEW_MAP)
jobs.write_text(text, encoding="utf-8")
print("jobs.py: pipeline + fusion profiles")

# ---- server endpoints
server = BACKEND / "server.py"
text = server.read_text(encoding="utf-8")

OLD_MODEL = """class SeparateRequest(BaseModel):"""
NEW_MODEL = '''class PipelineRequest(BaseModel):
    model: str = "bs_roformer_sw"
    vocal_parts: bool = False
    guitar_parts: bool = False
    drums_parts: bool = False
    guitar_mode: Literal["timbre", "duet"] = "timbre"
    harmony: bool = True
    harmony_default: bool = True
    force_stems: bool = False


class SeparateRequest(BaseModel):'''
assert text.count(OLD_MODEL) == 1
text = text.replace(OLD_MODEL, NEW_MODEL)

OLD_ROUTE = """    @application.post("/api/projects/{project_id}/drums-parts", status_code=202)"""
NEW_ROUTE = '''    @application.post("/api/projects/{project_id}/pipeline", status_code=202)
    def pipeline(project_id: str, body: PipelineRequest):
        from .stems_worker import MODELS
        if body.model not in MODELS:
            raise HTTPException(400, "未知的分轨模型")
        spec = body.model_dump()
        return application.state.jobs.submit(project_id, "pipeline", key_mode="continuous",
                                             model=body.model, pipeline=spec,
                                             force=body.force_stems)

    @application.post("/api/projects/{project_id}/fusion", status_code=202)
    def fusion(project_id: str, body: SeparateRequest | None = None):
        body = body or SeparateRequest()
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

    @application.post("/api/projects/{project_id}/drums-parts", status_code=202)'''
assert text.count(OLD_ROUTE) == 1
text = text.replace(OLD_ROUTE, NEW_ROUTE)
server.write_text(text, encoding="utf-8")
print("server.py: pipeline + fusion endpoints")
