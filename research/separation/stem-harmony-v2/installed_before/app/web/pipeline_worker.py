"""One-button pipeline: analysis -> separation -> optional sub-splits -> harmonic fusion.

The serial job queue means these cannot be separate jobs, so the pipeline runs each stage
worker in turn against the same job id (which also becomes the analysis run id).
"""
from __future__ import annotations

import time
from pathlib import Path

from .storage import atomic_json, now, read_json

STAGE_LABELS = {
    "analysis": "和声分析",
    "stems": "音轨分离",
    "vocal_parts": "人声细分",
    "guitar_parts": "吉他细分",
    "drums_parts": "鼓组细分",
    "fusion": "分轨融合和声",
}


def _worker(profile: str):
    if profile == "retained10":
        from .retained_worker import execute
        return execute
    if profile == "stems":
        from .stems_worker import execute
        return execute
    if profile == "vocal_parts":
        from .vocal_parts_worker import execute
        return execute
    if profile == "guitar_parts":
        from .guitar_parts_worker import execute
        return execute
    if profile == "drums_parts":
        from .drums_worker import execute
        return execute
    if profile == "fusion":
        from .fusion_worker import execute
        return execute
    raise ValueError("未知流程步骤：" + profile)


def execute(store, job_id):
    job_path = store.job_path(job_id)
    job = read_json(job_path)
    spec = dict(job.get("pipeline") or {})
    project_id = job["project_id"]
    model = spec.get("model") or "bs_roformer_sw"
    steps = [("retained10", "analysis")]
    steps.append(("stems", "stems"))
    if spec.get("vocal_parts"):
        steps.append(("vocal_parts", "vocal_parts"))
    if spec.get("guitar_parts"):
        steps.append(("guitar_parts", "guitar_parts"))
    if spec.get("drums_parts"):
        steps.append(("drums_parts", "drums_parts"))
    if spec.get("harmony", True):
        steps.append(("fusion", "fusion"))

    records = job.setdefault("stages", [])
    for index, (profile, key) in enumerate(steps, start=1):
        label = f"{index}/{len(steps)} {STAGE_LABELS.get(key, key)}"
        job = read_json(job_path)
        job.update(profile=profile, pipeline_step=key, pipeline_index=index,
                   stage_label=f"{label} …")
        if key == "stems":
            job["model"] = model
            job["force"] = bool(spec.get("force_stems"))
        if key == "vocal_parts":
            job["model"] = model
        if key == "guitar_parts":
            job["model"] = model
            job["guitar_mode"] = spec.get("guitar_mode") or "timbre"
        if key == "drums_parts":
            job["model"] = model
        if key == "fusion":
            job["model"] = model
            job["make_default"] = bool(spec.get("harmony_default", True))
        atomic_json(job_path, job)
        started = time.perf_counter()
        _worker(profile)(store, job_id)
        job = read_json(job_path)
        records.append(dict(id=key, label=label, status="completed",
                            seconds=round(time.perf_counter() - started, 1)))
        job["stages"] = records
        if key == "analysis":
            project = store.project(project_id)
            project["current_run"] = job_id
            store.save_project(project)
        atomic_json(job_path, job)

    job = read_json(job_path)
    job.update(profile="pipeline", stage="complete", stage_label="完整流程完成",
               pipeline_step="complete")
    atomic_json(job_path, job)
    return dict(stages=records)
