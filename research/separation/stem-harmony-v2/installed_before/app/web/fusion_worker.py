"""Drum-free harmonic fusion job: second pass over an existing analysis."""
from __future__ import annotations

import time
from pathlib import Path

from .fusion_engine import fuse_project
from .storage import atomic_json, now, read_json


def execute(store, job_id):
    job_path = store.job_path(job_id)
    job = read_json(job_path)
    project = store.project(job["project_id"])
    model = job.get("model") or "bs_roformer_sw"

    def publish(**update):
        job.update(**update)
        atomic_json(job_path, job)

    publish(stage="fusion", stage_label="正在融合混音与和声轨")
    started = time.perf_counter()
    meta = fuse_project(store, project, model, make_default=bool(job.get("make_default", True)))
    project["stem_fusion"] = dict(model=model, run_id=project.get("current_run"),
                                  updated_at=now(), segments=meta["segments"],
                                  seconds=meta["seconds"])
    store.save_project(project)
    publish(stage_label="分轨融合完成")
    return dict(meta, total_seconds=round(time.perf_counter() - started, 1))
