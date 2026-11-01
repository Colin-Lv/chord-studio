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
    from .stems_worker import MODELS
    if model not in MODELS:
        raise ValueError('未知的分轨模型')

    def publish(**update):
        job.update(**update)
        atomic_json(job_path, job)

    publish(stage="fusion", stage_label="正在识别分轨和弦")
    started = time.perf_counter()
    spec = job.get('pipeline') or {}
    if spec.get('harmony_method') == 'legacy_fusion':
        meta = fuse_project(store, project, model, make_default=bool(job.get('make_default', False)))
    else:
        from .stem_harmony import recognize, publish_result, VERSION
        from .stems_worker import stems_dir
        from .storage import signature
        if not project.get('current_run'):
            raise ValueError('请先完成和声分析，或选择完整流程')
        run_dir = store.run_dir(project['id'], project['current_run'])
        if not (run_dir / 'retained10/posterior.npz').exists():
            raise ValueError('此版本没有当前模型的声学证据，请先运行当前冻结模型')
        audio = store.project_dir(project['id']) / project['audio']
        base = stems_dir(signature(audio)['sha256'], model)
        if not (base / 'manifest.json').exists():
            from .stems_worker import execute as separate
            separate(store, job_id)
            job.update(read_json(job_path))
        output = run_dir / VERSION / model
        segments, meta = recognize(run_dir, base, output,
                                  lambda message: publish(stage='fusion', stage_label=message))
        if signature(audio)['sha256'] != meta['source_audio_sha256']:
            raise ValueError('分析期间原音频发生变化，未发布结果')
        if store.project(project['id']).get('current_run') != project['current_run']:
            raise ValueError('分析版本已改变，未发布结果，请重试')
        # Keep V2 available for comparison. The bass-aware revision is a separate
        # immutable source, so saved corrections and earlier results remain valid.
        meta['source_id'] = publish_result(run_dir, segments, meta, False)
        from .stem_bass_harmony import (recognize as recognize_bass,
                                        publish_result as publish_bass, VERSION as BASS_VERSION)
        segments, meta = recognize_bass(run_dir, base, run_dir / BASS_VERSION / model,
                                       lambda message: publish(stage='fusion', stage_label=message))
        if (signature(audio)['sha256'] != meta['source_audio_sha256']
                or store.project(project['id']).get('current_run') != project['current_run']):
            raise ValueError('分析版本或音频已改变，未发布低音和弦结果')
        meta['source_id'] = publish_bass(run_dir, segments, meta, bool(job.get('make_default', False)))
        project = store.project(project['id'])
    project["stem_fusion"] = dict(model=model, run_id=project.get("current_run"),
                                  updated_at=now(), segments=meta["segments"],
                                  seconds=meta["seconds"])
    store.save_project(project)
    publish(stage_label="分轨和弦识别完成")
    return dict(meta, total_seconds=round(time.perf_counter() - started, 1))
