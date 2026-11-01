"""Default two-branch harmony inference: frozen chord evidence + learned bass.

Separation is an internal prerequisite, not a user opt-in. The old 120-class
weights stay reproducible; their final public output now includes slash bass.
"""
from pathlib import Path
from .storage import read_json,atomic_json,signature
from . import stems_worker,stem_bass_harmony

VERSION='joint-harmony-v1'
DEFAULT_SEPARATOR='bs_roformer_sw'


def execute(store,job_id,run_dir=None):
    job_path=store.job_path(job_id);job=read_json(job_path)
    project=store.project(job['project_id']);original_run=project.get('current_run')
    folder=Path(run_dir) if run_dir is not None else store.run_dir(project['id'],project['current_run'])
    analysis=read_json(folder/'analysis.json')
    audio=store.project_dir(project['id'])/project['audio'];audio_hash=signature(audio)['sha256']
    if analysis.get('audio',{}).get('sha256')!=audio_hash or project['audio_identity']['sha256']!=audio_hash:
        raise ValueError('和弦与原音频身份不一致，不能合并指定低音')
    spec=job.get('pipeline') or {}
    model=spec.get('model') or job.get('model') or DEFAULT_SEPARATOR
    if model not in stems_worker.MODELS:raise ValueError('指定低音识别的分轨模型无效')

    def progress(message):
        current=read_json(job_path)
        if current.get('status')=='interrupted':raise RuntimeError('任务已中断')
        current.update(stage='joint_harmony',stage_label=message);atomic_json(job_path,current)

    progress('准备和弦与指定低音识别，复用或自动分离伴奏')
    # Explicit force applies to the pipeline's separation step. Here an already
    # prepared separation must be reused instead of running it a second time.
    stems_worker.execute(store,job_id,model_override=model,force_override=False)
    base=stems_worker.stems_dir(audio_hash,model)
    rows,meta=stem_bass_harmony.recognize(folder,base,folder/VERSION/model,progress)
    if (signature(audio)['sha256']!=audio_hash
            or store.project(project['id']).get('current_run')!=original_run):
        raise ValueError('识别期间音频或分析版本改变，未发布指定低音结果')
    if read_json(job_path).get('status')=='interrupted':raise RuntimeError('任务已中断')
    meta.update(pipeline=VERSION,automatic=True,separator=model,
                output_semantics='independent chord root/quality and observed bass; slash need not be chord tone')
    source_id=stem_bass_harmony.publish_result(folder,rows,meta,make_default=True,integrated=True)
    progress('和弦与指定低音识别完成')
    return dict(source_id=source_id,segments=len(rows),separator=model)
