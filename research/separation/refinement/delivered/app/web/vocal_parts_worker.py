"""Versioned second-level vocal estimates; never overwrite the parent stem."""
import os
from functools import lru_cache
from pathlib import Path
import subprocess
import time
from uuid import uuid4

from .storage import ROOT, atomic_json, now, read_json, signature
from .stems_worker import _annotate, _load, _runtime, _to_mp3, stems_dir, MODELS
from .vocal_parts_runner import MODEL, CONFIG

PARTS = (
    dict(id='lead_vocals', name='主唱（估计）', short='LEAD', instrument='主唱', color='#b6e7ce'),
    dict(id='backing_vocals', name='和声／人声余量', short='BACK', instrument='背景和声、混响与分离余量', color='#9bc3ea'),
)


@lru_cache(maxsize=16)
def _file_hash(path, size, mtime_ns):
    return signature(path)['sha256']


def model_fingerprint():
    folder=Path('D:/ChordStudio/data/models/audio_separator')
    result={}
    for key,name in [('weights_sha256',MODEL),('config_sha256',CONFIG)]:
        path=folder/name
        if not path.is_file():return None
        stat=path.stat()
        result[key]=_file_hash(str(path),stat.st_size,stat.st_mtime_ns)
    return result


def parts_manifest(sha, model):
    return stems_dir(sha,model)/'vocal_parts'/'manifest.json'


def valid_manifest(sha, model):
    if model not in MODELS:
        return None
    path=parts_manifest(sha,model)
    parent=stems_dir(sha,model)/'wav/vocals.wav'
    if not path.is_file() or not parent.is_file():
        return None
    data=read_json(path)
    if len(data.get('tracks',[])) != 2:
        return None
    if data.get('parent_sha256') != signature(parent)['sha256']:
        return None
    if data.get('inference',{}).get('runner_sha256') != signature(ROOT/'app/web/vocal_parts_runner.py')['sha256']:
        return None
    current=model_fingerprint()
    if current is None or any(data.get('inference',{}).get(k)!=v for k,v in current.items()):
        return None
    base=stems_dir(sha,model)
    if not all((base/t[k]).is_file() for t in data.get('tracks',[]) for k in ('file','wav')):
        return None
    return data


def execute(store, job_id):
    job_path=store.job_path(job_id);job=read_json(job_path)
    project=store.project(job['project_id'])
    audio=store.project_dir(project['id'])/project['audio']
    sha=signature(audio)['sha256'];model=job.get('model') or 'htdemucs'
    if model not in MODELS:raise ValueError('未知的分轨模型')
    base_dir=stems_dir(sha,model);parent=base_dir/'wav/vocals.wav'
    if not parent.is_file() or not (base_dir/'manifest.json').is_file():
        raise ValueError('请先生成人声轨，再细分主唱与和声')
    cached=valid_manifest(sha,model)
    if cached and not job.get('force'):
        return dict(cached,cached=True)
    parent_sha=signature(parent)['sha256']
    base=read_json(base_dir/'manifest.json')
    generation=uuid4().hex
    target=base_dir/'vocal_parts'/generation
    python,device=_runtime()
    job.update(stage='separate',stage_label='正在分离主唱与背景和声',device=device)
    atomic_json(job_path,job)
    env=os.environ.copy();env['PYTHONUTF8']='1'
    start=time.perf_counter()
    proc=subprocess.run([python,'-B',str(ROOT/'app/web/vocal_parts_runner.py'),
                         '--input',str(parent),'--outdir',str(target)],cwd=ROOT,
                        env=env,capture_output=True,text=True,timeout=3600,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    (target.parent/(generation+'.log')).write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
    if proc.returncode:raise RuntimeError('人声细分失败：'+(proc.stderr or proc.stdout)[-1500:])
    if signature(parent)['sha256']!=parent_sha:raise RuntimeError('人声源轨已变化，请重新运行细分')
    info=read_json(target/'inference.json')
    data,sr=_load(parent);duration=len(data)/sr
    bars=[float(b['start']) for b in base.get('bars',[])]
    tracks=[]
    for spec in PARTS:
        stem=spec['id'];wav=target/(stem+'.wav');mp3=target/(stem+'.mp3')
        _to_mp3(wav,mp3)
        track=dict(spec,parent_id='vocals',file=mp3.relative_to(base_dir).as_posix(),
                   wav=wav.relative_to(base_dir).as_posix())
        _annotate(wav,track,bars,duration);tracks.append(track)
    loudest=max(t['rms_db'] for t in tracks)
    for t in tracks:t['level_factor']=round(max(.05,min(1,10**((t['rms_db']-loudest)/20))),3)
    result=dict(schema='chord_studio_vocal_parts_v1',project_id=project['id'],
                audio_sha256=sha,parent_sha256=parent_sha,model=model,generation=generation,
                created_at=now(),cached=False,tracks=tracks,duration=duration,
                bars=base.get('bars',[]),inference=info,seconds=time.perf_counter()-start,
                note='主唱为模型估计；和声轨还包含混响、串音及分离余量，尚未拆成逐条和声。')
    atomic_json(parts_manifest(sha,model),result)
    job.update(stage_label='主唱与和声细分完成');atomic_json(job_path,job)
    return result
