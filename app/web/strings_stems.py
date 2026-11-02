"""Versioned seven-track results. Commit a manifest only after every asset is ready."""
from pathlib import Path
from functools import lru_cache
from uuid import uuid4
import os,subprocess,time
import numpy as np
from .storage import ROOT,atomic_json,now,read_json,signature
from app.runtime_paths import models_root
from .stems_worker import (MODELS,stems_dir,stems_for,data_root,_runtime,_decode_to_wav,
                         _ffmpeg,_run,_load,_annotate,beat_grid,_latest_beats_run)


@lru_cache(maxsize=64)
def _hash(path,size,mtime):return signature(path)['sha256']


def digest(path):
    path=Path(path);s=path.stat();return _hash(str(path),s.st_size,s.st_mtime_ns)


def fingerprint(model):
    if model not in {'strings_v1','strings_v2'}:raise ValueError('未知的提琴组版本')
    name='bs_bowed_str_gilliaaan' if model=='strings_v1' else 'bs_bowed_str2_gilliaaan'
    models=models_root()
    paths=[models/'bowed-strings'/(name+s) for s in ['.ckpt','_config.yaml']]
    paths += [models/'audio_separator'/('BS-Roformer-SW'+s) for s in ['.ckpt','.yaml']]
    paths += [ROOT/p for p in ['app/web/strings_runner.py','app/web/strings_stems.py',
                               'app/web/vendor/msst_model_utils.py','app/web/guitar_engine.py','app/web/separation_cache.py']]
    if not all(p.is_file() for p in paths):raise ValueError('提琴组或六轨模型文件未安装完整')
    return {p.name:digest(p) for p in paths}


def compatible_recipe(saved,current):
    if saved==current:return True
    # Only this explicitly verified orchestration migration can reuse old full
    # results. Future edits invalidate the alias unless revalidated and recorded.
    alias=ROOT/'app/web/models/strings_cache_migration.json'
    if not alias.is_file():return False
    migration=read_json(alias)
    if any(current.get(k)!=v for k,v in migration['new_code'].items()):return False
    if any(saved.get(k)!=v for k,v in migration['old_code'].items()):return False
    adjusted=dict(current)
    for key in migration['new_code']:adjusted.pop(key,None)
    adjusted.update(migration['old_code'])
    return adjusted==saved


def cached_manifest(sha,model,recipe):
    base=stems_dir(sha,model);path=base/'manifest.json'
    if not path.is_file():return None
    m=read_json(path)
    if m.get('audio_sha256')!=sha or not compatible_recipe(m.get('fingerprint',{}),recipe):return None
    if [t['id'] for t in m.get('tracks',[])]!=[t['id'] for t in stems_for(model)]:return None
    for t in m['tracks']:
        for field in ['file','wav']:
            p=(base/t[field]).resolve()
            if not p.is_relative_to(base.resolve()) or not p.is_file():return None
    reference=(base/m.get('reference_file','')).resolve()
    if not reference.is_relative_to(base.resolve()) or not reference.is_file():return None
    return m


def execute(store,job_id,model,force=False):
    job_path=store.job_path(job_id);job=read_json(job_path);project=store.project(job['project_id'])
    audio=store.project_dir(project['id'])/project['audio'];sha=signature(audio)['sha256']
    recipe=fingerprint(model);base=stems_dir(sha,model);started=time.perf_counter()
    def publish(label):
        current=read_json(job_path)
        if current.get('status')=='interrupted':raise RuntimeError('任务已中断')
        current.update(stage='separate',stage_label=label);atomic_json(job_path,current)
    def save_project(m):
        current=store.project(project['id']);saved=dict(current.get('stems') or {})
        saved[model]=dict(sha256=sha,manifest=str(base/'manifest.json'),updated_at=now(),model=model,bpm=m['bpm'])
        current['stems']=saved;store.save_project(current)
    cached=cached_manifest(sha,model,recipe)
    if cached and not force:
        save_project(cached);publish('复用已保存的提琴组与六轨');return dict(cached,cached=True)
    generation=uuid4().hex;out=base/'generations'/generation;out.mkdir(parents=True)
    source=out/'source.wav';wavdir=out/'wav'
    publish('正在准备原曲');_decode_to_wav(audio,source)
    python,device=_runtime();env=os.environ.copy();env['PYTHONUTF8']='1'
    command=[python,'-B',str(ROOT/'app/web/strings_runner.py'),'--input',str(source),
             '--outdir',str(wavdir),'--version',model[-2:],
             '--models',str(models_root()/'bowed-strings'),
             '--sw-models',str(models_root()/'audio_separator'),
             '--stage-cache',str(data_root()/'cache/separation-stages')]
    if force:command.append('--force')
    publish('正在提取提琴组，随后分离其余六轨')
    with (out/'inference.log').open('w',encoding='utf8') as log:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
            timeout=5400,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if result.returncode:raise RuntimeError('提琴组分离失败；已有结果保留。详情见 '+str(out/'inference.log'))
    info=read_json(wavdir/'inference.json');x,sr=_load(source);duration=len(x)/sr
    if info['output_frames']!=len(x):raise ValueError('原曲与分轨长度不一致')
    run=store.run_dir(project['id'],project['current_run']) if project.get('current_run') else None
    if run is None or not (run/'beats.json').is_file():run=_latest_beats_run(store.project_dir(project['id']))
    grid=beat_grid(source,duration,run);bars=[float(t) for t in grid['bars'] if 0<=t<duration]
    if not bars or bars[0]>0:bars.insert(0,0.)
    tracks=[];peak=float(np.max(np.abs(x)))
    for spec in stems_for(model):
        path=wavdir/(spec['id']+'.wav');y,rate=_load(path)
        if y.shape!=x.shape or rate!=sr or not np.isfinite(y).all():raise ValueError('音轨时长或数值异常：'+spec['id'])
        peak=max(peak,float(np.abs(y).max()))
    # One common preview gain, including reference and any later sub-splits.
    gain=min(1.,.75/max(peak,1e-9))
    publish('正在整理提琴组、试听与小节视图')
    def encode(wav,mp3):
        r=_run([_ffmpeg(),'-hide_banner','-loglevel','error','-y','-i',wav,'-af',f'volume={gain:.10f}',
                '-codec:a','libmp3lame','-b:a','192k',mp3])
        if r.returncode:raise RuntimeError('试听编码失败')
    for spec in stems_for(model):
        name=spec['id'];wav=wavdir/(name+'.wav');mp3=out/(name+'.mp3');encode(wav,mp3)
        track=dict(spec,file=mp3.relative_to(base).as_posix(),wav=wav.relative_to(base).as_posix())
        _annotate(wav,track,bars,duration);tracks.append(track)
    encode(source,out/'reference.mp3')
    m=dict(schema='chord_studio_stems_v1',project_id=project['id'],audio_sha256=sha,generation=generation,
        created_at=now(),cached=False,fingerprint=recipe,source_audio=project['audio'],
        model=dict(name=model,label=MODELS[model]['label'],device=device,seconds=round(time.perf_counter()-started,1),
                   package='Gilliaan BowedStrings + BS-Roformer-SW'),
        duration=duration,sample_rate=sr,bpm=round(float(grid['bpm']),2),bpm_source=grid['bpm_source'],
        bars=[dict(index=i+1,start=t,end=bars[i+1] if i+1<len(bars) else duration) for i,t in enumerate(bars)],
        beats=grid['beats'][:4000],tracks=tracks,preview_gain=gain,
        reference_file=(out/'reference.mp3').relative_to(base).as_posix(),
        reconstruction=dict(relative_db=info['sum_error_relative_db']),inference=info,
        note='提琴组直接来自原曲；其余六轨来自扣除提琴组后的音频。两个版本分别保存。')
    if signature(audio)['sha256']!=sha or fingerprint(model)!=recipe:raise RuntimeError('音频或模型已变化，未覆盖已保存结果')
    publish('正在保存提琴组与六轨')
    atomic_json(out/'manifest.json',m)
    atomic_json(base/'manifest.json',m)
    save_project(m);publish('提琴组与六轨已完成')
    return m
