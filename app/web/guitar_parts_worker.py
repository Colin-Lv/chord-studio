"""Source- and recipe-bound guitar parts; do not replace the parent guitar."""
from functools import lru_cache
from pathlib import Path
import os,subprocess,time
from uuid import uuid4
from .storage import ROOT,atomic_json,now,read_json,signature
from .stems_worker import MODELS,stems_dir,_runtime,_annotate,_load,_to_mp3,stem_wav
from app.runtime_paths import models_root

MODEL_DIR=models_root()/'guitar'
RECIPES={
    'timbre':dict(label='木吉他／电吉他',files=[f'bs_mega_53stem_{k}-guitar_mvsep{s}' for k in ['acoustic','electric'] for s in ['.ckpt','_config.yaml']],
        parts=[('acoustic_guitar','木吉他（估计）','ACOU','#d9b082'),('electric_guitar','电吉他（估计）','ELEC','#c49ce8'),('guitar_residual','吉他余量','REST','#8a9da6')],
        note='木吉他与电吉他为模型估计；余量保留未归类声音及分离误差，不代表第三把吉他。'),
    'duet':dict(label='旋律／伴奏吉他',files=['guitar_role_v1.ckpt','mbr_lead_rhythm_guitar_listra92_config.yaml'],
        parts=[('guitar_one','吉他旋律（估计）','LEAD','#d9b082'),('guitar_two','吉他伴奏（估计）','COMP','#c49ce8')],
        note='按演奏角色估计，不表示已确认有两把吉他；单把指弹也可能被拆开，相似声部仍可能串音。收起可恢复完整吉他。'),
}
@lru_cache(maxsize=32)
def _hash(path,size,mtime):return signature(path)['sha256']
def digest(path):
    stat=path.stat();return _hash(str(path),stat.st_size,stat.st_mtime_ns)
def fingerprint(mode):
    if mode not in RECIPES:return None
    files=[MODEL_DIR/n for n in RECIPES[mode]['files']]+[ROOT/'app/web/guitar_parts_runner.py',ROOT/'app/web/guitar_engine.py']
    if not all(p.is_file() for p in files):return None
    return {p.name:digest(p) for p in files}
def parts_manifest(sha,model,mode):return stems_dir(sha,model)/'guitar_parts'/mode/'manifest.json'
def valid_manifest(sha,model,mode):
    if model not in MODELS or mode not in RECIPES:return None
    base=stems_dir(sha,model);path=parts_manifest(sha,model,mode);parent=stem_wav(sha,model,'guitar')
    if not path.is_file() or not parent.is_file():return None
    manifest=read_json(path)
    if manifest.get('parent_sha256')!=digest(parent) or manifest.get('fingerprint')!=fingerprint(mode):return None
    if [t.get('id') for t in manifest.get('tracks',[])]!=[p[0] for p in RECIPES[mode]['parts']]:return None
    for t in manifest['tracks']:
        for key in ['file','wav']:
            path=(base/t.get(key,'')).resolve()
            if not path.is_relative_to(base.resolve()) or not path.is_file():return None
    return manifest
def execute(store,jobid):
    job_path=store.job_path(jobid);job=read_json(job_path);project=store.project(job['project_id'])
    model=job.get('model') or 'bs_roformer_sw';mode=job.get('guitar_mode') or 'timbre'
    if model not in MODELS or mode not in RECIPES:raise ValueError('未知的吉他细分选项')
    audio=store.project_dir(project['id'])/project['audio'];sha=signature(audio)['sha256']
    base=stems_dir(sha,model);parent=stem_wav(sha,model,'guitar')
    if not parent.is_file() or not (base/'manifest.json').is_file():raise ValueError('请先生成六轨精细版的吉他轨')
    cached=valid_manifest(sha,model,mode)
    if cached and not job.get('force'):return dict(cached,cached=True)
    identity=digest(parent);frozen=fingerprint(mode)
    if not frozen:raise ValueError('吉他模型文件尚未准备完整')
    gen=uuid4().hex;output=base/'guitar_parts'/gen;output.parent.mkdir(parents=True,exist_ok=True)
    python,device=_runtime();start=time.perf_counter()
    job.update(stage='separate',stage_label='正在细分吉他：'+RECIPES[mode]['label']);atomic_json(job_path,job)
    env=os.environ.copy();env['PYTHONUTF8']='1'
    with (output.parent/(gen+'.log')).open('w',encoding='utf-8') as log:
        result=subprocess.run([python,'-B',str(ROOT/'app/web/guitar_parts_runner.py'),'--input',str(parent),'--outdir',str(output),'--mode',mode,'--model-dir',str(MODEL_DIR)],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=3600,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if result.returncode:raise RuntimeError('吉他细分失败，详情见任务输出日志')
    if digest(parent)!=identity or fingerprint(mode)!=frozen:raise RuntimeError('源吉他或模型已更新，请重新细分')
    parent_manifest=read_json(base/'manifest.json');data,sr=_load(parent);duration=len(data)/sr
    bars=[float(b['start']) for b in parent_manifest.get('bars',[])];tracks=[]
    job.update(stage_label='正在整理吉他音轨和试听文件');atomic_json(job_path,job)
    for stem,name,short,color in RECIPES[mode]['parts']:
        wav=output/(stem+'.wav');mp3=output/(stem+'.mp3');_to_mp3(wav,mp3)
        t=dict(id=stem,name=name,short=short,color=color,instrument=name,parent_id='guitar',file=mp3.relative_to(base).as_posix(),wav=wav.relative_to(base).as_posix())
        _annotate(wav,t,bars,duration);tracks.append(t)
    loudest=max(t['rms_db'] for t in tracks)
    for t in tracks:t['level_factor']=round(max(.05,min(1,10**((t['rms_db']-loudest)/20))),3)
    manifest=dict(schema='chord_studio_guitar_parts_v1',project_id=project['id'],audio_sha256=sha,parent_sha256=identity,model=model,
        mode=mode,label=RECIPES[mode]['label'],generation=gen,created_at=now(),fingerprint=frozen,tracks=tracks,
        duration=duration,bars=parent_manifest.get('bars',[]),inference=read_json(output/'inference.json'),
        seconds=time.perf_counter()-start,note=RECIPES[mode]['note'])
    if read_json(job_path).get('status')=='interrupted':raise RuntimeError('任务已被服务重启中断')
    atomic_json(parts_manifest(sha,model,mode),manifest)
    return manifest
