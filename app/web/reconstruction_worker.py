"""Immutable whole-song side assets. Publish only a verified, complete build."""
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4
from .storage import ROOT, read_json, atomic_json, signature, now
from .piano_worker import latest as latest_piano, execute as transcribe_piano
from .stems_worker import stems_dir, execute as separate, _decode_to_wav

from app.runtime_paths import python_path
PYTHON=python_path('reconstruction')


def latest(store,project_id):
    project=store.project(project_id);record=project.get('song_reconstruction')
    if not record:raise FileNotFoundError('还没有整曲重构结果，请先生成')
    folder=store.project_dir(project_id)/'reconstruction'/record['generation']
    manifest=read_json(folder/'manifest.json')
    if signature(store.project_dir(project_id)/project['audio'])['sha256']!=manifest['audio_sha256']:
        raise FileNotFoundError('原曲已更换，请重新生成')
    return folder,manifest


def execute(store,job_id):
    job_path=store.job_path(job_id);job=read_json(job_path);pid=job['project_id']
    project=store.project(pid);audio=store.project_dir(pid)/project['audio'];audio_hash=signature(audio)['sha256']
    def stage(label):
        current=read_json(job_path);current.update(stage='reconstruction',stage_label=label);atomic_json(job_path,current)
    stage('准备已有分轨与钢琴 MIDI')
    separate(store,job_id,model_override='bs_roformer_sw',force_override=False)
    try:
        piano,pm=latest_piano(store,pid)
        if pm.get('input_kind')!='mix':raise FileNotFoundError('需要原曲的钢琴分轨')
    except FileNotFoundError:
        transcribe_piano(store,job_id);piano,pm=latest_piano(store,pid)
    stems=stems_dir(audio_hash,'bs_roformer_sw')/'wav'
    if signature(stems/'piano.wav')['sha256']!=pm['source_sha256']:
        transcribe_piano(store,job_id);piano,pm=latest_piano(store,pid)
    generation=uuid4().hex;output=store.project_dir(pid)/'reconstruction'/generation;output.mkdir(parents=True)
    original=output/'original-input.wav';_decode_to_wav(audio,original)
    beats=read_json(piano/'beats.json')
    harmony=None
    if project.get('current_run'):
        analysis_path=store.run_dir(pid,project['current_run'])/'analysis.json'
        if analysis_path.exists():
            analysis=read_json(analysis_path)
            if analysis.get('audio',{}).get('sha256')==audio_hash:
                selected=next((s for s in analysis.get('sources',[]) if s['id']==analysis.get('default_source')),None)
                if selected:harmony=dict(run=project['current_run'],source=selected['id'],rows=selected['rows'],
                                         status='existing model reference; not ground truth; no note modification')
    config=dict(project_id=pid,generation=generation,title=project['title'],output=str(output),stems=str(stems),piano=str(piano),
                piano_inference=pm['inference'],piano_generation=pm['generation'],audio_sha256=audio_hash,
                duration=pm['duration'],beats=beats,original_wav=str(original),analysis_run=project.get('current_run'),
                comparison=(job.get('pipeline') or {}).get('comparison',True),harmony=harmony,
                instrument_presence_review=project.get('instrument_presence_review'))
    atomic_json(output/'config.json',config)
    env=os.environ.copy();env['PYTHONUTF8']='1';env['WANDB_MODE']='disabled'
    with (output/'build.log').open('w',encoding='utf8') as log:
        process=subprocess.Popen([str(PYTHON),'-u','-m','app.web.reconstruction_build',str(output/'config.json')],cwd=ROOT,env=env,
                                 stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        start=time.monotonic()
        try:
            while process.poll() is None:
                if time.monotonic()-start>6600:raise TimeoutError('整曲重构超时')
                if read_json(job_path).get('status')=='interrupted':raise RuntimeError('任务已中断')
                if (output/'progress.json').exists():stage(read_json(output/'progress.json')['label'])
                time.sleep(2)
        except BaseException:
            # Kill the process tree so a failed job cannot keep a GPU child alive.
            if os.name=='nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
            else:process.kill()
            process.wait();raise
    if process.returncode:raise RuntimeError((output/'build.log').read_text(encoding='utf8',errors='replace')[-2000:])
    if signature(audio)['sha256']!=audio_hash:raise RuntimeError('重构期间原曲发生变化')
    manifest=read_json(output/'manifest.json')
    if not (output/'reconstruction.zip').is_file():raise RuntimeError('工程包没有生成完成')
    project=store.project(pid);project['song_reconstruction']=dict(generation=generation,updated_at=now());store.save_project(project)
    return manifest
