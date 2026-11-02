"""Versioned piano transcription side assets. Never replace a harmony run."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import shutil
import sys
from uuid import uuid4
from .storage import ROOT, atomic_json, read_json, signature, now
from .stems_worker import stems_dir, _decode_to_wav, _load, execute as separate

from app.runtime_paths import python_path
PIANO_PYTHON = python_path('piano')
POLICY = 'transkun-v2-beat-map-v1'
FILES = {'performance.mid', 'grid.mid', 'raw.mid', 'notes.json', 'manifest.json', 'preview.wav', 'source.wav'}


def latest(store, project_id):
    project = store.project(project_id)
    record = project.get('piano_transcription')
    if not record:
        raise FileNotFoundError('还没有钢琴 MIDI，请先生成')
    folder = store.project_dir(project_id) / 'piano' / record['generation']
    manifest = read_json(folder/'manifest.json')
    audio = store.project_dir(project_id)/project['audio']
    if signature(audio)['sha256'] != manifest['audio_sha256']:
        raise FileNotFoundError('原曲已更换，请重新生成钢琴 MIDI')
    return folder, manifest


def run_process(command, log_path, timeout=3600):
    env=os.environ.copy();env['PYTHONUTF8']='1'
    with Path(log_path).open('w',encoding='utf8') as log:
        result=subprocess.run([str(x) for x in command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                              timeout=timeout,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if result.returncode:
        raise RuntimeError('钢琴转写失败：'+Path(log_path).read_text(encoding='utf8',errors='replace')[-1500:])


def execute(store, job_id):
    from .piano_midi import export, explicit_grid
    job_path=store.job_path(job_id);job=read_json(job_path)
    project=store.project(job['project_id']);project_id=project['id']
    source_kind=(job.get('pipeline') or {}).get('input','mix')
    if source_kind not in {'mix','piano'}:
        raise ValueError('请选择原曲或钢琴独奏 / 分轨')
    if not PIANO_PYTHON.is_file():
        raise ValueError('钢琴转写运行环境尚未安装')
    audio=store.project_dir(project_id)/project['audio']
    audio_hash=signature(audio)['sha256']
    generation=uuid4().hex
    output=store.project_dir(project_id)/'piano'/generation;output.mkdir(parents=True)

    def stage(label):
        # Separation may have updated this same job record.
        current=read_json(job_path);current.update(stage='piano',stage_label=label);atomic_json(job_path,current)

    stage('准备钢琴音频')
    if source_kind=='mix':
        separate(store,job_id,model_override='bs_roformer_sw',force_override=False)
        parent_source=stems_dir(audio_hash,'bs_roformer_sw')/'wav/piano.wav'
        source=output/'source.wav'
        shutil.copyfile(parent_source,source)
    else:
        parent_source=audio
        source=output/'source.wav';_decode_to_wav(audio,source)
    source_hash=signature(source)['sha256']
    # Limit memory before inference; project separation already has its own limits.
    import soundfile as sf
    duration=sf.info(str(source)).duration
    if not 0 < duration <= 1200:
        raise ValueError('本版支持 20 分钟以内的音频')
    run=store.run_dir(project_id,project['current_run']) if project.get('current_run') else None
    analysis=read_json(run/'analysis.json') if run and (run/'analysis.json').is_file() else {}
    if analysis.get('audio',{}).get('sha256')!=audio_hash:
        run=None;analysis={}
    beat_path=run/'beats.json' if run else None
    beat_source='project beats.json'
    beat_error=None
    settings=job.get('pipeline') or {}
    if settings.get('bpm') is not None:
        beats=explicit_grid(settings['bpm'],settings['first_downbeat'],settings['beats_per_bar'],duration)
        beat_source='user constant tempo grid'
    elif beat_path and beat_path.is_file():
        beats=read_json(beat_path)
    else:
        stage('识别原曲拍点与小节')
        beat_path=output/'beats.json'
        try:
            run_process([sys.executable,'-B',ROOT/'app/downbeat_bridge.py',audio,'-o',beat_path],output/'beats.log',1000)
            beats=read_json(beat_path);beat_source='madmom downbeat tracker'
        except (RuntimeError,subprocess.TimeoutExpired) as exc:
            beats=[];beat_error=str(exc);beat_source='unavailable'
    if len(beats)<3:
        beats=[];beat_source='unavailable'
        beat_error=beat_error or '拍点不足，保留实际时间，未标记小节或量化。'
    stage('识别钢琴音符、力度、按键时长与踏板')
    run_process([PIANO_PYTHON,'-B','-m','app.web.piano_runner',source,output/'raw.mid'],output/'inference.log')
    chord_rows=[];harmony_source=None
    if analysis:
        harmony_source=analysis.get('default_source')
        selected=next((s for s in analysis.get('sources',[]) if s['id']==harmony_source),{})
        chord_rows=selected.get('rows',[])
    stage('保留演奏时间，并整理小节与节奏位置')
    report=export(output/'raw.mid',output,beats,duration,chord_rows)
    report['tempo_source']=beat_source
    atomic_json(output/'notes.json',report)
    atomic_json(output/'beats.json',beats)
    stage('生成 MIDI 钢琴试听')
    preview_error=None
    try:
        run_process([sys.executable,'-B','-m','app.web.piano_preview',output/'performance.mid',output/'preview.wav'],output/'preview.log')
    except RuntimeError as exc:
        preview_error=str(exc)  # MIDI remains downloadable if the local soundfont is absent.
    if signature(source)['sha256']!=source_hash or signature(audio)['sha256']!=audio_hash:
        raise RuntimeError('音频在转写期间发生变化，请重新生成')
    implementation={n:signature(ROOT/'app/web'/n)['sha256'] for n in ['piano_runner.py','piano_midi.py','piano_worker.py']}
    manifest=dict(schema=POLICY,generation=generation,project_id=project_id,created_at=now(),
                  audio_sha256=audio_hash,input_kind=source_kind,source_sha256=source_hash,
                  separation_model='bs_roformer_sw' if source_kind=='mix' else None,
                  source_path=str(source),parent_source_path=str(parent_source),duration=duration,notes=report['count'],
                  bars=len(report['bars']),bpm=report['quarter_note_bpm'],
                  pedal_events=len(report['controls']),swing_regions=report['swing'],
                  snapped_onsets=report['snapped_onsets'],maximum_onset_shift_ms=report['maximum_onset_shift_ms'],
                  beat_source=beat_source,beat_error=beat_error,beat_settings=settings,
                  analysis_run=project.get('current_run') if run else None,harmony_source=harmony_source,
                  inference=read_json(output/'raw.inference.json'),implementation=implementation,
                  preview_error=preview_error,
                  files={name:signature(output/name) for name in FILES-{'manifest.json'} if (output/name).is_file()},
                  limitations=['分轨可能漏音或串音；本曲没有逐音人工真值，未宣称零漏音。',
                               '原始演奏版保留模型时序；节拍整理版只在 40 毫秒内吸附，不保证所有音符均量化。',
                               '拍点与小节为估计；swing 与连续三连音后半拍有时无法仅凭起奏区分。'])
    if beat_error:
        manifest['limitations'].insert(0,'本次没有可靠拍点；两份 MIDI 均保留实际时间，小节与拍号未确认。')
    atomic_json(output/'manifest.json',manifest)
    if read_json(job_path).get('status')=='interrupted':
        raise RuntimeError('任务已中断，旧结果保持不变')
    project=store.project(project_id)
    project['piano_transcription']=dict(generation=generation,updated_at=now(),input_kind=source_kind)
    store.save_project(project)
    return manifest
