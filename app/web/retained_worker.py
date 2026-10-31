"""Adapt exact v88 audio predictions to the existing immutable timeline schema."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .storage import ROOT, PROFILES, atomic_json, now, read_json, signature

PHASES = {
    'decoded':'音频已解码，准备提取声学特征',
    'extracting_anchors':'madmom 正在提取和弦时间锚点',
    'extracting_deep':'正在提取 DeepChroma 特征',
    'extracting_nnls':'正在提取 NNLS 音符证据',
    'features_complete':'声学特征就绪，retained10 正在推理',
    'cache_reused':'复用同音频缓存，retained10 正在推理',
    'prediction_complete':'retained10 推理完成，正在建立时间线',
}


def execute(store, job_id):
    from .worker import lab_rows, build_bars, review_candidates
    job_path = store.job_path(job_id)
    job = read_json(job_path)
    key_mode = str(job.get('key_mode') or 'madmom')
    project = store.project(job['project_id'])
    folder = store.project_dir(project['id'])
    output = store.run_dir(project['id'], job_id)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    runtime_label = str(read_json(ROOT/'desktop_runtime.json').get('model_version') or 'retained10')
    warnings = [f'{runtime_label} 支持 maj/min/7/maj7/min7/9/maj9/min9/sus2/sus4 及 N；后验未校准，不能作为准确率。',
                f'主调与拍点由独立 madmom 模型估计，不属于 {runtime_label}；结构分段为相似性线索。']

    def publish():
        if read_json(job_path)['status'] == 'interrupted':
            raise RuntimeError('任务已中断，请重新分析。')
        atomic_json(job_path, job)

    def command(*args, timeout=900):
        return subprocess.run([str(value) for value in args],cwd=ROOT,check=True,timeout=timeout,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)

    def stage(name, label, action, optional=False):
        job.update(stage=name, stage_label=label)
        publish()
        record = dict(id=name,label=label,started_at=now())
        begin = time.perf_counter()
        try:
            value = action()
            record['status'] = 'completed'
            return value
        except Exception as exc:
            record.update(status='failed',error=str(exc))
            if not optional:
                job['error'] = str(exc)[-2200:]
                raise
            warnings.append(label+'未完成；和弦预测仍可用。详情见运行日志。')
            traceback.print_exc()
        finally:
            record['seconds'] = time.perf_counter()-begin
            job['stages'].append(record)
            publish()

    if job['download']:
        def download():
            target = output/'source'
            target.mkdir()
            command(os.environ.get('HARMONY_DOWNLOAD_PYTHON',sys.executable),'-B','-m','yt_dlp',
                    '--ignore-config','--no-playlist','--no-update','--no-cache-dir','--socket-timeout','20',
                    '--retries','2','--max-filesize','512M','--write-info-json','--no-progress',
                    '-f','bestaudio','-o',str(target/'audio.%(ext)s'),'--',project['source_url'])
            from .storage import AUDIO_EXTENSIONS
            files=[p for p in target.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS]
            if len(files)!=1:
                raise ValueError('下载未产生有效音频，可改为上传本地文件。')
            metadata=read_json(target/'audio.info.json')
            project.update(audio=str(files[0].relative_to(folder)),audio_identity=signature(files[0]),
                           original_filename=files[0].name,title=str(metadata.get('title') or project['title']))
            store.save_project(project)
        stage('download','正在下载 B 站音频',download)

    audio = folder/project['audio']
    prediction_dir = output/'retained10'
    cache = Path(os.environ['CHORD_DESKTOP_DATA'])/'audio-cache'

    def infer():
        args=[sys.executable,'-I','-B',str(ROOT/'inference_windowless.py'),'--audio',str(audio),
              '--output-dir',str(prediction_dir),'--cache-dir',str(cache),'--max-seconds','600',
              '--memory-budget-mib',str(read_json(ROOT/'desktop_runtime.json')['inference_memory_budget_mib'])]
        stdout=(output/'inference-entry.stdout').open('wb')
        stderr=(output/'inference-entry.stderr').open('wb')
        child=subprocess.Popen(args,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        last=''
        while child.poll() is None:
            time.sleep(.35)
            path=prediction_dir/'worker.stdout'
            if path.exists():
                lines=path.read_text(encoding='utf-8',errors='replace').splitlines()
                if lines and lines[-1]!=last:
                    last=lines[-1]
                    try:
                        phase=json.loads(last).get('phase')
                    except ValueError:
                        phase=None
                    if phase in PHASES:
                        job['stage_label']=PHASES[phase]
                        publish()
        stdout.close();stderr.close()
        status=read_json(prediction_dir/'run_status.json') if (prediction_dir/'run_status.json').is_file() else {}
        if child.returncode or not status.get('successful'):
            err=prediction_dir/'worker.stderr'
            detail=err.read_text(encoding='utf-8',errors='replace')[-1800:] if err.exists() else (output/'inference-entry.stderr').read_text(encoding='utf-8',errors='replace')[-1800:]
            if not detail: detail=f'推理入口退出码 {child.returncode}，未产生成功回执。'
            raise RuntimeError('retained10 音频推理失败：'+detail)
        payload=read_json(prediction_dir/'prediction.json')
        config=read_json(ROOT/'desktop_runtime.json')
        if payload['weights_sha256']!=config['weights_sha256'] or payload['input_sha256']!=project['audio_identity']['sha256']:
            raise ValueError('retained10 模型或输入身份不符，拒绝发布结果。')
        # Exact LAB copied; no resmoothing, relabeling, quantization, or key override.
        shutil.copyfile(prediction_dir/'prediction.lab',output/'retained10.lab')
        shutil.copyfile(Path(payload['cache_directory'])/'madmom.lab',output/'raw_madmom.lab')
        return payload
    prediction=stage('retained10','检查冻结模型并提取音频特征',infer)

    sys.path.insert(0,str(ROOT/'app'))
    import numpy as np
    from audio_waveform import peak_envelope
    duration=prediction['audio_seconds']
    runtime=read_json(ROOT/'inference/runtime.local.json')
    def waveform():
        result=subprocess.run([runtime['ffmpeg']['path'],'-hide_banner','-loglevel','error','-threads','1',
            '-i',str(audio),'-vn','-ac','1','-ar','22050','-f','f32le','pipe:1'],capture_output=True,check=True,timeout=120,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        y=np.frombuffer(result.stdout,dtype=np.float32)
        if abs(len(y)/22050-duration)>2/22050:
            raise ValueError('波形与预测时钟不一致。')
        return y
    samples=stage('waveform','生成播放波形',waveform)
    key_path=output/'key.json'
    stage('key','独立估计全曲主调',lambda:command(sys.executable,'-B',ROOT/'app/key_bridge.py',audio,'-o',key_path,timeout=600),optional=True)
    beat_path=output/'beats.json'
    stage('beats','定位拍点与小节',lambda:command(sys.executable,'-B',ROOT/'app/downbeat_bridge.py',audio,'-o',beat_path,'--timeout','900',timeout=900),optional=True)
    beats=read_json(beat_path) if beat_path.exists() else []
    key=read_json(key_path) if key_path.exists() else None
    downbeats=[float(row['time']) for row in beats if int(row.get('beat',0))==1] if isinstance(beats,list) else []
    stable_path=output/'retained10_stable.lab'
    stable=dict(applied=False,segments=None)
    def stabilize_timeline():
        from .stabilize_timeline import stabilize_lab
        cells=Path(prediction['cache_directory'])/'cells.json'
        if not cells.is_file():
            raise FileNotFoundError('声学缓存缺少 cells.json，跳过稳定时间轴')
        segments=stabilize_lab(lab_path=output/'retained10.lab',
                               posterior_path=prediction_dir/'posterior.npz',
                               cells_path=cells,
                               output_path=stable_path,
                               downbeats=downbeats)
        stable.update(applied=True,segments=len(segments))
        return segments
    # Derived only: prediction.lab, posterior.npz and the frozen weights stay untouched.
    stage('stabilize','生成稳定时间轴（派生）',stabilize_timeline,optional=True)
    span_path=output/'retained10_span.lab'
    def span_decode():
        from .span_pool import pool_lab
        source=stable_path if stable_path.exists() else output/'retained10.lab'
        if not beats:
            raise ValueError('缺少拍点，跳过小节池化')
        chroma=None
        try:
            import numpy as np
            deep=Path(prediction['cache_directory'])/'deep.npz'
            if deep.is_file():
                chroma=np.load(deep)['chroma']
        except Exception:
            chroma=None
        return pool_lab(source,span_path,beats,chroma=chroma)
    stage('span','生成小节池化时间轴（派生）',span_decode,optional=True)
    full_path=output/'retained10_full.lab'
    full=dict(applied=False,segments=None)
    def full_decode_stage():
        """Second-tier derived timeline: semi-Markov decode over phase-corrected half bars."""
        from .full_decode import decode as decode_full, write_lab as write_full_lab, PARAMS
        if not beats:
            raise ValueError('缺少拍点，跳过完整段解码')
        cache_dir=Path(prediction['cache_directory'])
        cells_path=cache_dir/'cells.json'
        deep_path=cache_dir/'deep.npz'
        notes_path=cache_dir/'notes.npz'
        for path in (cells_path,deep_path,notes_path):
            if not path.is_file():
                raise FileNotFoundError(f'声学缓存缺少 {path.name}，跳过完整段解码')
        cells=read_json(cells_path)
        deep=np.load(deep_path)
        notes=np.load(notes_path)
        rows=decode_full(
            posterior=np.load(prediction_dir/'posterior.npz')['chord'],
            starts=np.array([cell['start'] for cell in cells],dtype=float),
            ends=np.array([cell['end'] for cell in cells],dtype=float),
            chroma=deep['chroma'],
            act=notes['activations'],note_times=notes['times'],
            beats=np.array([float(row['time']) for row in beats],dtype=float),
            beat_pos=np.array([int(row.get('beat',0)) for row in beats],dtype=int))
        write_full_lab(full_path,rows)
        full.update(applied=True,segments=len(rows),params=dict(PARAMS))
        return rows
    stage('full','生成完整段解码时间轴（派生，第二档）',full_decode_stage,optional=True)
    runtime_config=read_json(ROOT/'desktop_runtime.json')
    model_label=str(runtime_config.get('model_version') or 'retained10')
    frozen_date=str(runtime_config.get('frozen_utc') or '')[:10]
    retained_name=f"{model_label} · 冻结 {frozen_date}" if frozen_date else model_label
    stable_name=f"{model_label} · 稳定时间轴（派生）"
    sources=[]
    span_name=f"{model_label} · 小节池化（派生）"
    full_name=f"{model_label} · 完整段解码（派生）"
    for source_id,name in [('retained10',retained_name),
                           ('retained10_stable',stable_name),
                           ('retained10_span',span_name),
                           ('retained10_full',full_name),
                           ('raw_madmom','madmom · 时间锚点对照')]:
        path=output/(source_id+'.lab')
        if not path.exists():
            continue
        sources.append(dict(id=source_id,name=name,path=path.name,**signature(path),rows=lab_rows(path)))
    for source in sources:
        review_candidates(source['rows'],sources)
    _ids={source['id'] for source in sources}
    # The full-segment decode keeps more chord tones than bar pooling (tone recall 0.922 vs
    # 0.827 on 虚拟) and has the better boundary F1, so it is the default; bar pooling stays
    # available as a switchable source and is the fallback when the decode cannot run.
    default_source=('retained10_full' if 'retained10_full' in _ids
                    else ('retained10_span' if 'retained10_span' in _ids
                          else ('retained10_stable' if stable['applied'] and 'retained10_stable' in _ids
                                else 'retained10')))
    display_rows=next((source['rows'] for source in sources if source['id']==default_source),sources[0]['rows'])
    # The published key must not depend on which display timeline is selected. Harmonic key
    # evidence always comes from the frozen stabilised timeline (frozen prediction when the
    # stabiliser is unavailable), never from a smoother such as retained10_span/full.
    key_rows=next((source['rows'] for source in sources if source['id']=='retained10_stable'),
                  next((source['rows'] for source in sources if source['id']=='retained10'),display_rows))
    from key_from_chords import infer as infer_key, key_name, parse_rows
    from key_decision import decide, guard_456
    parsed_chords,_=parse_rows([dict(start=row['start'],end=row['end'],label=row['label']) for row in key_rows])
    def chord_key_evidence():
        if len(parsed_chords)<3:
            raise ValueError('和弦太少，无法用和声证据推断主调')
        evidence=infer_key(parsed_chords)
        pc,mode=evidence['pred']
        raw_key=key_name(pc,mode)
        guarded,guard_info=guard_456(parsed_chords,raw_key)
        return dict(key=guarded,raw_key=raw_key,mode=mode,margin=round(float(evidence['pred_pair']['margin']),4),
                    method='harmonic_key_v1',chords=len(parsed_chords),guard=guard_info,
                    runner_up=[dict(key=key_name(c['pc'],c['mode']),score=round(float(c['score']),3)) for c in evidence['candidates'][:3]])
    chord_evidence=stage('key_chords','用和弦进行交叉验证主调',chord_key_evidence,optional=True)
    if key is not None and chord_evidence:
        decision=decide(key.get('key'),chord_evidence['key'],chord_evidence['margin'],parsed_chords)
        if decision['use_chord'] and decision['chord_key']!=key.get('key'):
            key=dict(key,key=decision['chord_key'],source='chords_coverage',cnn_key=key.get('key'))
        key=dict(key,chord_evidence=dict(chord_evidence,decision=decision,
                                         evidence_source=next((source['id'] for source in sources
                                                               if source['rows'] is key_rows),'unknown')))
    def experimental_key():
        """Continuous 24-candidate scoring (research prototype); never changes the default."""
        from .key_experimental import evaluate as evaluate_experimental_key
        if not key or not key.get('candidates'):
            raise ValueError('缺少声学候选分布，跳过连续评分主调')
        return evaluate_experimental_key(key_rows,key['candidates'],published_key=key.get('key'))
    experimental=stage('key_experimental','连续评分主调（实验，第二档）',experimental_key,optional=True)
    if key is not None and experimental:
        key=dict(key,experimental=experimental)
    if key_mode=='continuous':
        block=(key or {}).get('experimental')
        if block:
            gated=key.get('key')
            key=dict(key,key=block['key'],source='continuous_key_v1',experimental_mode=True,
                     acoustic_key=block.get('acoustic_key'),gated_key=gated)
            warnings.append('主调以连续评分（实验）为准：分数未校准，147 例为已暴露回归集（123/147，改对 3、改错 2）；'
                            'CNN 原判与和弦覆盖裁决结果保留在 key_analysis 中。')
        else:
            warnings.append('连续评分（实验）不可用，主调退回 CNN 与和弦覆盖裁决。')
    bars=build_bars(beats,duration,display_rows)
    def structure():
        from song_structure import analyze_song_structure
        return analyze_song_structure(dict(bars=bars,duration=duration,beat_source='madmom',key=(key or {}).get('key')),y=samples,sr=22050)
    structure_data=stage('structure','分析段落相似性与和声循环',structure,optional=True)
    if signature(audio)!=project['audio_identity']:
        raise ValueError('音频在分析中发生变化，结果未发布。')
    result=dict(schema='harmony_web_analysis_v1',project_id=project['id'],run_id=job_id,title=project['title'],
        profile='retained10',profile_name=retained_name,duration=duration,key_mode=key_mode,
        audio=dict(url=f"/api/projects/{project['id']}/audio",**project['audio_identity']),
        key_analysis=key,beats=beats,bars=bars,waveform=peak_envelope(samples,22050),structure=structure_data,
        sources=sources,default_source=default_source,warnings=warnings,reference_available=False,
        candidate_score_kind='uncalibrated_source_proposals_not_accuracy',
        model_identity={'retained10':dict(version=model_label,entry=str(runtime_config.get('entry_version') or 'v88'),frozen_utc=str(runtime_config.get('frozen_utc') or ''),weights_sha256=prediction['weights_sha256'],supported_qualities=prediction['supported_qualities'])},
        provenance=dict(created_at=now(),worker=signature(__file__),fitting=False,reference_inputs=False,
            elapsed_seconds=time.perf_counter()-started,stages=job['stages'],original_intervals_preserved=True,
            exact_v88_prediction='retained10/prediction.json',exact_v88_lab='retained10/prediction.lab',
            cache_reused=prediction['cache_reused'],default_changed_by_user_authorization=True,
            timeline_stabilization=dict(
                derived_from='retained10.lab',module='app/web/stabilize_timeline.py',
                applied=stable['applied'],segments=stable['segments'],
                downbeats_available=len(downbeats),
                params=dict(rel_threshold=0.50,max_short=0.40,uncertain_margin=2.0,
                            same_root_max=0.50,downbeat_tolerance=0.25)),
            full_decode=dict(applied=full['applied'],segments=full['segments'],
                module='app/web/full_decode.py',derived_from='posterior.npz + cells.json + deep.npz + notes.npz + beats.json',
                second_tier=True,default=False,params=full.get('params'))))
    atomic_json(output/'analysis.json',result)
