"""Frozen v90 chord evidence with an independently trained bass voice.

Keep the existing chord quality when the root is unchanged. A newly detected
bass may be an inversion or pedal, not a new root. Unknown bass preserves the
baseline. This derived source preserves frozen predictions and human key reviews.
"""
import hashlib
import time
from pathlib import Path
import numpy as np
import soundfile as sf

from . import bass_voice
from .stems_worker import resolve_stem
from .full_decode import pooled_label, tone_support, span_tone
from .fusion_engine import segments_from
from .stem_harmony import build_views, pool, validate_cells, LABELS
from .storage import atomic_json, atomic_text, read_json, signature, now

VERSION='stem-bass-harmony-v2'
SOURCE='stem_bass_harmony_v2'
DISPLAY_NAME='分轨和弦与低音 · V3.1（实验）'


def selected_bass_model(store, project):
    """Remember a previously active bass-aware source across a fresh analysis.

    This is a source preference, never a license to copy old prediction rows.
    A different recording or an explicitly selected baseline does not opt in.
    """
    if not project.get('current_run'):return None
    path=store.run_dir(project['id'],project['current_run'])/'analysis.json'
    if not path.exists():return None
    previous=read_json(path)
    source_id=previous.get('default_source','')
    meta=previous.get('stem_bass_harmony') or {}
    if (not source_id.startswith('stem_bass_harmony_') or meta.get('source_id')!=source_id
            or previous.get('audio',{}).get('sha256')!=project.get('audio_identity',{}).get('sha256')):
        return None
    from .stems_worker import MODELS
    model=meta.get('model')
    if isinstance(model,dict):model=model.get('name')
    return model if model in MODELS else None


def restore_selected_source(store, project, run_dir, model, publish=lambda message: None):
    """Recompute against this run's evidence; reuse only audio-identity stem cache."""
    from .stems_worker import stems_dir
    audio=store.project_dir(project['id'])/project['audio']
    digest=signature(audio)['sha256']
    if digest!=project['audio_identity']['sha256']:raise ValueError('原音频已改变，不能延续分轨来源')
    base=stems_dir(digest,model)
    if not (base/'manifest.json').exists():
        raise ValueError('此前选用的分轨缓存不可用，请重新识别分轨和弦')
    rows,meta=recognize(run_dir,base,Path(run_dir)/VERSION/model,publish)
    if signature(audio)['sha256']!=digest:raise ValueError('识别期间原音频发生变化')
    source_id=publish_result(run_dir,rows,meta,make_default=True)
    return dict(source_id=source_id,model=model,recomputed=True)


def refine_label(prior, original, tone, bass):
    if original in ('N','X') or not bass:
        return original
    one=np.zeros((1,12));one[0,bass_voice.NAMES.index(bass)]=1
    label=pooled_label(np.asarray(prior)[None,:],one,tone,[0],1.5,.7,.28,'compatible')
    # Repair a decoder root only when the acoustic root winner also supports
    # that repair. Bass alone cannot introduce a third, unsupported harmony.
    acoustic_root=bass_voice.NAMES[int(np.asarray(prior).reshape(12,10).sum(1).argmax())]
    if label.split(':')[0]!=original.split(':')[0] and label.split(':')[0]!=acoustic_root:
        label=original
    # A low note does not decide whether an unchanged chord has a seventh/ninth.
    if label.split(':')[0]==original.split(':')[0]:
        label=original.split('/')[0]
    return bass_voice.attach_bass(label,bass)


def recognize(run_dir, base, output, publish=lambda message: None, model_path=None):
    started=time.perf_counter();run_dir,base,output=map(Path,(run_dir,base,output))
    model_path=Path(model_path) if model_path else bass_voice.MODEL
    model_signature=signature(model_path)
    analysis=read_json(run_dir/'analysis.json');prediction=read_json(run_dir/'retained10/prediction.json')
    manifest=read_json(base/'manifest.json');duration=float(analysis['duration'])
    if not np.isfinite(duration) or not 0<duration<=600:
        raise ValueError('低音和弦识别支持十分钟以内的有效音频')
    if manifest['audio_sha256']!=prediction['input_sha256']:
        raise ValueError('分轨与和弦分析不是同一份音频')
    cache=Path(prediction['cache_directory'])
    with np.load(run_dir/'retained10/posterior.npz') as z:
        posterior,nprob=z['chord'],z['N'].reshape(-1)
    if (posterior.ndim!=2 or posterior.shape[1]!=120 or not np.isfinite(posterior).all()
            or np.any(posterior<0) or not np.allclose(posterior.sum(1),1,atol=1e-3)
            or len(nprob)!=len(posterior) or not np.isfinite(nprob).all()
            or np.any((nprob<0)|(nprob>1))):
        raise ValueError('和弦后验格式错误')
    bounds=validate_cells(read_json(cache/'cells.json'),len(posterior),duration)
    with np.load(cache/'notes.npz') as z:
        act,times=z['activations'],z['times']
    if act.shape!=(84,len(times)) or not np.isfinite(act).all() or np.any(act<0):
        raise ValueError('音符证据格式错误')
    beats=read_json(run_dir/'beats.json')
    beat_times=np.asarray([float(r['time']) for r in beats])
    if not np.isfinite(beat_times).all() or np.any(np.diff(beat_times)<=0):
        raise ValueError('拍点必须按时间递增')
    baseline=next((s['rows'] for s in analysis['sources'] if s['id']=='retained10_full'),None)
    if baseline is None:
        baseline=[dict(start=r['start'],end=r['end'],label=r['chord']) for r in prediction['segments']]
    publish('读取伴奏与低声部，保留钢琴和吉他中的低音')
    views,identities,groups=build_views(base,manifest,duration,output/'views')
    audio,rate=sf.read(views['accompaniment'],dtype='float32',always_2d=True)
    if 'bass' in views:
        bass,br=sf.read(views['bass'],dtype='float32',always_2d=True)
        if bass.shape!=audio.shape or br!=rate:
            raise ValueError('伴奏与贝斯的时间轴不一致')
        audio+=bass
    evidence=bass_voice.spectrum(audio,rate);model=bass_voice.load_model(model_path)
    support=tone_support(act)
    original_baseline=baseline
    from .harmony_boundaries import canonical_cuts,recover_sustained_chords
    def bass_at(a,b):return bass_voice.summarize_beats(model,evidence,a,b,beat_times)
    def coarse_label(row):
        a,b=row['start'],row['end']
        return refine_label(pool(posterior,bounds,a,b),row['label'],span_tone(support,times,a,b),bass_at(a,b)['note'])
    baseline,repairs=recover_sustained_chords(baseline,posterior,nprob,bounds,beat_times,coarse_label,bass_at)
    from .chord_onsets import align_recovered_onsets,resolve_recovered_attacks,reconcile_attack,spectral_profile,onset_evidence,POLICY as ONSET_POLICY
    onset_audio=onset_evidence(audio,rate) if repairs else None
    baseline,onsets=align_recovered_onsets(baseline,repairs,beat_times,
        lambda a,b:spectral_profile(onset_audio,a,b),lambda a,b:float(pool(nprob,bounds,a,b)))
    baseline,attacks=resolve_recovered_attacks(baseline,original_baseline,repairs,beat_times,onset_audio,
        lambda a,b:float(pool(nprob,bounds,a,b)),
        lambda attack,a,b:reconcile_attack(attack,pool(posterior,bounds,a,b),bass_at(a,b)))
    cuts,snaps=canonical_cuts(baseline,[r['time'] for r in beats if int(r['beat'])==1],duration)
    diagnostics,labels,spans=[],[],[]
    publish('分别识别和弦与低音，保留转位和持续低音')
    for a,b in zip(cuts,cuts[1:]):
        if b<=a:continue
        prior=pool(posterior,bounds,a,b)
        original=next((r['label'] for r in baseline if r['start']<=(a+b)/2<r['end']),LABELS[int(prior.argmax())])
        bass=bass_voice.summarize_beats(model,evidence,a,b,beat_times)
        label=refine_label(prior,original,span_tone(support,times,a,b),bass['note'])
        if pool(nprob,bounds,a,b)>=.5:label='N'
        diagnostics.append(dict(start=a,end=b,original=original,label=label,bass=bass))
        labels.append(label);spans.append((a,b))
    rows=segments_from(labels,spans)
    from .source_rows import canonical_rows
    rows=canonical_rows(rows,duration,complete=True)
    for name,digest in identities.items():
        if signature(resolve_stem(base,name))['sha256']!=digest:
            raise ValueError('分轨在识别期间发生变化')
    if signature(model_path)!=model_signature:
        raise ValueError('低音模型在识别期间发生变化')
    meta=dict(version=VERSION,timeline_policy='validated-timeline-v3',created_at=now(),model=manifest['model'],segments=len(rows),
              seconds=round(time.perf_counter()-started,2),source_audio_sha256=prediction['input_sha256'],
              input_stems=identities,groups=groups,bass_model=model_signature,
              implementation_sha256={p.name:signature(p)['sha256'] for p in
                                     [Path(__file__),Path(bass_voice.__file__),Path(__file__).with_name('full_decode.py'),
                                      Path(__file__).with_name('harmony_boundaries.py'),
                                      Path(__file__).with_name('source_rows.py'),Path(__file__).with_name('fusion_engine.py'),
                                      Path(__file__).with_name('chord_onsets.py')]},
              posterior_sha256=signature(run_dir/'retained10/posterior.npz')['sha256'],
              baseline_sha256=hashlib.sha256(__import__('json').dumps(original_baseline,sort_keys=True).encode()).hexdigest(),
              boundary_recovery=dict(grid_snaps=snaps,recovered_chords=repairs,grid_alias_tolerance_seconds=.020),
              onset_alignment=dict(policy=ONSET_POLICY,decisions=onsets,attacks=attacks),
              key_used=False,reference_inputs=False,fitting=False,
              limitations=['Bass scores are uncalibrated; uncertain bass is omitted',
                           'Relative lowest voice within E1–B5; not a complete polyphonic transcription',
                           'Chord roots still use frozen v90 evidence; 10 qualities only',
                           'Real-song regression is development evidence, not a blind benchmark'])
    atomic_json(output/'diagnostics.json',dict(meta=meta,segments=diagnostics))
    atomic_json(output/'result.json',dict(meta=meta,rows=rows))
    return rows,meta


def publish_result(run_dir,segments,meta,make_default=False,integrated=False):
    run_dir=Path(run_dir);path=run_dir/'analysis.json';result=read_json(path)
    from .source_rows import serialize_rows
    segments,content=serialize_rows(segments,result.get('duration'),complete=True)
    digest=hashlib.sha256(content.encode()).hexdigest()
    source_id=('retained10_bass' if integrated else SOURCE)+'_'+digest[:12]
    lab=run_dir/(source_id+'.lab');atomic_text(lab,content)
    sources=[s for s in result['sources'] if s['id']!=source_id]
    name=f'当前模型 · 和弦与指定低音（v90 + {bass_voice.VERSION}）' if integrated else DISPLAY_NAME
    sources.append(dict(id=source_id,name=name,path=lab.name,**signature(lab),rows=segments))
    result.update(sources=sources,stem_bass_harmony=dict(meta,source_id=source_id))
    if integrated:
        result['joint_harmony']=dict(meta,source_id=source_id,status='completed',automatic=True)
        result.setdefault('model_identity',{})['bass_voice']=dict(version=bass_voice.VERSION,
            weights_sha256=meta.get('bass_model',{}).get('sha256'),combined_with='retained10')
    if make_default:result['default_source']=source_id
    from .harmonic_key import refresh
    refresh(result,[source_id])
    atomic_json(path,result)
    return source_id
