"""Reproducible whole-song reconstruction in its isolated audio environment."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import numpy as np
import soundfile as sf
from .storage import atomic_json, read_json, signature, now, ROOT
from .reconstruction_midi import read_track, write_midi, verify_roundtrip
from .piano_midi import BeatMap
from app.runtime_paths import data_root

PARTS=[('piano','钢琴',0,'transkun'),('bass','贝斯',33,'basic'),('drums','鼓组',0,'yourmt3'),
       ('guitar','吉他估计',25,'yourmt3'),('other','其他音高声部估计',48,'basic'),('vocals','人声旋律估计',53,'basic')]


def excluded_families(config):
    review=config.get('instrument_presence_review') or {}
    if review.get('audio_sha256')!=config.get('audio_sha256'):return set()
    return {name for name,entry in review.get('instruments',{}).items()
            if entry.get('present') is False and entry.get('origin')=='user'}


def stage(output,label):
    atomic_json(output/'progress.json',dict(label=label,updated_at=now()))
    print(label,flush=True)


def infer(model,source,target,output):
    from .reconstruction_runner import MODELS
    source_identity=signature(source)
    if model=='basic':
        import basic_pitch
        weights=Path(basic_pitch.__file__).parent/'saved_models/icassp_2022/nmp.onnx'
    else:weights=MODELS/('adt-str' if model=='drums' else 'yourmt3')/'download-manifest.json'
    import hashlib
    identity=hashlib.sha256((source_identity['sha256']+signature(weights)['sha256']+signature(ROOT/'app/web/reconstruction_runner.py')['sha256']).encode()).hexdigest()
    cache=data_root()/'reconstruction-cache'/model/identity
    if (cache/'raw.mid').exists() and (cache/'raw.inference.json').exists():
        for suffix in ['.mid','.inference.json','.events.json']:
            if (cache/('raw'+suffix)).exists():shutil.copyfile(cache/('raw'+suffix),target.with_suffix(suffix))
        return
    # Resume only identical input/model settings within an unpublished generation.
    if target.exists() and target.with_suffix('.inference.json').exists():
        old=read_json(target.with_suffix('.inference.json'))
        if old.get('cache_identity')==identity:return
    env=os.environ.copy();env['PYTHONUTF8']='1';env['WANDB_MODE']='disabled'
    with target.with_suffix('.log').open('w',encoding='utf8') as log:
        result=subprocess.run([sys.executable,'-u','-m','app.web.reconstruction_runner',model,str(source),str(target)],
                              cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=5400,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if result.returncode:raise RuntimeError(target.with_suffix('.log').read_text(encoding='utf8',errors='replace')[-2000:])
    report=read_json(target.with_suffix('.inference.json'));report['cache_identity']=identity
    atomic_json(target.with_suffix('.inference.json'),report)
    cache.mkdir(parents=True,exist_ok=True)
    for suffix in ['.mid','.events.json','.inference.json']:
        if target.with_suffix(suffix).exists():shutil.copyfile(target.with_suffix(suffix),cache/('raw'+suffix))


def build(config):
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    duration=config['duration'];beats=config['beats'];grid=BeatMap(beats,duration)
    stems=Path(config['stems']);piano=Path(config['piano']);tracks=[];summaries=[];warnings=[]
    excluded=excluded_families(config)
    for ident,label,program,model in PARTS:
        stage(output,'转写 '+label)
        source=stems/(ident+'.wav');raw=output/(ident+'.'+model+'.raw.mid')
        if model=='transkun':shutil.copyfile(piano/'performance.mid',raw)
        else:infer(model,source,raw,output)
        found=read_track(raw,ident,label,program,duration)
        # Basic Pitch outputs a single instrument with no independent voice identity.
        for track in found:
            track.update(origin=model,source=ident,identity='instrument-family estimate',
                         included_in_reconstruction=ident not in excluded,
                         velocity_kind='fixed model velocity; not recovered dynamics' if model=='yourmt3' else 'model amplitude estimate' if model=='basic' else 'model velocity estimate')
            for note in track['notes']:note.update(grid.position(note['start']))
            write_midi([track],output/(track['id']+'.mid'),beats,duration)
            verify_roundtrip([track],output/(track['id']+'.mid'))
        tracks.extend(found)
        summaries.append(dict(id=ident,label=(label+'分离候选（已确认本曲无此乐器）') if ident in excluded else label,model=model,program=program,notes=sum(len(t['notes']) for t in found),
                              included_in_reconstruction=ident not in excluded,
                              track_ids=[t['id'] for t in found],source_sha256=signature(source)['sha256'],
                              inference=read_json(raw.with_suffix('.inference.json')) if model!='transkun' else config['piano_inference']))
    for part in summaries:
        alternative={'bass':'yourmt3','drums':'drums','guitar':'basic'}.get(part['id'])
        if not alternative:continue
        ident=part['id'];stage(output,'生成 '+part['label']+' 的独立候选')
        raw=output/(ident+'.'+alternative+'.raw.mid');infer(alternative,stems/(ident+'.wav'),raw,output)
        alternate=read_track(raw,ident+'_alt',part['label']+' · 独立候选',part['program'],duration)
        path=output/(ident+'.alternative.mid');write_midi(alternate,path,beats,duration)
        verification=verify_roundtrip(alternate,path)
        atomic_json(output/(ident+'.alternative.json'),dict(model=alternative,tracks=alternate,roundtrip=verification))
        part['alternative']=dict(model=alternative,notes=verification['notes'],midi_file=path.name,events_file=ident+'.alternative.json',
                                 render_file=ident+'.alternative.preview.wav')
    stage(output,'验证统一速度图与 MIDI 导出')
    selected_tracks=[t for t in tracks if t['included_in_reconstruction']]
    write_midi(selected_tracks,output/'performance.mid',beats,duration)
    check=verify_roundtrip(selected_tracks,output/'performance.mid')
    atomic_json(output/'events.json',dict(tracks=selected_tracks,excluded_tracks=[t for t in tracks if not t['included_in_reconstruction']],beats=beats,bars=grid.bar_rows,duration=duration,harmony=config.get('harmony'),
                                        timing='unquantized performance seconds; shared tempo map',roundtrip=check))
    if config.get('harmony'):atomic_json(output/'harmony-context.json',config['harmony'])
    # Independent full-mix model stays a separate candidate, never an automatic union.
    comparison=None
    if config.get('comparison',True):
        stage(output,'原曲多乐器模型独立对照')
        try:
            target=output/'yourmt3.raw.mid'
            infer('yourmt3',Path(config['original_wav']),target,output)
            candidate=read_track(target,'mix','原曲多乐器候选',None,duration)
            write_midi(candidate,output/'mix-candidate.mid',beats,duration)
            check2=verify_roundtrip(candidate,output/'mix-candidate.mid')
            atomic_json(output/'mix-candidate.json',dict(tracks=candidate,roundtrip=check2))
            comparison=dict(model='YourMT3',notes=check2['notes'],tracks=len(candidate),roundtrip=check2)
        except Exception as exc:
            warnings.append('原曲多乐器对照未成功：'+str(exc)[-900:])
    stage(output,'生成各声部重放与合奏')
    from .reconstruction_render import render,reference_audio,fit_level,SOUNDFONT
    rate=44100;n=round(duration*rate)
    midi_mix=np.zeros((n,2),dtype=np.float32);hybrid_mix=midi_mix.copy();source_mix=midi_mix.copy();mix_bound=midi_mix.copy()
    for part in summaries:
        ident=part['id'];reference=reference_audio(stems/(ident+'.wav'),duration,rate)
        rendered=np.zeros_like(reference)
        for track_id in part['track_ids']:rendered+=render(output/(track_id+'.mid'),duration,rate)
        gain,level=fit_level(rendered,reference);rendered*=gain
        part.update(level=level,source_file=ident+'.source.wav',render_file=ident+'.render.wav',default_mode='off' if ident in excluded else 'midi' if ident in {'piano','bass','drums'} else 'source')
        sf.write(output/part['source_file'],reference,rate,subtype='FLOAT')
        sf.write(output/part['render_file'],rendered,rate,subtype='FLOAT')
        mix_bound+=np.maximum(abs(reference),abs(rendered))
        if part.get('alternative'):
            alternate=render(output/part['alternative']['midi_file'],duration,rate)
            alt_gain,alt_level=fit_level(alternate,reference);alternate*=alt_gain
            # Reserve headroom even when the user selects an alternate for every track.
            mix_bound+=np.maximum(0,abs(alternate)-np.maximum(abs(reference),abs(rendered)))
            part['alternative']['level']=alt_level
            sf.write(output/(ident+'.alternative.wav'),alternate,rate,subtype='FLOAT')
        source_mix+=reference
        if part['included_in_reconstruction']:
            midi_mix+=rendered
            hybrid_mix+=rendered if part['default_mode']=='midi' else reference
    # The same attenuation for every mode and source prevents clipping without
    # changing balance or granting MIDI a loudness advantage.
    original=reference_audio(config['original_wav'],duration,rate)
    comparison_audio=None
    if comparison:
        stage(output,'渲染原曲多乐器独立对照')
        comparison_audio=render(output/'mix-candidate.mid',duration,rate)
        comparison_gain,comparison_level=fit_level(comparison_audio,original)
        comparison_audio*=comparison_gain;comparison['level']=comparison_level
    peak=max(float(np.max(abs(y))) for y in [mix_bound,original])
    if comparison_audio is not None:peak=max(peak,float(np.max(abs(comparison_audio))))
    attenuation=min(1.,.94/max(peak,1e-9))
    for name,y in [('midi-mix.wav',midi_mix),('hybrid-mix.wav',hybrid_mix),('source-mix.wav',source_mix),('original.wav',original)]:
        sf.write(output/name,y*attenuation,rate,subtype='PCM_24')
    if comparison_audio is not None:sf.write(output/'mix-candidate.wav',comparison_audio*attenuation,rate,subtype='PCM_24')
    from scipy.signal import resample_poly
    def preview(path):
        y,_=sf.read(path,dtype='float32',always_2d=True)
        sf.write(path.with_name(path.stem+'.preview.wav'),resample_poly(y,1,2,axis=0),22050,subtype='PCM_16')
    preview(output/'original.wav')
    if comparison_audio is not None:preview(output/'mix-candidate.wav')
    for part in summaries:
        # PCM16 is portable for browser/DAW. Preserve relative gain and shared attenuation.
        for key in ['source_file','render_file']:
            path=output/part[key];y,_=sf.read(path,dtype='float32',always_2d=True);sf.write(path,y*attenuation,rate,subtype='PCM_24')
            preview(path);part['full_'+key]=path.name;part[key]=path.stem+'.preview.wav'
        if part.get('alternative'):
            path=output/(part['id']+'.alternative.wav');y,_=sf.read(path,dtype='float32',always_2d=True);sf.write(path,y*attenuation,rate,subtype='PCM_24');preview(path)
    limitations=[
        '分离器输出槽位不证明原曲存在对应乐器；人工确认不存在的类别仅保留诊断候选，不进入默认 MIDI 或重构合奏。',
        '整曲 MIDI 是模型估计，尚无本曲逐音人工真值；音符数量和模型一致性不是准确率。',
        '钢琴保留已有 Transkun 演奏结果与踏板；其余轨道不使用和弦模板补音，也不删除非和弦音。',
        '鼓、吉他采用分轨 YourMT3；贝斯、其他、人声使用 Basic Pitch。串音、八度、复音和尾音可能出错，没有证明恢复了独立演奏者。',
        'YourMT3 不恢复逐音力度，本版这些声部保留其固定力度；不要把试听响度匹配当作真实力度恢复。',
        '人声 MIDI 仅近似音高事件，不含歌词、音素和真实歌声；混合重放保留其音频，纯 MIDI 不使用这些音频。',
        '音源为 GM 近似音色，非原工程插件；力度非原始 MIDI 真值。Basic Pitch 重叠音弯音不能独立编码，原始轮廓另存。',
        'ADT-STR 鼓组使用作者自定义类别转为 GM 代表音色，音符尾部固定为 100 毫秒触发长度，不是鼓声真实衰减时长。',
        '保留秒级演奏时间，不强制拉齐拍点；拍号、swing 与段落结构仍依赖估计。']
    manifest=dict(schema='song-reconstruction-v1',generation=config['generation'],project_id=config['project_id'],
                  title=config['title'],created_at=now(),audio_sha256=config['audio_sha256'],duration=duration,
                  piano_generation=config['piano_generation'],analysis_run=config.get('analysis_run'),
                  instrument_presence_review=config.get('instrument_presence_review'),excluded_families=sorted(excluded),
                  tracks=summaries,notes=check['notes'],bars=len(grid.bar_rows),roundtrip=check,comparison=comparison,
                  limitations=limitations,warnings=warnings,render=dict(engine='FluidSynth 2.6.2',soundfont=signature(SOUNDFONT),
                  sample_rate=rate,shared_attenuation=attenuation,unattenuated_peak=peak),files={})
    manifest['implementation']={p.name:signature(p)['sha256'] for p in (ROOT/'app/web').glob('reconstruction_*.py')}
    from importlib.metadata import version
    manifest['runtime']={name:version(name) for name in ['torch','transformers','basic-pitch','pretty_midi','mido','onnxruntime']}
    # Portable project: separate dry MIDI, rendered WAVs, source WAVs and full timing map.
    readme='''# Chord Studio 整曲重构\n\nperformance.mid：统一速度图的六类声部演奏 MIDI。\nmidi-mix.wav：完全 MIDI 重放（人声仅以乐器模拟音高）。\nhybrid-mix.wav：钢琴/贝斯/鼓 MIDI + 吉他/其他/人声音频。\n*.source.wav：分离估计；*.render.wav：该声部 MIDI 重放。所有 WAV 从原曲 0 秒开始。\nevents.json：逐音、力度、控制器、原始秒时间与小节位置。\n*.raw.mid：未经公共时间图再导出的模型结果。mix-candidate.mid：独立 YourMT3 对照，未合并。\n\nDAW 使用：先导入 performance.mid 并接受其速度图；WAV 放在时间 0。每条 MIDI 轨指定乐器，鼓用 GM 鼓组。音频与 MIDI 对照轨不要同时叠加当作重构结果。\n轨道名称为乐器类别估计，不代表原音乐工程的真实轨道实例。所有 MIDI 可在 DAW 编辑，源结果保持独立。\n'''
    (output/'README.md').write_text(readme,encoding='utf8')
    if excluded:
        with (output/'README.md').open('a',encoding='utf8') as f:
            f.write('\n人工确认不存在的乐器类别：'+', '.join(sorted(excluded))+'。这些类别的候选单轨文件仅供诊断，已从 performance.mid、midi-mix.wav、hybrid-mix.wav 排除；源分轨对照与未经人工校正的原曲模型候选保持原始内容。\n')
    atomic_json(output/'manifest.json',manifest)
    allowed={'.mid','.wav','.json','.md'}
    for path in output.iterdir():
        if path.suffix in allowed and path.name not in {'manifest.json','progress.json','config.json','original-input.wav'}:
            manifest['files'][path.name]=signature(path)
    atomic_json(output/'manifest.json',manifest)
    with zipfile.ZipFile(output/'reconstruction.zip','w',zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for name in [*manifest['files'],'manifest.json']:archive.write(output/name,name)
    manifest['files']['reconstruction.zip']=signature(output/'reconstruction.zip')
    atomic_json(output/'manifest.json',manifest)
    stage(output,'重构与导出完成')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);args=p.parse_args();build(read_json(args.config))
