"""Fixed Krumhansl-Kessler profile diagnostic on the existing audio frontend.

References: https://essentia.upf.edu/reference/std_Key.html and its documented
Krumhansl profile arrays. Existing Chord Studio uses the same published arrays.
Only profile statistics are implemented here; no Essentia source is copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from chord_recognizer import SAMPLE_RATE, decode_audio, extract_chroma

SCHEMA = 'fixed_key_profile_v1'
FRONTEND = 'existing centered fast chroma, 22050 Hz, hop 512'
PROFILES = {
    'major': [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88],
    'minor': [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17],
}
NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


def scores(chroma, duration, sample_rate=SAMPLE_RATE, hop=512):
    values=np.asarray(chroma,dtype=float)
    if values.ndim!=2 or values.shape[0]!=12 or not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Expected finite nonnegative 12 x frames chroma')
    if not np.isfinite(duration) or duration<=0 or sample_rate<=0 or hop<=0:
        raise ValueError('Invalid shared frontend time axis')
    times=np.arange(values.shape[1])*hop/sample_rate
    if len(times) and times[-1]>duration+1e-6:
        raise ValueError('Frame time exceeds decoded audio duration')
    if len(times)==0:
        return dict(schema=SCHEMA,key=None,tonic=None,mode=None,candidates=[],reason='Empty chroma',status='ambiguous')
    boundaries=np.r_[0.,(times[:-1]+times[1:])/2,duration]
    weights=np.diff(boundaries)
    mean=values@weights/weights.sum()
    centered=mean-mean.mean();norm=np.linalg.norm(centered)
    if norm<1e-12:
        return dict(schema=SCHEMA,key=None,tonic=None,mode=None,candidates=[],mean_chroma=mean.tolist(),reason='No discriminating pitch-class evidence',status='ambiguous')
    rows=[]
    for tonic in range(12):
        for mode,profile in PROFILES.items():
            vector=np.roll(np.asarray(profile),tonic);vector-=vector.mean()
            correlation=float(centered@vector/(norm*np.linalg.norm(vector)))
            rows.append(dict(key=NOTES[tonic]+('m' if mode=='minor' else ''),tonic=NOTES[tonic],mode=mode,score=correlation))
    rows.sort(key=lambda row:(-row['score'],row['key']))
    for row in rows:row['rank']=1+sum(other['score']>row['score']+1e-12 for other in rows)
    spread=rows[0]['score']-rows[-1]['score'];margin=rows[0]['score']-rows[1]['score']
    return dict(schema=SCHEMA,key=rows[0]['key'],tonic=rows[0]['tonic'],mode=rows[0]['mode'],candidates=rows,
                margin=margin,relative_margin=margin/spread if spread else None,status='ambiguous' if margin<.03 else 'resolved',
                mean_chroma=mean.tolist(),duration=duration,sample_rate=sample_rate,hop=hop,frames=len(times),
                frame_time_origin='sample zero; centered frames; exact-duration Voronoi frame weights',
                profile='fixed Krumhansl-Kessler',score_kind='within-source Pearson correlation, not a probability',
                profile_source='https://essentia.upf.edu/reference/std_Key.html')


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def analyze_audio(audio, output=None):
    began=time.perf_counter();audio=Path(audio)
    implementation={name:digest(Path(__file__).parent/name) for name in ['key_profile_baseline.py','chord_recognizer.py','audio_features.py']}
    contract=dict(audio_sha256=digest(audio),implementation=implementation,sample_rate=SAMPLE_RATE,hop=512,frontend=FRONTEND,profiles=PROFILES)
    if output and Path(output).is_file():
        cached=json.loads(Path(output).read_text(encoding='utf-8'))
        if cached.get('contract')==contract:return cached
    samples=decode_audio(audio,sample_rate=SAMPLE_RATE)
    if not len(samples):raise ValueError('Empty audio')
    chroma=extract_chroma(samples,SAMPLE_RATE,hop_length=512,mode='fast')
    result=scores(chroma,len(samples)/SAMPLE_RATE)
    result.update(contract=contract,audio=str(audio.resolve()),duration=len(samples)/SAMPLE_RATE,
                  elapsed_seconds=time.perf_counter()-began)
    if output:
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
        temporary=output.with_suffix(output.suffix+'.part')
        temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        temporary.replace(output)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio',type=Path);parser.add_argument('-o','--output',type=Path,required=True)
    args=parser.parse_args();result=analyze_audio(args.audio,args.output)
    print({name:result.get(name) for name in ['key','tonic','mode','margin','elapsed_seconds']})
