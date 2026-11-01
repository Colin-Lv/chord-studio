"""Independent bass-note estimator. A bass note never overwrites a chord root.

Uses an auditable, MIDI-trained note ranker on low-register spectral peaks.
Works on accompaniment as well as isolated bass: piano left hands need not be
routed to the separator's bass instrument stem. Scores are not calibrated truth.
"""
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly, stft

VERSION = 'bass-voice-v1'
NOTES = np.arange(28, 60)  # E1..B3; out-of-range bass is explicitly unsupported.
NAMES = ('C','C#','D','D#','E','F','F#','G','G#','A','A#','B')
MODEL = Path(__file__).parent / 'models/bass_voice_v1.joblib'


def spectrum(audio, sr):
    from math import gcd
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError('无效的低音输入')
    divisor = gcd(int(sr), 11025)
    y = resample_poly(y, 11025 // divisor, int(sr) // divisor)
    y = np.pad(y, (0, max(0, 8192-len(y))))
    f, times, z = stft(y, fs=11025, nperseg=8192, noverlap=7168)
    power = abs(z)**2
    # Keep actual frequencies, rather than folding away register before detection.
    values=[]
    for note in np.arange(28, 85):
        hz = 440 * 2**((note-69)/12)
        mask = abs(f-hz) <= max(11025/8192, hz*.025)
        values.append(power[mask].max(axis=0))
    return dict(times=times, notes=np.asarray(values), energy=power.sum(axis=0))


def feature_rows(evidence, start, end):
    times=evidence['times'];hop=1024/11025
    weights=np.maximum(0,np.minimum(times+hop/2,end)-np.maximum(times-hop/2,start))
    if weights.sum() <= 0:
        raise ValueError('低音区间没有音频覆盖')
    v=np.average(evidence['notes'],axis=1,weights=weights)
    energy=float(np.average(evidence['energy'],weights=weights))
    low=v[:len(NOTES)];peak=max(float(low.max()),1e-14)
    total=max(float(low.sum()),1e-14)
    rows=[]
    for i,note in enumerate(NOTES):
        neighbours=[max(0,i-2),max(0,i-1),min(len(v)-1,i+1),min(len(v)-1,i+2)]
        harmonics=[v[min(i+d,len(v)-1)] for d in (0,12,19,24)]
        below=low[:max(0,i-1)]
        rows.append([note/60, *np.log1p(np.asarray(harmonics)/peak*100),
                     low[i]/peak,low[i]/total,
                     np.log1p(low[i]/max(float(v[neighbours].mean()),1e-14)),
                     float(below.max(initial=0))/peak,float(below.sum())/total,
                     float((below>low[i]*.2).sum())/len(NOTES),
                     low[i]/max(energy,1e-14),total/max(energy,1e-14)])
    return np.asarray(rows,dtype=np.float32),energy


def estimate(model, evidence, start, end, min_score=.8, min_margin=.2):
    x,energy=feature_rows(evidence,start,end)
    if energy<1e-9:
        return dict(note=None,score=0.,margin=0.,reason='quiet')
    score=model.predict_proba(x)[:,list(model.classes_).index(1)]
    pc=np.array([score[NOTES%12==i].max() for i in range(12)])
    order=np.argsort(pc);best=int(order[-1]);margin=float(pc[best]-pc[order[-2]])
    accepted=bool(pc[best]>=min_score and margin>=min_margin)
    return dict(note=NAMES[best] if accepted else None,candidate=NAMES[best],
                score=float(pc[best]),margin=margin,reason='accepted' if accepted else 'uncertain')


def load_model(path=MODEL):
    import joblib
    return joblib.load(path)


def attach_bass(label, bass):
    """Absolute-note slash spelling is understood by the project's chord parser."""
    if label in ('N','X') or not bass or '/' in label:
        return label
    root=label.split(':')[0]
    return label if root==bass else f'{label}/{bass}'


def summarize_beats(model, evidence, start, end, beats):
    """Duration vote, not loudness vote: a loud passing tone cannot own a bar."""
    bounds=sorted({float(start),float(end),*[float(t) for t in beats if start<t<end]})
    votes={};details=[]
    for a,b in zip(bounds,bounds[1:]):
        if b-a<.08:
            continue
        result=estimate(model,evidence,a,b)
        details.append(dict(start=a,end=b,**result))
        if result['note']:
            votes[result['note']]=votes.get(result['note'],0.)+b-a
    order=sorted(votes,key=votes.get,reverse=True)
    winner=order[0] if order else None
    duration=end-start
    accepted=(winner is not None and votes[winner]>=duration*.5-1e-8
              and (len(order)<2 or votes[winner]>votes[order[1]]+1e-8))
    return dict(note=winner if accepted else None,
                coverage=votes.get(winner,0.)/max(duration,1e-9),beats=details)
