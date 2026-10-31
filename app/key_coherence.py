"""Experimental continuous key scoring; no song names, override gates or mode bonus.

The acoustic scores are uncalibrated evidence, not reported probabilities.
The chord channel measures explanatory cost and correctly voiced tonic arrivals.
"""
import math
import re
from types import SimpleNamespace

NAMES = ('C','C#','D','D#','E','F','F#','G','G#','A','A#','B')
MAJOR = {0,2,4,5,7,9,11}
MINOR = {0,2,3,5,7,8,10}

def key_id(label):
    m = re.fullmatch(r'([A-G])([#b]?)(m|:min|:minor| minor|:maj|:major| major)?', str(label).strip())
    if not m:
        return None
    pc = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[m[1]]
    pc += {'#':1,'b':-1,'':0}[m[2]]
    return pc % 12, 'minor' if m[3] in ('m',':min',':minor',' minor') else 'major'

def name(pc, mode):
    return NAMES[pc] + ('m' if mode == 'minor' else '')

def pcs(c):
    values = {(c.root+x) % 12 for x in [0,c.third,c.fifth,c.seventh,
              *[off for off,_ in c.colors]] if x is not None}
    if c.bass is not None:
        values.add(c.bass % 12)
    return values

def native(c, tonic, mode):
    intervals = {(p-tonic)%12 for p in pcs(c)}
    scale = MAJOR if mode == 'major' else MINOR
    if intervals <= scale:
        return True
    # Harmonic-minor raised seventh is contextual, not a universal extra scale tone.
    return mode == 'minor' and intervals <= scale|{11} and (
        ((c.root-tonic)%12 == 7 and c.third == 4) or
        ((c.root-tonic)%12 == 11 and c.third == 3 and c.fifth == 6))

def tonic_chord(c, tonic, mode):
    return c.root == tonic and c.third == (3 if mode == 'minor' else 4) and c.fifth == 7

def dominant_quality(c):
    # A major seventh or augmented triad is not evidence of a conventional V(7).
    # Altered dominants need a separate, explicit model; do not infer one here.
    return c.third == 4 and c.fifth == 7 and c.seventh in (None, 10)

def harmonic_spans(chords):
    """Coalesce adjacent identical harmonies without crossing gaps or mutating input."""
    spans=[]
    for c in chords:
        if not (math.isfinite(c.start) and math.isfinite(c.end) and 0 <= c.start < c.end):
            raise ValueError('Chord times must be finite with 0 <= start < end')
        if spans and c.start < spans[-1].end - 1e-5:
            raise ValueError('Chord spans must be ordered and non-overlapping')
        signature=(c.root,c.third,c.fifth,c.seventh,c.bass,tuple(sorted(c.colors)))
        if spans and abs(spans[-1].end-c.start)<1e-5 and spans[-1].signature==signature:
            spans[-1].end=c.end
            spans[-1].dur=c.end-spans[-1].start
        else:
            spans.append(SimpleNamespace(root=c.root,third=c.third,fifth=c.fifth,
                seventh=c.seventh,bass=c.bass,colors=list(c.colors),label=c.label,
                start=c.start,end=c.end,dur=c.end-c.start,signature=signature))
    return spans

def evidence(chords, tonic, mode):
    chords=harmonic_spans(chords)
    duration = sum(c.dur for c in chords) or 1.0
    cost = home = native_time = outside_mass = pitch_mass = 0.0
    exceptions = {}
    arrivals = returns = 0
    for i,c in enumerate(chords):
        scale = {(tonic+x)%12 for x in (MAJOR if mode=='major' else MINOR)}
        if mode=='minor' and native(c,tonic,mode):
            scale |= pcs(c)
        # This weighted pitch statistic is diagnostic only. Compatibility below
        # still classifies the complete chord, including its extensions.
        weighted=[(c.root,1.0)]
        for off,w in [(c.third,1.0),(c.fifth,.7),(c.seventh,.5),
                      *[(off,.25) for off,_ in c.colors]]:
            if off is not None:weighted.append(((c.root+off)%12,w))
        if c.bass is not None and c.bass!=c.root:weighted.append((c.bass,.5))
        pitch_mass += c.dur*sum(w for _,w in weighted)
        outside_mass += c.dur*sum(w for pc,w in weighted if pc not in scale)
        following = chords[i+1] if i+1 < len(chords) else None
        adjacent = following is not None and abs(c.end-following.start)<1e-5
        if native(c,tonic,mode):
            compatibility,role = 1.0,'native'
            native_time += c.dur
        elif (adjacent and dominant_quality(c) and following.third in (3,4)
              and following.fifth==7 and (c.root-following.root)%12==7
              and native(following,tonic,mode)):
            compatibility,role = .75,'resolved_secondary_dominant'
        elif native(c,tonic,'minor' if mode=='major' else 'major'):
            compatibility,role = .55,'parallel_borrowing'
        else:
            compatibility,role = .12,'unexplained'
        cost += c.dur*math.log(compatibility)
        if role!='native':
            k=(c.label,role);exceptions[k]=exceptions.get(k,0.0)+c.dur
        if tonic_chord(c,tonic,mode):
            home += c.dur
            if i and abs(chords[i-1].end-c.start)<1e-5:
                prev=chords[i-1]
                if (prev.root-tonic)%12==7 and dominant_quality(prev):
                    arrivals+=1
                elif (native(prev,tonic,mode) and prev.fifth==7 and
                      (prev.root-tonic)%12 in (2,5)):
                    returns+=1
    return dict(log_compatibility=cost/duration,outside_pitch_share=outside_mass/(pitch_mass or 1),tonic_share=home/duration,
                authentic_arrivals=arrivals,predominant_returns=returns,native_share=native_time/duration,
                exceptions=[dict(chord=k[0],role=k[1],seconds=v) for k,v in
                            sorted(exceptions.items(),key=lambda x:-x[1])])

def rank(chords, acoustic_candidates=()):
    """Continuous score: acoustic + compatibility + tonic occupancy + arrivals.

    Arrival counts are saturating support, not a claim of phrase-final cadences.
    No absent tonic penalty: not having a tonic does not establish another key.
    """
    chords=list(chords)
    scores={}
    for c in acoustic_candidates:
        key=key_id(c['key'])
        value=float(c['score'])
        if key is None or key in scores or not math.isfinite(value) or value<0:
            raise ValueError('Acoustic candidates require unique valid keys and finite nonnegative scores')
        scores[key]=value
    if scores and (len(scores)!=24 or not math.isfinite(sum(scores.values())) or sum(scores.values())<=0):
        raise ValueError('Provide all 24 acoustic scores with positive total, or omit the acoustic channel')
    total=sum(scores.values())
    if not chords and not total:
        return []
    out=[]
    for pc in range(12):
        for mode in ('major','minor'):
            e=evidence(chords,pc,mode)
            acoustic=math.log(max(scores.get((pc,mode),0)/total,1e-6)) if total else 0.0
            harmonic=(3*e['log_compatibility']+.6*e['tonic_share']+
                      .4*min(e['authentic_arrivals'],4)/4+.1*min(e['predominant_returns'],8))
            out.append(dict(key=name(pc,mode),score=acoustic+harmonic,
                            acoustic_term=acoustic,harmonic_term=harmonic,**e))
    return sorted(out,key=lambda r:-r['score'])
