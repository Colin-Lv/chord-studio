"""Chord-first, segmental tonal inference. No song IDs or acoustic-key override.

Root, chord quality, extensions and observed slash bass retain separate semantics.
The learned candidate ranker is trained outside inference. Scores are not accuracy.
"""
from __future__ import annotations
import json,math,hashlib
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from .key_chord_adapter import parse_rows
from . import key_coherence as theory

VERSION='harmonic-key-reasoner-v3'
MODEL=Path(__file__).parent/'web/models/harmonic_key_v3.json'
KEYS=[(pc,mode) for mode in ('major','minor') for pc in range(12)]
FEATURES=['native','borrowed','secondary','unexplained','outside','tonic','opposite_tonic',
          'dominant','authentic_arrival','plagal_arrival','predominant_to_dominant',
          'tonic_departure','entry_tonic','ending_tonic']


def key_name(index):return theory.name(*KEYS[index])


def chord_features(chords):
    """Local explanatory features; transitions never cross silence/gaps."""
    n=len(chords);f=np.zeros((n,24,len(FEATURES)));roles=[]
    for i,c in enumerate(chords):
        prev=chords[i-1] if i and abs(chords[i-1].end-c.start)<1e-5 else None
        following=chords[i+1] if i+1<n and abs(c.end-chords[i+1].start)<1e-5 else None
        explanations=[]
        for k,(tonic,mode) in enumerate(KEYS):
            scale={(tonic+p)%12 for p in (theory.MAJOR if mode=='major' else theory.MINOR)}
            native=theory.native(c,tonic,mode)
            secondary=(not native and following is not None and theory.dominant_quality(c)
                       and (c.root-following.root)%12==7 and following.third in (3,4)
                       and following.fifth==7 and theory.native(following,tonic,mode))
            borrowed=not native and not secondary and theory.native(c,tonic,'minor' if mode=='major' else 'major')
            outside=len(theory.pcs(c)-scale)/max(1,len(theory.pcs(c)))
            if native:outside=0.
            home=theory.tonic_chord(c,tonic,mode)
            opposite=theory.tonic_chord(c,tonic,'minor' if mode=='major' else 'major')
            dominant=(c.root-tonic)%12==7 and theory.dominant_quality(c)
            auth=prev is not None and home and (prev.root-tonic)%12==7 and theory.dominant_quality(prev)
            plagal=prev is not None and home and (prev.root-tonic)%12==5 and theory.native(prev,tonic,mode)
            pred=prev is not None and dominant and (prev.root-tonic)%12 in (2,5) and theory.native(prev,tonic,mode)
            departure=prev is not None and theory.tonic_chord(prev,tonic,mode) and native
            f[i,k,:12]=[native,borrowed,secondary,not(native or borrowed or secondary),outside,
                         home,opposite,dominant,auth,plagal,pred,departure]
            explanations.append('调内' if native else '已解决的次属和弦' if secondary else '平行调借用' if borrowed else '含需解释的变化音')
        roles.append(explanations)
    return f,roles


def segment_features(chords):
    if not chords:return np.zeros((24,len(FEATURES)))
    chords=theory.harmonic_spans(chords)
    f,_=chord_features(chords);w=np.array([c.dur for c in chords])
    result=np.average(f,axis=0,weights=w)
    result[:,12]=f[0,:,5]*w[0]/w.sum();result[:,13]=f[-1,:,5]*w[-1]/w.sum()
    return result


def load_model(path=MODEL):
    model=json.loads(Path(path).read_text(encoding='utf8'))
    if model['features']!=FEATURES or len(model['weights'])!=len(FEATURES):
        raise ValueError('调性排序模型特征版本不匹配')
    return model


def validate(rows,duration):
    if not math.isfinite(duration) or duration<=0:raise ValueError('无效音频时长')
    end=0.
    for row in rows:
        a,b=float(row['start']),float(row['end'])
        if not math.isfinite(a+b) or a<end-1e-6 or a<0 or b<=a or b>duration+.05:
            raise ValueError('和弦时间轴必须有序、无重叠且位于音频内')
        end=b


def analyze(rows,duration,bars=(),sections=(),acoustic=None,model=None):
    validate(rows,duration);chords,stats=parse_rows(rows)
    chords=theory.harmonic_spans(chords)
    if len(chords)<2:
        return dict(method=VERSION,status='insufficient',segments=[],changes=[],main_key=None,
                    reason='有效和弦不足，无法建立主音中心',parse=stats)
    model=model or load_model();coef=np.asarray(model['weights']);f,roles=chord_features(chords)
    durations=np.array([c.dur for c in chords]);n=len(chords)
    terminal=np.array([chord_features([c])[0][0,:,:4] for c in chords])
    bar_lengths=[b['end']-b['start'] for b in bars if b['end']>b['start']]
    unit=float(np.median(bar_lengths)) if bar_lengths else float(np.median(durations))
    unit=max(.25,unit)
    # All valid durations count. Adjacent cell subdivision cannot multiply evidence.
    mass=durations/unit
    cumulative=np.concatenate([np.zeros((1,24,len(FEATURES))),np.cumsum(f*mass[:,None,None],axis=0)])
    cm=np.r_[0,np.cumsum(mass)]
    cuts=list(range(n+1));boundary_search='all chord boundaries'
    if n>500:
        # Keep every chord observation. Restrict only the quadratic search of
        # possible tonal-region boundaries for very fragmented source tracks.
        proposed=[float(b['start']) for b in bars if 'start' in b]
        if not proposed or len(proposed)>400:proposed=np.linspace(chords[0].start,chords[-1].end,400)
        starts=np.array([c.start for c in chords])
        cuts=sorted({0,n,*[int(np.searchsorted(starts,t)) for t in proposed]})
        boundary_search='bar/time proposals snapped to chord boundaries (all observations retained)'
    # A normalization per observation is constant across competing partitions.
    # Normalizing each region's mean logits would reward moving a chromatic
    # chord into an incompatible region merely to sharpen that region's rank.
    normalizer=np.r_[0,np.cumsum(logsumexp(f@coef,axis=1)*mass)]
    section_starts=[float(s['start']) for s in sections if 'start' in s]
    change=float(model.get('change_cost',5.));short=float(model.get('short_cost',2.))
    transition=np.full((24,24),change)
    for a,(ta,ma) in enumerate(KEYS):
        for b,(tb,mb) in enumerate(KEYS):
            # Relative and parallel changes remain allowed, never prohibited.
            shared=len({(ta+x)%12 for x in (theory.MAJOR if ma=='major' else theory.MINOR)}&
                       {(tb+x)%12 for x in (theory.MAJOR if mb=='major' else theory.MINOR)})
            transition[a,b]+= .15*(7-shared)
    np.fill_diagonal(transition,np.inf)
    dp=np.full((n+1,24),-np.inf);parents={};span_cache={}
    for stop,j in enumerate(cuts[1:],1):
        for i in cuts[:stop]:
            # One isolated chord supplies chord quality, not an independent
            # tonal progression. It remains explained by an adjoining region.
            # Long holds and repeated frame labels cannot change this fact.
            if j-i<2:continue
            size=cm[j]-cm[i]
            features=(cumulative[j]-cumulative[i])/size
            # Do not count an arrival from a chord outside this proposed region.
            features[:,8:12]-=f[i,:,8:12]*mass[i]/size
            # A resolved secondary dominant needs its resolution INSIDE the
            # region, just as an arrival needs its predecessor inside it.
            features[:,:4]+=(terminal[j-1]-f[j-1,:,:4])*mass[j-1]/size
            features[:,12]=f[i,:,5]*mass[i]/size;features[:,13]=f[j-1,:,5]*mass[j-1]/size
            logits=features@coef
            scores=logits*size-(normalizer[j]-normalizer[i])
            scores-=short*max(0,3-size)**2
            span_cache[i,j]=(logits,scores)
            if i==0:values=scores;previous=np.full(24,-1)
            else:
                discount=.5 if any(abs(chords[i].start-t)<.15 for t in section_starts) else 0.
                choices=dp[i,:,None]-transition+discount
                previous=choices.argmax(0);values=choices[previous,np.arange(24)]+scores
            better=values>dp[j]
            for k in np.flatnonzero(better):parents[j,int(k)]=(i,int(previous[k]))
            dp[j]=np.maximum(dp[j],values)
    end_key=int(dp[n].argmax());parts=[];j=n;k=end_key
    while j:
        i,previous=parents[j,k];parts.append((i,j,k));j,k=i,previous
    parts.reverse();segments=[]
    for index,(i,j,k) in enumerate(parts):
        logits,scores=span_cache[i,j];order=np.argsort(logits)[::-1]
        start=chords[i].start;end=chords[j-1].end
        # Region boundaries bracket the first explanatory chord, never copied from annotations.
        if index+1<len(parts):end=chords[parts[index+1][0]].start
        rivals=[int(x) for x in order if int(x)!=k]
        alternative=rivals[0];margin=float(logits[k]-logits[alternative])
        evidence=[]
        region_f,region_roles=chord_features(chords[i:j])
        for t in range(i,j):
            c=chords[t];rel=(c.root-KEYS[k][0])%12
            evidence.append(dict(start=c.start,end=c.end,chord=c.label,degree_semitones=rel,
                                 role=region_roles[t-i][k],tonic=bool(region_f[t-i,k,5]),arrival=bool(region_f[t-i,k,8])))
        alternatives=[]
        tonic,mode=KEYS[k]
        relative=((tonic+9)%12+12) if mode=='major' else (tonic+3)%12
        parallel=tonic+(12 if mode=='major' else 0)
        for rival in dict.fromkeys([k,*rivals[:2],relative,parallel]):
            alternatives.append(dict(key=key_name(rival),score=round(float(logits[rival]),5),
                                     explanations=[dict(chord=chords[t].label,role=region_roles[t-i][rival]) for t in range(i,j)]))
        alternatives_keep=[]
        for neighbour in [parts[index-1][2] if index else None,parts[index+1][2] if index+1<len(parts) else None]:
            if neighbour is not None and neighbour!=k and neighbour not in alternatives_keep:
                alternatives_keep.append(neighbour)
        keep=[dict(key=key_name(kk),local_fit_loss=round(float(scores[k]-scores[kk]),4),
                   explanations=[dict(chord=chords[t].label,role=region_roles[t-i][kk]) for t in range(i,j)]) for kk in alternatives_keep]
        segments.append(dict(start=round(start,6),end=round(end,6),key=key_name(k),
                             evidence_seconds=round(float(durations[i:j].sum()),6),
                             status='ambiguous' if margin<1 else 'supported',margin=round(margin,4),
                             evidence=evidence,candidates=alternatives,keep_neighbour_key=keep,
                             start_bar=next((b.get('bar') for b in bars if b['start']<=start<b['end']),None),
                             end_bar=next((b.get('bar') for b in bars if b['start']<end<=b['end']+.001),None)))
    changes=[]
    for prev,cur in zip(segments,segments[1:]):
        t=cur['start'];near=[c for c in chords if c.start<=t+.001 and c.end>=t-.001]
        previous_index=KEYS.index(theory.key_id(prev['key']))
        new_index=KEYS.index(theory.key_id(cur['key']))
        distinctive=next((c.start for x,c in enumerate(chords) if t<=c.start<cur['end']
                          and f[x,new_index,0] and not f[x,previous_index,0]),None)
        arrival=next((e['start'] for e in cur['evidence'] if e['arrival']),None)
        changes.append(dict(time=t,from_key=prev['key'],to_key=cur['key'],status=cur['status'],
                            first_distinctive_chord=distinctive,first_authentic_arrival=arrival,
                            boundary_interval=[max(prev['start'],near[0].start if near else t),
                                               min(cur['end'],near[-1].end if near else t)],
                            evidence=cur['evidence'][:4],interpretation='转调候选'))
    shares={}
    for segment in segments:shares[segment['key']]=shares.get(segment['key'],0.)+segment['evidence_seconds']
    main=max(shares,key=shares.get)
    # Report transient applied dominants independently of the decoded region path.
    tonicizations=[]
    for a,b in zip(chords,chords[1:]):
        if abs(a.end-b.start)>1e-5 or not theory.dominant_quality(a) or (a.root-b.root)%12!=7:continue
        if b.third not in (3,4) or b.fifth!=7:continue
        target=theory.name(b.root,'minor' if b.third==3 else 'major')
        region=next((s for s in segments if s['start']<=a.start<s['end']),None)
        if region and b.end<=region['end']+1e-6 and target!=region['key']:
            tonicizations.append(dict(start=a.start,end=b.end,target=target,context=region['key'],chords=[a.label,b.label]))
    section_keys=[]
    for section in sections:
        a,b=section.get('start',0),section.get('end',0)
        overlaps=[dict(key=s['key'],start=max(a,s['start']),end=min(b,s['end'])) for s in segments if min(b,s['end'])>max(a,s['start'])]
        section_keys.append(dict(label=section.get('name') or section.get('label') or '段落',start=a,end=b,keys=overlaps))
    # Preserve short contrasting passages for review even if the global path
    # keeps its tonal context. They are hypotheses, not extra modulation votes.
    excursions=[]
    for i,j,k in parts:
        begin=None
        for t in range(i,j+1):
            outside=t<j and not f[t,k,0]
            if outside and begin is None:begin=t
            if begin is not None and (not outside or (t+1<j and chords[t].end<chords[t+1].start-1e-5)):
                stop=t if not outside else t+1
                local=chords[begin:stop];features=segment_features(local);scores=features@coef
                ranked=[int(x) for x in np.argsort(scores)[::-1][:3]]
                local_f,local_roles=chord_features(local)
                tonic,mode=KEYS[k]
                parallel=tonic+(12 if mode=='major' else 0)
                parallel_relative=(tonic+3)%12 if mode=='major' else (tonic+9)%12+12
                candidates=[]
                for candidate in dict.fromkeys([k,*ranked,parallel,parallel_relative]):
                    candidates.append(dict(key=key_name(candidate),score=round(float(scores[candidate]),4),
                        explanations=[dict(chord=c.label,role=local_roles[x][candidate]) for x,c in enumerate(local)]))
                excursions.append(dict(start=local[0].start,end=local[-1].end,context=key_name(k),
                    interpretation='保留原调上下文；对照借用、主音化与短转调解释',
                    candidates=candidates,chords=[c.label for c in local],
                    independent_progression=len(local)>1))
                begin=None
    gaps=[];cursor=0.
    for c in chords:
        if c.start>cursor+1e-5:gaps.append(dict(start=cursor,end=c.start))
        cursor=c.end
    if cursor<duration:gaps.append(dict(start=cursor,end=duration))
    return dict(method=VERSION,status='computed',segments=segments,changes=changes,uncovered=gaps,
                start_key=segments[0]['key'],end_key=segments[-1]['key'],main_key=main,
                key_seconds=shares,summary_method='duration of decoded tonal regions',
                tonicizations=tonicizations,excursions=excursions,sections=section_keys,parse=stats,
                coverage=sum(durations)/duration,analyzed_range=[chords[0].start,chords[-1].end],
                boundary_search=boundary_search,
                acoustic_reference=acoustic,acoustic_used_for_decoding=False,
                score_kind='uncalibrated candidate rank; not probability',
                model_sha256=hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest(),
                limitations=['和弦错误会传播到调性判断','主音化与转调可能有多种合理解释',
                             '尚未使用独立旋律转录消歧','主要调性按局部段落时长汇总；不是唯一全曲调性'])
