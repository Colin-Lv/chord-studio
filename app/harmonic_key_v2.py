"""Experimental hierarchical tonic/mode evidence, independent of key resolver.

Compatibility, centre evidence and temporal grammar are separate quantities.
No candidate receives a fixed major/minor preference. Unknown quality is masked.
The old key engine and every frozen model remain untouched.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math

import numpy as np

from chord_studio import normalize_external_chord_label, parse_chord_name

SCHEMA = 'harmonic_tonic_mode_v2'
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR = {0: 'major', 2: 'minor', 4: 'minor', 5: 'major', 7: 'major', 9: 'minor', 11: 'dim'}
NATURAL_MINOR = {0: 'minor', 2: 'dim', 3: 'major', 5: 'minor', 7: 'minor', 8: 'major', 10: 'major'}
# These coexist with natural minor; harmonic minor never replaces the v chord.
HARMONIC_MINOR = {7: {'major'}, 11: {'dim'}}
BORROWED = {
    'major': {(1, 'major'), (3, 'major'), (5, 'minor'), (8, 'major'), (10, 'major')},
    'minor': {(0, 'major'), (2, 'minor'), (5, 'major'), (9, 'dim')},
}
DEGREES = {0: 'I', 1: 'bII', 2: 'II', 3: 'bIII', 4: 'III', 5: 'IV', 6: '#IV', 7: 'V', 8: 'bVI', 9: 'VI', 10: 'bVII', 11: 'VII'}
QUALITIES = ['major', 'minor', 'dim', 'aug', 'sus2', 'sus4', 'unknown']


@dataclass(frozen=True)
class Policy:
    root_weight: float = .4
    quality_weight: float = .6
    tonic_duration_weight: float = .35
    mode_tonic_quality_weight: float = .25
    cadence_weight: float = .25
    pair_weight: float = .08
    loop_weight: float = .18
    ending_weight: float = .06
    phrase_ending_weight: float = .04
    local_weight: float = .08
    entry_weight: float = .015
    window_bars: int = 8
    step_bars: int = 4
    local_margin: float = .03
    review_margin: float = .03
    transition_cost: float = .06

    def __post_init__(self):
        for name,value in asdict(self).items():
            if not math.isfinite(value) or value < 0:raise ValueError('Invalid policy field: '+name)
        if self.window_bars < 1 or self.step_bars < 1 or int(self.window_bars)!=self.window_bars or int(self.step_bars)!=self.step_bars:
            raise ValueError('Window and step must be positive integers')


DEFAULT_POLICY = Policy()
STAGES = ['A_fit', 'B_duration', 'C_cadence', 'D_loop', 'E_ending', 'F_local', 'G_full']


def quality_of(label):
    _, suffix = parse_chord_name(label.split('/', 1)[0])
    if 'm7b5' in suffix or 'hdim' in suffix or 'dim' in suffix or '°' in suffix or 'ø' in suffix:
        return 'dim'
    if 'sus2' in suffix:
        return 'sus2'
    if 'sus' in suffix:
        return 'sus4'
    if 'aug' in suffix or suffix.startswith('+'):
        return 'aug'
    if suffix.startswith('m') and not suffix.startswith('maj'):
        return 'minor'
    if suffix == '' or suffix.startswith(('maj', 'add', '6', '7', '9', '11', '13')):
        return 'major'
    return 'unknown'


def compatibility(interval, quality, mode):
    """Root membership and quality agreement, without functional bonuses."""
    scale = MAJOR if mode == 'major' else NATURAL_MINOR
    allowed = {scale[interval]} if interval in scale else set()
    if mode == 'minor':
        allowed |= HARMONIC_MINOR.get(interval, set())
    borrowed = (interval, quality) in BORROWED[mode]
    root = 1. if allowed else .35 if any(i == interval for i, _ in BORROWED[mode]) else -.35
    valid = quality not in {'unknown', 'sus2', 'sus4'}
    agreement = 1. if quality in allowed else .35 if borrowed else -.6
    if not valid:
        agreement = 0.
    degree = DEGREES[interval]
    if mode == 'minor':
        degree = {0:'I',2:'II',3:'III',5:'IV',7:'V',8:'VI',10:'VII',11:'vii'}.get(interval,degree)
    if quality == 'minor':degree=degree.lower()
    elif quality == 'dim':degree=degree.lower()+'°'
    return dict(root_compatibility=root, quality_compatibility=agreement, quality_valid=valid,
                allowed_qualities=sorted(allowed), borrowed=borrowed,
                functional_degree=degree, interval=interval, mode=mode)


def compatibility_table():
    entries = []
    triads = []
    for tonic in range(12):
        for mode in ['major', 'minor']:
            key = NOTES[tonic] + ('m' if mode == 'minor' else '')
            scale = MAJOR if mode == 'major' else NATURAL_MINOR
            triads.append(dict(key=key, natural_diatonic=[dict(root=NOTES[(tonic+i)%12], quality=q, interval=i) for i, q in scale.items()],
                               additional_minor=[dict(root=NOTES[(tonic+i)%12], quality=q, interval=i) for i, qs in HARMONIC_MINOR.items() for q in sorted(qs)] if mode == 'minor' else []))
            for root in range(12):
                for quality in QUALITIES:
                    entries.append(dict(key=key, tonic=NOTES[tonic], chord_root=NOTES[root], chord_quality=quality,
                                        **compatibility((root-tonic)%12, quality, mode)))
    return dict(schema='harmonic_compatibility_table_v2', entries=entries, theoretical_triads=triads,
                semantics='Scores describe compatibility only; tonic attraction is a separate feature')


def event_rows(events, duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid duration')
    rows = []; previous = 0.
    for event in events:
        start, end = float(event['start']), float(event['end'])
        if not math.isfinite(start+end) or start < previous-1e-6 or start < 0 or end <= start:
            raise ValueError('Invalid or overlapping chord intervals')
        # Only rounding-level overlaps from official annotations are tolerated.
        start = max(start, previous); previous = end
        label = normalize_external_chord_label(event.get('label', event.get('chord', 'X')), 'sharps')
        start, end = min(duration, start), min(duration, end)
        if end <= start or label in {'N', 'X'}:
            continue
        root, _ = parse_chord_name(label)
        if root is None:
            continue
        quality = quality_of(label)
        if rows and (rows[-1]['root'], rows[-1]['quality']) == (root, quality) and abs(rows[-1]['end']-start) < 1e-6:
            rows[-1]['end'] = end; rows[-1]['duration'] = end-rows[-1]['start']
        else:
            rows.append(dict(start=start, end=end, duration=end-start, root=root, quality=quality, original=label))
    return rows


def rotation_family(sequence):
    sequence = tuple(sequence)
    return min(sequence[i:]+sequence[:i] for i in range(len(sequence))) if sequence else ()


# Tokens retain both relative root and chord quality. Repeated rotations of
# one family do not create multiple votes for the same observed window.
LOOPS = {
    'major': {rotation_family(loop) for loop in [((0,'major'),(7,'major'),(9,'minor'),(5,'major')),
                                                ((0,'major'),(9,'minor'),(5,'major'),(7,'major'))]},
    'minor': {rotation_family(loop) for loop in [((0,'minor'),(10,'major'),(8,'major')),
                                                ((8,'major'),(10,'major'),(0,'minor')),
                                                ((0,'minor'),(8,'major'),(3,'major'),(10,'major')),
                                                ((0,'minor'),(5,'minor'),(10,'major'),(3,'major'))]},
}


def loop_evidence(tokens, mode):
    return loop_evidence_runs([tokens],mode)


def loop_evidence_runs(runs, mode):
    densities = []
    for length in sorted({len(family) for family in LOOPS[mode]}):
        opportunities = hits = 0
        for tokens in runs:
            for i in range(len(tokens)-length+1):
                opportunities += 1
                hits += rotation_family(tokens[i:i+length]) in LOOPS[mode]
        if opportunities:densities.append(hits/opportunities)
    return max(densities,default=0.)


def components(rows, tonic, mode, bars=()):
    total = sum(row['duration'] for row in rows)
    if not total:
        return {}
    token_rows = [((row['root']-tonic)%12, row['quality']) for row in rows]
    runs=[]
    for i,(row,token) in enumerate(zip(rows,token_rows)):
        if i==0 or row['start']-rows[i-1]['end']>1e-6:runs.append([])
        if not runs[-1] or runs[-1][-1]!=token:runs[-1].append(token)
    tokens=[token for run in runs for token in run]
    adjacent=[pair for run in runs for pair in zip(run,run[1:])]
    allowed_tonic = 'major' if mode == 'major' else 'minor'
    tonic_quality = lambda token: token == (0, allowed_tonic)
    dominant = lambda token: token[0] == 7 and token[1] in ({'major'} if mode == 'major' else {'major','minor'})
    leading = lambda token: token == (11, 'dim')
    predominant = lambda token: token[0] in {2,5} and compatibility(token[0], token[1], mode)['quality_compatibility'] == 1.
    arrivals = sum(tonic_quality(b) and (dominant(a) or leading(a)) for a,b in adjacent)
    pair_hits = sum((predominant(a) and dominant(b)) or (dominant(a) and tonic_quality(b)) or
                    (a[0] == 5 and tonic_quality(b) and compatibility(a[0],a[1],mode)['quality_compatibility'] == 1.) or
                    (mode == 'minor' and a == (10,'major') and tonic_quality(b))
                    for a,b in adjacent)
    fits = [compatibility(token[0],token[1],mode) for token in token_rows]
    quality_seconds = sum(row['duration'] for row,fit in zip(rows,fits) if fit['quality_valid'])
    root_fit = sum(row['duration']*fit['root_compatibility'] for row,fit in zip(rows,fits))/total
    quality_fit = sum(row['duration']*fit['quality_compatibility'] for row,fit in zip(rows,fits))/quality_seconds if quality_seconds else 0.
    tonic_seconds = sum(row['duration'] for row in rows if row['root'] == tonic)
    aligned = sum(row['duration']*(1 if row['quality'] == allowed_tonic else -1)
                  for row in rows if row['root'] == tonic and row['quality'] in {'major','minor'})
    # Measured bar endpoints are weak phrase proxies, never confirmed sections.
    ends = [bar['end'] for i,bar in enumerate(bars) if (i+1)%4 == 0]
    end_hits = sum(any(row['start'] < end <= row['end']+1e-7 and row['root'] == tonic and row['quality'] == allowed_tonic for row in rows) for end in ends)
    return dict(root_fit=root_fit, quality_fit=quality_fit, quality_coverage=quality_seconds/total,
                tonic_root_duration=tonic_seconds/total, tonic_quality_alignment=aligned/total,
                cadence=arrivals/max(1,len(adjacent)), cadence_count=arrivals,
                pair=pair_hits/max(1,len(adjacent)), loop=loop_evidence_runs(runs,mode),
                ending=float(tonic_quality(tokens[-1])), phrase_ending=end_hits/len(ends) if ends else 0.,
                entry=float(tonic_quality(tokens[0])), token_count=len(tokens), contiguous_runs=len(runs))


def global_scores(features, policy, stage, local=None):
    level = STAGES.index(stage)
    tonic_scores = []; modes = {}
    for tonic in range(12):
        by_mode = {}
        for mode in ['major','minor']:
            f = features[tonic][mode]
            fit = policy.root_weight*f['root_fit'] + policy.quality_weight*f['quality_fit']
            centre = fit
            mode_score = fit
            if level >= 1:
                centre += policy.tonic_duration_weight*f['tonic_root_duration']
                mode_score += policy.mode_tonic_quality_weight*f['tonic_quality_alignment']
            if level >= 2:
                grammar = policy.cadence_weight*f['cadence'] + policy.pair_weight*f['pair']
                centre += grammar; mode_score += grammar
            if level >= 3:
                centre += policy.loop_weight*f['loop']; mode_score += policy.loop_weight*f['loop']
            if level >= 4:
                ending = policy.ending_weight*f['ending'] + policy.phrase_ending_weight*f['phrase_ending']
                centre += ending; mode_score += ending
            if level >= 6:
                centre += policy.entry_weight*f['entry']; mode_score += policy.entry_weight*f['entry']
            by_mode[mode] = dict(mode=mode, score=mode_score, centre_score=centre, components=f)
        tonic_score = max(row['centre_score'] for row in by_mode.values())
        if level >= 5 and local:
            tonic_score += policy.local_weight*local['tonic_shares'].get(NOTES[tonic],0.)
        tonic_scores.append(dict(tonic=NOTES[tonic], pitch_class=tonic, score=tonic_score))
        ordered = sorted(by_mode.values(), key=lambda row:(-row['score'],row['mode']))
        for i,row in enumerate(ordered):row['rank']=i+1
        modes[NOTES[tonic]] = ordered
    ordered = sorted(tonic_scores,key=lambda row:(-row['score'],row['pitch_class']))
    for i,row in enumerate(ordered):row['rank']=i+1
    return ordered, modes


def checked_bars(bars, duration):
    checked=[];previous=0.;numbers=set()
    for i,bar in enumerate(bars):
        a,b=float(bar['start']),float(bar['end']);number=int(bar.get('bar',i+1))
        if not math.isfinite(a+b) or a < previous-1e-6 or a < 0 or b <= a or number in numbers:
            raise ValueError('Invalid bar grid')
        previous=b;numbers.add(number)
        if a < duration:checked.append(dict(bar=number,start=a,end=min(b,duration)))
    return checked


def feature_bundle(events, duration, bars=(), policy=DEFAULT_POLICY):
    rows=event_rows(events,duration);bars=checked_bars(bars,duration)
    if not rows:return dict(available=False,duration=duration,rows=[],bars=bars)
    features={tonic:{mode:components(rows,tonic,mode,bars) for mode in ['major','minor']} for tonic in range(12)}
    windows=[]
    if len(bars) >= policy.window_bars:
        starts=list(range(0,len(bars)-policy.window_bars+1,policy.step_bars))
        tail=len(bars)-policy.window_bars
        if starts[-1] != tail:starts.append(tail)
        for start in starts:
            group=bars[start:start+policy.window_bars];a,b=group[0]['start'],group[-1]['end']
            clips=[dict(row,start=max(a,row['start']),end=min(b,row['end']),duration=min(b,row['end'])-max(a,row['start'])) for row in rows if min(b,row['end'])>max(a,row['start'])]
            coverage=sum(row['duration'] for row in clips)/(b-a)
            if coverage < .5:continue
            fs={tonic:{mode:components(clips,tonic,mode,group) for mode in ['major','minor']} for tonic in range(12)}
            windows.append(dict(start=a,end=b,start_index=start,end_index=start+len(group)-1,features=fs,coverage=coverage))
    return dict(available=True,duration=duration,rows=rows,bars=bars,features=features,windows=windows,
                pitched_coverage=sum(row['duration'] for row in rows)/duration,
                root_count=len({row['root'] for row in rows}))


def local_timeline(bundle, policy):
    windows=bundle.get('windows',[])
    if not windows:return dict(available=False,windows=[],segments=[],tonic_shares={},decisive_bars=0,valid_bars=0)
    emission=[];rankings=[]
    for window in windows:
        tonic,modes=global_scores(window['features'],policy,'E_ending')
        rankings.append((tonic,modes));emission.append([next(row['score'] for row in tonic if row['pitch_class']==pc) for pc in range(12)])
    emission=np.asarray(emission);values=emission[0].copy();parents=[]
    transition=np.full((12,12),-policy.transition_cost);np.fill_diagonal(transition,0.)
    for observation in emission[1:]:
        choices=values[:,None]+transition
        previous=choices.argmax(axis=0);parents.append(previous)
        values=choices[previous,np.arange(12)]+observation
    path=[int(values.argmax())]
    for previous in reversed(parents):path.append(int(previous[path[-1]]))
    path.reverse();out=[]
    for window,(tonics,modes),selected,obs in zip(windows,rankings,path,emission):
        margin=float(obs[selected]-max(v for i,v in enumerate(obs) if i!=selected))
        mode=modes[NOTES[selected]][0]['mode']
        out.append(dict(start=window['start'],end=window['end'],start_index=window['start_index'],end_index=window['end_index'],tonic=NOTES[selected],mode=mode,key=NOTES[selected]+('m' if mode=='minor' else ''),margin=margin,stable=margin>=policy.local_margin))
    assigned=[]
    for i,bar in enumerate(bundle['bars']):
        matching=[window for window in out if window['start_index']<=i<=window['end_index']]
        if not matching:continue
        support=sum(max(0.,min(bar['end'],row['end'])-max(bar['start'],row['start'])) for row in bundle['rows'])
        if support/(bar['end']-bar['start'])<.5:continue
        nearest=min(matching,key=lambda row:abs((row['start']+row['end'])/2-(bar['start']+bar['end'])/2))
        assigned.append(dict(**bar,tonic=nearest['tonic'],mode=nearest['mode'],key=nearest['key'],stable=nearest['stable']))
    counts=Counter(row['tonic'] for row in assigned if row['stable']);n=len(assigned)
    segments=[]
    for row in assigned:
        status='supported' if row['stable'] else 'tentative'
        if segments and segments[-1]['key']==row['key'] and segments[-1]['status']==status and abs(segments[-1]['end']-row['start'])<1e-6:
            segments[-1]['end']=row['end'];segments[-1]['bars']+=1
        else:segments.append(dict(start=row['start'],end=row['end'],tonic=row['tonic'],mode=row['mode'],key=row['key'],status=status,bars=1))
    return dict(available=bool(assigned),windows=out,bar_assignments=assigned,segments=segments,
                tonic_shares={key:count/n for key,count in counts.items()} if n else {},
                decisive_bars=sum(counts.values()),valid_bars=n,
                dependence='Same chords and measured grid as global evidence, not an independent vote')


def analyze_bundle(bundle, policy=DEFAULT_POLICY, stage='G_full', include_details=True):
    if not bundle['available']:
        return dict(schema=SCHEMA,key=None,tonic=None,mode=None,status='ambiguous',tonic_candidates=[],mode_candidates_by_tonic={},final_key_candidates=[],window_evidence={},reason='No pitched chord evidence')
    local=local_timeline(bundle,policy) if STAGES.index(stage)>=5 else None
    tonics,modes=global_scores(bundle['features'],policy,stage,local)
    tonic=tonics[0]['tonic'];mode=modes[tonic][0]['mode']
    tonic_margin=tonics[0]['score']-tonics[1]['score'];mode_margin=modes[tonic][0]['score']-modes[tonic][1]['score']
    final=[]
    for row in tonics:
        for conditional in modes[row['tonic']]:
            final.append(dict(key=row['tonic']+('m' if conditional['mode']=='minor' else ''),tonic=row['tonic'],mode=conditional['mode'],rank=len(final)+1,tonic_score=row['score'],conditional_mode_score=conditional['score']))
    ambiguous=tonic_margin<policy.review_margin or mode_margin<policy.review_margin or bundle['root_count']<3
    if not include_details:
        modes={key:[{k:v for k,v in row.items() if k!='components'} for row in candidates] for key,candidates in modes.items()}
    return dict(schema=SCHEMA,key=final[0]['key'],tonic=tonic,mode=mode,status='ambiguous' if ambiguous else 'resolved',
                tonic_candidates=tonics,mode_candidates_by_tonic=modes,final_key_candidates=final,
                margins=dict(tonic=tonic_margin,conditional_mode=mode_margin),window_evidence=local or {},
                policy=asdict(policy),stage=stage,pitched_coverage=bundle['pitched_coverage'],
                score_semantics='Choose tonic using maximum conditional centre evidence; choose mode only within that tonic. Scores are not probabilities.',
                phrase_evidence='Fixed four-bar endings are weak proxies; no verse/chorus labels are inferred')


def analyze(events,duration,bars=(),policy=DEFAULT_POLICY,stage='G_full',include_details=True):
    return analyze_bundle(feature_bundle(events,duration,bars,policy),policy,stage,include_details)


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    from key_fusion import read_raw_lab
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--neutral',type=Path,required=True)
    parser.add_argument('--contract',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    neutral=json.loads(args.neutral.read_text(encoding='utf-8'))
    if args.contract:
        from evaluate_harmonic_key_v2 import load_contract
        policy=Policy(**load_contract(args.contract)['policy'])
    else:policy=DEFAULT_POLICY
    result=analyze(read_raw_lab(args.raw),neutral['duration'],neutral.get('bars',[]),policy)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print({name:result[name] for name in ['key','tonic','mode','status']})
