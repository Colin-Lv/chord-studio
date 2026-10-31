"""Primary CNN/harmonic agreement, auxiliary evidence, calibrated conflict gates.

No reference, song title, artist, or dataset identifier is an inference feature.
Without a compatible calibrated artifact disagreements remain explicitly ambiguous.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

import key_fusion as legacy
from chord_studio import parse_chord_name

SCHEMA="key_disagreement_resolver_v2"
FEATURE_SCHEMA="key_conflict_features_v2"
DEFAULT_MODEL=Path(__file__).resolve().parents[1]/"benchmarks/models/key_resolver_v2.json"


def relation(a,b):
    if not a or not b:return "missing"
    a,b=legacy.key_identity(a),legacy.key_identity(b)
    if a==b:return "same_key"
    root,mode=parse_chord_name(a);other,other_mode=parse_chord_name(b)
    if root==other:return "parallel_mode" if mode!=other_mode else "same_tonic"
    delta=(other-root)%12
    if mode!=other_mode and (delta==9 if mode=="" else delta==3):return "relative_major_minor"
    if mode==other_mode and delta in {5,7}:return "perfect_fifth"
    return "other"


def features(evidence):
    a=evidence['madmom_cnn'];h=evidence['raw_chord_functional'];j=evidence['functional_joint'];w=evidence['window_timeline']
    result={}
    def put(name,value):
        result[name]=float(value) if value is not None and math.isfinite(float(value)) else 0.
        result[name+"__valid"]=float(value is not None and math.isfinite(float(value)))
    for prefix,source in [('cnn',a),('harmonic',h)]:
        rows=source.get('candidates',[])
        for name,value in [('top_score',rows[0]['score'] if rows else None),('second_score',rows[1]['score'] if len(rows)>1 else None),('margin',source.get('margin')),('relative_margin',source.get('relative_margin'))]:put(prefix+'_'+name,value)
    def candidate(source,key):return next((r for r in source.get('candidates',[]) if r['key']==key),{})
    hr=candidate(a,h.get('key'));put('harmonic_rank_in_cnn',hr.get('rank'))
    put('harmonic_cnn_gap',a['candidates'][0]['score']-hr['score'] if hr and a.get('candidates') else None)
    for name in ['decisive_coverage','unresolved_coverage']:put(name,w.get(name))
    for role,key in [('cnn',a.get('key')),('harmonic',h.get('key'))]:
        put(role+'_minor_mode',float(key.endswith('m')) if key else None)
        d=h.get('breakdown',{}).get(key,{})
        durations=d.get('function_seconds',{});total=sum(durations.values())
        for degree in ['I','V','IV','II']:
            put(role+'_'+degree+'_duration',durations.get(degree,0)/total if total else None)
        put(role+'_predominant_duration',(durations.get('II',0)+durations.get('IV',0))/total if total else None)
        for degree in ['I','V']:
            put(role+'_strongest_function_'+degree,float(d.get('strongest_function')==degree) if d else None)
        for name in ['duration_weighted_fit','out_of_scale_fraction','cadence_vi_count','ending_tonic','common_pair_hits','common_loop_hits']:
            value=d.get(name)
            if name=='ending_tonic':value=float('I' in d.get('ending_functions',[])) if d else None
            put(role+'_'+name,value)
        put(role+'_common_pair_fraction',d.get('common_pair_hits',0)/d['pair_total'] if d.get('pair_total') else None)
        put(role+'_out_of_key_seconds',d.get('out_of_scale_fraction',0)*total if d else None)
        metrics=w.get('candidate_metrics',{}).get(key,{})
        for name in ['stable_coverage','candidate_decisive_share','top_competitor_share_valid','top_competitor_share_decisive','candidate_vs_competitor_gap']:
            put(role+'_'+name,metrics.get(name))
        segments=[s for s in w.get('segments',[]) if s['key']==key and s.get('status')=='supported']
        put(role+'_segment_count',len(segments) if w.get('available') else None)
        put(role+'_longest_segment_seconds',max((s['end']-s['start'] for s in segments),default=0) if w.get('available') else None)
        put(role+'_joint_rank',candidate(j,key).get('rank'))
        put(role+'_joint_support',float(j.get('key')==key) if j.get('available') else None)
    put('joint_relative_margin',j.get('relative_margin'))
    put('modulation',float(w.get('modulation_detected',False)) if w.get('available') else None)
    put('pitched_coverage',h.get('pitched_coverage'))
    for name in ['perfect_fifth','relative_major_minor','parallel_mode','same_tonic','other']:
        result['relation_'+name]=float(relation(a.get('key'),h.get('key'))==name)
    return result


def load_model(path=None):
    path=Path(path or DEFAULT_MODEL)
    if not path.is_file():return None
    model=json.loads(path.read_text(encoding='utf-8'))
    if model.get('feature_schema')!=FEATURE_SCHEMA or model.get('schema')!=SCHEMA:
        raise ValueError('Incompatible key resolver artifact')
    # Bind every feature-producing module, not only the vector dimension.
    expected_modules={'key_disagreement.py','key_fusion.py','detect_key_timeline.py','chord_studio.py'}
    if set(model.get('implementation',{}))!=expected_modules:
        raise ValueError('Incomplete resolver implementation contract')
    for name,expected in model['implementation'].items():
        actual=hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError('Resolver implementation changed: '+name)
    model['_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    return model


def gate_scores(model,values):
    import numpy as np
    x=np.array([values[name] for name in model['feature_names']],dtype=float)
    result={}
    for name,head in model['heads'].items():
        if not head.get('available'):result[name]=None;continue
        z=(x-np.array(head['mean']))/np.array(head['scale'])
        logit=float(z@np.array(head['coef'])+head['intercept'])
        result[name]=1/(1+math.exp(-max(-700.,min(700.,logit))))
    return result


def primary_ready(evidence):
    """Shared input-quality requirement for learned and rule conflict gates."""
    a,h=evidence['madmom_cnn'],evidence['raw_chord_functional']
    return bool(a.get('complete_24') and h.get('complete_24') and (a.get('margin') or 0)>0 and (h.get('margin') or 0)>0
        and h.get('pitched_coverage',0)>=legacy.POLICY.min_pitched_coverage
        and h.get('event_count',0)>=legacy.POLICY.min_events and h.get('root_count',0)>=legacy.POLICY.min_roots)


def decision(evidence,model=None):
    a,h=evidence['madmom_cnn'],evidence['raw_chord_functional']
    provisional=a.get('key') or h.get('key') or evidence['functional_joint'].get('key')
    if not primary_ready(evidence):return provisional,'ambiguous',['Primary evidence missing, tied or insufficient'],{}
    if a['key']==h['key']:return a['key'],'agreement',['Primary CNN and global harmony agree; joint and timeline are auxiliary'],{}
    vector=features(evidence)
    if model is None:return provisional,'ambiguous',['Primary sources disagree; no calibrated resolver artifact'],dict(features=vector)
    expected_models=model.get('cnn_model_sha256')
    observed_models=sorted(item['sha256'] for item in (a.get('original') or {}).get('model_files',[]))
    if expected_models is not None and observed_models!=expected_models:
        return provisional,'ambiguous',['CNN weight identity differs from calibration; resolver withheld'],dict(features=vector)
    scores=gate_scores(model,vector)
    accepted={name:score is not None and score>=model['thresholds'][name] for name,score in scores.items()}
    audit=dict(features=vector,gate_scores=scores,thresholds=model['thresholds'],model_sha256=model.get('_sha256'),
               score_kind='logistic discrimination score; cutoff selected on calibration, not calibrated probability')
    timeline=evidence['window_timeline']
    audit['support_summary']=dict(cnn=a['key'],harmonic=h['key'],joint=evidence['functional_joint'].get('key'),
        decisive_coverage=timeline.get('decisive_coverage'),unresolved_coverage=timeline.get('unresolved_coverage'),
        harmonic_support=timeline.get('candidate_metrics',{}).get(h['key'],{}),
        cnn_margin=a.get('margin'),harmonic_relative_margin=h.get('relative_margin'))
    if accepted.get('harmonic') and not accepted.get('cnn'):
        return h['key'],'disagreement_resolved_harmonic',[
            f"Harmonic gate {scores['harmonic']} >= {model['thresholds']['harmonic']}; CNN gate {scores['cnn']} did not pass {model['thresholds']['cnn']}",
            f"calibration artifact: {model.get('_sha256')}"],audit
    if accepted.get('cnn') and not accepted.get('harmonic'):
        return a['key'],'disagreement_resolved_cnn',[
            f"CNN gate {scores['cnn']} >= {model['thresholds']['cnn']}; harmonic gate {scores['harmonic']} did not pass {model['thresholds']['harmonic']}"],audit
    return provisional,'ambiguous',['Neither gate or both gates passed; no unique resolution'],audit


def analyze(cnn,neutral,raw_events,*,requested_key=None,model=None):
    old=legacy.analyze_key_evidence(cnn,neutral,raw_events,requested_key=requested_key)
    if requested_key:return old
    key,path,reasons,audit=decision(old['evidence'],model)
    resolved=path!='ambiguous'
    old.update(key=key,source=SCHEMA,decision_path=path,status='resolved' if resolved else 'ambiguous',
        needs_review=not resolved or old['evidence']['window_timeline'].get('modulation_detected',False),
        reason=reasons,resolver=audit,evidence_roles=dict(primary=['madmom_cnn','raw_chord_functional'],
            auxiliary=['functional_joint','window_timeline','cadences','loops']),
        legacy_gate_checks=old.pop('gate_checks'),gate_checks={},policy_status='experimental disagreement resolver; unchanged default CNN')
    old['legacy_policy']=old.pop('policy')
    old['policy']=dict(primary_min_events=legacy.POLICY.min_events,primary_min_roots=legacy.POLICY.min_roots,
        primary_min_pitched_coverage=legacy.POLICY.min_pitched_coverage,disagreement_thresholds=model['thresholds'] if model else None,
        model_sha256=model.get('_sha256') if model else None)
    old['candidates'].sort(key=lambda row:(row['key']!=key,row['source_ranks']['raw_chord_functional'] or 99,row['key']))
    old['conflict']['cnn_overridden']=path=='disagreement_resolved_harmonic'
    old['conflict']['primary_disagreement']=old['evidence']['madmom_cnn'].get('key')!=old['evidence']['raw_chord_functional'].get('key')
    return old
