"""Fit two small correctness heads; choose abstention cutoffs only on calibration.

Sealed-test rows are never opened here. Pure rule gate is an explicit comparator.
Artifacts are portable numeric JSON, not executable pickles.
"""
from collections import Counter
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import key_disagreement as resolver
from evaluate_global_key_fusion import signature


def read_cases(split_path,root,phases):
    split=json.loads(Path(split_path).read_text(encoding='utf-8'));cases=[];missing=[]
    for row in split['rows']:
        if not row['selected'] or row['split'] not in phases:continue
        path=Path(root)/row['id']/'case.json'
        if not path.exists():missing.append(row['id']);continue
        case=json.loads(path.read_text(encoding='utf-8'))
        if case['group']!=row['group'] or case['reference']!=row['reference'] or case['split']!=row['split']:raise ValueError('Split/case mismatch')
        cases.append(case)
    return cases,missing


def rule_decision(e,parameters):
    key,path,reasons,_=resolver.decision(e)
    if path=='agreement' or not resolver.primary_ready(e):return key,path,reasons
    a,h,w=e['madmom_cnn'],e['raw_chord_functional'],e['window_timeline']
    metrics=w.get('candidate_metrics',{}).get(h.get('key'),{})
    rank=next((r['rank'] for r in a.get('candidates',[]) if r['key']==h.get('key')),99)
    checks=dict(decisive_coverage=(w.get('decisive_coverage') or 0)>=parameters['coverage'],
        decisive_dominance=(metrics.get('candidate_decisive_share') or 0)>=parameters['dominance'],
        competitor_gap=(metrics.get('candidate_vs_competitor_gap') or 0)>=parameters['gap'],
        cnn_candidate_support=rank<=2,harmonic_margin=(h.get('relative_margin') or 0)>=.05)
    if a.get('key')!=h.get('key') and all(checks.values()):return h['key'],'disagreement_resolved_harmonic',[checks]
    return key,'ambiguous',[checks]


def evaluate_cases(cases,model=None,rule=None):
    rows=[]
    for case in cases:
        e=case['evidence'];cnn=e['madmom_cnn'].get('key');harmonic=e['raw_chord_functional'].get('key')
        if rule is not None:key,path,reason=rule_decision(e,rule);audit={}
        else:key,path,reason,audit=resolver.decision(e,model)
        reference=case['reference']
        rows.append(dict(id=case['id'],dataset=case['dataset'],genre=case.get('genre','unknown'),reference=reference,cnn=cnn,
            harmonic=harmonic,final=key,decision_path=path,reason=reason,audit=audit,
            correct=key==reference,cnn_correct=cnn==reference,harmonic_correct=harmonic==reference,
            resolved=path!='ambiguous',confusion=resolver.relation(reference,key)))
    n=len(rows);resolved=sum(r['resolved'] for r in rows);wrong=sum(r['resolved'] and not r['correct'] for r in rows)
    matrix={method:dict(Counter(f"{r['reference']} -> {r[method]}" for r in rows)) for method in ['cnn','harmonic','final']}
    source_metrics={}
    for method in ['cnn','harmonic']:
        emitted=sum(r[method] is not None for r in rows)
        errors=sum(r[method] is not None and r[method]!=r['reference'] for r in rows)
        source_metrics[method]=dict(accuracy=sum(r[method]==r['reference'] for r in rows)/n if n else None,
            resolved_coverage=emitted/n if n else None,ambiguous_rate=(n-emitted)/n if n else None,
            wrong_but_resolved=errors,wrong_but_resolved_rate=errors/n if n else None,
            corrected=sum(not r['cnn_correct'] and r[method]==r['reference'] for r in rows),
            broken=sum(r['cnn_correct'] and r[method]!=r['reference'] for r in rows),
            confusions=dict(Counter(resolver.relation(r['reference'],r[method]) for r in rows)),
            resolution_semantics='baseline emits its top key without a confidence gate')
    return dict(n=n,accuracy=sum(r['correct'] for r in rows)/n if n else None,
        accuracy_semantics='includes provisional CNN answers on ambiguous cases; inspect resolved coverage separately',
        source_metrics=source_metrics,
        cnn_accuracy=sum(r['cnn_correct'] for r in rows)/n if n else None,harmonic_accuracy=sum(r['harmonic_correct'] for r in rows)/n if n else None,
        resolved_coverage=resolved/n if n else None,ambiguous_rate=(n-resolved)/n if n else None,
        wrong_but_resolved=wrong,wrong_but_resolved_rate=wrong/n if n else None,wrong_among_resolved=wrong/resolved if resolved else None,
        corrected=sum(not r['cnn_correct'] and r['correct'] for r in rows),broken=sum(r['cnn_correct'] and not r['correct'] for r in rows),
        confusions=dict(Counter(r['confusion'] for r in rows)),confusion_matrix=matrix,rows=rows)


def objective(report):
    return (report['broken'],-report['corrected'],report['wrong_but_resolved'],-(report['resolved_coverage'] or 0))


def train(split_path,root,output):
    cases,missing=read_cases(split_path,root,{'development','calibration'})
    if missing:raise ValueError('Incomplete development/calibration: '+','.join(missing))
    implementation={name:signature(Path(__file__).parent/name)['sha256'] for name in ['key_fusion.py','detect_key_timeline.py','chord_studio.py']}
    if any(case.get('evidence_implementation')!=implementation for case in cases):
        raise ValueError('Stale evidence implementation; run run_key_resolver_audio.py --evidence-only first')
    development=[r for r in cases if r['split']=='development'];calibration=[r for r in cases if r['split']=='calibration']
    if {r['group'] for r in development}&{r['group'] for r in calibration}:raise ValueError('Group leakage')
    disagree=lambda rows:[r for r in rows if r['evidence']['madmom_cnn']['key']!=r['evidence']['raw_chord_functional']['key']]
    train_rows=disagree(development);cal_rows=disagree(calibration)
    if not train_rows:raise ValueError('No development disagreements')
    names=sorted(resolver.features(train_rows[0]['evidence']));x=np.array([[resolver.features(r['evidence'])[n] for n in names] for r in train_rows])
    scaler=StandardScaler().fit(x);z=scaler.transform(x);heads={}
    for name,source in [('cnn','madmom_cnn'),('harmonic','raw_chord_functional')]:
        y=np.array([int(r['evidence'][source]['key']==r['reference']) for r in train_rows]);counts=Counter(y.tolist())
        available=len(train_rows)>=20 and min(counts.get(0,0),counts.get(1,0))>=5
        head=dict(available=available,development_class_counts=dict(counts))
        if available:
            lr=LogisticRegression(C=1.,class_weight='balanced',max_iter=2000,random_state=20261006).fit(z,y)
            head.update(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=lr.coef_[0].tolist(),intercept=float(lr.intercept_[0]))
        heads[name]=head
    model=dict(schema=resolver.SCHEMA,feature_schema=resolver.FEATURE_SCHEMA,feature_names=names,heads=heads,thresholds=dict(cnn=1.01,harmonic=1.01))
    # Small calibration partitions do not justify an automatic override merely
    # because the training head can fit. Count both positive and negative cases.
    grids={};coverage={}
    for name,source in [('cnn','madmom_cnn'),('harmonic','raw_chord_functional')]:
        counts=Counter(int(r['evidence'][source]['key']==r['reference']) for r in cal_rows)
        enough=len(cal_rows)>=12 and min(counts.get(0,0),counts.get(1,0))>=3
        grids[name]=[.5,.6,.7,.8,.9,.95,.99,1.01] if enough and heads[name]['available'] else [1.01]
        coverage[name]=dict(class_counts=dict(counts),sufficient=enough)
    candidates=[]
    for cnn,harmonic in itertools.product(grids['cnn'],grids['harmonic']):
        model['thresholds']=dict(cnn=cnn,harmonic=harmonic)
        report=evaluate_cases(calibration,model)
        candidates.append((objective(report),dict(model['thresholds']),report))
    _,thresholds,chosen=min(candidates,key=lambda r:(r[0],-r[1]['harmonic'],-r[1]['cnn']))
    model['thresholds']=thresholds
    rule_options=[]
    for coverage_min,dominance,gap in itertools.product([.25,.5,.75],[.6,.75,.9],[.2,.4,.6]):
        parameters=dict(coverage=coverage_min,dominance=dominance,gap=gap)
        report=evaluate_cases(calibration,rule=parameters);rule_options.append((objective(report),parameters,report))
    # Explicit always-abstain conflict baseline, available even on tiny data.
    parameters=dict(coverage=1.01,dominance=1.01,gap=1.01)
    report=evaluate_cases(calibration,rule=parameters);rule_options.append((objective(report),parameters,report))
    _,best_rule,rule_report=min(rule_options,key=lambda r:(r[0],-r[1]['coverage'],-r[1]['dominance'],-r[1]['gap']))
    model.update(family='two_logistic_correctness_heads',calibration_support=coverage,split=signature(split_path),
        cnn_model_sha256=sorted(item['sha256'] for item in cases[0]['evidence']['madmom_cnn']['original'].get('model_files',[])),
        training_ids=[r['id'] for r in train_rows],calibration_ids=[r['id'] for r in calibration],
        development_disagreements=len(train_rows),calibration_disagreements=len(cal_rows),
        selection_objective='min broken, max corrected, min wrong resolved, max resolved; calibration only',
        implementation={n:signature(Path(__file__).parent/n)['sha256'] for n in ['key_disagreement.py','key_fusion.py','detect_key_timeline.py','chord_studio.py']},
        input_cases=[signature(Path(root)/r['id']/'case.json') for r in cases],
        training_implementation=signature(__file__),
        calibration_search=[dict(thresholds=thresholds,metrics={k:v for k,v in report.items() if k not in {'rows','confusion_matrix'}}) for _,thresholds,report in candidates],
        rule_calibration_search=[dict(parameters=parameters,metrics={k:v for k,v in report.items() if k not in {'rows','confusion_matrix'}}) for _,parameters,report in rule_options],
        pure_rule_baseline=best_rule,calibration_comparison=dict(logistic=chosen,rule=rule_report),
        score_kind='discrimination scores with calibrated cutoffs, not guaranteed probabilities',
        independent_test_used=False,deployment_approved=False)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if any(sorted(item['sha256'] for item in case['evidence']['madmom_cnn']['original'].get('model_files',[]))!=model['cnn_model_sha256'] for case in cases):
        raise ValueError('Mixed CNN weight versions in training/calibration')
    if output.exists():raise ValueError('Refusing to replace a frozen resolver; use a new experiment path')
    output.write_text(json.dumps(model,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:model[k] for k in ['development_disagreements','calibration_disagreements','thresholds','calibration_support','pure_rule_baseline']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();train(a.split,a.cases,a.output)
