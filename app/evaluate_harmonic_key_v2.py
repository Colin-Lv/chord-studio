"""Develop on isolated harmonic_dev, freeze, then open the new sealed scores.

Old GiantSteps180/Beatles17/Xuni are regression only. Resolver remains frozen.
The tiny grid selects harmonic coefficients, never a song-specific rule.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import time

import harmonic_key_v2 as h
from chord_studio import parse_chord_name
from evaluate_global_key_fusion import signature
from key_disagreement import relation
from key_fusion import key_identity, read_raw_lab
from key_profile_baseline import PROFILES

CODE_FILES = ['harmonic_key_v2.py','key_profile_baseline.py','chord_studio.py','chord_recognizer.py','audio_features.py','evaluate_harmonic_key_v2.py',
              'key_fusion.py','key_disagreement.py','evaluate_global_key_fusion.py','prepare_harmonic_key_data.py',
              'evaluate_harmonic_billboard_v2.py','evaluate_billboard_key_reasoning.py','run_harmonic_key_regressions.py',
              'finish_harmonic_key_experiment.py']
METHODS = ['cnn','harmonic_v1','chroma','harmonic_v2']


def implementation():
    return {name:signature(Path(__file__).parent/name)['sha256'] for name in CODE_FILES}


def records(split_path,acoustics,profiles,phases):
    manifest=json.loads(Path(split_path).read_text(encoding='utf-8'));result=[]
    # Filter before opening predictions: sealed outcomes are not read for tuning.
    selected=[row for row in manifest['rows'] if row['harmonic_selected'] and row['harmonic_split'] in phases]
    for row in selected:
        directory=Path(acoustics)/row['id']
        paths=dict(raw=directory/'raw.lab',cnn=directory/'cnn.json',neutral=directory/'neutral_analysis.json',profile=Path(profiles)/(row['id']+'.json'))
        if not all(path.is_file() for path in paths.values()):raise FileNotFoundError('Incomplete inputs: '+row['id'])
        cnn=json.loads(paths['cnn'].read_text(encoding='utf-8'));neutral=json.loads(paths['neutral'].read_text(encoding='utf-8'))
        profile=json.loads(paths['profile'].read_text(encoding='utf-8'));cache=json.loads((directory/'acoustics.json').read_text(encoding='utf-8'))
        if cache['audio']['sha256']!=profile['contract']['audio_sha256']:raise ValueError('Audio differs between evidence sources')
        expected={name:signature(Path(__file__).parent/name)['sha256'] for name in ['key_profile_baseline.py','chord_recognizer.py','audio_features.py']}
        if profile['contract']['implementation']!=expected or profile['contract']['profiles']!=PROFILES:raise ValueError('Stale profile cache')
        for name,cache_key in [('raw','raw'),('cnn','cnn'),('neutral','neutral')]:
            if signature(paths[name])['sha256']!=cache['outputs'][cache_key]['sha256']:raise ValueError('Acoustic cache changed')
        if abs(neutral['duration']-profile['duration'])>1/22050+1e-6:raise ValueError('Shared frontend duration mismatch')
        bundle=h.feature_bundle(read_raw_lab(paths['raw']),neutral['duration'],neutral.get('bars',[]))
        old=json.loads((directory/'case.json').read_text(encoding='utf-8'))
        result.append(dict(id=row['id'],group=row['group'],split=row['harmonic_split'],genre=row['genre'],
                           reference=row['reference'],bundle=bundle,cnn=key_identity(cnn['key']),
                           harmonic_v1=old['evidence']['raw_chord_functional']['key'],chroma=profile['key'],
                           cnn_models=cnn.get('model_files'),
                           inputs={name:signature(path) for name,path in paths.items()},audio=cache['audio']))
    return result


def metrics(rows,method):
    n=len(rows);full=tonic=mode=available=0;confusions={};matrix={};classes={'major':0,'minor':0,'missing':0}
    for row in rows:
        prediction=row[method];reference=row['reference']
        a,am=parse_chord_name(reference);b,bm=parse_chord_name(prediction) if prediction else (None,None)
        available+=prediction is not None;full+=prediction==reference;tonic+=a==b;mode+=am==bm
        classes['missing' if prediction is None else 'minor' if bm=='m' else 'major']+=1
        category='exact' if prediction==reference else relation(reference,prediction)
        confusions[category]=confusions.get(category,0)+1
        pair=f'{reference} -> {prediction}';matrix[pair]=matrix.get(pair,0)+1
    return dict(n=n,full_key_accuracy=full/n if n else None,tonic_accuracy=tonic/n if n else None,
                mode_accuracy=mode/n if n else None,mode_given_correct_tonic=full/tonic if tonic else None,
                correct_tonic_count=tonic,full_key_correct=full,prediction_coverage=available/n if n else None,
                predicted_modes=classes,confusions=confusions,confusion_matrix=matrix)


def summary(rows,methods=METHODS):
    return {method:dict(overall=metrics(rows,method),
                        major=metrics([row for row in rows if not row['reference'].endswith('m')],method),
                        minor=metrics([row for row in rows if row['reference'].endswith('m')],method)) for method in methods}


def predictions(records,policy,stage='G_full',details=False):
    rows=[]
    for record in records:
        prediction=h.analyze_bundle(record['bundle'],policy,stage,include_details=details)
        row={key:value for key,value in record.items() if key!='bundle'}
        row.update(harmonic_v2=prediction['key'],tonic=prediction['tonic'],mode=prediction['mode'],
                   status=prediction['status'],margins=prediction.get('margins'),prediction=prediction if details else None)
        rows.append(row)
    return rows


def ablation(records,policy):
    return {stage:summary(predictions(records,policy,stage),['harmonic_v2'])['harmonic_v2'] for stage in h.STAGES}


def select(split_path,acoustics,profiles,output):
    output=Path(output)
    if output.exists():raise ValueError('Refusing to overwrite frozen harmonic contract')
    source=records(split_path,acoustics,profiles,{'harmonic_dev','harmonic_calibration'})
    dev=[row for row in source if row['split']=='harmonic_dev'];cal=[row for row in source if row['split']=='harmonic_calibration']
    if not dev or not cal:raise ValueError('Missing isolated development or calibration partition')
    candidates=[]
    for duration,cadence,loop in itertools.product([.2,.35,.6],[.15,.35],[.1,.2]):
        policy=h.Policy(tonic_duration_weight=duration,cadence_weight=cadence,loop_weight=loop)
        report=summary(predictions(dev,policy),['harmonic_v2'])['harmonic_v2']
        major=report['major']['full_key_accuracy'];minor=report['minor']['full_key_accuracy']
        balanced=(major+minor)/2 if major is not None and minor is not None else report['overall']['full_key_accuracy']
        candidates.append(dict(policy=asdict(policy),report=report,objective=[balanced,min(major or 0,minor or 0),report['overall']['full_key_accuracy']]))
    # First maximum uses the predefined grid order as a reproducible tie breaker.
    best=max(candidates,key=lambda row:tuple(row['objective']));policy=h.Policy(**best['policy'])
    contract=dict(schema='harmonic_engine_contract_v2',prediction_schema=h.SCHEMA,policy=asdict(policy),
                  implementation=implementation(),split=signature(split_path),frozen_epoch=time.time(),
                  selection='development balanced major/minor full-key accuracy, then minimum class accuracy, then overall; calibration is validation only',
                  candidates=candidates,development_ids=[row['id'] for row in dev],calibration_ids=[row['id'] for row in cal],
                  dev_ablation=ablation(dev,policy),calibration_ablation=ablation(cal,policy),
                  calibration_summary=summary(predictions(cal,policy)),
                  inputs=[dict(id=row['id'],inputs=row['inputs'],audio=row['audio'],cnn_models=row['cnn_models']) for row in source],
                  new_sealed_predictions_read=False,resolver_training=False,deployment=False,
                  confidence_note='Harmonic scores and review margins are not calibrated confidence')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(contract,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(dict(policy=contract['policy'],dev=best['report'],calibration=contract['calibration_summary']['harmonic_v2']))
    return contract


def load_contract(path):
    contract=json.loads(Path(path).read_text(encoding='utf-8'))
    if contract['schema']!='harmonic_engine_contract_v2' or contract['prediction_schema']!=h.SCHEMA:raise ValueError('Invalid harmonic contract')
    if contract['implementation']!=implementation():raise ValueError('Harmonic implementation changed after freeze')
    return contract


def oracle(rows):
    n=len(rows)
    combinations=[['cnn','harmonic_v2'],['cnn','chroma'],['cnn','harmonic_v2','chroma']]
    return {' ∪ '.join(methods):dict(correct=sum(any(row[method]==row['reference'] for method in methods) for row in rows),
                                   n=n,accuracy=sum(any(row[method]==row['reference'] for method in methods) for row in rows)/n if n else None)
            for methods in combinations}


def save_report(rows,output,metadata):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(schema='harmonic_key_evaluation_v2',summary=summary(rows),oracles=oracle(rows),
                per_genre={genre:summary([row for row in rows if row['genre']==genre]) for genre in sorted({row['genre'] for row in rows})},
                rows=rows,**metadata)
    cnn=report['summary']['cnn']['overall']['full_key_correct'];union=report['oracles']['cnn ∪ harmonic_v2']['correct']
    report['resolver_prerequisite']=dict(cnn_correct=cnn,union_correct=union,complementary_candidates=union-cnn,
        necessary_oracle_condition=union>cnn,sufficient_for_training=False,
        decision='No resolver training in this round; any later training needs fresh validation, adequate class support and explicit deployment review')
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    fields=['id','reference','cnn','harmonic_v1','chroma','harmonic_v2','tonic','mode','status','genre']
    with output.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    for method in METHODS:
        matrix=report['summary'][method]['overall']['confusion_matrix'];keys=[root+suffix for suffix in ['', 'm'] for root in h.NOTES]
        with output.with_name(output.stem+'_'+method+'_confusion.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['reference / prediction']+keys+['missing'])
            for ref in keys:writer.writerow([ref]+[matrix.get(f'{ref} -> {key}',0) for key in keys]+[matrix.get(f'{ref} -> None',0)])
    print(json.dumps(dict(summary=report['summary'],oracles=report['oracles'],prerequisite=report['resolver_prerequisite']),ensure_ascii=False))
    return report


def evaluate(split_path,acoustics,profiles,contract_path,output,phase='harmonic_new_sealed'):
    contract=load_contract(contract_path)
    if signature(split_path)['sha256']!=contract['split']['sha256']:raise ValueError('Split changed after freeze')
    source=records(split_path,acoustics,profiles,{phase})
    if phase=='harmonic_new_sealed' and {row['id'] for row in source}&set(contract['development_ids']+contract['calibration_ids']):raise ValueError('Sealed track leakage')
    rows=predictions(source,h.Policy(**contract['policy']),details=True)
    return save_report(rows,output,dict(phase=phase,contract=signature(contract_path),split=signature(split_path),
        evaluation_epoch=time.time(),new_acoustic_inference=False,new_training=False,test_used_for_selection=False,
        independent_of_old_resolver_components=True,upstream_cnn_pretraining_overlap='not exhaustively verified'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['select','evaluate'])
    parser.add_argument('--split',type=Path,required=True);parser.add_argument('--acoustics',type=Path,required=True)
    parser.add_argument('--profiles',type=Path,required=True);parser.add_argument('--contract',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['harmonic_dev','harmonic_calibration','harmonic_new_sealed'],default='harmonic_new_sealed')
    args=parser.parse_args()
    if args.action=='select':select(args.split,args.acoustics,args.profiles,args.output)
    else:
        if not args.contract:parser.error('--contract is required after freezing')
        evaluate(args.split,args.acoustics,args.profiles,args.contract,args.output,args.phase)
