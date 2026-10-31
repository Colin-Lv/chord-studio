"""Versioned candidate-relative key features and offline linear scorer.

Only fixed CNN top3 candidates. No reference, identity, local timeline, chroma,
loop term, user corrections or data-role input is consumed by features.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import numpy as np
from app.web.storage import ROOT, atomic_json, read_json, signature

SCHEMA='cnn_top3_key_features_v1'
NOTES=('C','C#','D','D#','E','F','F#','G','G#','A','A#','B')
PC={n:i for i,n in enumerate(NOTES)}|dict(Db=1,Eb=3,Gb=6,Ab=8,Bb=10)
CNN_FIELDS=('cnn_score','cnn_rank','cnn_gap_from_first','cnn_gap_to_next','cnn_same_tonic_mode_margin','candidate_minor')
HARMONIC_FIELDS=('global_tonic_rank','global_tonic_gap_from_first','global_hierarchical_rank','global_conditional_mode_margin',
    'tonic_root_duration','tonic_quality_alignment','root_fit','quality_fit','quality_coverage','ending','phrase_ending','cadence','pair')
FEATURES={'M1':list(CNN_FIELDS),'M2':list(CNN_FIELDS)+list(HARMONIC_FIELDS)+[n+'_valid' for n in HARMONIC_FIELDS]}
EPS=1e-12
CODE_FILES=('app/key_top3_reranker_v1.py','app/train_key_top3_reranker_v1.py')


def canonical(key):
    if not isinstance(key,str) or not key:raise ValueError('Missing complete key')
    minor=key.endswith('m');root=key[:-1] if minor else key
    if root not in PC:raise ValueError('Invalid complete key: '+key)
    return NOTES[PC[root]]+('m' if minor else '')


def candidates(cnn):
    values=cnn.get('candidates',[])
    if len(values)!=24:raise ValueError('CNN requires the complete 24-key vector')
    rows=[]
    for i,row in enumerate(values):
        score=float(row['score'])
        if not math.isfinite(score) or score<0:raise ValueError('Invalid CNN score')
        rows.append(dict(key=canonical(row['key']),score=score,original_position=i+1))
    if len({r['key'] for r in rows})!=24:raise ValueError('CNN keys collapse or repeat after spelling normalization')
    # Exact score descending; ties retain the original frozen vector order.
    rows.sort(key=lambda row:(-row['score'],row['original_position']))
    if canonical(cnn['key'])!=rows[0]['key']:raise ValueError('CNN top1 differs from its vector')
    return [dict(row,rank=i+1) for i,row in enumerate(rows)]


def target_labels(keys,reference):
    if reference is None:raise ValueError('Unsupervisable reference; never generate three false negatives')
    ref=canonical(reference)
    return [int(key==ref) for key in keys]


def global_view(harmonic):
    """Read-only view: remove inherited local/Viterbi and loop contributions.

    Frozen predictions remain intact. These explicit derived ranks use the
    existing frozen global coefficients; no new coefficient is selected here.
    Tied evidence has a shared competition rank, independent of tonic spelling.
    """
    if not harmonic or not harmonic.get('key'):return {}
    if harmonic.get('schema')!='harmonic_tonic_mode_v2' or harmonic.get('stage')!='G_full':raise ValueError('Wrong frozen harmonic schema/stage')
    policy=harmonic['policy'];view={}
    for tonic in NOTES:
        modes={}
        for row in harmonic['mode_candidates_by_tonic'][tonic]:
            loop=policy['loop_weight']*row['components']['loop']
            modes[row['mode']]=dict(score=row['score']-loop,centre=row['centre_score']-loop,components=row['components'])
        if set(modes)!={'major','minor'}:raise ValueError('Incomplete conditional mode evidence')
        view[tonic]=dict(modes=modes,score=max(r['centre'] for r in modes.values()))
    for tonic,item in view.items():
        item['rank']=1+sum(other['score']>item['score']+EPS for other in view.values())
        item['gap']=max(other['score'] for other in view.values())-item['score']
    return view


def feature_rows(cnn,harmonic=None):
    vector=candidates(cnn);top=vector[:3];scores={r['key']:r['score'] for r in vector};view=global_view(harmonic);result=[]
    for row in top:
        key=row['key'];minor=key.endswith('m');tonic=key.removesuffix('m');opposite=tonic if minor else tonic+'m'
        values=[row['score'],row['rank'],vector[0]['score']-row['score'],row['score']-vector[row['rank']]['score'],scores[key]-scores[opposite],float(minor)]
        fields={}
        if view:
            mode='minor' if minor else 'major';other='major' if minor else 'minor';v=view[tonic];m=v['modes'][mode];f=m['components']
            mode_rank=1+int(v['modes'][other]['score']>m['score']+EPS)
            fields=dict(global_tonic_rank=v['rank'],global_tonic_gap_from_first=v['gap'],
                global_hierarchical_rank=2*(v['rank']-1)+mode_rank,global_conditional_mode_margin=m['score']-v['modes'][other]['score'],
                **{n:f.get(n) for n in HARMONIC_FIELDS[4:]})
        numeric=[];valid=[]
        for name in HARMONIC_FIELDS:
            value=fields.get(name)
            ok=value is not None and isinstance(value,(int,float)) and math.isfinite(value)
            numeric.append(float(value) if ok else np.nan);valid.append(float(ok))
        result.append(dict(key=key,cnn_rank=row['rank'],M1=values,M2=values+numeric+valid))
    return result


def fit_model(rows,method,C):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    X=np.asarray([f[method] for r in rows for f in r['features']],dtype=float)
    y=np.asarray([v for r in rows for v in target_labels([f['key'] for f in r['features']],r['reference'])])
    if set(y)!={0,1}:raise ValueError('Training lacks positive/negative candidate labels')
    pipeline=Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True)),
                       ('scaler',StandardScaler()),('lr',LogisticRegression(C=C,l1_ratio=0,solver='lbfgs',max_iter=2000,tol=1e-8,class_weight=None))])
    pipeline.fit(X,y,lr__sample_weight=np.repeat(1/3,len(y)))
    scaler=pipeline.named_steps['scaler'];lr=pipeline.named_steps['lr']
    model=dict(schema='offline_key_top3_lr_v1',feature_schema=SCHEMA,method=method,C=C,features=FEATURES[method],
               imputer_fill=0,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coefficient=lr.coef_[0].tolist(),intercept=float(lr.intercept_[0]),
               score_kind='uncalibrated_candidate_discriminant',automatic_override_enabled=False,
               training_ids=[r['id'] for r in rows],training_groups=sorted({r['group'] for r in rows}),
               preprocessing_fit_ids=[r['id'] for r in rows],sample_weight_per_song=1.,iterations=lr.n_iter_.tolist())
    predicted=pipeline.decision_function(X)
    reconstructed=decision_scores(model,[f for r in rows for f in r['features']])
    if not np.allclose(predicted,reconstructed,atol=1e-12,rtol=1e-12):raise ValueError('Serialized scoring differs from fitted LR')
    return model


def decision_scores(model,features):
    X=np.asarray([row[model['method']] for row in features],dtype=float)
    X=np.where(np.isnan(X),model['imputer_fill'],X)
    return ((X-np.asarray(model['mean']))/np.asarray(model['scale']))@np.asarray(model['coefficient'])+model['intercept']


def rank(model,features):
    scores=decision_scores(model,features)
    if not np.isfinite(scores).all():raise ValueError('Nonfinite discriminant')
    order=sorted(range(len(features)),key=lambda i:(-float(scores[i]),features[i]['cnn_rank']))
    return [dict(key=features[i]['key'],cnn_rank=features[i]['cnn_rank'],score=float(scores[i]),rank=j+1) for j,i in enumerate(order)]


def load_model(path):
    model=read_json(path)
    if model['schema']!='offline_key_top3_lr_v1' or model['feature_schema']!=SCHEMA or model['features']!=FEATURES[model['method']]:raise ValueError('Wrong model/schema')
    for name,digest in model.get('implementation',{}).items():
        if signature(ROOT/name)['sha256']!=digest:raise ValueError('Reranker implementation changed: '+name)
    contract=model.get('harmonic_contract')
    if contract and signature(contract['path'])['sha256']!=contract['sha256']:raise ValueError('Frozen upstream contract changed')
    if contract:
        for name,digest in read_json(contract['path'])['implementation'].items():
            if signature(ROOT/'app'/name)['sha256']!=digest:raise ValueError('Frozen upstream implementation changed')
    return model


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--cnn',type=Path,required=True);p.add_argument('--harmonic',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();model=load_model(args.model);features=feature_rows(read_json(args.cnn),read_json(args.harmonic) if args.harmonic else None)
    ranked=rank(model,features)
    atomic_json(args.output,dict(schema='offline_top3_key_prediction_v1',key=ranked[0]['key'],ranking=ranked,
                  feature_schema=SCHEMA,features=[{**r,'M2':[None if not math.isfinite(v) else v for v in r['M2']]} for r in features],
                  inputs={name:dict(path=str(path),**signature(path)) for name,path in dict(cnn=args.cnn,model=args.model,harmonic=args.harmonic).items() if path},
                  score_kind='uncalibrated_candidate_discriminant',automatic_override_enabled=False,new_acoustic_inference=False))
    print(dict(method=model['method'],key=ranked[0]['key'],automatic_override_enabled=False))


if __name__=='__main__':main()
