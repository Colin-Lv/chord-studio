"""Learn candidate-key ranking and select segment costs using symbolic controls only."""
import sys,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from app.harmonic_key_v3 import FEATURES,KEYS,MODEL,segment_features,analyze,key_name
from app.key_chord_adapter import parse_rows
NAMES=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def events(tokens,tonic=0,start=0,step=2):
    return [dict(start=start+i*step,end=start+(i+1)*step,label=f'{NAMES[(tonic+r)%12]}:{q}') for i,(r,q) in enumerate(tokens)]

MAJOR=[[(0,'maj'),(5,'maj'),(7,'7'),(0,'maj')],[(0,'maj'),(9,'min'),(2,'min7'),(7,'7'),(0,'maj7')],
       [(5,'maj7'),(7,'7'),(4,'min7'),(9,'min7'),(2,'min7'),(7,'7'),(0,'maj')],
       [(0,'maj'),(10,'maj'),(5,'maj'),(0,'maj')],[(0,'maj'),(5,'min'),(0,'maj')],
       [(0,'maj'),(7,'7'),(0,'maj')]+[(5,'maj'),(7,'maj'),(9,'min7')]*3+[(5,'maj'),(7,'7'),(0,'maj')]]
MINOR=[[(0,'min'),(5,'min'),(7,'7'),(0,'min')],[(0,'min'),(8,'maj'),(10,'maj'),(0,'min')],
       [(0,'min7'),(3,'maj'),(5,'min7'),(7,'7'),(0,'min')],
       [(0,'min'),(2,'hdim7'),(7,'7'),(0,'min')],
       [(0,'min'),(7,'7'),(0,'min')]+[(8,'maj'),(10,'maj'),(3,'maj')]*3+[(5,'min'),(7,'7'),(0,'min')]]

def corpus(seed,count):
    rng=np.random.default_rng(seed);out=[]
    for i in range(count):
        minor=int(rng.integers(2));tonic=int(rng.integers(12));templates=MINOR if minor else MAJOR
        tokens=[]
        for repeat in range(int(rng.integers(2,5))):tokens+=templates[int(rng.integers(len(templates)))]
        rotation=int(rng.integers(len(tokens)))
        tokens=tokens[rotation:]+tokens[:rotation]
        # An occasional secondary dominant must not relabel a whole region.
        if i%3==0:
            at=int(rng.integers(1,len(tokens)-1));r,q=tokens[at]
            tokens.insert(at,((r+7)%12,'7'))
        rows=events(tokens,tonic,step=float(rng.choice([1.,2.,3.])))
        out.append(dict(rows=rows,key=tonic+12*minor))
    return out

def paired_cases(seed,count):
    rng=np.random.default_rng(seed);out=[]
    for i in range(count):
        root=int(rng.integers(12));minor=i%3!=0
        target=(root+(7 if i%3==0 else 9 if i%3==1 else 0))%12
        target_index=target+12*minor;template=MINOR[0] if minor else MAJOR[0]
        step=float(rng.choice([1.5,2.,3.]))
        first=events(MAJOR[0]*2,root,step=step);start=first[-1]['end']
        bridge=events([(7,'7'),(0,'min' if minor else 'maj')],target,start,step)
        # Same prefix / tonicization; only following persistence differs.
        temporary=first+bridge+events(MAJOR[0]*2,root,bridge[-1]['end'],step)
        sustained=first+bridge+events(template*3,target,bridge[-1]['end'],step)
        out.extend([dict(kind='temporary',rows=temporary,keys=[key_name(root)]),
                    dict(kind='modulation',rows=sustained,keys=[key_name(root),key_name(target_index)],boundary=start)])
    return out

def fit():
    datasets={s:corpus(seed,n) for s,seed,n in [('train',91371,180),('validation',61327,40),('test',82319,40)]}
    x=[];y=[]
    for item in datasets['train']:
        f=segment_features(parse_rows(item['rows'])[0]);x.extend(f);y.extend([int(k==item['key']) for k in range(24)])
    candidates=[];best=None
    for c in [.1,1.,10.]:
        model=LogisticRegression(C=c,class_weight='balanced',max_iter=2000,random_state=20261101).fit(x,y)
        hits=sum(int((segment_features(parse_rows(item['rows'])[0])@model.coef_[0]).argmax())==item['key'] for item in datasets['validation'])
        candidates.append(dict(C=c,validation_hits=hits,total=40))
        if best is None or hits>best[0]:best=(hits,model,c)
    model=dict(schema='harmonic-key-ranker-v3',features=FEATURES,weights=best[1].coef_[0].tolist(),
               C=best[2],training='synthetic symbolic controls; no user-song fitting',change_cost=5.,short_cost=2.)
    controls=paired_cases(12319,5);cost_results=[]
    for cost in [3.,5.,8.,12.,18.,24.,32.,48.]:
        model['change_cost']=cost;correct=0
        for item in controls:
            rows=item['rows'];result=analyze(rows,rows[-1]['end'],model=model)
            correct+=[s['key'] for s in result['segments']]==item['keys']
        # Stationary pieces are essential negative controls for segmentation.
        # Candidate-ranking accuracy alone cannot detect over-segmentation.
        stationary=0
        for item in datasets['validation'][:16]:
            rows=item['rows'];result=analyze(rows,rows[-1]['end'],model=model)
            stationary += [s['key'] for s in result['segments']]==[key_name(item['key'])]
        cost_results.append(dict(cost=cost,correct=correct+stationary,total=len(controls)+16,
                                 paired_correct=correct,stationary_correct=stationary))
    # Balance detection against false alarms; the number of stationary examples
    # must not make a "never modulate" decoder win model selection.
    model['change_cost']=max(cost_results,key=lambda r:(r['paired_correct']/len(controls)+r['stationary_correct']/16,-r['cost']))['cost']
    MODEL.parent.mkdir(exist_ok=True);MODEL.write_text(json.dumps(model,indent=2)+'\n')
    report=dict(candidate_selection=candidates,cost_selection=cost_results,model=model,tests=[],stationary_test=[])
    for item in datasets['test']:
        f=segment_features(parse_rows(item['rows'])[0]);report['tests'].append(dict(truth=key_name(item['key']),predicted=key_name(int((f@np.array(model['weights'])).argmax()))))
        result=analyze(item['rows'],item['rows'][-1]['end'],model=model)
        report['stationary_test'].append(dict(truth=[key_name(item['key'])],predicted=[s['key'] for s in result['segments']]))
    report['paired_test']=[]
    for item in paired_cases(99217,6):
        result=analyze(item['rows'],item['rows'][-1]['end'],model=model)
        report['paired_test'].append(dict(kind=item['kind'],truth=item['keys'],predicted=[s['key'] for s in result['segments']],
                                          boundaries=[s['time'] for s in result['changes']],expected_boundary=item.get('boundary')))
    HERE.mkdir(exist_ok=True);(HERE/'training_report.json').write_text(json.dumps(report,indent=2))
    (HERE/'symbolic_data.json').write_text(json.dumps(dict(datasets=datasets,validation_pairs=controls,test_pairs=paired_cases(99217,6)),indent=2))
    print('ranker test',sum(r['truth']==r['predicted'] for r in report['tests']),'/40',flush=True)
    print('paired test',report['paired_test'],flush=True);print('selected',model,flush=True)
if __name__=='__main__':fit()
