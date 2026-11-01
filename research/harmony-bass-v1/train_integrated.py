"""Expand high-register coverage, with separate validation and held-out timbres."""
import sys,importlib.util,json
from pathlib import Path
import numpy as np,soundfile as sf,joblib
from sklearn.ensemble import ExtraTreesClassifier
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]))
from app.web import bass_voice as bass
from app.web.storage import atomic_json,signature
import validate_integrated as challenge
OUT=HERE/'integrated-training';OUT.mkdir(exist_ok=True)


def generate():
    for split,programs,seed in [('train',[0,3,4,5,24,25,28,29,32,33,48,80],2026111310),
            ('validation',[2,26,50,81],2026112310),('test',[6,7,30,31],2026113310)]:
        folder=OUT/split
        if not (folder/'manifest.json').exists():challenge.main(programs,folder,seed)


def dataset(split):
    folder=OUT/split;xs=[];ys=[];truth=[];records=[]
    for group in json.loads((folder/'manifest.json').read_text()):
        audio,rate=sf.read(folder/(group['id']+'.wav'));evidence=bass.spectrum(audio,rate)
        for event in group['events']:
            for i in range(4):
                a=event['start']+i*.5;b=a+.5
                x,_=bass.feature_rows(evidence,a,b);xs.append(x)
                ys.append((bass.NOTES==event['bass_midi']).astype(int));truth.append(-1 if event['truth'] is None else bass.NAMES.index(event['truth']))
            records.append(dict(group=group['id'],kind=group['id'].split('-',1)[1],**event))
    return np.array(xs),np.array(ys),np.array(truth),records


def scores(model,x):
    raw=model.predict_proba(x.reshape(-1,x.shape[-1]))[:,list(model.classes_).index(1)].reshape(-1,len(bass.NOTES))
    return np.stack([raw[:,bass.NOTES%12==pc].max(1) for pc in range(12)],axis=1)


def aggregate(pc,threshold,margin):
    # Equal half-second beats, exactly the deployed duration-vote rule.
    result=[]
    for group in pc.reshape(-1,4,12):
        candidates=group.argmax(1);votes=np.bincount(candidates,minlength=12);winner=int(votes.argmax())
        mask=candidates==winner;values=group[mask];margins=np.sort(values,axis=1)[:,-1]-np.sort(values,axis=1)[:,-2]
        ok=(votes[winner]>=2 and np.count_nonzero(votes==votes[winner])==1
            and values[:,winner].mean()>=threshold and margins.mean()>=margin)
        result.append(winner if ok else -1)
    return np.array(result)


def train():
    old=np.load(HERE/'training-register-v2/features.npz')
    x,y,_,_=dataset('train');ox=old['x'][old['splits']=='train'];oy=old['y'][old['splits']=='train']
    x=np.concatenate([x,ox]);y=np.concatenate([y,oy]);vx,vy,vt,vr=dataset('validation');truth=vt[::4]
    candidates=[];best=None
    for leaf in [2,8]:
        model=ExtraTreesClassifier(n_estimators=160,min_samples_leaf=leaf,class_weight='balanced',n_jobs=4,random_state=20261103)
        model.fit(x.reshape(-1,x.shape[-1]),y.ravel());model.n_jobs=1
        pc=scores(model,vx)
        old_pc=scores(model,old['x'][old['splits']=='validation']);old_truth=old['labels'][old['splits']=='validation']
        for threshold in [.4,.5,.6,.7,.8]:
            for margin in [.05,.1,.2]:
                predicted=aggregate(pc,threshold,margin);accepted=predicted>=0
                order=np.sort(old_pc,axis=1);old_accepted=(order[:,-1]>=threshold)&((order[:,-1]-order[:,-2])>=margin)
                correct=int((predicted[accepted]==truth[accepted]).sum())+int((old_pc.argmax(1)[old_accepted]==old_truth[old_accepted]).sum())
                total=int(accepted.sum()+old_accepted.sum());precision=correct/max(total,1)
                item=dict(leaf=leaf,threshold=threshold,margin=margin,accepted=total,correct=correct,precision=precision)
                candidates.append(item)
                if precision>=.99 and (best is None or (total,precision)>(best[0],best[1])):best=(total,precision,model,threshold,margin)
        print('fit leaf',leaf,flush=True)
    if best is None:raise RuntimeError('No validation point meets precision target; keep the installed model')
    _,_,model,threshold,margin=best;model.bass_min_score_=threshold;model.bass_min_margin_=margin
    path=OUT/'bass_voice_v3.joblib';joblib.dump(model,path)
    reports={}
    for split in ['validation','test']:
        x,y,truth,records=dataset(split);predicted=aggregate(scores(model,x),threshold,margin);true=truth[::4]
        for row,pred in zip(records,predicted):row['predicted_bass']=None if pred<0 else bass.NAMES[pred]
        accepted=predicted>=0
        reports[split]=dict(n=len(true),correct=int((predicted==true).sum()),accepted=int(accepted.sum()),
            accepted_correct=int((predicted[accepted]==true[accepted]).sum()),abstained=int(((predicted<0)&(true>=0)).sum()),
            silence_false_positive=int(((predicted>=0)&(true<0)).sum()),
            high_register=dict(n=sum(r['kind']=='high_piano' for r in records),correct=sum(r['truth']==r['predicted_bass'] for r in records if r['kind']=='high_piano')),rows=records)
    report=dict(training='old train split + new rendered MIDI beat features; no real-song fitting',
        selection=candidates,model=signature(path),threshold=threshold,margin=margin,reports=reports)
    atomic_json(OUT/'report.json',report);print(json.dumps({s:{k:v for k,v in r.items() if k!='rows'} for s,r in reports.items()}),flush=True)
if __name__=='__main__':generate();train()
