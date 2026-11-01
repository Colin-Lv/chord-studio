"""Second candidate: feature-corruption training, still song-disjoint."""
import json,sys
from pathlib import Path
import numpy as np,joblib
from sklearn.ensemble import ExtraTreesClassifier
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[2]))
from app.web.stem_learned import features
from app.web.stem_harmony import INTERVALS,QUALITIES
from app.web.storage import atomic_json,signature
OUT=HERE/'training'


def corrupt(x,y,seed):
    rng=np.random.default_rng(seed);out=x.copy()
    for i,row in enumerate(out):
        t=row[:12]*rng.lognormal(0,.8,12)*(rng.random(12)>.15)
        t+=rng.dirichlet(np.ones(12)*.25)*rng.uniform(0,.5)
        row[:12]=t/max(t.sum(),1e-9)
        b=row[12:24].copy()
        if rng.random()<.3:
            b[:]=0;b[(y[i]//10+rng.choice(INTERVALS[QUALITIES[y[i]%10]]))%12]=1
        b+=rng.dirichlet(np.ones(12)*.3)*rng.uniform(0,.5)
        row[12:24]=b/max(b.sum(),1e-9)
    return out


def batch(model,x):
    f=np.stack([features(r[:12],r[12:24],r[24],r[25]) for r in x])
    raw=model.predict_proba(f.reshape(-1,38)).reshape(len(x),12,-1)
    probs=np.zeros((len(x),12,10))
    for j,c in enumerate(model.classes_):
        if c:probs[:,:,int(c)-1]=raw[:,:,j]
    return probs.reshape(len(x),120).argmax(1)


def main():
    d=dict(np.load(OUT/'data.npz'));x=d['x'];y=d['y'];split=json.load(open(OUT/'split.json'))
    masks={k:np.array([split[s]==k for s in d['song']]) for k in ['train','validation','test']}
    tx=x[masks['train']];ty=y[masks['train']]
    tx=np.concatenate([tx]+[corrupt(tx,ty,100+i) for i in range(5)]);ty=np.tile(ty,6)
    f=np.stack([features(r[:12],r[12:24],r[24],r[25]) for r in tx])
    target=np.zeros((len(ty),12),int);target[np.arange(len(ty)),ty//10]=ty%10+1
    best=None;selection=[]
    for leaf in [8,32]:
        model=ExtraTreesClassifier(n_estimators=160,min_samples_leaf=leaf,max_features=.8,class_weight='balanced',n_jobs=4,random_state=20261101)
        model.fit(f.reshape(-1,38),target.reshape(-1));model.n_jobs=1
        vx=x[masks['validation']];vy=y[masks['validation']]
        clean=np.mean(batch(model,vx)==vy);noisy=np.mean(batch(model,corrupt(vx,vy,999))==vy)
        metric=(clean+noisy)/2;selection.append(dict(leaf=leaf,clean=float(clean),corrupted=float(noisy)))
        print(selection[-1],flush=True)
        if best is None or metric>best[0]:best=(metric,model)
    model=best[1];joblib.dump(model,OUT/'candidate.joblib');reports={}
    for part in ['validation','test']:
        vx=x[masks[part]];vy=y[masks[part]]
        reports[part]=dict(n=len(vy),clean=float(np.mean(batch(model,vx)==vy)),corrupted=float(np.mean(batch(model,corrupt(vx,vy,999))==vy)))
    atomic_json(OUT/'robust_metrics.json',dict(selection=selection,reports=reports,model_sha256=signature(OUT/'candidate.joblib')['sha256'],
        limitations='Feature perturbations are not real separator errors. Test songs exposed in first round; second round is development evidence.'))
    print(reports,flush=True)


if __name__=='__main__':main()
