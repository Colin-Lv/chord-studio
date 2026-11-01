"""Audit the old negative result with overlap alignment, keeping its decoder fixed."""
import json,sys
from pathlib import Path
import numpy as np
here=Path(__file__).resolve().parent;sys.path.insert(0,str(here.parents[2]))
from app.web.storage import Store,read_json,atomic_json
from app.web.stem_harmony import pool
from app.web.fusion_engine import full_decode_segments
from evaluate import read_lab,score
s=Store(Path('D:/ChordStudio/data/projects'));p=s.project('aa5a41797bb343e590abcf4d8b27ae54');r=s.run_dir(p['id'],p['current_run'])
h=Path('D:/ChordStudio/data/stems/d4867eec9bec73fb4dbf0575cff26e3e6550cdba9553c8f886b3a4adab6ef72f/bs_roformer_sw/fusion/retained10')
def load(d):
    pred=read_json(d/'prediction.json');cells=read_json(Path(pred['cache_directory'])/'cells.json')
    with np.load(d/'posterior.npz') as z:prob=z['chord']
    return prob,np.array([[c['start'],c['end']] for c in cells])
a,ab=load(r/'retained10');b,bb=load(h)
aligned=np.array([pool(b,bb,start,end) for start,end in ab])
truth=read_lab(here.parents[1]/'golden/reference_golden.lab')
result=[]
for lam in [0,.25,.65]:
    logits=(1-lam)*np.log(np.maximum(a,1e-9))+lam*np.log(np.maximum(aligned,1e-9))
    prob=np.exp(logits-logits.max(axis=1,keepdims=True));prob/=prob.sum(axis=1,keepdims=True)
    rows=full_decode_segments(r,prob)
    result.append(dict(lambda_=lam,**score(rows,truth)))
atomic_json(here/'aligned_legacy_scores.json',result)
print(result)
