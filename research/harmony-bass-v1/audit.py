"""Read-only diagnosis of the reported passage; no assumed reference labels."""
import json, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from app.web.storage import read_json,atomic_json
from app.web.full_decode import NAMES,QUALITIES,span_bass,tone_support,span_tone,pooled_label,PARAMS,tone_log_score
from app.web.stem_harmony import pool,extract_notes,pitch_profile,build_views
PROJECT=Path('D:/ChordStudio/data/projects/c1ebfe4bb5924849b320be11b0aa1f9e')
LABELS=[f'{r}:{q}' for r in NAMES for q in QUALITIES]

def load():
    p=read_json(PROJECT/'project.json');run=PROJECT/'runs'/p['current_run']
    pred=read_json(run/'retained10/prediction.json');cache=Path(pred['cache_directory'])
    cells=read_json(cache/'cells.json');bounds=np.array([[c['start'],c['end']] for c in cells])
    notes=dict(np.load(cache/'notes.npz'));post=np.load(run/'retained10/posterior.npz')['chord']
    return p,run,pred,cache,cells,bounds,notes,post

def top(x,names,k=4):return {names[i]:round(float(x[i]),4) for i in np.argsort(x)[-k:][::-1]}

def main():
    p,run,pred,cache,cells,bounds,notes,post=load()
    analysis=read_json(run/'analysis.json');print('run',run,'cache',cache,'default',analysis['default_source'],flush=True)
    print('sources',[(s['id'],len(s['rows'])) for s in analysis['sources']],flush=True)
    print('explicit slash cells',sum('/' in c['chord'] for c in cells),'/',len(cells),flush=True)
    base=Path(p['stems']['bs_roformer_sw']['manifest']).parent
    views,ids,groups=build_views(base,read_json(base/'manifest.json'),analysis['duration'],HERE/'views')
    evidence={}
    for role,audio in views.items():
        print('extract',role,flush=True);evidence[role],_=extract_notes(audio,HERE/'cache')
    support=tone_support(notes['activations']);nt=notes['times']
    bass_cells=np.array([span_bass(notes['activations'],nt,a,b) for a,b in bounds])
    times=[17.81,20.81,23.83,26.82,29.81,32.83,35.82,38.81,41.81]
    rows=[]
    for a,b in zip(times,times[1:]):
        ix=np.where((bounds[:,0]<b)&(bounds[:,1]>a))[0]
        pri=pool(post,bounds,a,b);bb=pool(bass_cells,bounds,a,b);tone=span_tone(support,nt,a,b)
        row=dict(start=a,end=b,posterior=top(pri,LABELS),mix_bass=top(bb,NAMES),mix_tone=top(tone,NAMES,7))
        row['relabel_ablation']={str((lam,mu)):pooled_label(post,bass_cells,tone,ix,lam,mu,PARAMS['tau']) for lam,mu in [(1.5,.7),(0,.7),(1.5,0),(0,0)]}
        for role,e in evidence.items():
            v,active=pitch_profile(e,a,b,bass=role=='bass');row[role]=top(v,NAMES,7);row[role+'_activity']=active
        row['source_labels']={s['id']:[(r['start'],r['end'],r['label']) for r in s['rows'] if r['start']<b and r['end']>a] for s in analysis['sources']}
        rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    atomic_json(HERE/'audit.json',dict(run=str(run),cache=str(cache),rows=rows,inputs=ids,groups=groups))

if __name__=='__main__':main()
