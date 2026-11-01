"""Train from existing rendered audio, splitting all versions by original song."""
import os
os.environ['OMP_NUM_THREADS']='4'
import json,sys,time,hashlib
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from app.web.stem_harmony import extract_notes,pitch_profile,QUALITIES,score_chords,LABELS
from app.web.stem_learned import features,predict
from app.web.storage import atomic_json,signature
OUT=HERE/'training';OUT.mkdir(exist_ok=True)
BASE=Path('D:/ChordStudio/research/boundary-dataset')
EXT=Path('D:/ChordStudio/research/extension-dataset')


def prepare():
    data=OUT/'data.npz'
    if data.exists(): return dict(np.load(data,allow_pickle=False))
    samples=json.loads((BASE/'dataset.json').read_text(encoding='utf8'))['samples']
    eligible=[s for s in samples if s['texture']=='block' and (BASE/'stems'/s['sample_id']/'stem_COMP.wav').exists() and (BASE/'stems'/s['sample_id']/'stem_BASS.wav').exists()]
    songs=sorted({s['source_song'] for s in eligible});rng=np.random.default_rng(20261101);rng.shuffle(songs);songs=songs[:150]
    split={song:('train' if i<100 else 'validation' if i<125 else 'test') for i,song in enumerate(songs)}
    atomic_json(OUT/'split.json',split)
    selected=[(BASE,s,False) for s in eligible if s['source_song'] in split]
    exts=json.loads((EXT/'dataset.json').read_text(encoding='utf8'))['samples']
    selected += [(EXT,dict(s,source_song=s['sample_id'].split('-')[1]),True) for s in exts if '-block' in s['sample_id'] and s['sample_id'].split('-')[1] in split]
    rows=[];labels=[];groups=[];domains=[];identities=[]
    for i,(base,s,extension) in enumerate(selected):
        audio=base/s['audio'] if extension else BASE/'stems'/s['sample_id']/'stem_COMP.wav'
        bass_audio=audio if extension else BASE/'stems'/s['sample_id']/'stem_BASS.wav'
        comp,_=extract_notes(audio,OUT/'cache')
        bass=comp if extension else extract_notes(bass_audio,OUT/'cache')[0]
        lab=base/s['labels'];chords=json.loads(lab.read_text(encoding='utf8'))['chords']
        for c in chords:
            if c['quality'] not in QUALITIES:continue
            start=max(0,float(c['start']));end=min(float(c['end']),float(comp['duration']),float(bass['duration']))
            if end-start<.15:continue
            t,ta=pitch_profile(comp,start,end);b,ba=pitch_profile(bass,start,end,bass=True)
            rows.append(np.r_[t,b,ta,ba]);labels.append(int(c['root_pc'])*10+QUALITIES.index(c['quality']));groups.append(s['source_song']);domains.append('mixed_extension' if extension else 'rendered_stems')
        identities.append(dict(sample=s['sample_id'],audio=signature(audio)['sha256'],bass=signature(bass_audio)['sha256'],labels=signature(lab)['sha256']))
        if (i+1)%10==0:print('prepared',i+1,'/',len(selected),'chords',len(rows),flush=True)
    np.savez_compressed(data,x=np.array(rows),y=labels,song=groups,domain=domains)
    atomic_json(OUT/'inputs.json',identities)
    return dict(np.load(data,allow_pickle=False))


def main():
    d=prepare();split=json.loads((OUT/'split.json').read_text());x=d['x'];y=d['y'];groups=d['song']
    masks={k:np.array([split[s]==k for s in groups]) for k in ['train','validation','test']}
    f=np.stack([features(r[:12],r[12:24],r[24],r[25]) for r in x])
    target=np.zeros((len(y),12),dtype=int);target[np.arange(len(y)),y//10]=y%10+1
    results=[];best=None
    for leaf in [2,8,20]:
        model=ExtraTreesClassifier(n_estimators=160,min_samples_leaf=leaf,max_features=.8,class_weight='balanced',n_jobs=4,random_state=20261101)
        model.fit(f[masks['train']].reshape(-1,38),target[masks['train']].reshape(-1))
        model.n_jobs=1  # Small per-span inference avoids thread startup and sklearn/joblib warnings.
        ids=np.where(masks['validation'])[0];pred=[]
        for i in ids:pred.append(predict(model,x[i,:12],x[i,12:24],x[i,24],x[i,25]).argmax())
        acc=float(np.mean(np.array(pred)==y[ids]));results.append(dict(leaf=leaf,validation_exact=acc));print(results[-1],flush=True)
        if best is None or acc>best[0]:best=(acc,model,leaf)
    model=best[1];joblib.dump(model,OUT/'candidate.joblib')
    reports={}
    for part in ['validation','test']:
        ids=np.where(masks[part])[0];pred=np.array([predict(model,x[i,:12],x[i,12:24],x[i,24],x[i,25]).argmax() for i in ids])
        reports[part]=dict(chords=len(ids),songs=len(set(groups[ids])),exact=float(np.mean(pred==y[ids])),root=float(np.mean(pred//10==y[ids]//10)),
            by_quality={q:dict(n=int(np.sum(y[ids]%10==j)),exact=float(np.mean(pred[y[ids]%10==j]==y[ids][y[ids]%10==j]))) for j,q in enumerate(QUALITIES) if np.any(y[ids]%10==j)})
        rule=np.array([score_chords(np.ones(120)/120,'',x[i,:12],x[i,12:24],x[i,24],x[i,25]).argmax() for i in ids])
        reports[part]['same_evidence_rule_exact']=float(np.mean(rule==y[ids]))
        reports[part]['song_macro_exact']=float(np.mean([np.mean(pred[groups[ids]==s]==y[ids][groups[ids]==s]) for s in set(groups[ids])]))
    atomic_json(OUT/'metrics.json',dict(selection=results,selected_leaf=best[2],reports=reports,train_chords=int(masks['train'].sum()),
        data_sha256=signature(OUT/'data.npz')['sha256'],model_sha256=signature(OUT/'candidate.joblib')['sha256'],
        limitation='Synthetic stems and mixed extension renders, not separator outputs; true boundaries used. Not end-to-end or blind real-song accuracy.'))
    print(reports,flush=True)


if __name__=='__main__': main()
