"""Development regression, separate from synthetic held-out evaluation."""
import sys,json
from pathlib import Path
import numpy as np,soundfile as sf
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from app.web.bass_voice import spectrum,summarize_beats,load_model,attach_bass,NAMES
from app.web.stem_bass_harmony import refine_label
from app.web.full_decode import pooled_label,tone_support,span_tone
from app.web.storage import read_json,atomic_json
from app.web.stem_harmony import pool,build_views
from app.web.fusion_engine import segments_from
sys.path.insert(0,str(HERE.parent/'separation/stem-harmony-v2'))
from evaluate import score,read_lab

def main():
    model=load_model(HERE/'training-register-v2/bass_voice_v2.joblib');all_results={}
    for pid in ['c1ebfe4bb5924849b320be11b0aa1f9e','aa5a41797bb343e590abcf4d8b27ae54']:
        base=Path('D:/ChordStudio/data/projects')/pid;p=read_json(base/'project.json');run=base/'runs'/p['current_run']
        analysis=read_json(run/'analysis.json');pred=read_json(run/'retained10/prediction.json');cache=Path(pred['cache_directory'])
        cells=read_json(cache/'cells.json');cb=np.array([[r['start'],r['end']] for r in cells]);post=np.load(run/'retained10/posterior.npz')['chord']
        nprob=np.load(run/'retained10/posterior.npz')['N'].reshape(-1)
        notes=np.load(cache/'notes.npz');support=tone_support(notes['activations']);beats=read_json(run/'beats.json')
        original=next(s['rows'] for s in analysis['sources'] if s['id']=='retained10_full')
        manifest=Path(p['stems']['bs_roformer_sw']['manifest'])
        views,_,_=build_views(manifest.parent,read_json(manifest),analysis['duration'],HERE/pid/'views')
        # Include real bass when present, preserve relative source levels.
        y,sr=sf.read(views['accompaniment']);
        if 'bass' in views:y+=sf.read(views['bass'])[0]
        evidence=spectrum(y,sr);bounds=sorted({0.,analysis['duration'],*[r[k] for r in original for k in ('start','end')],*[r['time'] for r in beats if r['beat']==1]})
        labels=[];spans=[];details=[]
        for a,b in zip(bounds,bounds[1:]):
            if b-a<1e-8:continue
            old=next((r['label'] for r in original if r['start']<=(a+b)/2<r['end']),'N')
            bass=summarize_beats(model,evidence,a,b,[r['time'] for r in beats]);label=old
            if bass['note'] and old!='N':
                label=refine_label(pool(post,cb,a,b),old,span_tone(support,notes['times'],a,b),bass['note'])
            if pool(nprob,cb,a,b)>=.5:label='N'
            labels.append(label);spans.append((a,b));details.append(dict(start=a,end=b,original=old,label=label,bass=bass))
        rows=segments_from(labels,spans);result=dict(rows=rows,details=details)
        if pid.startswith('aa5'):
            # Reference lacks complete bass labels: compare core harmony separately.
            core=segments_from([r['label'].split('/')[0] for r in rows],[(r['start'],r['end']) for r in rows])
            result['core_metrics']=score(core,read_lab(HERE.parent/'golden/reference_golden.lab'))
            print('xuni',result['core_metrics'],flush=True)
        else:
            print('heart',[(r['start'],r['end'],r['original'],r['label'],r['bass']['note']) for r in details if 17<r['start']<42],flush=True)
        atomic_json(HERE/pid/'candidate.json',result);all_results[pid]=result
    atomic_json(HERE/'candidate_results.json',all_results)
if __name__=='__main__':main()
