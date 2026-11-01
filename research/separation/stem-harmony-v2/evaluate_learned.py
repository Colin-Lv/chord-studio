"""Fixed real-audio regression after model selection; no fitting on Xuni."""
import json,sys
from pathlib import Path
import joblib,numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[2]))
from app.web.stem_learned import predict
from app.web.stem_harmony import LABELS,score_chords,pool
from app.web.fusion_engine import segments_from
from app.web.storage import Store,read_json,atomic_json
from evaluate import read_lab,score
MODEL=HERE/'training/candidate.joblib'
# Fixed prior to looking at this regression, not selected on the song.
LEARNED_WEIGHT=.5


def evaluate():
    model=joblib.load(MODEL);store=Store(Path('D:/ChordStudio/data/projects'))
    project=store.project('aa5a41797bb343e590abcf4d8b27ae54');run=store.run_dir(project['id'],project['current_run'])
    pred=read_json(run/'retained10/prediction.json');cells=read_json(Path(pred['cache_directory'])/'cells.json')
    bounds=np.array([[r['start'],r['end']] for r in cells])
    with np.load(run/'retained10/posterior.npz') as z:posterior=z['chord'];n=z['N'].reshape(-1)
    diag=read_json(HERE/'xuni/diagnostics.json')['segments']
    paths={k:[] for k in ['baseline','v2','learned_only','learned_fusion']};spans=[]
    for r in diag:
        a,b=r['start'],r['end'];spans.append((a,b));prior=pool(posterior,bounds,a,b)
        learned=predict(model,r['tones'],r['bass'],r['accompaniment_activity'],r['bass_activity'])
        fixed=score_chords(prior,r['baseline'],np.array(r['tones']),np.array(r['bass']),r['accompaniment_activity'],r['bass_activity'])
        labels=dict(baseline=r['baseline'],v2=r['label'],learned_only=LABELS[int(learned.argmax())],
                    learned_fusion=LABELS[int((fixed+LEARNED_WEIGHT*np.log(learned+1e-9)).argmax())])
        if pool(n,bounds,a,b)>=.5: labels={k:'N' for k in labels}
        for k in paths:paths[k].append(labels[k])
    truth=read_lab(HERE.parents[1]/'golden/reference_golden.lab')
    timelines={k:segments_from(labels,spans) for k,labels in paths.items()}
    result=dict(weight=LEARNED_WEIGHT,metrics={k:score(v,truth) for k,v in timelines.items()},timelines=timelines)
    atomic_json(HERE/'training/real_regression.json',result);print(result['metrics'],flush=True)


if __name__=='__main__':evaluate()
