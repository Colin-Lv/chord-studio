import sys,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]))
from app.web import full_decode as fd
from app.web.storage import read_json,atomic_json
from app.web.fusion_engine import full_decode_segments
sys.path.insert(0,str(HERE.parent/'separation/stem-harmony-v2'))
from evaluate import read_lab,score

def main():
    result={}
    original=fd.PARAMS.copy()
    for pid in ['c1ebfe4bb5924849b320be11b0aa1f9e','aa5a41797bb343e590abcf4d8b27ae54']:
        base=Path('D:/ChordStudio/data/projects')/pid;p=read_json(base/'project.json');run=base/'runs'/p['current_run']
        post=np.load(run/'retained10/posterior.npz')['chord'];out={}
        for lam in [1.5,0]:
            fd.PARAMS['lambda_bass']=lam;rows=full_decode_segments(run,post)
            out[str(lam)]=dict(rows=rows)
            if pid.startswith('aa5'):
                out[str(lam)]['metrics']=score(rows,read_lab(HERE.parent/'golden/reference_golden.lab'))
            print(pid,lam,out[str(lam)].get('metrics',[(r['start'],r['end'],r['label']) for r in rows if r['start']<42 and r['end']>17]),flush=True)
        result[pid]=out
    fd.PARAMS.update(original);atomic_json(HERE/'decoder_ablation.json',result)
if __name__=='__main__':main()
