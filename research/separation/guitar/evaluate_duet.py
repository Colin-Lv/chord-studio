import sys,json,time,argparse
from pathlib import Path
import numpy as np,soundfile as sf,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'app/web'))
from guitar_engine import load_model,separate
from evaluate_candidates import scores
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(4)
ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default='D:/ChordStudio/data/models/guitar/duet_adapt/best.ckpt');ap.add_argument('--name',default='duet_adapt');ap.add_argument('--fixed-role',action='store_true');args=ap.parse_args()
model,cfg,device=load_model('lead')
model.load_state_dict(torch.load(args.checkpoint,map_location=device,weights_only=True),strict=True)
rows=[]
for case in json.loads((ROOT/'benchmark/protocol.json').read_text())['cases']:
    folder=Path(case['folder']);x,sr=sf.read(folder/'mix.wav',dtype='float32');a,_=sf.read(folder/'source1.wav',dtype='float32');b,_=sf.read(folder/'source2.wav',dtype='float32')
    t=time.perf_counter();swaps=[]
    y=separate(model,x,cfg['audio']['chunk_size'],device,align_sources=not args.fixed_role,progress=lambda f,s:swaps.append(s))
    out=ROOT/'benchmark_outputs'/case['id']/args.name;out.mkdir(parents=True,exist_ok=True)
    sf.write(out/'target.wav',y,sr,subtype='FLOAT');sf.write(out/'residual.wav',x-y,sr,subtype='FLOAT')
    row=dict(case=case['id'],seconds=time.perf_counter()-t,swaps=max(swaps),**scores(y,x,a,b));rows.append(row)
    (out/'scores.json').write_text(json.dumps(row,indent=2));print(case['id'],row['si_sdr'],flush=True)
(ROOT/(args.name+'_scores.json')).write_text(json.dumps(rows,indent=2))
