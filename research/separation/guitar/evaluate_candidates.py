import argparse,json,gc,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from experiment_models import load_model,separate
ROOT=Path(__file__).resolve().parent
def sisdr(e,t):
    e=e.reshape(-1).astype('float64');t=t.reshape(-1).astype('float64')
    e-=e.mean();t-=t.mean()
    if np.dot(t,t)<1e-12:return None
    s=t*np.dot(t,e)/(np.dot(t,t)+1e-20)
    return float(10*np.log10((np.dot(s,s)+1e-20)/(np.dot(e-s,e-s)+1e-20)))
def snr(e,t):
    e=e.astype('float64');t=t.astype('float64')
    if np.sum(t*t)<1e-12:return None
    return float(10*np.log10((np.sum(t*t)+1e-20)/(np.sum((t-e)**2)+1e-20)))
def scores(y,x,a,b):
    rem=x-y
    matrix=[[sisdr(y,a),sisdr(y,b)],[sisdr(rem,a),sisdr(rem,b)]]
    return dict(si_sdr=matrix,snr=[[snr(y,a),snr(y,b)],[snr(rem,a),snr(rem,b)]],
        mixture_baseline=[sisdr(x,a),sisdr(x,b)],
        energy_fraction=[float(np.sum(y*y)/(np.sum(x*x)+1e-20)),float(np.sum(rem*rem)/(np.sum(x*x)+1e-20))],
        sum_error=float(np.max(np.abs(x-y-rem))))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=['acoustic','electric','lead']);args=ap.parse_args()
    torch.set_num_threads(4)
    cases=json.loads((ROOT/'benchmark/protocol.json').read_text(encoding='utf-8'))['cases']
    rows=[]
    for key in args.models:
        model,cfg,device=load_model(key)
        for c in cases:
            source=Path(c['folder']);out=ROOT/'benchmark_outputs'/c['id']/key;out.mkdir(parents=True,exist_ok=True)
            x,sr=sf.read(source/'mix.wav',dtype='float32');a,_=sf.read(source/'source1.wav',dtype='float32');b,_=sf.read(source/'source2.wav',dtype='float32')
            t=time.perf_counter()
            if (out/'target.wav').exists():y,_=sf.read(out/'target.wav',dtype='float32')
            else:
                y=separate(model,x,cfg['audio']['chunk_size'],device)
                sf.write(out/'target.wav',y,sr,subtype='FLOAT');sf.write(out/'residual.wav',x-y,sr,subtype='FLOAT')
            row=dict(case=c['id'],kind=c['kind'],model=key,seconds=time.perf_counter()-t,**scores(y,x,a,b))
            rows.append(row);(out/'scores.json').write_text(json.dumps(row,indent=2))
            print(c['id'],key,row['si_sdr'],flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    (ROOT/'candidate_scores.json').write_text(json.dumps(rows,indent=2))
