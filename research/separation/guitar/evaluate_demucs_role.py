"""Independent community HTDemucs role candidate, strict state loading."""
import json,time
from pathlib import Path
import numpy as np,soundfile as sf,torch,yaml
from demucs.htdemucs import HTDemucs
from demucs.apply import apply_model
from evaluate_candidates import scores
ROOT=Path(__file__).resolve().parent;MODELS=Path('D:/ChordStudio/data/models/guitar');KEY='demucs4_lead_rhythm_guitar_drypaint'
torch.set_num_threads(4)
cfg=yaml.safe_load((MODELS/(KEY+'_config.yaml')).read_text())
params=cfg['htdemucs'].copy();assert params.pop('num_subbands')==1
params['dconv_init']=float(params['dconv_init'])
model=HTDemucs(sources=cfg['training']['instruments'],samplerate=44100,segment=3,**params)
model.load_state_dict(torch.load(MODELS/(KEY+'.ckpt'),map_location='cpu',weights_only=True),strict=True)
model.eval().requires_grad_(False).cuda();rows=[]
for case in json.loads((ROOT/'benchmark/protocol.json').read_text())['cases']:
    folder=Path(case['folder']);x,sr=sf.read(folder/'mix.wav',dtype='float32');a,_=sf.read(folder/'source1.wav',dtype='float32');b,_=sf.read(folder/'source2.wav',dtype='float32')
    t=time.perf_counter();mix=torch.from_numpy(x.T.copy());ref=mix.mean(0);mean=ref.mean();std=ref.std().clamp_min(1e-8)
    with torch.inference_mode():
        out=apply_model(model,((mix-mean)/std).unsqueeze(0),shifts=0,split=True,overlap=.5,device='cuda',segment=3)[0].cpu()*std+mean
    y=out[0].numpy().T.copy();folder=ROOT/'benchmark_outputs'/case['id']/'demucs_role';folder.mkdir(parents=True,exist_ok=True)
    sf.write(folder/'target.wav',y,sr,subtype='FLOAT');sf.write(folder/'residual.wav',x-y,sr,subtype='FLOAT');sf.write(folder/'native_rhythm.wav',out[1].numpy().T,sr,subtype='FLOAT')
    row=dict(case=case['id'],seconds=time.perf_counter()-t,**scores(y,x,a,b));rows.append(row)
    (folder/'scores.json').write_text(json.dumps(row,indent=2));print(case['id'],row['si_sdr'],flush=True)
(ROOT/'demucs_role_scores.json').write_text(json.dumps(rows,indent=2))
