"""Conservative local Banquet BYO-query runner, batch=1, float WAV.

Uses upstream architecture and strict checkpoint loading. Query identity does not
guarantee performer identity; residual is explicitly algebraic.
"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml


def read(path):
    x,sr=sf.read(path,dtype='float32',always_2d=True)
    if x.shape[1]==1:x=np.repeat(x,2,axis=1)
    if x.shape[1]!=2 or not len(x) or not np.isfinite(x).all():raise ValueError('Invalid audio')
    tensor=torch.from_numpy(x.T.copy())
    return torchaudio.functional.resample(tensor,sr,44100) if sr!=44100 else tensor


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.repo.resolve()))
    # The checkpoint already contains PaSST. Avoid downloading a second checkpoint.
    import hear21passt.base as hb
    hb.get_model_passt=partial(hb.get_model_passt,pretrained=False)
    from core.models.e2e.bandit.bandit import PasstFiLMConditionedBandit
    from core.types import SimpleishNamespace
    torch.set_num_threads(4)
    config=yaml.safe_load((args.repo/'config/models/bandit-query.yml').read_text())
    model=PasstFiLMConditionedBandit(**config['kwargs'])
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    states=ckpt['state_dict']
    weights={k[len('model.'):]:v for k,v in states.items() if k.startswith('model.')}
    model.load_state_dict(weights,strict=True)
    del ckpt,states,weights
    device='cuda' if torch.cuda.is_available() else 'cpu';model=model.eval().to(device)
    mix=read(args.input);query=read(args.query)
    qlen=441000
    query=query.repeat(1,(qlen+query.shape[-1]-1)//query.shape[-1])[:,:qlen].unsqueeze(0).to(device)
    n=mix.shape[-1];chunk=6*44100;hop=3*44100
    if args.official:
        from types import SimpleNamespace
        from core.models.ebase import EndToEndLightningSystem
        # Upstream reflect-padding needs a long enough input. Pad short pilots
        # with silence to 40 s; retain/score only the original sample coordinates.
        extended=torch.nn.functional.pad(mix,(0,max(0,40*44100-n))).unsqueeze(0).to(device)
        wrapper=SimpleNamespace(inference=SimpleNamespace(fs=44100,chunk_size_seconds=6.0,
            hop_size_seconds=.5,batch_size=1),device=torch.device(device),forward=model)
        batch=dict(mixture={'audio':extended},query={'audio':query},metadata={'stem':['target']},estimates={})
        start=time.perf_counter()
        result=EndToEndLightningSystem.chunked_inference(wrapper,batch)
        target=result.estimates['target']['audio'].squeeze(0).cpu()[:,:n]
        sf.write(args.output/'target.wav',target.T.numpy(),44100,subtype='FLOAT')
        sf.write(args.output/'residual.wav',(mix-target).T.numpy(),44100,subtype='FLOAT')
        sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        meta=dict(official_chunked_inference=True,chunk_seconds=6,hop_seconds=.5,batch_size=1,
            right_silence_padding_samples=max(0,40*44100-n),seconds=time.perf_counter()-start,
            samples=n,sample_rate=44100,device=device,strict_load=True,
            checkpoint_sha256=sha(args.checkpoint),input_sha256=sha(args.input),query_sha256=sha(args.query),
            runner_sha256=sha(__file__),peak_vram_mb=torch.cuda.max_memory_allocated()/1024**2)
        (args.output/'inference.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        return
    # Half-chunk exterior padding keeps all saved samples away from zero weights.
    padded=torch.nn.functional.pad(mix,(hop,chunk));total=torch.zeros_like(padded)
    count=torch.zeros(padded.shape[-1]);window=torch.hann_window(chunk).clamp_min(1e-5)
    start=time.perf_counter()
    with torch.inference_mode():
        for pos in range(0,hop+n,hop):
            part=padded[:,pos:pos+chunk].unsqueeze(0).to(device)
            batch=SimpleishNamespace(mixture={'audio':part},query={'audio':query},estimates={})
            estimate=model(batch).estimates['target'].audio.squeeze(0).cpu()
            if estimate.shape!=part.squeeze(0).shape:raise ValueError('Model length mismatch')
            total[:,pos:pos+chunk]+=estimate*window
            count[pos:pos+chunk]+=window
            print('seconds',min(pos/44100,n/44100),'/',n/44100,flush=True)
    target=(total/count.clamp_min(1e-8))[:,hop:hop+n]
    if not torch.isfinite(target).all():raise ValueError('Non-finite model output')
    sf.write(args.output/'target.wav',target.T.numpy(),44100,subtype='FLOAT')
    sf.write(args.output/'residual.wav',(mix-target).T.numpy(),44100,subtype='FLOAT')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    meta=dict(repo_commit=subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'],text=True).strip(),
              checkpoint_sha256=sha(args.checkpoint),input_sha256=sha(args.input),query_sha256=sha(args.query),
              runner_sha256=sha(__file__),seconds=time.perf_counter()-start,device=device,
              peak_vram_mb=torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else 0,
              sample_rate=44100,samples=n,chunk_seconds=6,hop_seconds=3,
              strict_load=True,note='Target estimate and mixture-minus-target; not independent sources or a verified track identity')
    (args.output/'inference.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for key in ['input','query','output']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--official',action='store_true')
    ap.add_argument('--repo',type=Path,default=Path(__file__).parent/'query-bandit')
    ap.add_argument('--checkpoint',type=Path,default=Path('D:/ChordStudio/data/models/banquet/ev-pre-aug.verified.ckpt'))
    main(ap.parse_args())
