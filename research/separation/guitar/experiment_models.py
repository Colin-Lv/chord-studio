"""Strict local candidate inference. Outputs preserve input gain and stereo timing."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import yaml

MODELS={
 'lead':('mbr_lead_rhythm_guitar_listra92','mel'),
 'acoustic':('bs_mega_53stem_acoustic-guitar_mvsep','bs'),
 'electric':('bs_mega_53stem_electric-guitar_mvsep','bs'),
}
class ConfigLoader(yaml.SafeLoader):pass
ConfigLoader.add_constructor('tag:yaml.org,2002:python/tuple',lambda loader,node:tuple(loader.construct_sequence(node)))

def load_model(key,folder=Path('D:/ChordStudio/data/models/guitar')):
    from audio_separator.separator.uvr_lib_v5.roformer.mel_band_roformer import MelBandRoformer
    from audio_separator.separator.uvr_lib_v5.roformer.bs_roformer import BSRoformer
    name,kind=MODELS[key]
    cfg=yaml.load((folder/(name+'_config.yaml')).read_text(),Loader=ConfigLoader)
    model=(MelBandRoformer if kind=='mel' else BSRoformer)(**cfg['model'])
    if kind=='mel':
        # audio-separator 0.48 accepts this setting but omits it when creating
        # its mask heads. Honour the checkpoint config, without changing the
        # installed module or loading any parameters non-strictly.
        from audio_separator.separator.uvr_lib_v5.roformer.mel_band_roformer import MaskEstimator
        model.mask_estimators=torch.nn.ModuleList([
            MaskEstimator(dim=cfg['model']['dim'],dim_inputs=head.dim_inputs,
                depth=cfg['model']['mask_estimator_depth'],
                mlp_expansion_factor=cfg['model'].get('mlp_expansion_factor',4))
            for head in model.mask_estimators])
    sd=torch.load(folder/(name+'.ckpt'),map_location='cpu',weights_only=True)
    model.load_state_dict(sd,strict=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    return model.eval().requires_grad_(False).to(device),cfg,device

def separate(model,x,chunk,device):
    length=len(x);hop=chunk//2
    padded=np.pad(x.T,((0,0),(hop,chunk)))
    result=np.zeros_like(padded);count=np.zeros(padded.shape[-1],dtype='float32')
    window=np.maximum(np.hanning(chunk),1e-5).astype('float32')
    scale=max(1.,float(np.abs(x).max()))
    with torch.inference_mode():
        for pos in range(0,hop+length,hop):
            part=torch.from_numpy(padded[:,pos:pos+chunk].copy()/scale).unsqueeze(0).to(device)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
                estimate=model(part)
            estimate=estimate.squeeze(0).float().cpu().numpy()
            if estimate.ndim==3:estimate=estimate[0]
            # BSRoFormer ISTFT rounds to a hop boundary; pad only its tail.
            if estimate.shape[-1]>chunk:raise ValueError('Model output too long')
            if estimate.shape[-1]<chunk:estimate=np.pad(estimate,((0,0),(0,chunk-estimate.shape[-1])))
            if not np.isfinite(estimate).all():raise ValueError('Nonfinite output')
            result[:,pos:pos+chunk]+=estimate*window*scale
            count[pos:pos+chunk]+=window
    return (result/count.clip(1e-8))[:,hop:hop+length].T.copy()

def main(args):
    torch.set_num_threads(4)
    model,cfg,device=load_model(args.model)
    for source in args.inputs:
        x,sr=sf.read(source,dtype='float32',always_2d=True)
        if x.shape[1]==1:x=np.repeat(x,2,axis=1)
        assert sr==44100 and x.shape[1]==2
        output=args.out/source.stem/args.model;output.mkdir(parents=True,exist_ok=False)
        start=time.perf_counter()
        est=separate(model,x,args.chunk or cfg['audio']['chunk_size'],device)
        sf.write(output/'target.wav',est,sr,subtype='FLOAT')
        sf.write(output/'residual.wav',x-est,sr,subtype='FLOAT')
        meta=dict(source=str(source),model=args.model,seconds=time.perf_counter()-start,
            chunk=args.chunk or cfg['audio']['chunk_size'],device=device,
            peak_vram_mb=torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else 0,
            input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),strict_load=True,
            relative_energy_db=float(10*np.log10((np.square(est).sum()+1e-20)/(np.square(x).sum()+1e-20))))
        (output/'inference.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        print(json.dumps(meta),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS,required=True)
    ap.add_argument('--inputs',type=Path,nargs='+',required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--chunk',type=int);main(ap.parse_args())
