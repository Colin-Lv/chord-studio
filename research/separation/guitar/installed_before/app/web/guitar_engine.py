"""Guitar separator kernels: strict checkpoints, common gain and aligned overlap-add."""
from pathlib import Path
import numpy as np
import torch
import yaml

MODELS={
 'lead':('mbr_lead_rhythm_guitar_listra92','mel'),
 'acoustic':('bs_mega_53stem_acoustic-guitar_mvsep','bs'),
 'electric':('bs_mega_53stem_electric-guitar_mvsep','bs'),
}
class ConfigLoader(yaml.SafeLoader):pass
ConfigLoader.add_constructor('tag:yaml.org,2002:python/tuple',lambda loader,node:tuple(loader.construct_sequence(node)))

def load_model(key,folder=Path('D:/ChordStudio/data/models/guitar'),checkpoint=None):
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
    sd=torch.load(checkpoint or folder/(name+'.ckpt'),map_location='cpu',weights_only=True)
    model.load_state_dict(sd,strict=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    return model.eval().requires_grad_(False).to(device),cfg,device

def separate(model,x,chunk,device,align_sources=False,progress=None):
    length=len(x);hop=chunk//2
    padded=np.pad(x.T,((0,0),(hop,chunk)))
    result=np.zeros_like(padded);count=np.zeros(padded.shape[-1],dtype='float32')
    window=np.maximum(np.hanning(chunk),1e-5).astype('float32')
    scale=max(1.,float(np.abs(x).max()))
    swaps=0
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
            estimate*=scale
            if align_sources and pos>0:
                # Compare both assignments in the shared overlap. This fixes
                # chunk permutation; it cannot identify players after silence.
                valid=count[pos:pos+hop]>1e-4
                if valid.any():
                    anchor=result[:,pos:pos+hop][:,valid]/count[pos:pos+hop][valid]
                    candidate=estimate[:,:hop][:,valid]
                    complement=padded[:,pos:pos+hop][:,valid]-candidate
                    if np.mean((complement-anchor)**2)<np.mean((candidate-anchor)**2):
                        estimate=padded[:,pos:pos+chunk]-estimate;swaps+=1
            result[:,pos:pos+chunk]+=estimate*window
            count[pos:pos+chunk]+=window
            if progress:progress(min(1.,(pos+hop)/max(length,1)),swaps)
    return (result/count.clip(1e-8))[:,hop:hop+length].T.copy()

