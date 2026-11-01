"""Second-level guitar separation, float PCM, immutable outputs."""
import argparse,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import soundfile as sf

def main(source,output,mode,model_dir):
    import torch
    from guitar_engine import load_model,separate
    torch.set_num_threads(4)
    parent,sr=sf.read(source,dtype='float32',always_2d=True)
    if sr!=44100 or parent.shape[1]!=2 or not len(parent) or not np.isfinite(parent).all():
        raise ValueError('Expected finite nonempty stereo 44100 Hz guitar stem')
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();swaps=0
    def progress(fraction,count):
        nonlocal swaps
        swaps=count
        print(json.dumps(dict(progress=round(fraction,3),overlap_swaps=count)),flush=True)
    estimates={};used=[]
    if mode=='timbre':
        for key in ['acoustic','electric']:
            model,cfg,device=load_model(key,model_dir)
            print(key,flush=True)
            estimates[key+'_guitar']=separate(model,parent,cfg['audio']['chunk_size'],device,progress=progress)
            del model;gc.collect()
            if device=='cuda':torch.cuda.empty_cache()
        estimates['guitar_residual']=parent-estimates['acoustic_guitar']-estimates['electric_guitar']
        used=['bs_mega_53stem_'+key+'-guitar_mvsep'+suffix for key in ['acoustic','electric'] for suffix in ['.ckpt','_config.yaml']]
    elif mode=='duet':
        model,cfg,device=load_model('lead',model_dir,checkpoint=model_dir/'guitar_role_v1.ckpt')
        # Fixed role training: output is melody, complement is accompaniment.
        # Permutation alignment would incorrectly change the semantic labels.
        estimates['guitar_one']=separate(model,parent,cfg['audio']['chunk_size'],device,align_sources=False,progress=progress)
        estimates['guitar_two']=parent-estimates['guitar_one']
        used=['guitar_role_v1.ckpt','mbr_lead_rhythm_guitar_listra92_config.yaml']
    else:raise ValueError('Unknown guitar split mode')
    for key,audio in estimates.items():
        if audio.shape!=parent.shape or not np.isfinite(audio).all():raise ValueError('Invalid guitar estimate')
        sf.write(output/(key+'.wav'),audio,sr,subtype='FLOAT')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    meta=dict(mode=mode,source_sha256=sha(source),model_files={n:sha(model_dir/n) for n in used},
        runner_sha256=sha(__file__),engine_sha256=sha(Path(__file__).with_name('guitar_engine.py')),
        samples=len(parent),sample_rate=sr,channels=2,device=device,torch=torch.__version__,
        seconds=time.perf_counter()-started,overlap_swaps=swaps,
        peak_vram_mb=torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else 0,
        sum_max_abs_error=float(np.max(np.abs(parent-sum(estimates.values())))),
        notes='Timbre outputs are independent estimates plus algebraic residual. Role outputs estimate melody and accompaniment, not verified performer identities.')
    (output/'inference.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--outdir',type=Path,required=True)
    ap.add_argument('--mode',choices=['timbre','duet'],required=True)
    ap.add_argument('--model-dir',type=Path,default=Path('D:/ChordStudio/data/models/guitar'))
    a=ap.parse_args();main(a.input,a.outdir,a.mode,a.model_dir)
