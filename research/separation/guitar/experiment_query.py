import argparse,sys,time,json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'Mus3D/src'))
from source_embedding_generator.separator_io import load_separator
import query_conditioned_bslocoformer.bslocoformer as backbone
# Upstream forces CUDA FlashAttention, unavailable for these tensor strides on
# this Windows build. Keep identical attention mathematics with PyTorch's fallback.
backbone.sdpa_kernel=lambda *a,**k:torch.nn.attention.sdpa_kernel([
    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
    torch.nn.attention.SDPBackend.MATH])

ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--query',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args();torch.set_num_threads(4)
model=load_separator('D:/ChordStudio/data/models/mus3d/BS-Locoformer_query',device='cuda',use_ema=False)
x,sr=sf.read(args.input,dtype='float32',always_2d=True);q,qr=sf.read(args.query,dtype='float32',always_2d=True)
assert sr==qr==44100
q=np.tile(q,(int(np.ceil(441000/len(q))),1))[:441000]
query=torch.from_numpy(q.T.copy()).unsqueeze(0).cuda()
chunk=441000;hop=220500;n=len(x);pad=np.pad(x.T,((0,0),(hop,chunk)))
total=np.zeros_like(pad);count=np.zeros(pad.shape[-1]);window=np.maximum(np.hanning(chunk),1e-5)
t=time.perf_counter()
with torch.inference_mode():
 emb=model.query_adapter(model.query_model(query)).float()
 for pos in range(0,n+hop,hop):
  a=torch.from_numpy(pad[:,pos:pos+chunk].copy()).unsqueeze(0).cuda()
  with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(a,emb,query_is_embedding=True,level=torch.zeros(1,dtype=torch.long,device='cuda'))
  y=pred.squeeze(0).float().cpu().numpy();total[:,pos:pos+chunk]+=y*window;count[pos:pos+chunk]+=window
target=(total/count.clip(1e-8))[:,hop:hop+n].T
args.out.mkdir(parents=True,exist_ok=False)
sf.write(args.out/'target.wav',target,sr,subtype='FLOAT');sf.write(args.out/'residual.wav',x-target,sr,subtype='FLOAT')
meta=dict(source=str(args.input),query=str(args.query),seconds=time.perf_counter()-t,peak_vram_mb=torch.cuda.max_memory_allocated()/1024**2)
(args.out/'inference.json').write_text(json.dumps(meta,indent=2));print(meta)
