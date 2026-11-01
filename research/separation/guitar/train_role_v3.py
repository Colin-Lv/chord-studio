"""Fixed-role adaptation using real GuitarSet performance labels; player-isolated split. Model output is SOLO, complement COMP. Includes one-source negatives."""
import argparse, json, random, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from experiment_models import load_model
from scipy.signal import resample_poly
ROOT=Path(__file__).resolve().parent;DATA=Path('D:/ChordStudio/data/research/guitar')
OUT=Path('D:/ChordStudio/data/models/guitar/role_adapt_v3');OUT.mkdir(exist_ok=True)
def read(p):
 x,sr=sf.read(p,dtype='float32',always_2d=True)
 if sr!=44100:
  import math
  d=math.gcd(sr,44100);x=resample_poly(x,44100//d,sr//d,axis=0)
 if x.shape[1]==1:x=np.repeat(x,2,axis=1)
 return torch.from_numpy(x.T.copy())
def pairs():
 result={'train':[],'valid':[]}
 for player in ['00','01','02','04','03']:
  for song in ['Rock2-142-D','Jazz1-200-B','Funk1-97-C','BN1-129-Eb']:
   paths=[DATA/'guitarset/audio'/f'{player}_{song}_{role}_mic.wav' for role in ['comp','solo']]
   if all(p.exists() for p in paths):result['valid' if player=='03' else 'train'].append(paths)
 return result
def crop(pair,rng,augment=True):
 n=132300;a,b=pair;length=min(a.shape[-1],b.shape[-1]);pos=rng.randrange(max(1,length-n))
 a=a[:,pos:pos+n].clone();b=b[:,pos:pos+n].clone()
 if augment:
  a*=10**rng.uniform(-.3,.3);b*=10**rng.uniform(-.3,.3)
  if rng.random()<.2:
   if rng.random()<.5:a*=0
   else:b*=0
 gain=.12/(torch.sqrt((a+b).square().mean())+1e-8)
 return torch.stack([a*gain,b*gain]).unsqueeze(0).cuda()
def loss(y,targets):
 x=targets.sum(1);other=x-y
 denom=targets.square().mean((2,3)).clamp_min(1e-7)
 def snr(e,t,d):
  e=e.float().flatten(1);t=t.float().flatten(1)
  e=e-e.mean(1,keepdim=True);t=t-t.mean(1,keepdim=True)
  projection=t*((e*t).sum(1,keepdim=True)/(t.square().sum(1,keepdim=True)+1e-8))
  target_energy=projection.square().mean(1).clamp_min(1e-8)
  direction=10*torch.log10((e-projection).square().mean(1)/target_energy+.001)
  amplitude=(e-t).square().mean(1)/d
  active=t.square().mean(1)>1e-7
  return torch.where(active,direction+.2*amplitude,((e-t).square().mean(1)/(x.float().square().mean((1,2)).clamp_min(1e-7)))*20)
 forward=(snr(y,targets[:,0],denom[:,0])+snr(other,targets[:,1],denom[:,1]))*.5
 reverse=(snr(y,targets[:,1],denom[:,1])+snr(other,targets[:,0],denom[:,0]))*.5
 return reverse.mean() # output is always solo; complement is accompaniment
ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=600);args=ap.parse_args()
torch.set_num_threads(4);torch.manual_seed(20261101);rng=random.Random(20261101)
paths=pairs();assert len(paths['train'])==16 and len(paths['valid'])>=2
(OUT/'protocol.json').write_text(json.dumps({k:[[str(p) for p in pair] for pair in v] for k,v in paths.items()},indent=2))
data={k:[[read(p) for p in pair] for pair in v] for k,v in paths.items()}
model,cfg,device=load_model('lead');assert device=='cuda'
for name,p in model.named_parameters():p.requires_grad=True
params=[p for p in model.parameters() if p.requires_grad]
optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=1e-4)
scaler=torch.amp.GradScaler('cuda');history=[];best=float('inf');t=time.perf_counter()
validation=[crop(pair,random.Random(i*321+j),False) for i,pair in enumerate(data['valid']) for j in range(3)]
def validate():
 model.eval()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
  return float(torch.stack([loss(model(batch.sum(1)),batch) for batch in validation]).mean())
baseline=validate();best=baseline;torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},OUT/'best.ckpt');print('baseline validation',baseline,flush=True)
for step in range(1,args.steps+1):
 model.eval() # no dropout/BN state changes, mask heads still receive gradients
 batch=crop(rng.choice(data['train']),rng)
 optimizer.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.float16):value=loss(model(batch.sum(1)),batch)
 if not torch.isfinite(value):raise RuntimeError('nonfinite training loss')
 scaler.scale(value).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(params,5.)
 scaler.step(optimizer);scaler.update()
 if step%50==0 or step==args.steps:
  val=validate();row=dict(step=step,train=float(value.detach()),valid=val,seconds=time.perf_counter()-t)
  history.append(row);print(json.dumps(row),flush=True)
  if val<best:
   best=val;torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},OUT/'best.ckpt')
  (OUT/'training.json').write_text(json.dumps(dict(baseline_validation=baseline,history=history,
   trainable_parameters=sum(p.numel() for p in params),criterion='held-out player03 fixed-role SI-SDR plus amplitude/absence loss, not test score',peak_vram_mb=torch.cuda.max_memory_allocated()/1024**2),indent=2))
