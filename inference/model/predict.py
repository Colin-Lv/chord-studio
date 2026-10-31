"""Portable frozen retained10 research predictor. Uses audio features, never reference labels."""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name,'1')
import argparse,json,hashlib,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.special import softmax,expit
sys.path.insert(0,str(Path(__file__).resolve().parent))
from chord_semantics_v2 import parse_chord

NAMES=('C','C#','D','D#','E','F','F#','G','G#','A','A#','B')
QUALITIES=('maj','min','7','maj7','min7','9','maj9','min9','sus2','sus4')
INDICES=np.array([[(j//12)*12+(j%12+r)%12 if j<132 else 132+(j-132+r)%12 if j<144 else j for j in range(149)] for r in range(12)])
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def decode(probability,cells,nmask):
    result=[]
    for cell,index,is_n in zip(cells,probability.argmax(1),nmask):
        root,quality=divmod(int(index),10);label='N' if is_n else NAMES[root]+':'+QUALITIES[quality]
        original=parse_chord(cell['chord'])
        if not is_n and original.state=='chord' and original.explicit_bass:label+='/'+NAMES[original.bass_pc]
        if result and result[-1]['chord']==label and result[-1]['end']==cell['start']:result[-1]['end']=cell['end']
        else:result.append(dict(start=cell['start'],end=cell['end'],chord=label))
    return result

class Retained10:
    """Offline, whole-clip routed chord model on the fixed 149/12/4 feature contract."""
    def __init__(self,bundle_dir=None):
        self.directory=Path(bundle_dir) if bundle_dir is not None else Path(__file__).resolve().parent
        self.manifest=read(self.directory/'manifest.json')
        for name,digest in self.manifest['files_sha256'].items():
            if sha(self.directory/name)!=digest:raise ValueError('Frozen bundle file differs: '+name)
        self.arrays=load_npz(self.directory/'retained10.npz');self.seeds=self.manifest['seeds']
        if self.manifest['recipe']!={'quality':'deep10','mix':'router','N_head':'prior'}:raise ValueError('Unsupported recipe')
        if tuple(self.manifest['qualities'])!=QUALITIES:raise ValueError('Quality order differs')
        self.source=self._component('source')
    def _component(self,prefix):
        return {key[len(prefix)+2:]:v for key,v in self.arrays.items() if key.startswith(prefix+'__')}
    def _source(self,x):
        p=self.source;h=[];lo=[]
        # Preserve the original matrix batch shape and float32 order.
        for i in range(0,len(x),128):
            z=x[i:i+128,INDICES].reshape(-1,149)
            a=np.maximum(z@p['w0']+p['b0'],0);b=np.maximum(a@p['w1']+p['b1'],0)
            h.append(b.reshape(-1,12,48));lo.append((b@p['w2']+p['b2']).reshape(-1,120))
        return np.concatenate(h),np.concatenate(lo)
    def predict_arrays(self,features,cells,deep_chroma,energy):
        x=np.asarray(features);dc=np.asarray(deep_chroma);energy=np.asarray(energy)
        n=len(cells)
        if n<1 or x.shape!=(n,149) or dc.ndim!=2 or dc.shape[1]!=12 or len(dc)<1 or energy.shape!=(n,4):raise ValueError('Expected n×149, frames×12 at10Hz, n×4 and n audio cells')
        if x.dtype!=np.float32 or dc.dtype!=np.float32 or energy.dtype!=np.float32:raise ValueError('The frozen frontend contract requires float32 input arrays')
        if not all(np.isfinite(a).all() for a in (x,dc,energy)):raise ValueError('Nonfinite audio features')
        starts=np.array([c['start'] for c in cells]);ends=np.array([c['end'] for c in cells])
        if not np.isfinite(starts).all() or not np.isfinite(ends).all() or np.any(ends<=starts) or np.max(abs(starts[1:]-ends[:-1]),initial=0)>1e-8:raise ValueError('Audio cells must be finite, increasing and contiguous')
        h,lo=self._source(x);times=np.arange(len(dc))/10;deep=[]
        for c in cells:
            middle=(c['start']+c['end'])/2;a,b=np.searchsorted(times,[max(0,middle-.25),middle+.25]);a=min(a,len(dc)-1);b=max(a+1,min(b,len(dc)));deep.append(dc[a:b].mean(0))
        deep=np.asarray(deep,'float32');pr=np.exp(lo-lo.max(1,keepdims=True));pr/=pr.sum(1,keepdims=True);ordered=np.sort(pr,axis=1);g=x[:,:132].reshape(-1,11,12)
        common=np.concatenate((h.mean(1),g.mean(2),g.std(2),g.max(2),x[:,148:149],pr.max(1,keepdims=True),(-np.sum(pr*np.log(np.maximum(pr,1e-30)),1)/np.log(120))[:,None],(ordered[:,-1]-ordered[:,-2])[:,None],(ends-starts)[:,None]),1).astype('float32')
        order=np.array([[(r+j)%12 for j in range(12)] for r in range(12)]);quality_x=np.concatenate((h,deep[:,order]),2);gate_x=np.concatenate((common,np.sort(deep,axis=1)),1)
        clip_x=np.concatenate((gate_x,energy),1).astype('float64');w=ends-starts;w/=w.sum();mean=w@clip_x;std=np.sqrt(np.maximum(0,w@np.square(clip_x)-mean**2));clip=np.r_[mean,std].astype('float32')
        source=softmax(lo.astype('float64'),axis=1);probabilities=[];nprobabilities=[];weights=[]
        for seed in self.seeds:
            pn=self._component('N_'+str(seed));pq=self._component('quality_'+str(seed));pg=self._component('router_'+str(seed))
            z=((gate_x-pn['mean'])/pn['std']).copy();z[:,-12:]=0;npred=expit((np.maximum(z@pn['w0']+pn['b0'],0)@pn['w1']+pn['b1']).ravel())
            residual=(np.maximum(((quality_x-pq['mean'])/pq['std'])@pq['w0']+pq['b0'],0)@pq['w1']+pq['b1']).reshape(-1,120)
            adapted=softmax((lo+residual).astype('float64'),axis=1)
            weight=float(expit((np.maximum(((clip[None,:]-pg['mean'])/pg['std'])@pg['w0']+pg['b0'],0)@pg['w1']+pg['b1']).ravel())[0])
            probabilities.append((1-weight)*source+weight*adapted);nprobabilities.append(npred);weights.append(weight)
        probability=np.mean(probabilities,axis=0);nprobability=np.mean(nprobabilities,axis=0)
        if not np.isfinite(probability).all() or np.max(abs(probability.sum(1)-1))>1e-12 or not np.isfinite(nprobability).all():raise ValueError('Invalid model output')
        return dict(segments=decode(probability,cells,nprobability>=.5),chord_probabilities=probability,N_probabilities=nprobability,real_expert_weights=weights)
    def predict_folder(self,folder):
        folder=Path(folder)
        return self.predict_arrays(load_npz(folder/'features.npz')['features'],read(folder/'cells.json'),load_npz(folder/'deep.npz')['chroma'],load_npz(folder/'energy.npz')['energy'])

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--features',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent);a=parser.parse_args();output=a.output.resolve();lab=output.with_suffix('.lab')
    if output.suffix!='.json' or output.exists() or lab.exists():raise ValueError('Choose new JSON and LAB output paths')
    model=Retained10(a.bundle);r=model.predict_folder(a.features);payload=dict(created_utc=datetime.now(timezone.utc).isoformat(),research_only=True,default_changed=False,reference_inputs=False,bundle_version=model.manifest['bundle_version'],weights_sha256=model.manifest['files_sha256']['retained10.npz'],frozen_candidate_bundle_sha256=model.manifest['frozen_candidate_bundle_sha256'],supported_qualities=QUALITIES,real_expert_weights=r['real_expert_weights'],segments=r['segments'],limitations=model.manifest['limitations']);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='\n') as f:json.dump(payload,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    with lab.open('x',encoding='utf-8',newline='\n') as f:f.write(''.join(f'{s["start"]:.17g} {s["end"]:.17g} {s["chord"]}\n' for s in r['segments']))
    print(json.dumps(dict(JSON=str(output),LAB=str(lab),research_only=True,new_fits=0,new_updates=0)))
if __name__=='__main__':main()
