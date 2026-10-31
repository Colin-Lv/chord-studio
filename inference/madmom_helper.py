"""Existing madmom0.16.1 runtime, isolated process, one fixed audio task."""
import os,sys,json,hashlib,warnings
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
warnings.filterwarnings('ignore',message='pkg_resources is deprecated')
import numpy as np
import madmom
from madmom.audio.signal import Signal
request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));folder=Path(request['folder']);runtime=request['runtime']
if madmom.__version__!=runtime['madmom_version']:raise ValueError('Different madmom version')
from madmom import models
found={'CNN':models.CHORDS_CNN_FEAT[0],'CRF':models.CHORDS_CFCRF[0],'DeepChroma':models.CHROMA_DNN[0]}
for name,path in found.items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=runtime['madmom_models'][name]['sha256']:raise ValueError('Different upstream weight: '+name)
if sys.argv[2]=='anchors':
    from madmom.features.chords import CNNChordFeatureProcessor,CRFChordRecognitionProcessor
    from madmom.processors import SequentialProcessor
    data=np.load(request['mono'],allow_pickle=False);segments=SequentialProcessor([CNNChordFeatureProcessor(),CRFChordRecognitionProcessor()])(data)
    with (folder/'madmom.lab').open('x',encoding='utf-8',newline='\n') as f:
        for start,end,label in segments:f.write(f'{float(start):.3f} {float(end):.3f} {label}\n')
    print(json.dumps(dict(segments=len(segments),samples=len(data))),flush=True)
elif sys.argv[2]=='deep':
    from madmom.audio.chroma import DeepChromaProcessor
    signal=Signal(request['audio'],sample_rate=44100,num_channels=1);chroma=np.asarray(DeepChromaProcessor()(signal),dtype='float32')
    if chroma.ndim!=2 or chroma.shape[1]!=12 or not np.isfinite(chroma).all() or chroma.min()<0 or chroma.max()>1:raise ValueError('Invalid DeepChroma')
    np.savez_compressed(folder/'deep.npz',chroma=chroma,fps=np.array(10.),samples=np.array(len(signal)),sample_rate=np.array(44100));print(json.dumps(dict(frames=len(chroma),samples=len(signal))),flush=True)
else:raise ValueError('Unknown fixed helper action')
