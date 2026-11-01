"""New final evaluation cases, excluded from every adaptation run."""
import json
from pathlib import Path
import numpy as np,soundfile as sf
ROOT=Path(__file__).resolve().parent;DATA=Path('D:/ChordStudio/data/research/guitar/guitarset/audio')
protocol=json.loads((ROOT/'benchmark/protocol.json').read_text())
cases=[c for c in protocol['cases'] if not c['id'].startswith('final_')]
for song in ['Funk1-97-C','BN1-129-Eb','Rock3-117-Bb','Jazz3-150-C']:
    sources=[DATA/f'05_{song}_{role}_mic.wav' for role in ['comp','solo']]
    audio=[]
    for p in sources:
        x,sr=sf.read(p,dtype='float32',always_2d=True);assert sr==44100
        x=x[2*sr:14*sr];x=np.repeat(x,2,axis=1) if x.shape[1]==1 else x
        x*=.075/np.sqrt(np.mean(x*x)+1e-20);audio.append(x)
    n=min(map(len,audio));a,b=[x[:n] for x in audio]
    variants=[('mix',a,b)]
    if song in ['Rock3-117-Bb','Jazz3-150-C']:
        variants += [('comp_only',a,np.zeros_like(b)),('solo_only',np.zeros_like(a),b)]
    for variant,audio_a,audio_b in variants:
        name='final_05_'+song+'_'+variant;out=ROOT/'benchmark'/name;out.mkdir(exist_ok=True)
        gain=.7/max(1.,float(np.max(np.abs(audio_a+audio_b))));audio_a=audio_a*gain;audio_b=audio_b*gain
        for key,x in [('mix',audio_a+audio_b),('source1',audio_a),('source2',audio_b)]:sf.write(out/(key+'.wav'),x,sr,subtype='FLOAT')
        cases.append(dict(id=name,kind='final_role_'+variant,folder=str(out),sources=list(map(str,sources)),
            duration=n/sr,split='player05 excluded from adaptation; Rock3/Jazz3 also unseen composition'))
protocol['cases']=cases
protocol['final_test_note']='Final cases predeclared before evaluation of adapted model. Pilot cases repeatedly inspected; do not call them a blind test. No claim about upstream pretrained model data overlap.'
(ROOT/'benchmark/protocol.json').write_text(json.dumps(protocol,indent=2))
print(len(cases),'cases total')
