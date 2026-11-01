"""Two guitars in overlapping registers, and a known synthetic vocal remix.

Controls measure only these constructed mixtures, not commercial-master accuracy.
"""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np
import soundfile as sf
import librosa

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'synth6'))
from build_synth6 import R


def main():
    out=ROOT/'controls';out.mkdir(exist_ok=False)
    sampler=R.make_sampler_class(R.load_sf2_module())(str(R.find_soundfont(None)))
    sr=44100;n=sr*14
    def guitar(program, melody, query=False):
        x=np.zeros(n,dtype=np.float32)
        for i in range(24):
            pitch=melody[i%len(melody)]+(2 if query else 0)
            note,_=sampler.render(program,pitch,90,.38,sr=sr,bank=0,gain=.5)
            pos=int((.25+i*.5)*sr);end=min(n,pos+len(note));x[pos:end]+=note[:end-pos]
        x*=.09/max(float(np.sqrt(np.mean(x*x))),1e-8)
        return np.repeat(x[:,None],2,axis=1)
    parts={'acoustic':guitar(25,[60,64,67,64,62,65,69,65]),
           'electric':guitar(29,[67,65,64,62,64,65,67,69])}
    for name,x in parts.items():sf.write(out/(name+'.wav'),x,sr,subtype='FLOAT')
    sf.write(out/'guitar_mix.wav',sum(parts.values()),sr,subtype='FLOAT')
    sf.write(out/'query_acoustic.wav',guitar(25,[57,60,64,60],True),sr,subtype='FLOAT')
    sf.write(out/'query_electric.wav',guitar(29,[69,67,65,64],True),sr,subtype='FLOAT')
    # A pitch-shifted remix with known waveforms; not an independently recorded duet.
    voice,_=sf.read(ROOT/'vocal-float-pilot/lead_vocals.wav',dtype='float32',always_2d=True)
    lead=voice[sr*3:sr*15].mean(axis=1)
    harmony=librosa.effects.pitch_shift(lead,sr=sr,n_steps=4)*.5
    lead=np.repeat(lead[:,None],2,axis=1)
    harmony=np.stack([harmony*.5,harmony],axis=1)
    for name,x in [('voice_lead_truth',lead),('voice_harmony_truth',harmony),('voice_mix',lead+harmony),('voice_solo',lead)]:
        sf.write(out/(name+'.wav'),x,sr,subtype='FLOAT')
    meta=dict(guitar_soundfont=str(sampler.path),guitar_programs=[25,29],guitar_registers_overlap=True,
              guitar_spatial_layout='identical centred stereo; no panning shortcut',
              query='same timbre, different notes; not the ground-truth waveform',
              voice='estimated lead excerpt remixed with pitch-shifted duplicate; synthetic stress test only',
              files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.wav')})
    (out/'protocol.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')


if __name__=='__main__':main()
