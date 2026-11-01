"""Deterministic remixes of real recorded guitars; keep waveforms and identities."""
from pathlib import Path
import json,hashlib
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
ROOT=Path(__file__).resolve().parent;DATA=Path('D:/ChordStudio/data/research/guitar');OUT=ROOT/'benchmark'
OUT.mkdir(exist_ok=True)
def read(p,start=0,duration=12):
 x,sr=sf.read(p,dtype='float32',always_2d=True)
 x=x[int(start*sr):int((start+duration)*sr)]
 if sr!=44100:
  from math import gcd
  g=gcd(sr,44100);x=resample_poly(x,44100//g,sr//g,axis=0).astype('float32')
 if x.shape[1]==1:x=np.repeat(x,2,axis=1)
 return x
def rms(x):return float(np.sqrt(np.mean(x*x)+1e-20))
cases=[]
def case(name,a,b,sources,kind):
 n=min(len(a),len(b));a=a[:n];b=b[:n];gain=.7/max(1.,float(np.max(np.abs(a+b))))
 a=a*gain;b=b*gain;folder=OUT/name;folder.mkdir(exist_ok=True)
 for key,x in [('mix',a+b),('source1',a),('source2',b)]:sf.write(folder/(key+'.wav'),x,44100,subtype='FLOAT')
 cases.append(dict(id=name,kind=kind,sources=sources,duration=n/44100,folder=str(folder.resolve()),source_rms=[rms(a),rms(b)]))
wood=DATA/'GuitarDuets/Real/Valses Poeticos 6/guitar2.wav'
for i,number in enumerate(['01','02','03']):
 electric=DATA/'egset12'/(number+'.wav')
 if not electric.exists():continue
 a=read(wood,start=3+i*4);b=read(electric,start=3)
 a*=.075/rms(a);b*=.075/rms(b)
 # First source wood, second electric. These labels are timbre, not role.
 case('real_timbre_'+number,a,b,[str(wood),str(electric)],'acoustic_electric_remix')
 if i==0:
  case('wood_only',a,np.zeros_like(a),[str(wood),None],'negative_acoustic')
  case('electric_only',np.zeros_like(b),b,[None,str(electric)],'negative_electric')
  case('real_timbre_quiet_electric',a,b*.5,[str(wood),str(electric)],'acoustic_electric_remix')
for folder in sorted((DATA/'GuitarDuets/Real/test').glob('*')):
 if all((folder/(s+'.wav')).exists() for s in ['guitar1','guitar2']):
  a=read(folder/'guitar1.wav',3,16);b=read(folder/'guitar2.wav',3,16)
  case('duet_'+folder.name,a,b,[str(folder/'guitar1.wav'),str(folder/'guitar2.wav')],'real_duet')
for player in ['00','03','05']:
 for song in ['Rock2-142-D','Jazz1-200-B']:
  comp=DATA/'guitarset/audio'/f'{player}_{song}_comp_mic.wav'
  solo=DATA/'guitarset/audio'/f'{player}_{song}_solo_mic.wav'
  if not (comp.exists() and solo.exists()):continue
  a=read(comp,2,12);b=read(solo,2,12);a*=.075/rms(a);b*=.075/rms(b)
  case('role_'+player+'_'+song,a,b,[str(comp),str(solo)],'same_guitar_comp_solo_remix')
(OUT/'protocol.json').write_text(json.dumps(dict(cases=cases,note='Exact linear remixes. No per-output normalization. Classical training recordings may contain microphone bleed; real_duet uses the dedicated test folders. Guitar1/2 are source identities, not lead/rhythm labels.'),ensure_ascii=False,indent=2),encoding='utf-8')
print('prepared',len(cases),'cases')
