"""Fixed-role reference scoring; no best-permutation relabeling."""
import json
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parent


def read(path):
    return sf.read(path,dtype='float64',always_2d=True)[0]


def si_sdr(truth,estimate):
    if truth.shape!=estimate.shape:raise ValueError('Different time/channel coordinates')
    t=truth.reshape(-1).copy();e=estimate.reshape(-1).copy()
    t-=t.mean();e-=e.mean()
    power=np.dot(t,t)
    if power<1e-15:raise ValueError('Cannot score SI-SDR against silence')
    p=np.dot(t,e)/power*t
    return float(10*np.log10(max(np.dot(p,p),1e-20)/max(np.dot(e-p,e-p),1e-20)))


def main():
    control=ROOT/'controls';result={}
    if (ROOT/'voice-control-mix/inference.json').exists():
        truth=[read(control/n) for n in ['voice_lead_truth.wav','voice_harmony_truth.wav']]
        estimates=[read(ROOT/'voice-control-mix'/n) for n in ['lead_vocals.wav','backing_vocals.wav']]
        mix=read(control/'voice_mix.wav');solo=read(control/'voice_solo.wav')
        false=read(ROOT/'voice-control-solo/backing_vocals.wav')
        result['vocals']=dict(rows=['lead_truth','shifted_harmony_truth'],columns=['lead_estimate','residual'],
            si_sdr_matrix=[[si_sdr(t,y) for y in estimates] for t in truth],
            mixture_baseline=[si_sdr(t,mix) for t in truth],
            solo_false_residual_db=float(10*np.log10(np.sum(false**2)/np.sum(solo**2))))
    for query in ['acoustic','electric']:
        path=ROOT/('guitar-'+query)
        if not (path/'inference.json').exists():continue
        truth=[read(control/(name+'.wav')) for name in ['acoustic','electric']]
        estimates=[read(path/'target.wav'),read(path/'residual.wav')]
        mix=read(control/'guitar_mix.wav')
        result['guitar_'+query]=dict(rows=['acoustic','electric'],columns=['query_target','residual'],
            si_sdr_matrix=[[si_sdr(t,y) for y in estimates] for t in truth],
            mixture_baseline=[si_sdr(t,mix) for t in truth],
            sum_max_abs_error=float(np.max(np.abs(mix-sum(estimates)))))
    path=ROOT/'guitar-absent'
    if (path/'inference.json').exists():
        y=read(path/'target.wav');x=read(control/'acoustic.wav')
        result['absent_electric_query']=dict(target_energy_relative_db=float(10*np.log10(np.sum(y*y)/np.sum(x*x))))
    (ROOT/'control_scores.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
