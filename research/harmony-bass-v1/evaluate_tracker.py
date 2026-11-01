"""Evaluate the deployed beat aggregation on held-out rendered MIDI, not just the ranker."""
import sys
from pathlib import Path
import numpy as np,soundfile as sf
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]))
from app.web.bass_voice import load_model,spectrum,summarize_beats,NAMES
from app.web.storage import read_json,atomic_json
OUT=HERE/'training-register-v2'
model=load_model(OUT/'bass_voice_v2.joblib');results=[]
for item in read_json(OUT/'manifest.json'):
    if item['split']!='test':continue
    y,sr=sf.read(OUT/'audio'/(item['group']+'.wav'));e=spectrum(y,sr)
    for event in item['events']:
        a,b=event['start'],event['end'];r=summarize_beats(model,e,a,b,np.linspace(a,b,5))
        truth=None if event['bass']<0 else NAMES[event['bass']]
        results.append(dict(group=item['group'],kind=event['kind'],truth=truth,**r))
pitched=[r for r in results if r['truth'] is not None];accepted=[r for r in results if r['note'] is not None]
high=[r for r in results if r['kind']==5]
report=dict(n=len(results),pitched=len(pitched),pitched_correct=sum(r['note']==r['truth'] for r in pitched),
            accepted=len(accepted),accepted_correct=sum(r['note']==r['truth'] for r in accepted),
            high_register_n=len(high),high_register_correct=sum(r['note']==r['truth'] for r in high),
            unpitched_false_positives=sum(r['note'] is not None for r in results if r['truth'] is None),rows=results)
atomic_json(OUT/'tracker_metrics.json',report);print({k:v for k,v in report.items() if k!='rows'},flush=True)
