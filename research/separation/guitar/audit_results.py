"""Aggregate actual, independently estimated timbre outputs and downloaded assets."""
import json,hashlib
from pathlib import Path
import numpy as np,soundfile as sf
from evaluate_candidates import sisdr,snr
ROOT=Path(__file__).resolve().parent
rows=[]
for case in json.loads((ROOT/'benchmark/protocol.json').read_text())['cases']:
    if case['kind'] not in ['acoustic_electric_remix','negative_acoustic','negative_electric']:continue
    folder=Path(case['folder']);out=ROOT/'benchmark_outputs'/case['id']
    if not all((out/key/'target.wav').exists() for key in ['acoustic','electric']):continue
    x,sr=sf.read(folder/'mix.wav',dtype='float32')
    a,_=sf.read(folder/'source1.wav',dtype='float32');b,_=sf.read(folder/'source2.wav',dtype='float32')
    wood,_=sf.read(out/'acoustic/target.wav',dtype='float32');electric,_=sf.read(out/'electric/target.wav',dtype='float32')
    rest=x-wood-electric
    row=dict(case=case['id'],kind=case['kind'],
        acoustic_si_sdr=sisdr(wood,a),electric_si_sdr=sisdr(electric,b),
        input_si_sdr=[sisdr(x,a),sisdr(x,b)],
        acoustic_energy_fraction=float(np.sum(wood**2)/(np.sum(x**2)+1e-20)),
        electric_energy_fraction=float(np.sum(electric**2)/(np.sum(x**2)+1e-20)),
        residual_energy_fraction=float(np.sum(rest**2)/(np.sum(x**2)+1e-20)),
        reconstruction_error=float(np.max(np.abs(x-wood-electric-rest))))
    rows.append(row)
(ROOT/'timbre_pipeline_scores.json').write_text(json.dumps(rows,indent=2))
assets=[]
for folder in [Path('D:/ChordStudio/data/models/guitar'),Path('D:/ChordStudio/data/models/mus3d'),Path('D:/ChordStudio/data/research/guitar')]:
    for p in folder.rglob('*.download.json'):
        assets.append(json.loads(p.read_text()))
(ROOT/'completed_downloads.json').write_text(json.dumps(assets,indent=2))
print(json.dumps([r for r in rows if 'timbre' in r['case'] or 'only' in r['case']],indent=2))
print('Verified whole-file downloads:',len(assets))
