"""Independent NNLS note-template evidence; no instrument or voice separation.

Invoke as a script so the existing standalone audio bridge can be imported.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from harmonic_note_features import extract_note_evidence, MIDI_NOTES, SAMPLE_RATE, sha256_file


def summary(evidence, start, end, minor_pc=7, major_pc=8):
    mask=(evidence.times>=start)&(evidence.times<end)
    result={}
    for band,lo,hi in [('all',21,104),('low',21,47),('middle',48,71),('high',72,104)]:
        rows=(MIDI_NOTES>=lo)&(MIDI_NOTES<=hi)
        values=evidence.activations[rows][:,mask].mean(axis=1)
        notes=MIDI_NOTES[rows]
        minor=float(values[notes%12==minor_pc].sum());major=float(values[notes%12==major_pc].sum())
        result[band]=dict(minor_pc=minor_pc,major_pc=major_pc,minor_activation=minor,major_activation=major,
                         major_fraction_of_pair=major/(minor+major) if minor+major else None,
                         top_notes=[dict(midi=int(notes[i]),activation=float(values[i])) for i in np.argsort(-values)[:6]])
    return result


def synth(notes):
    t=np.arange(3*SAMPLE_RATE)/SAMPLE_RATE;y=np.zeros_like(t)
    for midi in notes:
        freq=440*2**((midi-69)/12)
        for h in range(1,7):
            y+=.65**(h-1)*np.sin(2*np.pi*freq*h*t+.1*h)
    y*=np.minimum(np.clip(t/.05,0,1),np.clip((3-t)/.05,0,1))
    return (.6*y/max(np.abs(y).max(),1)).astype(np.float32)


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    result=dict(synthetic={},recordings=[],scope='note-template activations, not a verified melody/bass part or chord probability')
    for label,notes in [('E_root_overtones',[52]),('E_major',[52,56,59]),('E_minor',[52,55,59])]:
        path=out/(label+'.wav');sf.write(path,synth(notes),SAMPLE_RATE,subtype='FLOAT')
        e=extract_note_evidence(path,cache_dir=out/'cache')
        result['synthetic'][label]=dict(notes=notes,evidence=summary(e,.5,2.5),metadata=e.metadata)
    # Check directional discrimination, not a claim of real-recording accuracy.
    maj=result['synthetic']['E_major']['evidence']['middle']['major_fraction_of_pair']
    minor=result['synthetic']['E_minor']['evidence']['middle']['major_fraction_of_pair']
    if not maj>minor:raise AssertionError('Synthetic major/minor discrimination failed')
    for sid,minor_pc,major_pc in [('191347',6,7),('853834',7,8)]:
        path=Path('D:/ChordStudio/_archive/2026-10-30/benchmarks/external_key_data/audio')/(sid+'.mp3')
        provenance=json.loads((Path('benchmarks/reports/harmonic_key_v2_20261006/acoustics')/sid/'raw.lab.provenance.json').read_text())
        assert sha256_file(path)==provenance['audio']['sha256']
        print('NNLS extraction:',sid,flush=True)
        e=extract_note_evidence(path,cache_dir=out/'cache')
        windows=[(0,e.metadata['duration_seconds'])]+([(47.1,65.1),(65.1,100)] if sid=='853834' else [(0,32),(32,64),(64,96)])
        row=dict(id=sid,metadata=e.metadata,windows=[dict(start=a,end=b,evidence=summary(e,a,b,minor_pc,major_pc)) for a,b in windows])
        result['recordings'].append(row)
        print(json.dumps(dict(id=sid,all=row['windows'][0]['evidence']),ensure_ascii=False),flush=True)
    result['implementation_sha256']=sha256_file(__file__)
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);main(ap.parse_args().out)
