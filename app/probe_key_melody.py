"""Role-isolated evidence probe, not an audio key recognizer or a trained fusion.

Run once per output path. Thresholds are fixed before looking at outcomes.
POP909 MELODY is a separate arranged MIDI part. key_audio is a reference
extracted from audio, not a claim of independently adjudicated human truth.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pretty_midi
from app.key_coherence import key_id, name

PROTOCOL = dict(rest_min_quarter_beats=1.0, note_min_quarter_beats=0.5,
                end_weight_cap_quarter_beats=2.0, final_note_in_rest_histogram=False,
                experiment='diagnostic_only_no_fusion_no_tuning',
                population='all 909; exclude missing MELODY and multi-key references explicitly',
                comparison='relative-key pair given by reference; NOT 24-class key accuracy')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def features(notes, tempo_times, tempi):
    duration=np.zeros(12)
    endings=np.zeros(12)
    events=[]
    # Merge tied/repeated adjacent pitches before deciding whether a rest follows.
    spans=[]
    for n in sorted(notes,key=lambda n:(n.start,n.end,n.pitch)):
        if spans and n.pitch==spans[-1][2] and n.start<=spans[-1][1]+1e-5:
            spans[-1][1]=max(spans[-1][1],n.end)
        else:spans.append([n.start,n.end,n.pitch])
    for i,(start,end,pitch) in enumerate(spans):
        duration[pitch%12]+=end-start
        if i+1==len(spans):continue  # No observed following silence at the end.
        tempo=float(tempi[max(0,np.searchsorted(tempo_times,end,side='right')-1)])
        quarter=60/tempo
        length=(end-start)/quarter
        gap=(spans[i+1][0]-end)/quarter
        if gap>=PROTOCOL['rest_min_quarter_beats'] and length>=PROTOCOL['note_min_quarter_beats']:
            weight=min(length,PROTOCOL['end_weight_cap_quarter_beats'])
            endings[pitch%12]+=weight
            events.append(dict(start=start,end=end,pitch=pitch,gap_quarters=gap,weight=weight))
    def norm(a):return (a/a.sum()).tolist() if a.sum()>0 else None
    return dict(duration_pc=norm(duration),pre_rest_pc=norm(endings),pre_rest_events=events,
                final_pc=spans[-1][2]%12 if spans else None,notes=len(spans))

def preference(hist, truth, other):
    if hist is None:return 'missing'
    delta=hist[truth]-hist[other]
    return 'reference' if delta>1e-9 else 'alternative' if delta < -1e-9 else 'tie'

def run(out):
    out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2),encoding='utf-8')
    root=Path('D:/ChordStudio/_archive/2026-10-30/benchmarks/raw/pop909/POP909')
    manifest=Path('benchmarks/manifest.pop909_synth.csv')
    failures_path=Path('D:/ChordStudio/research/key-v3/runs/failures.pop909.json')
    failures={r['id']:r for r in json.loads(failures_path.read_text(encoding='utf-8'))}
    rows=[]; excluded=[]
    for rec in csv.DictReader(manifest.open(encoding='utf-8-sig')):
        sid=rec['id'].split('_')[-1]
        midi=root/sid/(sid+'.mid'); keyfile=root/sid/'key_audio.txt'
        ref=key_id(rec['key'])
        keys={key_id(line.split()[-1]) for line in keyfile.read_text().splitlines() if line.strip()}
        if keys!={ref} or ref is None:
            excluded.append(dict(id=rec['id'],reason='multiple or inconsistent global reference',keys=list(keys)))
            continue
        music=pretty_midi.PrettyMIDI(str(midi))
        tracks=[i for i in music.instruments if i.name=='MELODY' and not i.is_drum]
        if len(tracks)!=1 or not tracks[0].notes:
            excluded.append(dict(id=rec['id'],reason='missing or ambiguous MELODY track'));continue
        f=features(tracks[0].notes,*music.get_tempo_changes())
        relative=((ref[0]+9)%12,'minor') if ref[1]=='major' else ((ref[0]+3)%12,'major')
        r=dict(id=rec['id'],reference=name(*ref),relative=name(*relative),
               midi_sha256=digest(midi),reference_sha256=digest(keyfile),**f)
        r['relative_comparison']={field:preference(f[field],ref[0],relative[0]) for field in ['duration_pc','pre_rest_pc']}
        if rec['id'] in failures:
            old=failures[rec['id']]; other=key_id(old['pred'])
            r['old_failure']=old
            r['old_prediction_comparison']={field:preference(f[field],ref[0],other[0]) for field in ['duration_pc','pre_rest_pc']}
        rows.append(r)
    from collections import Counter
    groups={'all_single_key':rows,'old_failures':[r for r in rows if 'old_failure' in r],
            'old_low_tonic':[r for r in rows if r.get('old_failure',{}).get('tonic_truth',1)<.10],
            'old_no_cadence':[r for r in rows if r.get('old_failure',{}).get('cadences_truth',1)==0]}
    summary={}
    for group,subset in groups.items():
        comparison='relative_comparison' if group=='all_single_key' else 'old_prediction_comparison'
        summary[group]=dict(n=len(subset),**{field:dict(Counter(r[comparison][field] for r in subset))
                                                 for field in ['duration_pc','pre_rest_pc']})
    result=dict(protocol=PROTOCOL,summary=summary,excluded=excluded,rows=rows,
                implementation_sha256=digest(Path(__file__)),manifest_sha256=digest(manifest),
                old_failure_sha256=digest(failures_path),pretty_midi_version=pretty_midi.__version__)
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=summary,excluded=excluded),ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    run(ap.parse_args().output)
