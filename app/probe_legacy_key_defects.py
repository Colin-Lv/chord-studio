"""Executable counterexamples against frozen code; no production writes."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from app.key_coherence import key_id, name
from app.audit_key_revision import sha, write

def run(folder):
    sys.path.insert(0,str(folder/'before'))
    import key_from_chords as kc
    import key_decision as kd
    import key_layers as kl
    parse=lambda rows:kc.parse_rows(rows)[0]
    result={}
    result['aliases']={label:kd.guard_456(parse([(0,1,'F'),(1,2,'G'),(2,3,label)]),'Am')
                       for label in ['Am','A:min']}
    result['cadence_across_silence']=kl._cadences(parse([(0,1,'G'),(1,2,'N'),(2,3,'C')]),0)
    result['suspended_chord_counted_as_cadence']=kl._cadences(parse([(0,1,'Gsus4'),(1,2,'C')]),0)
    inputs=json.loads((folder/'inputs.json').read_text(encoding='utf-8'))
    girl=next(r for r in inputs['songs'] if r['id']=='0bc0304e01ed40d8baac6b76575b25b6')
    chords=parse(girl['rows']); inference=kc.infer(chords)
    raw=kc.key_name(*inference['pred']); guarded,guard=kd.guard_456(chords,raw)
    before=kd.decide('E',guarded,inference['pred_pair']['margin'],chords)
    real=kl.score
    def changed_bonus(*args):
        value=real(*args);value.update(idiom=-999999,frame_bonus=-999999);return value
    try:
        kl.score=changed_bonus
        after=kd.decide('E',guarded,inference['pred_pair']['margin'],chords)
    finally:kl.score=real
    result['bad_girl_causal_probe']=dict(raw_chord_key=raw,guarded_key=guarded,guard=guard,
        original_use_chord=before['use_chord'],changed_bonus_use_chord=after['use_chord'],
        same_reason=before['reason']==after['reason'])
    failure_path=Path('D:/ChordStudio/research/key-v3/runs/failures.pop909.json')
    failures=json.loads(failure_path.read_text(encoding='utf-8'))
    result['failure_counts']=dict(n=len(failures),below_ten_percent=sum(r['tonic_truth']<.1 for r in failures),
                                 below_five_percent=sum(r['tonic_truth']<.05 for r in failures),
                                 no_cadence=sum(r['cadences_truth']==0 for r in failures))
    root=Path('D:/ChordStudio/_archive/2026-10-30/benchmarks/raw/pop909/POP909')
    scope=[]
    manifest=Path('benchmarks/manifest.pop909_synth.csv')
    for rec in csv.DictReader(manifest.open(encoding='utf-8-sig')):
        p=root/rec['id'].split('_')[-1]/'key_audio.txt'
        totals=defaultdict(float);longest=(-1,None)
        for line in p.read_text().splitlines():
            start,end,label=line.split();duration=float(end)-float(start);key=key_id(label)
            if key is None or duration<=0:raise ValueError(str(p))
            totals[key]+=duration
            if duration>longest[0]:longest=(duration,key)
        maximum=max(totals.values());winners=[k for k,v in totals.items() if abs(v-maximum)<1e-8]
        scope.append(dict(id=rec['id'],reference=rec['key'],unique_keys=len(totals),
                          longest_segment=name(*longest[1]),total_duration_winners=[name(*k) for k in winners],
                          longest_is_total_winner=longest[1] in winners,reference_sha256=sha(p)))
    result['reference_scope']=dict(n=len(scope),multiple_keys=sum(r['unique_keys']>1 for r in scope),
                                   longest_vs_total_different=[r for r in scope if not r['longest_is_total_winner']],rows=scope,
                                   note='Neither longest segment nor cumulative duration establishes musical global tonic')
    result['input_hashes']=dict(failure_json=sha(failure_path),manifest=sha(manifest),code=sha(Path(__file__)))
    write(folder/'legacy_probes.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='reference_scope'},ensure_ascii=False))
    print(json.dumps({k:v for k,v in result['reference_scope'].items() if k!='rows'},ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);run(ap.parse_args().folder)
