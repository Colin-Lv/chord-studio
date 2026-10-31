"""Paired old/strict-parser replay with unchanged key scores and decisions."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
import mir_eval
from app.key_chord_adapter import parse_rows as strict_rows, adapt
from app.key_coherence import rank, key_id, pcs

SOURCE=Path('D:/ChordStudio/research/key-stage2-20261101')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'))
    assert sha(SOURCE/'inputs.json')==protocol['input_sha256']
    assert sha(SOURCE/'pop_inputs.json')==protocol['pop_sha256']
    for filename in ['inputs.json','pop_inputs.json']:
        shutil.copy2(SOURCE/filename,out/filename)
    code=out/'code';code.mkdir()
    for filename in ['key_coherence.py','key_chord_adapter.py','chord_semantics_v2.py','chord_semantics_v3.py','run_key_parser_stage3.py']:
        shutil.copy2(Path('app')/filename,code/filename)
    shutil.copytree(SOURCE/'baseline',out/'baseline',ignore=shutil.ignore_patterns('__pycache__'))
    sys.path.insert(0,str((out/'baseline').resolve()))
    import key_from_chords as kc
    import key_decision as decision
    data=json.loads((out/'inputs.json').read_text(encoding='utf-8'))
    data['pop']=json.loads((out/'pop_inputs.json').read_text(encoding='utf-8'))
    result={};labels=Counter()
    for group in ['cases','songs','pop']:
        records=[]
        for row in data[group]:
            for r in row['rows']:labels[r.get('label') if isinstance(r,dict) else r[2]]+=1
            record={k:row[k] for k in ['id','title','reference','single_key','run_id','source_id'] if k in row}
            for tag,parser in [('legacy',kc.parse_rows),('strict',strict_rows)]:
                chords,stats=parser(row['rows']);ev=kc.infer(chords)
                raw=kc.key_name(*ev['pred']);guarded,guard=decision.guard_456(chords,raw)
                item=dict(parse_stats=stats,raw_chord_key=raw,guarded=guarded)
                if group!='pop':
                    cnn=max(row['candidates'],key=lambda r:r['score'])['key']
                    d=decision.decide(cnn,guarded,ev['pred_pair']['margin'],chords)
                    ranked=rank(chords,row['candidates'])
                    item.update(cnn=cnn,legacy_rules=guarded if d['use_chord'] else cnn,
                                coherence=ranked[0]['key'],ranking=ranked)
                record[tag]=item
            records.append(record)
        result[group]=records
    changes=[]
    for label,n in labels.items():
        a=kc.parse_chord_label(label);b,status=adapt(label)
        old=sorted(pcs(a)) if a else None;new=sorted(pcs(b)) if b else None
        if old!=new:
            changes.append(dict(label=label,occurrences=n,legacy_pcs=old,strict_pcs=new,diagnostic=status))
    result['changed_labels']=sorted(changes,key=lambda r:-r['occurrences'])
    result['summary']={}
    for group,methods in [('cases',['cnn','legacy_rules','coherence']),('pop',['raw_chord_key','guarded'])]:
        groups=['all'] if group=='cases' else ['all','single_key','multi_key']
        for subset in groups:
            rows=[r for r in result[group] if subset=='all' or r['single_key']==(subset=='single_key')]
            result['summary'][group+'_'+subset]={tag:{method:dict(hits=sum(key_id(r[tag][method])==key_id(r['reference']) for r in rows),n=len(rows)) for method in methods} for tag in ['legacy','strict']}
    result['coherence_changes']=[dict(id=r['id'],reference=r['reference'],before=r['legacy']['coherence'],after=r['strict']['coherence']) for r in result['cases'] if r['legacy']['coherence']!=r['strict']['coherence']]
    result['metadata']=dict(status='exposed_regression_no_weight_changes',mir_eval=mir_eval.__version__,
         hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and '__pycache__' not in str(p)})
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=result['summary'],changed_labels=len(changes),coherence_changes=result['coherence_changes'],
          songs=[dict(title=r['title'],before=r['legacy']['coherence'],after=r['strict']['coherence']) for r in result['songs']]),ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);run(ap.parse_args().out)
