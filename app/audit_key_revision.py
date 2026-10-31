"""Freeze inputs and code before editing; compare on the identical frozen inputs.

python -m app.audit_key_revision freeze PATH
python -m app.audit_key_revision compare PATH
Known regression data only. Never reads a project's human-edited chord track.
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

BACKEND = Path('C:/Users/33573/AppData/Local/Programs/Chord Studio/backend/app')
CASES = Path('benchmarks/reports/harmonic_key_v2_20261006/acoustics')
PROJECTS = Path('D:/ChordStudio/data/projects')
PIDS = ['aa5a41797bb343e590abcf4d8b27ae54', '0bc0304e01ed40d8baac6b76575b25b6',
        '087d8b0c86414fe6b675cc622f505128']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def module(path, alias):
    spec = importlib.util.spec_from_file_location(alias, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def freeze(out):
    out.mkdir(parents=True, exist_ok=False)
    (out/'before').mkdir()
    provenance = []
    paths = [Path('app/key_coherence.py'), Path('app/evaluate_key_coherence.py'),
             Path('tests/test_key_coherence.py'), Path(__file__)]
    paths += [BACKEND/n for n in ['key_from_chords.py', 'key_decision.py', 'key_layers.py',
                                  'web/retained_worker.py']]
    paths += [Path('D:/ChordStudio/research/key-v3')/n for n in
              ['LAYERS_REPORT.md','FAILURE_ANALYSIS.md','failure_analysis.py','layers.py']]
    paths += [Path('docs/key_coherence_review_20261031.md'),
              Path('D:/ChordStudio/research/key-coherence-v4/replay.json')]
    for path in paths:
        target = out/'before'/path.name
        shutil.copy2(path, target)
        provenance.append(dict(source=str(path.resolve()), snapshot=str(target.relative_to(out)), sha256=sha(target)))
    parser = module(out/'before/key_from_chords.py', 'frozen_parser')
    inputs, exclusions = [], []
    for path in sorted(CASES.glob('*/case.json')):
        case = json.loads(path.read_text(encoding='utf-8'))
        raw = path.parent/'raw.lab'
        if not raw.exists():
            exclusions.append(dict(path=str(path.resolve()), reason='missing raw.lab'))
            continue
        inputs.append(dict(id=case['id'], split=case['split'], reference=case['reference'],
                           rows=parser.load_lab(str(raw)), candidates=case['evidence']['madmom_cnn']['candidates'],
                           case_sha256=sha(path), raw_sha256=sha(raw)))
    songs = []
    for pid in PIDS:
        p = PROJECTS/pid/'project.json'
        project = json.loads(p.read_text(encoding='utf-8'))
        folder = PROJECTS/pid/'runs'/project['current_run']
        analysis = json.loads((folder/'analysis.json').read_text(encoding='utf-8'))
        key = json.loads((folder/'key.json').read_text(encoding='utf-8'))
        # The historical third project has v1, not retained10_stable.
        # Explicit selection avoids silently evaluating a different model source.
        source_id = 'v1' if pid == '087d8b0c86414fe6b675cc622f505128' else 'retained10_stable'
        source = next(s for s in analysis['sources'] if s['id']==source_id)
        songs.append(dict(id=pid,title=project['title'],run_id=project['current_run'],
                          source_id=source['id'], rows=source['rows'],candidates=key['candidates'],
                          project_sha256=sha(p),analysis_sha256=sha(folder/'analysis.json'),
                          key_sha256=sha(folder/'key.json')))
    write(out/'inputs.json',dict(cases=inputs,songs=songs,exclusions=exclusions))
    write(out/'manifest.json',dict(status='exposed_regression_not_held_out',code=provenance,
                                   input_sha256=sha(out/'inputs.json'),python=sys.version))
    print(json.dumps(dict(path=str(out),cases=len(inputs),songs=len(songs),exclusions=exclusions)))

def compare(out):
    if (out/'comparison.json').exists() or (out/'key_coherence_after.py').exists():
        raise FileExistsError('Comparison already exists; use a new experiment directory')
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    assert sha(out/'inputs.json')==manifest['input_sha256']
    for entry in manifest['code']:
        assert sha(out/entry['snapshot'])==entry['sha256'], entry['snapshot']
    before=module(out/'before/key_coherence.py','before_coherence')
    after=module(Path('app/key_coherence.py'),'after_coherence')
    sys.path.insert(0,str(out/'before'))
    import key_from_chords as parser
    import key_decision as production
    data=json.loads((out/'inputs.json').read_text(encoding='utf-8'))
    result={}
    for group in ['cases','songs']:
        records=[]
        for row in data[group]:
            chords,stats=parser.parse_rows(row['rows'])
            ref=after.key_id(row.get('reference'))
            if group=='cases' and ref is None:raise ValueError(row['id'])
            cnn=max(row['candidates'],key=lambda r:r['score'])['key']
            inferred=parser.infer(chords)
            ck,guard=production.guard_456(chords,parser.key_name(*inferred['pred']))
            decision=production.decide(cnn,ck,inferred['pred_pair']['margin'],chords)
            record={k:v for k,v in row.items() if k not in ('rows','candidates')}
            record.update(cnn=cnn,parse_stats=stats,production=ck if decision['use_chord'] else cnn,
                          guard=guard,decision=decision)
            for tag,impl in [('before',before),('after',after)]:
                ranks=impl.rank(chords,row['candidates'])
                record[tag]=ranks[0]['key']
                record[tag+'_ranking']=ranks
            records.append(record)
        result[group]=records
    result['summary']={}
    for split in ['all']+sorted({r['split'] for r in result['cases']}):
        subset=[r for r in result['cases'] if split=='all' or r['split']==split]
        result['summary'][split]={}
        for method in ['cnn','production','before','after']:
            hits=sum(after.key_id(r[method])==after.key_id(r['reference']) for r in subset)
            fixed=[r['id'] for r in subset if after.key_id(r['cnn'])!=after.key_id(r['reference'])==after.key_id(r[method])]
            broken=[r['id'] for r in subset if after.key_id(r['cnn'])==after.key_id(r['reference'])!=after.key_id(r[method])]
            result['summary'][split][method]=dict(hits=hits,n=len(subset),fixed_vs_cnn=fixed,broken_vs_cnn=broken)
    result['changed']=[r['id'] for r in result['cases'] if r['before']!=r['after']]
    result['after_sha256']=sha(Path('app/key_coherence.py'))
    shutil.copy2('app/key_coherence.py',out/'key_coherence_after.py')
    write(out/'comparison.json',result)
    print(json.dumps(dict(summary=result['summary'],changed=result['changed'],
                         songs=[{k:r[k] for k in ['title','cnn','production','before','after']} for r in result['songs']]),ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['freeze','compare'])
    ap.add_argument('directory',type=Path)
    args=ap.parse_args()
    (freeze if args.action=='freeze' else compare)(args.directory)
