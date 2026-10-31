"""Frozen-input, isolated repair of the installed key rules and gate ablations.

prepare creates an immutable experiment; worker imports only its chosen copy.
No installed files or user projects are changed.
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

from app.key_coherence import key_id

SNAPSHOT = Path('D:/ChordStudio/research/key-audit-20261031-r1')
POP_ROOT = Path('D:/ChordStudio/_archive/2026-10-30/benchmarks/raw/pop909/POP909')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Patch precondition failed: ' + old[:100])
    return text.replace(old, new)


def build_repair(folder):
    decision = (folder/'key_decision.py').read_text(encoding='utf-8')
    decision = replace_once(decision, 'NOTE_PC =',
        'from key_transition_semantics import major_minor_flag, dominant_arrivals\n\nNOTE_PC =')
    decision = replace_once(decision, '        minor = "min" in chord.label',
        '        minor = major_minor_flag(chord)\n        if minor is None:\n            continue')
    start = decision.index('    count = 0\n', decision.index('def _authentic_cadences'))
    end = decision.index('\n\n\ndef guard_456', start)
    decision = decision[:start] + '    return len(dominant_arrivals(chords, minor_tonic, "minor"))\n' + decision[end:]
    decision = decision.replace('authentic V-i cadence(s)', 'adjacent V-i arrival(s)')
    decision = decision.replace('not a 4-5-6 idiom', 'not a 4-5-6 inventory')
    decision = decision.replace('4-5-6 idiom without V-i', '4-5-6 inventory without adjacent V-i')
    decision = replace_once(decision, '    return dict(use_chord=use_chord, cnn_key=cnn_key, chord_key=chord_key,',
        '    return dict(use_chord=use_chord, cnn_key=cnn_key, chord_key=chord_key,\n'
        '                coverage_ok=coverage_ok, veto=veto, force=force,')
    # Replace misleading introductory claims, not the thresholds or decisions.
    stop = decision.index('from __future__')
    decision = ('"""Experimental repair of legacy decision semantics.\n'
                'The 456 inventory heuristic and coverage/veto/force policy are retained\n'
                'for controlled ablation, not endorsed as validated music theory.\n"""\n' + decision[stop:])
    (folder/'key_decision.py').write_text(decision, encoding='utf-8')

    layers = (folder/'key_layers.py').read_text(encoding='utf-8')
    layers = replace_once(layers, 'from collections import defaultdict',
        'from collections import defaultdict\nfrom key_transition_semantics import tonic_triad, major_minor_flag, dominant_arrivals')
    layers = replace_once(layers,
        'if chord.root == tonic and (chord.kind in ("min",) if mode == "minor" else chord.kind not in ("min",)):',
        'if tonic_triad(chord, tonic, mode):')
    layers = replace_once(layers, 'signature[(degree, accidental, chord.kind in ("min",))]',
        'signature[(degree, accidental, major_minor_flag(chord))]')
    layers = layers.replace('c.kind == "min"', 'major_minor_flag(c) is True')
    layers = replace_once(layers, 'cadences = _cadences(chords, tonic)', 'cadences = _cadences(chords, tonic, mode)')
    layers = replace_once(layers, 'minor_cadence = _cadences(chords, (tonic + 9) % 12)',
        'minor_cadence = _cadences(chords, (tonic + 9) % 12, "minor")')
    start = layers.index('def _cadences(')
    end = layers.index('\n\ndef adjust(', start)
    layers = layers[:start] + ('def _cadences(chords, tonic: int, mode: str) -> int:\n'
        '    """Legacy field name; only counts adjacent harmonic arrivals."""\n'
        '    return len(dominant_arrivals(chords, tonic, mode))\n') + layers[end:]
    layers = replace_once(layers, '    return dict(key=key, penalties=penalties,',
        '    return dict(key=key, pattern_scope="unordered_inventory",\n'
        '                cadence_scope="adjacent_harmony_only", penalties=penalties,')
    # Fix a pre-existing argument-count error in the optional, unused adjust API.
    layers = layers.replace('_pc_mode(chord_key, NAMES)', '_pc_mode(chord_key)')
    layers = layers.replace('_pc_mode(cnn_key, NAMES)', '_pc_mode(cnn_key)')
    (folder/'key_layers.py').write_text(layers, encoding='utf-8')
    shutil.copy2('app/key_transition_semantics.py', folder/'key_transition_semantics.py')


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((SNAPSHOT/'manifest.json').read_text(encoding='utf-8'))
    assert sha(SNAPSHOT/'inputs.json') == manifest['input_sha256']
    for entry in manifest['code']:
        assert sha(SNAPSHOT/entry['snapshot']) == entry['sha256']
    shutil.copy2(SNAPSHOT/'inputs.json', out/'inputs.json')
    for arm in ['baseline', 'repaired']:
        folder = out/arm
        folder.mkdir()
        for filename in ['key_from_chords.py', 'key_decision.py', 'key_layers.py']:
            shutil.copy2(SNAPSHOT/'before'/filename, folder/filename)
    build_repair(out/'repaired')
    pop = []
    source = Path('benchmarks/manifest.pop909_synth.csv')
    for rec in csv.DictReader(source.open(encoding='utf-8-sig')):
        lab = Path('benchmarks/imported/pop909')/rec['id']/'chords.lab'
        keys = POP_ROOT/rec['id'].split('_')[-1]/'key_audio.txt'
        events = [line.split() for line in keys.read_text().splitlines() if line.strip()]
        unique = {key_id(row[2]) for row in events}
        if None in unique or key_id(rec['key']) is None:
            raise ValueError(rec['id'])
        pop.append(dict(id=rec['id'],reference=rec['key'],single_key=len(unique)==1,
                        rows=[line.split(maxsplit=2) for line in lab.read_text().splitlines() if line.strip()],
                        lab_sha256=sha(lab),key_sha256=sha(keys),key_events=events))
    write(out/'pop_inputs.json', pop)
    protocol = dict(status='exposed_regression_not_blind_test',
        arms=['baseline', 'repaired', 'repaired_no_guard', 'repaired_no_force', 'repaired_neither'],
        thresholds='unchanged; no fitted weights',
        pop_task='agreement with imported longest-segment reference; separately report single-key songs',
        input_sha256=sha(out/'inputs.json'),pop_sha256=sha(out/'pop_inputs.json'),
        code={str(p.relative_to(out)):sha(p) for arm in ['baseline','repaired'] for p in (out/arm).glob('*.py')})
    write(out/'protocol.json', protocol)
    print('Prepared frozen 147 cases, 3 project versions and 909 symbolic records:', out)


def worker(out, arm):
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    assert sha(out/'inputs.json') == protocol['input_sha256']
    assert sha(out/'pop_inputs.json') == protocol['pop_sha256']
    for path, digest in protocol['code'].items():
        assert sha(out/path) == digest
    sys.path.insert(0, str((out/arm).resolve()))
    import key_from_chords as kc
    import key_decision as kd
    data = json.loads((out/'inputs.json').read_text(encoding='utf-8'))
    result = {}
    for group in ['cases', 'songs']:
        records = []
        for row in data[group]:
            chords, stats = kc.parse_rows(row['rows'])
            inference = kc.infer(chords)
            raw = kc.key_name(*inference['pred'])
            guarded, guard = kd.guard_456(chords, raw)
            cnn = max(row['candidates'], key=lambda c:c['score'])['key']
            record = {k:v for k,v in row.items() if k not in ('rows','candidates')}
            record.update(cnn=cnn,raw_chord=raw,guard=guard,parse_stats=stats,methods={})
            combinations = [(True, True)] if arm=='baseline' else [(True,True),(False,True),(True,False),(False,False)]
            for use_guard, use_force in combinations:
                ck = guarded if use_guard else raw
                decision = kd.decide(cnn, ck, inference['pred_pair']['margin'], chords)
                use_chord = decision['use_chord'] if use_force else decision['coverage_ok'] and not decision['veto']
                label = arm if use_guard and use_force else ('repaired_neither' if not use_guard and not use_force else
                        'repaired_no_guard' if not use_guard else 'repaired_no_force')
                record['methods'][label] = dict(key=ck if use_chord else cnn,decision=decision,
                                               effective_use_chord=use_chord,guard_enabled=use_guard,force_enabled=use_force)
            records.append(record)
        result[group] = records
    pop_records = []
    for row in json.loads((out/'pop_inputs.json').read_text(encoding='utf-8')):
        chords,stats=kc.parse_rows(row['rows'])
        ev=kc.infer(chords);raw=kc.key_name(*ev['pred'])
        guarded,guard=kd.guard_456(chords,raw)
        pop_records.append(dict(id=row['id'],reference=row['reference'],single_key=row['single_key'],
                                raw=raw,guarded=guarded,guard=guard,parse_stats=stats))
    result['pop']=pop_records
    write(out/(arm+'_results.json'),result)
    print(arm, 'completed', len(result['cases']),len(pop_records))


def summarize(out):
    base=json.loads((out/'baseline_results.json').read_text(encoding='utf-8'))
    fixed=json.loads((out/'repaired_results.json').read_text(encoding='utf-8'))
    result=dict(recordings={},pop={},songs=[])
    for dataset in [base,fixed]:
        methods=list(dataset['cases'][0]['methods'])
        for method in methods:
            rows=dataset['cases']
            hits=lambda r:key_id(r['methods'][method]['key'])==key_id(r['reference'])
            result['recordings'][method]=dict(hits=sum(hits(r) for r in rows),n=len(rows),
                fixed_vs_cnn=[r['id'] for r in rows if hits(r) and key_id(r['cnn'])!=key_id(r['reference'])],
                broken_vs_cnn=[r['id'] for r in rows if not hits(r) and key_id(r['cnn'])==key_id(r['reference'])])
    for tag,dataset in [('baseline',base),('repaired',fixed)]:
        result['pop'][tag]={}
        for group in ['all','single_key','multi_key']:
            rows=[r for r in dataset['pop'] if group=='all' or r['single_key']==(group=='single_key')]
            result['pop'][tag][group]={m:dict(hits=sum(key_id(r[m])==key_id(r['reference']) for r in rows),n=len(rows)) for m in ['raw','guarded']}
    for b,f in zip(base['songs'],fixed['songs']):
        assert b['id']==f['id']
        result['songs'].append(dict(id=b['id'],title=b['title'],cnn=b['cnn'],
            methods={k:v['key'] for k,v in (b['methods']|f['methods']).items()}))
    result['recording_changes']=[dict(id=b['id'],reference=b['reference'],before=b['methods']['baseline']['key'],after=f['methods']['repaired']['key'])
            for b,f in zip(base['cases'],fixed['cases']) if b['methods']['baseline']['key']!=f['methods']['repaired']['key']]
    write(out/'summary.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','worker','summarize']);ap.add_argument('output',type=Path)
    ap.add_argument('--arm',choices=['baseline','repaired'])
    a=ap.parse_args()
    if a.action=='prepare':prepare(a.output)
    elif a.action=='summarize':summarize(a.output)
    else:worker(a.output,a.arm)
