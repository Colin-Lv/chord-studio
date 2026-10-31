"""Create inspectable phrase contexts from pinned human structure annotations.

No key prediction is changed. Structural phrase labels are not cadence labels.
PIANO's lowest sounding note is a proxy, not an independently annotated bass part.
"""
import argparse
import csv
from difflib import SequenceMatcher
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pretty_midi
from app.key_coherence import key_id, name

SIDS=['001','002','005','006','064','067','070','143']
POP=Path('D:/ChordStudio/_archive/2026-10-30/benchmarks/raw/pop909/POP909')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_structure(text, units_per_bar=16):
    text=text.strip()
    parts=re.findall(r'([A-Za-z])(\d+)',text)
    if ''.join(a+b for a,b in parts)!=text or not parts:
        raise ValueError('Invalid structure annotation')
    segments=[];position=0
    for label,length in parts:
        length=int(length)
        if length<=0:raise ValueError('Non-positive phrase length')
        end=position+length*units_per_bar
        segments.append(dict(label=label,start=position,end=end,melodic=label.isupper()))
        position=end
    return segments


def read_melody(path):
    notes=[];position=0
    for line in path.read_text().splitlines():
        pitch,duration=map(int,line.split())
        if not 0<=pitch<=127 or duration<=0:raise ValueError(str(path))
        if pitch:notes.append(dict(start=position,end=position+duration,pitch=pitch))
        position+=duration
    return notes,position


def read_chords(path):
    chords=[];position=0
    for line in path.read_text().splitlines():
        match=re.fullmatch(r'(\S+)\s+\[([^\]]*)\]\s+(-?\d+)\s+([\d.]+)\s*',line)
        if match is None:raise ValueError(line)
        duration=float(match[4])*4
        chords.append(dict(start=position,end=position+duration,label=match[1],
                           pcs=[int(p) for p in match[2].split(',') if p.strip()],root=int(match[3])))
        position+=duration
    return chords


def align_melody(music, notes):
    raw=sorted(next(i for i in music.instruments if i.name=='MELODY').notes,key=lambda n:n.start)
    matcher=SequenceMatcher(None,[n['pitch'] for n in notes],[n.pitch for n in raw],autojunk=False)
    offsets=[]
    for a,b,size in matcher.get_matching_blocks():
        for i in range(size):offsets.append(music.time_to_tick(raw[b+i].start)/music.resolution*4-notes[a+i]['start'])
    if not offsets:return dict(accepted=False,reason='no matching pitch sequence')
    center=float(np.median(offsets));residual=np.abs(np.array(offsets)-center)
    coverage=len(offsets)/max(len(raw),len(notes))
    p95=float(np.quantile(residual,.95))
    return dict(offset_sixteenths=center,matched_fraction=coverage,residual_p95_sixteenths=p95,
                residual_max_sixteenths=float(max(residual)),accepted=coverage>=.95 and p95<=1,
                method='pitch-sequence matches; median tick offset; not a forced bar alignment')


def phrase_contexts(segments, notes, melody_end, chords):
    result=[]
    for segment in segments:
        if not segment['melodic']:continue
        present=[n for n in notes if n['start']<segment['end'] and n['end']>segment['start']]
        if not present:
            result.append(dict(**segment,status='no_MELODY_note'));continue
        last=present[-1]
        at=min(last['end'],segment['end'])-1e-6
        chord=next((c for c in chords if c['start']<=at<c['end']),None)
        status='crosses_boundary' if last['end']>segment['end'] else 'tail_not_encoded' if melody_end<segment['end'] else 'observable'
        result.append(dict(**segment,status=status,ending=last,pc=last['pitch']%12,
            rest_to_boundary_sixteenths=max(0,segment['end']-last['end']),
            previous_pitches=[n['pitch'] for n in present[-4:]],chord_at_note_end=chord))
    return result


def plot_contexts(path, sid, notes, chords, contexts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=[c for c in contexts if 'ending' in c][:2]
    fig,axes=plt.subplots(len(chosen),1,figsize=(11,3*len(chosen)),squeeze=False)
    for ax,c in zip(axes[:,0],chosen):
        left=max(c['start'],c['end']-32);right=c['end']+4
        for note in notes:
            if note['start']<right and note['end']>left:
                ax.plot([max(left,note['start'])/4,min(right,note['end'])/4],[note['pitch']]*2,
                        color='#167b7b',linewidth=3,solid_capstyle='butt')
        for h in chords:
            if h['start']<right and h['end']>left:
                ax.axvline(max(left,h['start'])/4,color='#bbbbbb',linewidth=.5)
                ax.text(max(left,h['start'])/4,.98,h['label'],transform=ax.get_xaxis_transform(),fontsize=8,rotation=30,va='top')
        ax.axvline(c['end']/4,color='#b64230',linestyle='--')
        ax.set_title(f'{sid}: human label 1, {c["label"]}, boundary={c["end"]/4:g} quarter beats; {c["status"]}')
        ax.set(xlim=(left/4,right/4),ylabel='MIDI pitch',xlabel='Processed quarter-beat position')
        ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def run(root):
    source=root/'structure-source';out=root/'structure-packet';out.mkdir(exist_ok=False)
    source_manifest=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
    for record in source_manifest['files']:assert sha(source/record['path'])==record['sha256']
    code=(source/'seg.cpp').read_text()
    triples=set(map(int,re.search(r'time_signature_inThree\s*=\s*\{([^}]+)',code)[1].split(',')))
    manifest={r['id']:r for r in csv.DictReader(Path('benchmarks/manifest.pop909_synth.csv').open(encoding='utf-8-sig'))}
    prior=json.loads(Path('D:/ChordStudio/research/key-audit-20261031-r1/melody-probe/results.json').read_text(encoding='utf-8'))
    old={r['id']:r for r in prior['rows']}
    records=[]
    for sid in SIDS:
        folder=source/'POP909'/sid
        notes,melody_end=read_melody(folder/'melody.txt');chords=read_chords(folder/'finalized_chord.txt')
        midi=POP/sid/(sid+'.mid');music=pretty_midi.PrettyMIDI(str(midi))
        alignment=align_melody(music,notes)
        annotations={}
        for version in [1,2]:
            segments=parse_structure((folder/f'human_label{version}.txt').read_text(),12 if int(sid) in triples else 16)
            annotations[str(version)]=phrase_contexts(segments,notes,melody_end,chords)
        # Only expose original-MIDI seconds/bass where the offset model passed.
        for contexts in annotations.values():
            for c in contexts:
                if 'ending' not in c or not alignment['accepted']:continue
                coordinate=min(c['ending']['end'],c['end'])+alignment['offset_sixteenths']
                tick=max(0,int(round(coordinate*music.resolution/4)))
                seconds=float(music.tick_to_time(tick))
                active=[n.pitch for i in music.instruments if i.name=='PIANO' for n in i.notes if n.start<seconds and n.end>=seconds-1e-4]
                c.update(original_midi_seconds=seconds,lowest_sounding_piano_pitch=min(active) if active else None,
                         bass_evidence_scope='lowest sounding PIANO note proxy; not confirmed bass voice')
        ref=key_id(manifest['pop909_'+sid]['key'])
        alt=((ref[0]+9)%12,'minor') if ref[1]=='major' else ((ref[0]+3)%12,'major')
        comparison={}
        for version,contexts in annotations.items():
            observable=[c for c in contexts if c['status']=='observable']
            counts={name(*k):sum(c['pc']==k[0] for c in observable) for k in [ref,alt]}
            delta=counts[name(*ref)]-counts[name(*alt)]
            comparison[version]=dict(counts=counts,observable=len(observable),
                preference='reference' if delta>0 else 'relative' if delta<0 else 'tie',
                note='given candidate pair diagnostic, not 24-class key accuracy')
        plot_contexts(out/(sid+'.png'),sid,notes,chords,annotations['1'])
        records.append(dict(id='pop909_'+sid,reference=name(*ref),relative=name(*alt),
            group='old_failure' if 'old_failure' in old['pop909_'+sid] else 'control',
            annotations=annotations,comparison=comparison,alignment=alignment,
            original_time_signatures=[dict(n=t.numerator,d=t.denominator,time=float(t.time)) for t in music.time_signature_changes],
            source_units_per_bar=12 if int(sid) in triples else 16,source_seg_sha256=sha(source/'seg.cpp'),
            original_midi_sha256=sha(midi),upstream_commit=source_manifest['commit']))
    result=dict(status='diagnostic_not_training_labels_or_key_predictions',records=records,
                implementation_sha256=sha(Path(__file__)))
    (out/'packet.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([dict(id=r['id'],alignment=r['alignment'],comparison=r['comparison']) for r in records],ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);run(ap.parse_args().root)
