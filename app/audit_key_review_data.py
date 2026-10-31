"""Read frozen predictions, audit finite candidates, inventory future data roles.

References are joined only for this offline audit; no inference or fitting.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

from app.web.storage import ROOT, atomic_json, now, read_json, signature
from app.key_review_adapter import contributions

FIELDS = 'item_id audio_sha256 recording_id work_group artist_group source license_status excerpt_start excerpt_end domain original_mode_label tonic mode label_scope annotation_status reviewer prediction_visible data_role role_history group_id'.split()
NOTES=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
PC={n:i for i,n in enumerate(NOTES)} | dict(Db=1,Eb=3,Gb=6,Ab=8,Bb=10)


def canonical(key):
    if not key:return None
    minor=key.endswith('m');root=key[:-1] if minor else key
    return NOTES[PC[root]]+('m' if minor else '')


def verify(identity):
    actual=signature(identity['path'])
    if actual['sha256']!=identity['sha256']:raise ValueError('Input hash changed: '+identity['path'])


def audit(frozen, output):
    report=read_json(frozen/'harmonic_new_sealed.json');rows=[];groups=defaultdict(list)
    labels={row['id']:row for row in read_json(frozen/'split.json')['rows']}
    for row in report['rows']:
        for value in row['inputs'].values():verify(value)
        verify(row['audio'])
        cnn=read_json(row['inputs']['cnn']['path']);p=row['prediction']
        c=[canonical(x['key']) for x in cnn['candidates']];h=[canonical(x['key']) for x in p['final_key_candidates']]
        ref=canonical(row['reference']);cc=c[0]==ref;hc=h[0]==ref
        group=('both_correct' if cc and hc else 'cnn_only' if cc else 'harmonic_only' if hc else 'both_wrong')
        groups[group].append(row['id'])
        neutral=read_json(row['inputs']['neutral']['path'])
        clock=read_json(row['inputs']['profile']['path'])
        rank=lambda candidates: candidates.index(ref)+1 if ref in candidates else None
        item=dict(id=row['id'],group=group,reference=ref,genre=row['genre'],cnn_top6=c[:6],harmonic_top6=h[:6],
                  cnn_rank=rank(c),harmonic_rank=rank(h),prediction_agrees=c[0]==h[0],
                  fixed_top3_union=list(dict.fromkeys(c[:3]+h[:3])),
                  cnn_tonic_correct=c[0].removesuffix('m')==ref.removesuffix('m'),
                  harmonic_tonic_correct=h[0].removesuffix('m')==ref.removesuffix('m'),
                  input_time_delta=neutral['duration']-clock['duration'],
                  annotation_scope='Accepted full excerpt major/minor family; original subtype retained in split; no new human verification',
                  input_hashes_validated=True,required_grid=dict(duration=neutral['duration'],bars=neutral.get('bars',[])),
                  tonic_candidates_retained=len(p['tonic_candidates']),full_keys_retained=len(h))
        item['reference_outside_top3_union']=ref not in item['fixed_top3_union']
        item['original_mode_label']=labels[row['id']].get('original_key')
        item['annotation_confidence']=labels[row['id']].get('annotation_confidence')
        item['reference_tonic_rank']=next(x['rank'] for x in p['tonic_candidates'] if x['tonic']==ref.removesuffix('m'))
        item['reference_conditional_mode_rank']=next(x['rank'] for x in p['mode_candidates_by_tonic'][ref.removesuffix('m')] if x['mode']==('minor' if ref.endswith('m') else 'major'))
        rows.append(item)
        if group!='both_correct':
            atomic_json(output/'evidence'/group/(row['id']+'.json'),dict(audit=item,cnn_evidence=cnn,
                        harmonic_evidence=p,weighted_contributions=contributions(p),original_row=row,reference_used_only_for_offline_audit=True))
    stats={}
    for source,field in [('cnn','cnn_top6'),('harmonic_v2','harmonic_top6')]:
        for k in [1,2,3,6]:
            stats[f'{source}_top{k}']=dict(hits=sum(x['reference'] in x[field][:k] for x in rows),n=len(rows),candidate_budget=k)
    sizes=Counter(len(x['fixed_top3_union']) for x in rows)
    stats['fixed_top3_union']=dict(hits=sum(x['reference'] in x['fixed_top3_union'] for x in rows),n=len(rows),
                 maximum_budget=6,actual_counts=dict(sizes),mean_candidates=sum(len(x['fixed_top3_union']) for x in rows)/len(rows),
                 note='Candidate coverage only, not top1 accuracy; single-source top6 has the same maximum budget.')
    stats['matched_actual_union_budget']={method:dict(hits=sum(x['reference'] in x[field][:len(x['fixed_top3_union'])] for x in rows),n=len(rows),
          budget='For each song, same actual candidate count as the fixed top3 union') for method,field in [('cnn','cnn_top6'),('harmonic_v2','harmonic_top6')]}
    billboard=read_json(frozen/'billboard.json')['overall']['tonic_change_detection']
    fp=billboard['reference:False predicted:True'];tn=billboard['reference:False predicted:False'];tp=billboard['reference:True predicted:True'];fn=billboard['reference:True predicted:False']
    result=dict(schema='frozen_key_candidate_audit_v1',purpose='known regression diagnostic; no model selection',
         frozen_report=signature(frozen/'harmonic_new_sealed.json'),groups=dict(groups),counts={k:len(v) for k,v in groups.items()},
         agreement=dict(Counter('agree' if x['prediction_agrees'] else 'disagree' for x in rows)),topk=stats,
         agreement_by_correctness={name:dict(Counter('agree' if x['prediction_agrees'] else 'disagree' for x in rows if x['group']==name)) for name in groups},
         hierarchy=dict(all_twelve_tonics_retained=all(x['tonic_candidates_retained']==12 for x in rows),
                        all_twenty_four_keys_retained=all(x['full_keys_retained']==24 for x in rows),
                        ranking='All modes of rank1 tonic, then all modes of rank2 tonic, etc. Not a flat sum of uncalibrated cross-source scores.'),
         metrics=report['summary'],billboard=dict(TP=tp,FP=fp,TN=tn,FN=fn,false_positive_rate=fp/(fp+tn),
                  precision=tp/(tp+fp),recall=tp/(tp+fn),scope='whether reference tonic changes; no audio, full mode or precise modulation boundary truth'),
         negative_ablations='Frozen dev/cal loop and local/Viterbi did not add full-key hits; cadence lost one dev hit. No revised weights tested here.',
         hypothetical_v2_top1_override=dict(corrections=2,harmful_changes=11,net_correct=-9,performed=False),
         rows=rows,new_acoustic_inference=False,new_training=False)
    atomic_json(output/'candidate_audit.json',result)
    return result


def inventory(frozen, output):
    registry=[];gaps=[]
    def add(item):
        if not item.get('audio_path') or not Path(item['audio_path']).is_file():
            gaps.append(dict(item_id=item['item_id'],reason='missing_audio',audio_path=item.get('audio_path')));return
        item['audio_sha256']=signature(item['audio_path'])['sha256']
        item.setdefault('recording_id',item['audio_sha256']);item.setdefault('role_history',[])
        item.update(data_role='retired_known_development' if item.get('known') else 'pending_review_no_evaluation_role',
                    group_id=item.get('group_id'),prediction_visible=bool(item.get('known')))
        if not item['group_id']:
            item['group_id']=None  # Unknown artist/work cannot establish independence.
            gaps.append(dict(item_id=item['item_id'],reason='work_artist_version_group_not_verified'))
        item['role_history'].append(dict(at=None,role=item.get('historical_split') or 'known_regression' if item.get('known') else 'unassigned',history_source=item['source']))
        item['role_history'].append(dict(at=now(),role=item['data_role'],reason='New role registry only; historical splits unchanged',training_in_this_round=False))
        if item.get('annotation_status')!='historical_official_family_label':
            gaps.append(dict(item_id=item['item_id'],reason='needs_recording_specific_human_tonic_mode_scope_and_basis'))
        if item['license_status']=='unverified':gaps.append(dict(item_id=item['item_id'],reason='usage_rights_not_verified; no redistribution approval implied'))
        registry.append({k:item.get(k) for k in FIELDS}|{k:v for k,v in item.items() if k not in FIELDS})
    # All prior/new key splits are now known; explicitly retire for future work.
    split=read_json(frozen/'split.json')
    for row in split['rows']:
        if not (row.get('selected') or row.get('harmonic_selected')):continue
        if row.get('harmonic_split') is None and not row.get('selected'):continue
        reference=canonical(row.get('reference'))
        if not reference:continue
        tokens=row.get('group_tokens',[])
        support_paths=[frozen/'acoustics'/row['id']/'raw.lab.provenance.json',ROOT/'benchmarks/reports/key_resolver_20261006/giantsteps'/row['id']/'raw.lab.provenance.json']
        support=next((p for p in support_paths if p.is_file()),None)
        duration=None
        if support:
            identity=read_json(support)
            if signature(row['audio'])['sha256']!=identity['audio']['sha256']:raise ValueError('Inventory audio differs from cached support')
            duration=identity['duration']
        add(dict(item_id='giantsteps_'+row['id'],audio_path=row['audio'],work_group=[t for t in tokens if 'work:' in t],
                 artist_group=[t for t in tokens if t.startswith(('artist:','artist_name:'))],group_id=row['group'],
                 source='GiantSteps+ official archive; previously evaluated',license_status='source_declares_CC_BY_SA_4.0; audio-use scope requires respecting source terms',
                 excerpt_start=0,excerpt_end=duration,domain='EDM/'+row['genre'],original_mode_label=row.get('original_key'),
                 tonic=reference.removesuffix('m'),mode='minor' if reference.endswith('m') else 'major',
                 label_scope='annotated_dataset_excerpt',annotation_status='historical_official_family_label',
                 reviewer='dataset_annotation; not newly reviewed',known=True,
                 historical_split=row.get('harmonic_split') or row.get('split'),
                 support_provenance=signature(support) if support else None,
                 upstream_exposure='old resolver features/gates or harmonic_v2 dev/cal/test inspected; cannot be new independent test'))
    # User song files exist; CSV key is not promoted to recording-specific gold.
    manifest=ROOT/'benchmarks/manifest.user4_corrected.csv'
    with manifest.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            add(dict(item_id='user_'+row['id'],audio_path=str(ROOT/'benchmarks'/row['audio']),
                     work_group=row['title'],artist_group=row['artist'],group_id='user:'+row['artist']+':'+row['title'],
                     source='user diagnostic manifest',license_status='unverified',domain='pop_folk_pending_confirmation',
                     original_mode_label=None,tonic=None,mode=None,label_scope='unknown_recording_excerpt',
                     annotation_status='pending_human_review',reviewer=None,known=True,historical_key=row.get('key'),
                     note='Includes known multi-key/conflicting annotations; do not use historical CSV key as unique complete-key gold.'))
    for path in (ROOT/'projects/web').glob('*/project.json'):
        p=read_json(path)
        if not p.get('audio'):continue
        end=None
        if p.get('current_run'):
            result_path=path.parent/'runs'/p['current_run']/'analysis.json'
            if result_path.exists():end=read_json(result_path).get('duration')
        add(dict(item_id='web_'+p['id'],audio_path=str(path.parent/p['audio']),recording_id=p['audio_identity']['sha256'],
                 work_group=None,artist_group=None,group_id=None,source=p.get('source_url') or 'user upload',
                 license_status='unverified',excerpt_start=0,excerpt_end=end,domain='target_domain_pending_confirmation',
                 tonic=None,mode=None,original_mode_label=None,label_scope='full_decoded_recording' if end else 'unknown',
                 annotation_status='pending_human_review',reviewer=None,known=bool(p.get('current_run')),
                 project_id=p['id'],title=p['title'],prediction_exposure='model output already available; blindness not established'))
    holdings=[]
    for name,domain in [('manifest.isophonics_beatles_all.csv','known_rock'),('manifests/guitarset_v1.1.0/all_eligible_comp.csv','instrumental_guitar')]:
        manifest=ROOT/'benchmarks'/name
        with manifest.open(encoding='utf-8-sig',newline='') as stream:
            entries=list(csv.DictReader(stream))
        present=[x for x in entries if (ROOT/'benchmarks'/x['audio']).is_file()]
        references=[x for x in present if (ROOT/'benchmarks'/x['chord_reference']).is_file()]
        gaps.extend(dict(item_id=x['id'],reason='manifest_audio_missing',audio_path=str(ROOT/'benchmarks'/x['audio'])) for x in entries if x not in present)
        holdings.append(dict(manifest=name,manifest_identity=signature(manifest),entries=len(entries),audio_present=len(present),
             chord_reference_present=len(references),domain=domain,full_key_reference_verified=False,
             resolver_eligible=False,reason='Known upstream rock/chord data or guitar performed sheet-informed chords; no newly verified target-domain full-recording key gold',license_status='not re-audited this round'))
    # New, conservative connected components; never rewrite historical groups.
    parents=list(range(len(registry)));seen={}
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i,row in enumerate(registry):
        tokens=['audio:'+row['audio_sha256'],'recording:'+row['recording_id']]
        for kind,field in [('artist','artist_group'),('work','work_group')]:
            value=row.get(field)
            if value is None:continue
            for token in value if isinstance(value,list) else [value]:
                tokens.append(kind+':'+str(token).strip().casefold())
        if row.get('group_id'):tokens.append('historical_component:'+str(row['group_id']))
        for token in tokens:
            if token in seen:parents[find(i)]=find(seen[token])
            else:seen[token]=i
    members=defaultdict(list)
    for i,row in enumerate(registry):members[find(i)].append(row['item_id'])
    for i,row in enumerate(registry):
        row['historical_group_id']=row['group_id']
        row['group_id']=hashlib.sha256('|'.join(sorted(members[find(i)])).encode()).hexdigest()
        row['group_verification']='declared_metadata_only; aliases and unlisted versions not exhaustively verified'
    # Avoid treating waveform/audio variants as independent material.
    duplicates=defaultdict(list)
    for row in registry:duplicates[row['audio_sha256']].append(row['item_id'])
    by=lambda field:{value:len({json.dumps(x.get('work_group') or x['audio_sha256'],sort_keys=True,ensure_ascii=False) for x in registry if str(x.get(field))==value}) for value in sorted({str(x.get(field)) for x in registry})}
    summary=dict(schema='future_key_data_inventory_v1',items=len(registry),unique_audio=len(duplicates),
                 distinct_work_counts=dict(by_role=by('data_role'),by_domain=by('domain'),by_mode=by('mode')),
                 identity_duplicates={k:v for k,v in duplicates.items() if len(v)>1},
                 new_independent_calibration_items=0,new_independent_test_items=0,new_human_labels=0,
                 correctness='No correctness computed for pending human reviews; complete frozen 33-song correctness groups are in candidate_audit.json.',
                 field_coverage={name:dict(present=sum(x.get(name) not in (None,'',[]) for x in registry),total=len(registry),
                       note='Field present, not new human verification') for name in ['tonic','mode','original_mode_label','reviewer','work_group','artist_group','excerpt_end']},
                 excluded=dict(Billboard='symbolic only; no matched audio or complete mode truth',
                  Beatles='known rock regression / chord training; not new pop/folk calibration',GuitarSet='instrumental guitar domain; chord annotations are not full-recording key gold',
                  POP909='weak model-derived chord labels; not independent human key truth'),
                 counts_are='Work identifiers or audio-hash fallback; aliases/version groups pending, therefore not certified independent work counts',
                 additional_audio_holdings=holdings,
                 giantsteps_holdings=dict(audio_files_present=len(list((ROOT/'benchmarks/external_key_data/audio').glob('*.mp3'))),
                    accepted_family_labels=len(split['rows']),registered_evaluated_items=327,
                    eligible_not_in_registered_known_pool=[row['id'] for row in split['rows'] if not row.get('selected') and not row.get('harmonic_selected')],
                    note='Other eligible EDM files are outside the evaluated role registry, and connected to old components; not silently assigned to independent calibration/test.'),
                 proposed_connected_groups=len(members),group_isolation_certified=False,
                 protocol='group by transitive artist/work/version/audio connections across ALL upstream selection and learning; unverified groups ineligible for cal/test')
    output.mkdir(exist_ok=True)
    (output/'data_roles.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in registry)+'\n',encoding='utf-8')
    with (output/'data_roles.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items() if k in FIELDS} for row in registry)
    atomic_json(output/'data_inventory.json',summary);atomic_json(output/'data_gaps.json',dict(gaps=gaps))
    atomic_json(output/'review_queue.json',dict(items=[x for x in registry if x['annotation_status']=='pending_human_review'],
        sampling='Target pop/folk review queue; not representative calibration/test. Preserve all four correctness groups in known dev.',
        new_calibration_test_pool=[]))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen',type=Path,default=ROOT/'benchmarks/reports/harmonic_key_v2_20261006')
    parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/reports/key_review_data_20261006')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);start=time.perf_counter();cpu=time.process_time()
    a=audit(args.frozen,args.output);d=inventory(args.frozen,args.output)
    import psutil
    memory=psutil.Process().memory_info()._asdict()
    atomic_json(args.output/'audit_execution.json',dict(argv=[__import__('sys').executable,'-B','-m','app.audit_key_review_data',*__import__('sys').argv[1:]],elapsed_seconds=time.perf_counter()-start,
                cpu_seconds=time.process_time()-cpu,process_memory_bytes=memory,memory_scope='audit process only; Windows peak_wset is OS-recorded process working set peak',
                new_acoustic_inference=False,new_training=False,groups=a['counts'],inventory=d['items']))
    print(json.dumps(dict(groups=a['counts'],topk=a['topk'],billboard=a['billboard'],inventory=d),ensure_ascii=False))


if __name__=='__main__':main()
