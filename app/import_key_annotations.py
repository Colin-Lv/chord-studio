"""Validate JSONL/CSV listening labels and create a new audit-preserving import.

Does not modify any historical split, train models, or assign calibration/test.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from app.web.storage import now, signature


def validate(row):
    for field in ('item_id','audio_sha256','recording_id','reviewer','basis','label_scope','annotation_status','prediction_visible'):
        if field not in row or row[field] in (None,''):raise ValueError('Missing '+field)
    if len(row['audio_sha256'])!=64 or any(c not in '0123456789abcdef' for c in row['audio_sha256']):raise ValueError('Invalid audio SHA256')
    a,b=row.get('excerpt_start'),row.get('excerpt_end')
    if isinstance(a,bool) or isinstance(b,bool) or not isinstance(a,(int,float)) or not isinstance(b,(int,float)) or not math.isfinite(a+b) or not 0<=a<b:raise ValueError('Invalid excerpt')
    if row.get('mode') not in (None,'','major','minor'):raise ValueError('Unknown mode family')
    if row['label_scope'] not in ('full_recording','excerpt'):raise ValueError('Unknown label scope')
    if row['annotation_status'] not in ('single_key','ambiguous','multi_key','unknown'):raise ValueError('Unknown annotation status')
    if row['annotation_status']=='single_key' and not row.get('tonic'):raise ValueError('Single-key annotation needs tonic')
    if row.get('tonic') not in (None,'','C','C#','Db','D','D#','Eb','E','F','F#','Gb','G','G#','Ab','A','A#','Bb','B'):raise ValueError('Invalid tonic')
    if not isinstance(row['prediction_visible'],bool):raise ValueError('prediction_visible must be boolean, never inferred from agreement')
    if row.get('data_role') in ('calibration','test','sealed'):raise ValueError('Import cannot certify untouched group isolation; assign a role only with a separate frozen allocation')
    return {**row,'data_role':'pending_review_no_evaluation_role','independent_gold':False,
            'role_history':row.get('role_history',[])+[dict(at=now(),role='pending_review_no_evaluation_role',reason='Validated import; provenance/group isolation still requires audit')]}


def annotation_rows(annotation,all_latest_ranges=False):
    """Explicitly select ranges; never interpret repeated locks as new recordings."""
    def row(event):
        if (not isinstance(event,dict) or not isinstance(event.get('label'),dict)
                or type(event.get('revision')) is not int or event['revision']<1
                or type(event.get('prediction_visible_before_lock')) is not bool):
            raise ValueError('Invalid listening revision or exposure state')
        value=dict(event['label'],item_id=annotation.get('project_id','imported'),audio_sha256=annotation['audio_sha256'],
                   prediction_visible=event['prediction_visible_before_lock'],annotation_revision=event['revision'],
                   annotation_run_id=annotation.get('run_id'),recording_group_id=annotation['audio_sha256'],
                   prior_exposure_unknown=event.get('prior_exposure_unknown',True),
                   prior_exposure_self_report=event.get('prior_exposure_self_report','unknown'),
                   entry_context=event.get('entry_context'),based_on_revision=event.get('based_on_revision'))
        for field in ('recording_id','reviewer','basis'):
            if not isinstance(value.get(field),str) or not value[field].strip():raise ValueError('Invalid '+field)
        validate(value)
        return value
    locked=annotation.get('locked_label')
    if not all_latest_ranges:
        return [dict(row(locked),original_history=annotation['history'])]
    history=annotation.get('history')
    if not isinstance(history,list) or not history:raise ValueError('No listening history for multi-range selection')
    latest={};revisions={};previous=0;last_lock=None
    for event in history:
        if (not isinstance(event,dict) or type(event.get('revision')) is not int or event['revision']<=previous
                or event.get('action') not in ('lock','reveal')):
            raise ValueError('History revisions must remain ordered and unique')
        previous=event['revision']
        if event['action']!='lock':continue
        value=row(event);key=(value['recording_id'],value['excerpt_start'],value['excerpt_end'])
        latest[key]=value;revisions.setdefault(key,[]).append(event['revision']);last_lock=event
    if not latest or last_lock!=locked or annotation.get('revision')!=previous:
        raise ValueError('History and current locked revision disagree')
    return [dict(latest[key],related_revisions=revisions[key],annotation_counting_unit='latest_per_exact_range_not_independent_recording') for key in sorted(latest)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--all-latest-ranges',action='store_true',help='For an annotation JSON, import the latest lock per exact recording/range; preserve shared history, keep all rows pending review')
    args=p.parse_args()
    provenance_path=args.output.with_suffix('.provenance.json')
    if args.output.exists() or provenance_path.exists():raise ValueError('Refusing to overwrite existing import or provenance')
    if args.all_latest_ranges and args.input.suffix.lower()!='.json':raise ValueError('--all-latest-ranges requires an annotation JSON')
    annotation=None
    if args.input.suffix.lower()=='.csv':
        with args.input.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        for row in rows:
            for name in ('excerpt_start','excerpt_end'):row[name]=float(row[name])
            value=row['prediction_visible'].lower()
            if value not in ('true','false'):raise ValueError('Explicit CSV prediction_visible true/false required')
            row['prediction_visible']=value=='true'
            if row.get('role_history'):row['role_history']=json.loads(row['role_history'])
    else:
        obj=args.input.read_text(encoding='utf-8-sig')
        if args.input.suffix.lower()=='.json':
            annotation=json.loads(obj);rows=annotation_rows(annotation,args.all_latest_ranges)
        else:rows=[json.loads(line) for line in obj.splitlines() if line.strip()]
    validated=[validate(row) for row in rows]
    shared={}
    if args.all_latest_ranges:
        history=annotation['history']
        history_hash=hashlib.sha256(json.dumps(history,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
        shared=dict(annotation_history=history,annotation_history_sha256=history_hash,
                    annotation_identity={k:annotation.get(k) for k in ('schema','project_id','run_id','audio_sha256','duration','revision')})
        for row in validated:row.update(annotation_history_ref=provenance_path.name,annotation_history_sha256=history_hash)
    if len({(r['item_id'],r['recording_id'],r['excerpt_start'],r['excerpt_end']) for r in validated})!=len(validated):raise ValueError('Duplicate annotation ranges')
    serialized='\n'.join(json.dumps(r,ensure_ascii=False,allow_nan=False) for r in validated)+'\n'
    if provenance_path.exists():raise ValueError('Refusing to overwrite existing provenance')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # JSONL contains fields in their native types; source stays untouched.
    with args.output.open('x',encoding='utf-8') as f:f.write(serialized)
    provenance=dict(source=signature(args.input),source_path=str(args.input),
                output=signature(args.output),rows=len(validated),imported_at=now(),new_training=False,
                selection_policy='latest_per_exact_range' if args.all_latest_ranges else 'last_locked_label' if annotation is not None else 'provided_rows',
                distinct_audio_identity_count=len({r['audio_sha256'] for r in validated}),
                count_note='Distinct audio byte identities only; does not establish independent musical works or adjudicated truth',**shared)
    # Exclusive creation also protects a companion file created after preflight.
    with provenance_path.open('x',encoding='utf-8') as f:json.dump(provenance,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(dict(imported=len(validated),independent_gold=0,training=False))


if __name__=='__main__':main()
