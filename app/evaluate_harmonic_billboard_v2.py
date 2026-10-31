"""Known Billboard chord/grid-to-tonic diagnostic after harmonic code freeze.

There is no independent mode reference and no original audio in this input.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from chord_studio import parse_chord_name
from evaluate_billboard_key_reasoning import annotation
from evaluate_harmonic_key_v2 import load_contract
from evaluate_global_key_fusion import signature
from harmonic_key_v2 import Policy,analyze
from key_fusion import read_raw_lab


def evaluate(split_path,contract_path,old_report,output):
    contract=load_contract(contract_path);policy=Policy(**contract['policy'])
    manifest=json.loads(Path(split_path).read_text(encoding='utf-8'))
    previous=json.loads(Path(old_report).read_text(encoding='utf-8'));baseline={row['id']:row for row in previous['songs']}
    songs=[];failures=[];began=time.perf_counter()
    for row in manifest['rows']:
        if row['duplicate_of']:continue
        try:
            bars,tonics,duration=annotation(Path(row['salami']).read_text(encoding='utf-8-sig'))
            events=read_raw_lab(row['lab']);last=0.;adjustments=0
            for event in events:
                if event['start']<last:
                    if last-event['start']>1e-6:raise ValueError('Substantive LAB overlap')
                    event['start']=last;adjustments+=1
                last=event['end']
            events=[event for event in events if event['end']>event['start']]
            prediction=analyze(events,duration,bars,policy,include_details=False)
            # Join tonic references only after producing global and local output.
            weights=Counter()
            for segment in tonics:
                if segment['tonic'] is not None:weights[segment['tonic']]+=segment['end']-segment['start']
            if not weights:raise ValueError('Missing known tonic reference')
            target=max(weights,key=weights.get);root=parse_chord_name(prediction['tonic'])[0] if prediction['tonic'] else None
            good=covered=supported_good=supported_covered=0.
            for segment in prediction['window_evidence'].get('segments',[]):
                candidate=parse_chord_name(segment['tonic'])[0]
                for reference in tonics:
                    overlap=max(0.,min(segment['end'],reference['end'])-max(segment['start'],reference['start']))
                    if reference['tonic'] is not None:
                        good+=overlap*(candidate==reference['tonic']);covered+=overlap
                        if segment['status']=='supported':supported_good+=overlap*(candidate==reference['tonic']);supported_covered+=overlap
            support=[segment['tonic'] for segment in prediction['window_evidence'].get('segments',[]) if segment['status']=='supported']
            changes=sum(a!=b for a,b in zip(support,support[1:]))
            ref_changes=sum(a['tonic'] is not None and b['tonic'] is not None and a['tonic']!=b['tonic'] for a,b in zip(tonics,tonics[1:]))
            old=baseline.get(row['id'],{})
            songs.append(dict(id=row['id'],old_partition=row['split'],reference_tonic=target,tonic=prediction['tonic'],
                predicted_mode_unscored=prediction['mode'],correct=root==target,old_correct=old.get('global_tonic_correct'),
                fifth_error=root is not None and (root-target)%12 in {5,7},local_correct_seconds=good,local_covered_seconds=covered,
                supported_correct_seconds=supported_good,supported_covered_seconds=supported_covered,
                reference_seconds=sum(weights.values()),reference_tonic_changes=ref_changes,predicted_tonic_changes=changes,
                roundoff_adjustments=adjustments,inputs=[signature(row['lab']),signature(row['salami'])]))
        except Exception as exc:failures.append(dict(id=row['id'],error=repr(exc)))
    def summary(rows):
        n=len(rows);covered=sum(row['local_covered_seconds'] for row in rows);supported=sum(row['supported_covered_seconds'] for row in rows);reference=sum(row['reference_seconds'] for row in rows)
        return dict(n=n,tonic_accuracy=sum(row['correct'] for row in rows)/n if n else None,
            old_tonic_accuracy=sum(bool(row['old_correct']) for row in rows)/n if n else None,fifth_errors=sum(row['fifth_error'] for row in rows),
            corrected=sum(row['correct'] and row['old_correct'] is False for row in rows),broken=sum(not row['correct'] and row['old_correct'] is True for row in rows),
            local_tonic_accuracy=sum(row['local_correct_seconds'] for row in rows)/covered if covered else None,
            supported_local_accuracy=sum(row['supported_correct_seconds'] for row in rows)/supported if supported else None,
            supported_local_coverage=supported/reference if reference else None,
            tonic_change_detection=dict(Counter(f"reference:{row['reference_tonic_changes']>0} predicted:{row['predicted_tonic_changes']>0}" for row in rows)),
            confusion_matrix=dict(Counter(f"{row['reference_tonic']} -> {row['tonic']}" for row in rows)))
    result=dict(schema='billboard_harmonic_tonic_v2',purpose='known symbolic diagnostic; previous partitions are all regression',
        overall=summary(songs),old_partitions={phase:summary([row for row in songs if row['old_partition']==phase]) for phase in ['development','calibration','sealed_test']},
        songs=songs,failures=failures,mode_accuracy=None,audio_evaluation=False,reference_features='oracle chord annotations + phrase-interpolated grid',
        split=signature(split_path),contract=signature(contract_path),baseline=signature(old_report),elapsed_seconds=time.perf_counter()-began,
        no_threshold_selection=True,no_mode_reference_derived_from_chords=True)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(result['overall']);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['split','contract','old-report','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();evaluate(args.split,args.contract,args.old_report,args.output)
