"""Symbolic chord-to-tonic diagnostic; never an acoustic/full-key evaluation.

Reference chords and phrase-interpolated bars are explicit oracle inputs here.
Tonic labels are withheld from harmonic inference. No CNN is fabricated.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from chord_studio import parse_chord_name
from key_fusion import read_raw_lab, score_timed_chords_by_key, summarize_key_windows
from evaluate_global_key_fusion import signature


def annotation(text):
    lines=[];tonic=None;pending=False;tonics=[]
    for line in text.splitlines():
        match=re.match(r'#\s*tonic:\s*(.*)',line)
        if match:tonic=match[1].strip();pending=True;continue
        match=re.match(r'(\d+(?:\.\d*)?(?:e[-+]?\d+)?)\s+(.+)',line,re.I)
        if not match:continue
        t=float(match[1]);body=match[2]
        if pending:
            root=parse_chord_name(tonic)[0] if tonic and tonic!='?' else None
            tonics.append(dict(start=t,tonic=root,original=tonic));pending=False
        lines.append((t,body))
    bars=[]
    for (start,body),(end,_) in zip(lines,lines[1:]):
        if '|' not in body or end<=start:continue
        count=max(0,body.count('|')-1)
        repeat=re.search(r'\|\s*x\s*(\d+)',body)
        if repeat:count*=int(repeat[1])
        for i in range(count):
            bars.append(dict(bar=len(bars)+1,start=start+(end-start)*i/count,end=start+(end-start)*(i+1)/count))
    duration=lines[-1][0] if lines else 0
    for i,row in enumerate(tonics):row['end']=tonics[i+1]['start'] if i+1<len(tonics) else duration
    return bars,tonics,duration


def evaluate(split_path,output):
    split=json.loads(Path(split_path).read_text(encoding='utf-8'));songs=[];failed=[]
    for row in split['rows']:
        if row['duplicate_of']:continue
        try:
            text=Path(row['salami']).read_text(encoding='utf-8-sig');bars,tonics,duration=annotation(text)
            events=read_raw_lab(row['lab']);previous=0.;adjustments=0
            for event in events:
                if event['start']<previous:
                    if previous-event['start']>1e-6:raise ValueError('Substantive LAB overlap')
                    event['start']=previous;adjustments+=1
                previous=event['end']
            events=[e for e in events if e['end']>e['start']]
            scores=score_timed_chords_by_key(events,duration);windows=summarize_key_windows(events,bars,duration)
            # Only now join the held-out tonic annotations for evaluation.
            weights=Counter()
            for segment in tonics:
                if segment['tonic'] is not None:weights[segment['tonic']]+=segment['end']-segment['start']
            if not weights:raise ValueError('No known tonic reference')
            target=max(weights,key=weights.get);pred=parse_chord_name(scores['key'])[0]
            good=covered=supported_good=supported_covered=0.
            for segment in windows['segments']:
                root=parse_chord_name(segment['key'])[0]
                for ref in tonics:
                    overlap=max(0.,min(segment['end'],ref['end'])-max(segment['start'],ref['start']))
                    if ref['tonic'] is not None:
                        covered+=overlap;good+=overlap*(root==ref['tonic'])
                        if segment.get('status')=='supported':
                            supported_covered+=overlap;supported_good+=overlap*(root==ref['tonic'])
            supported=[s for s in windows['segments'] if s.get('status')=='supported']
            tonic_changes=sum(a['tonic'] is not None and b['tonic'] is not None and a['tonic']!=b['tonic'] for a,b in zip(tonics,tonics[1:]))
            supported_roots=[parse_chord_name(s['key'])[0] for s in supported]
            predicted_tonic_changes=sum(a!=b for a,b in zip(supported_roots,supported_roots[1:]))
            songs.append(dict(id=row['id'],split=row['split'],group=row['group'],global_tonic_reference=target,
                global_prediction=scores['key'],global_tonic_correct=pred==target,perfect_fifth_error=(pred-target)%12 in {5,7},
                local_tonic_correct_seconds=good,local_reference_covered_seconds=covered,
                local_supported_correct_seconds=supported_good,local_supported_covered_seconds=supported_covered,
                known_reference_seconds=sum(weights.values()),reference_tonic_changes=tonic_changes,predicted_tonic_changes=predicted_tonic_changes,
                predicted_modulation=windows['modulation_detected'],interpolation_roundoff_adjustments=adjustments,
                decisive_coverage=windows['decisive_coverage'],inputs=[signature(row['lab']),signature(row['salami'])]))
        except Exception as exc:failed.append(dict(id=row['id'],error=f'{type(exc).__name__}: {exc}'))
    summaries={}
    for phase in ['development','calibration','sealed_test']:
        records=[r for r in songs if r['split']==phase];covered=sum(r['local_reference_covered_seconds'] for r in records)
        summaries[phase]=dict(n=len(records),global_tonic_accuracy=sum(r['global_tonic_correct'] for r in records)/len(records) if records else None,
            perfect_fifth_errors=sum(r['perfect_fifth_error'] for r in records),
            tonic_confusion_matrix=dict(Counter(f"{r['global_tonic_reference']} -> {parse_chord_name(r['global_prediction'])[0]}" for r in records)),
            local_tonic_accuracy=sum(r['local_tonic_correct_seconds'] for r in records)/covered if covered else None,
            supported_local_tonic_accuracy=sum(r['local_supported_correct_seconds'] for r in records)/sum(r['local_supported_covered_seconds'] for r in records) if sum(r['local_supported_covered_seconds'] for r in records) else None,
            supported_local_coverage=sum(r['local_supported_covered_seconds'] for r in records)/sum(r['known_reference_seconds'] for r in records) if sum(r['known_reference_seconds'] for r in records) else None,
            tonic_change_detection=dict(Counter(f"reference:{r['reference_tonic_changes']>0} predicted:{r['predicted_tonic_changes']>0}" for r in records)),
            failed=sum(next(r['split'] for r in split['rows'] if r['id']==f['id'])==phase for f in failed))
    result=dict(schema='billboard_symbolic_tonic_v1',split=signature(split_path),summaries=summaries,songs=songs,failures=failed,
        audio_evaluation=False,mode_accuracy=None,relative_mode_confusion=None,
        unmeasured=['relative/parallel-mode confusion: mode references unavailable','borrowed-chord accuracy: no borrowed-chord ground truth','loop-rotation robustness: not measured on this corpus'],
        input_kind='reference chords + phrase-interpolated bars, not raw acoustic predictions',
        warning='This tests rule reasoning with oracle chord/grid input; cannot train or calibrate a CNN disagreement resolver',
        code=[signature(Path(__file__).parent/n) for n in ['key_fusion.py','detect_key_timeline.py','evaluate_billboard_key_reasoning.py']])
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(summaries,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();evaluate(a.split,a.output)
