"""Read-only pairwise source differences, never correctness or confidence scores."""
import math

from fastapi import HTTPException
from app.chord_semantics_v2 import parse_chord
from .editor import context


REASONS={
    'root':'根音不同',
    'harmony':'和弦音或扩展标注不同',
    'bass':'低音标注不同或一方未标注',
    'chord_vs_no_chord':'和弦与 N 不同',
    'unknown_label':'含未知或无法解析的标签',
    'missing_prediction':'至少一个来源未覆盖',
}


def differences(left,right):
    a,b=parse_chord(left),parse_chord(right)
    reasons=[]
    if a.state=='missing' or b.state=='missing':reasons.append('missing_prediction')
    if a.state=='X' or b.state=='X':reasons.append('unknown_label')
    if {a.state,b.state}=={'N','chord'}:reasons.append('chord_vs_no_chord')
    if a.state==b.state=='chord':
        if a.root_pc!=b.root_pc:reasons.append('root')
        if (a.chord_bitmap,a.extension_degrees)!=(b.chord_bitmap,b.extension_degrees):reasons.append('harmony')
        if (a.explicit_bass,a.bass_pc)!=(b.explicit_bass,b.bass_pc):reasons.append('bass')
    return reasons


def compare_rows(left,right,duration):
    if type(duration) not in {int,float} or not math.isfinite(duration) or duration<=0:
        raise ValueError('无效录音时间范围')
    boundaries={0.,duration}
    for rows in (left,right):
        previous=0.
        if not isinstance(rows,list):raise ValueError('无效来源时间轴')
        for row in rows:
            if (not isinstance(row,dict) or any(type(row.get(k)) not in {int,float} or not math.isfinite(row[k]) for k in ('start','end'))
                    or row['start']<previous or row['end']<=row['start'] or not isinstance(row.get('label'),str) or not row['label']):
                raise ValueError('来源区间无效或重叠')
            previous=row['end']
            boundaries.update(max(0.,min(duration,row[k])) for k in ('start','end'))
    ordered=sorted(boundaries);indices=[0,0];segments=[];matched=0.;flagged=0.;totals={k:0. for k in REASONS}
    for start,end in zip(ordered,ordered[1:]):
        pair=[]
        for side,rows in enumerate((left,right)):
            while indices[side]<len(rows) and rows[indices[side]]['end']<=start:indices[side]+=1
            i=indices[side];row=rows[i] if i<len(rows) and rows[i]['start']<=start else None
            pair.append(dict(label=row['label'] if row else None,source_index=i if row else None))
        reasons=differences(pair[0]['label'],pair[1]['label']);seconds=end-start
        if reasons:
            flagged+=seconds
            for reason in reasons:totals[reason]+=seconds
            segments.append(dict(start=start,end=end,left=pair[0],right=pair[1],reasons=reasons))
        else:matched+=seconds
    return dict(segments=segments,summary=dict(duration=duration,matching_label_seconds=matched,
        flagged_seconds=flagged,reason_seconds=totals,intervals=len(segments)),reason_labels=REASONS,
        interpretation='Label differences and missing/unknown predictions only. Agreement is not accuracy; reasons can overlap in time.')


def comparison(store,project_id,left_id,right_id):
    _,left_result,left,duration,audio=context(store,project_id,left_id)
    _,right_result,right,right_duration,right_audio=context(store,project_id,right_id)
    if ((left_result['run_id'],duration,audio)!=(right_result['run_id'],right_duration,right_audio)
            or left_result.get('audio',{}).get('sha256')!=audio or right_result.get('audio',{}).get('sha256')!=audio):
        raise HTTPException(409,'比较期间项目身份发生变化，请重新读取')
    payload=compare_rows(left['rows'],right['rows'],duration)
    return dict(schema='source_comparison_v1',project_id=project_id,run_id=left_result['run_id'],audio_sha256=audio,
        duration=duration,left={k:left[k] for k in ('id','name','sha256')},right={k:right[k] for k in ('id','name','sha256')},
        identical_source_bytes=left['sha256']==right['sha256'],read_only=True,independent_votes=False,**payload)
