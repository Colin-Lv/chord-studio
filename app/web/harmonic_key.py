"""Publish chord-first tonal timelines alongside each immutable chord source."""
import hashlib
import json
from ..harmonic_key_v3 import analyze, VERSION, load_model


def refresh(result, source_ids=None):
    """In-memory update; caller owns the analysis file and atomic publication.

    Human reviews/corrections are separate records and never read or changed.
    Repeated refresh preserves the original acoustic/global diagnostic once.
    """
    # Older partial publications can lack a verifiable clock. Preserve their
    # existing result rather than manufacturing a timeline from row endpoints.
    if not isinstance(result.get('duration'),(int,float)) or not result['duration']>0:
        result['harmonic_key_status']=dict(status='unavailable',reason='缺少有效音频时长')
        return result
    if 'key_analysis_before_v3' not in result:
        result['key_analysis_before_v3']=result.get('key_analysis')
    if result.get('key_fusion'):
        result.setdefault('key_fusion_before_v3',result.pop('key_fusion'))
    acoustic=result.get('key_analysis_before_v3')
    model=load_model()
    timelines=result.setdefault('harmonic_keys',{})
    valid_ids={s['id'] for s in result['sources']}
    for stale in set(timelines)-valid_ids:del timelines[stale]
    for source in result['sources']:
        if source_ids is not None and source['id'] not in source_ids:continue
        rows=source['rows']
        identity=hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        try:
            timeline=analyze(rows,float(result['duration']),result.get('bars') or [],
                             (result.get('structure') or {}).get('sections') or [],model=model)
        except ValueError as exc:
            timeline=dict(method=VERSION,status='unavailable',segments=[],changes=[],reason=str(exc))
        timeline.update(source_id=source['id'],source_name=source.get('name',source['id']),
                        source_sha256=source.get('sha256'),rows_sha256=identity,
                        audio_sha256=result.get('audio',{}).get('sha256'),run_id=result.get('run_id'),
                        label_status='model_inference_unverified',input_kind='model_chord_sequence')
        timelines[source['id']]=timeline
    active=timelines.get(result.get('default_source'),{})
    result['harmonic_key_default_source']=result.get('default_source')
    if active.get('status')=='computed':
        result['key_analysis']=dict(key=active['main_key'],source=VERSION,
            start_key=active['start_key'],end_key=active['end_key'],
            summary_method='predominant key by covered duration; see local timeline',
            candidates=[dict(key=k,seconds=v) for k,v in sorted(active['key_seconds'].items(),key=lambda x:-x[1])],
            acoustic_reference=acoustic,acoustic_used_for_decoding=False,
            source_id=active['source_id'],label_status='model_inference_unverified')
    else:
        result['key_analysis']=dict(key=None,source=VERSION,status=active.get('status','unavailable'),
                                  acoustic_reference=acoustic,reason=active.get('reason','尚无可用和弦序列'))
    return result
