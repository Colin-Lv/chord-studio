"""Listen/lock/reveal annotation snapshots. No prediction-derived gold labels."""
import math
import re
from fastapi import HTTPException
from .storage import atomic_json, now, read_json


def location(store, project_id):
    project=store.project(project_id)
    result=store.result(project_id)
    if (project.get('id')!=project_id or result.get('project_id')!=project_id
            or result.get('run_id')!=project.get('current_run')):
        raise HTTPException(409,'调性标注的项目与分析身份不一致；原记录已保留')
    return result,store.run_dir(project_id,result['run_id'])/'key_annotation.json'


def state(store, project_id):
    result,path=location(store,project_id)
    from .key_review import checked_path
    from .editor import file_identity
    p=store.project(project_id)
    sha=result.get('audio',{}).get('sha256')
    duration=result.get('duration')
    if (not sha or p.get('audio_identity',{}).get('sha256')!=sha
            or type(duration) not in {int,float} or not math.isfinite(duration) or duration<=0):
        raise HTTPException(409,'调性标注的音频或时间范围无法验证；原记录已保留')
    try:
        actual=file_identity(checked_path(store.project_dir(project_id),p['audio']))
    except (OSError,KeyError,ValueError) as exc:
        raise HTTPException(409,'标注音频无法验证；原记录已保留') from exc
    if actual!=sha:raise HTTPException(409,'音频已变更；原标注记录已保留')
    if path.exists():
        try:
            record=read_json(path)
        except (OSError,ValueError) as exc:
            raise HTTPException(409,'已保存标注无法解析；原记录已保留') from exc
        required=dict(schema='recording_key_annotation_v1',run_id=result['run_id'],audio_sha256=sha)
        if (not isinstance(record,dict) or any(record.get(key)!=value for key,value in required.items())
                or ('project_id' in record and record['project_id']!=project_id)
                or ('duration' in record and (type(record['duration']) not in {int,float} or record['duration']!=duration))
                or type(record.get('revision')) is not int or record['revision']<0
                or type(record.get('prediction_visible')) is not bool or not isinstance(record.get('history'),list)
                or 'prior_prediction_exposure_known' in record and type(record['prior_prediction_exposure_known']) is not bool):
            raise HTTPException(409,'已保存标注的项目、分析、音频、时长或版本不一致；原记录已保留')
        # Older records bind run/audio but lack project/duration. Read them by
        # value without rewriting; the next explicit save adds these bindings.
        locked=record.get('locked_label')
        if locked is not None:
            label=locked.get('label',{}) if isinstance(locked,dict) else {}
            if not isinstance(label,dict):
                raise HTTPException(409,'已保存听辨标签格式无效；原记录已保留')
            a,b=label.get('excerpt_start'),label.get('excerpt_end')
            if (type(a) not in {int,float} or type(b) not in {int,float} or not math.isfinite(a+b)
                    or not 0<=a<b<=duration+1e-6
                    or label.get('label_scope')=='full_recording' and (a!=0 or abs(b-duration)>1e-6)):
                raise HTTPException(409,'已保存听辨范围与当前录音不一致；原记录已保留')
    else:
        record=dict(schema='recording_key_annotation_v1',revision=0,history=[],locked_label=None,prediction_visible=False)
    known_exposure=record.get('prior_prediction_exposure_known',False) or record['prediction_visible'] or any(
        isinstance(event,dict) and event.get('prediction_visible_before_lock') is True for event in record['history'])
    # Return no model/result key or candidates before an explicit reveal.
    return {**record, 'project_id':project_id,'run_id':result['run_id'],'audio_sha256':sha,
            'audio_url':result['audio']['url'],'duration':result['duration'],'title':result['title'],
            'prior_prediction_exposure_known':known_exposure,
            'independent_gold':False,'blindness_note':'仅本页隐藏预测；此录音先前是否已见预测尚未核实。锁定不自动赋予独立金标准资格。'}


def update(store,project_id,body):
    current=state(store,project_id)
    if any(body[k]!=current[k] for k in ('run_id','audio_sha256','revision')):raise ValueError('CONFLICT: 标注运行/音频/版本已改变')
    result,path=location(store,project_id)
    record={k:current[k] for k in ('schema','revision','history','locked_label','prediction_visible','project_id','run_id','audio_sha256','duration','prior_prediction_exposure_known')}
    action=body['action'];event=dict(at=now(),action=action,revision=current['revision']+1)
    if action=='lock':
        based_on=body.get('based_on_revision')
        if based_on is not None:
            if type(based_on) is not int or not any(isinstance(prior,dict) and prior.get('action')=='lock'
                    and prior.get('revision')==based_on for prior in current['history']):
                raise ValueError('CONFLICT: 依据的历史听辨记录不存在，请保留草稿并重新读取')
        entry=body.get('entry_context')
        if entry is not None:
            if not isinstance(entry,dict) or entry.get('origin')!='workbench':raise ValueError('无效听辨入口')
            if any(entry.get(key)!=current[key] for key in ('project_id','run_id','audio_sha256')):
                raise ValueError('CONFLICT: 所选片段链接与当前项目、运行或音频身份不一致')
            left,right=entry.get('excerpt_start'),entry.get('excerpt_end')
            if (type(left) not in {int,float} or type(right) not in {int,float} or not math.isfinite(left+right)
                    or not 0<=left<right<=current['duration']):raise ValueError('无效入口片段时间范围')
        prior=body.get('prior_exposure','unknown')
        if prior not in ('unknown','seen','not_seen'):raise ValueError('无效此前预测暴露状态')
        known_exposure=current['prior_prediction_exposure_known'] or entry is not None or prior=='seen'
        label=body.get('label',{})
        required=['recording_id','reviewer','basis','label_scope','annotation_status','excerpt_start','excerpt_end']
        if any(label.get(k) in (None,'') for k in required):raise ValueError('请填写录音版本、时段、标注者、依据与范围')
        a,b=label['excerpt_start'],label['excerpt_end']
        if isinstance(a,bool) or isinstance(b,bool) or not isinstance(a,(float,int)) or not isinstance(b,(float,int)) or not math.isfinite(a+b) or not 0<=a<b<=result['duration']+1e-6:raise ValueError('无效标注时间范围')
        if label['label_scope'] not in ('full_recording','excerpt'):raise ValueError('无效标注范围')
        if label['label_scope']=='full_recording' and (a!=0 or abs(b-result['duration'])>1e-6):raise ValueError('整曲标注必须覆盖整个解码时段')
        if label.get('tonic') and not re.fullmatch(r'(?:C|C#|Db|D|D#|Eb|E|F|F#|Gb|G|G#|Ab|A|A#|Bb|B)',label['tonic']):raise ValueError('无效主音')
        if label.get('mode') not in (None,'','major','minor'):raise ValueError('大小调族须为 major/minor 或未知')
        if label['annotation_status'] not in ('single_key','ambiguous','multi_key','unknown'):raise ValueError('无效标注状态')
        if label['annotation_status']=='single_key' and not label.get('tonic'):raise ValueError('单一主调需要主音；mode 可以未知')
        event.update(label=label,prediction_visible_before_lock=known_exposure,
                     prediction_visible_on_annotation_page=current['prediction_visible'],
                     entry_context=entry,prior_exposure_self_report=prior,
                     based_on_revision=based_on,
                     prior_exposure_unknown=not known_exposure and prior=='unknown',
                     label_origin='human_listening',independent_gold=False)
        record['prior_prediction_exposure_known']=known_exposure
        record['locked_label']=event.copy()
    elif action=='reveal':
        if not current['locked_label']:raise ValueError('请先听辨、填写并锁定，再查看预测')
        record['prediction_visible']=True
        event.update(label_status='prediction_exposed',independent_gold=False)
    else:raise ValueError('无效标注操作')
    record['revision']+=1;record['history'].append(event);atomic_json(path,record)
    return state(store,project_id)


def predictions(store,project_id):
    if not state(store,project_id)['prediction_visible']:raise ValueError('预测尚未解锁')
    from .key_review import state as review
    result=store.result(project_id)
    return dict(cnn=result.get('key_analysis'),review=review(store,project_id),note='已见预测的修订会记录暴露状态；不自动进入校准或测试。')
