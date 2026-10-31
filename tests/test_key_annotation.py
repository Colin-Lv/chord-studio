import pytest
from tests.test_key_review_workbench import client, act
from app.import_key_annotations import validate

HEADERS={'X-Harmony-Client':'studio'}


def test_blind_lock_reveal_and_exposure_history(client):
    c,_,pid,_=client;url=f'/api/projects/{pid}/key-annotation'
    # fixture lacks presentation fields; add only title/audio route, not answers.
    store=c.app.state.store;result=store.result(pid);result.update(title='fixture');result['audio']['url']=f'/api/projects/{pid}/audio'
    from app.web.storage import atomic_json
    atomic_json(store.run_dir(pid,result['run_id'])/'analysis.json',result)
    s=c.get(url).json();assert not {'key','cnn','suggestion'} & s.keys() and not s['prediction_visible']
    assert c.get(url+'/predictions').status_code==400
    payload={k:s[k] for k in ('run_id','audio_sha256','revision')}
    assert c.post(url,headers=HEADERS,json=payload|dict(action='reveal')).status_code==400
    label=dict(recording_id='version1',reviewer='test',basis='listening fixture',label_scope='excerpt',annotation_status='single_key',excerpt_start=0,excerpt_end=4,tonic='C',mode=None)
    locked=c.post(url,headers=HEADERS,json=payload|dict(action='lock',label=label)).json()
    assert locked['locked_label']['prediction_visible_before_lock'] is False
    assert c.get(url+'/predictions').status_code==400
    payload={k:locked[k] for k in ('run_id','audio_sha256','revision')}
    revealed=c.post(url,headers=HEADERS,json=payload|dict(action='reveal')).json()
    assert revealed['prediction_visible'] and not revealed['independent_gold']
    assert c.get(url+'/predictions').status_code==200
    payload={k:revealed[k] for k in ('run_id','audio_sha256','revision')}
    revision=c.post(url,headers=HEADERS,json=payload|dict(action='lock',label=label|dict(mode='major'))).json()
    assert revision['locked_label']['prediction_visible_before_lock'] is True
    assert revision['history'][0]['label']['mode'] is None
    assert c.post(url,headers=HEADERS,json=payload|dict(action='lock',label=label)).status_code==409


@pytest.mark.parametrize('change',[dict(excerpt_end=float('inf')),dict(mode='inferred'),dict(prediction_visible='false'),dict(data_role='sealed'),dict(tonic='H'),dict(reviewer='')])
def test_annotation_import_rejects_invalid_or_uncertified(change):
    row=dict(item_id='item',audio_sha256='a'*64,recording_id='recording',reviewer='reviewer',basis='listening',label_scope='excerpt',annotation_status='single_key',excerpt_start=0,excerpt_end=4,tonic='C',mode=None,prediction_visible=False)
    with pytest.raises(ValueError):validate(row|change)


def test_unknown_fields_not_fabricated():
    row=dict(item_id='item',audio_sha256='a'*64,recording_id='recording',reviewer='reviewer',basis='uncertain',label_scope='excerpt',annotation_status='unknown',excerpt_start=0,excerpt_end=4,tonic=None,mode=None,prediction_visible=False)
    valid=validate(row)
    assert valid['tonic'] is None and valid['mode'] is None and not valid['independent_gold']
