"""Structural tests, not acoustic accuracy or extra hyperparameter search."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from app.key_top3_reranker_v1 import (CNN_FIELDS,HARMONIC_FIELDS,FEATURES,NOTES,PC,canonical,
    candidates,feature_rows,target_labels,fit_model,rank,load_model)
from app.train_key_top3_reranker_v1 import freeze_folds,connect_groups
from app.web.storage import ROOT,atomic_json,read_json


def cnn_fixture():
    keys=[n+s for s in ('','m') for n in NOTES]
    keys=['G','C','Em']+[k for k in keys if k not in ['G','C','Em']]
    return dict(key='G',candidates=[dict(key=k,score=(24-i)/300) for i,k in enumerate(keys)])


def harmonic_fixture():
    # Structural invariance tests need complete evidence, not a private song.
    modes={}
    for i,tonic in enumerate(NOTES):
        modes[tonic]=[]
        for j,mode in enumerate(['major','minor']):
            components={field:(i+1)/(20+k+j) for k,field in enumerate(HARMONIC_FIELDS[4:])}
            components['loop']=.1*(i%3)
            modes[tonic].append(dict(mode=mode,score=.8-i*.02-j*.01+.2*components['loop'],
                centre_score=.9-i*.03+.2*components['loop'],components=components))
    return dict(schema='harmonic_tonic_mode_v2',stage='G_full',key='C',policy={'loop_weight':.2},
                mode_candidates_by_tonic=modes)


def toy_records(n=12):
    c=cnn_fixture();f=feature_rows(c)
    return [dict(id=str(i),group='group'+str(i//2),reference=['G','C','Em','D'][i%4],features=copy.deepcopy(f)) for i in range(n)]


def shift(key,n):return NOTES[(PC[key.removesuffix('m')]+n)%12]+('m' if key.endswith('m') else '')


def test_budget_no_reference_insertion_and_miss_supervision():
    rows=feature_rows(cnn_fixture(),harmonic_fixture());assert [r['key'] for r in rows]==['G','C','Em']
    assert target_labels([r['key'] for r in rows],'D')==[0,0,0]
    with pytest.raises(ValueError):target_labels([r['key'] for r in rows],None)


def test_equivalent_spelling_distinct_mode_and_invalid_vector():
    assert canonical('Eb')=='D#' and canonical('Ebm')=='D#m'
    c=cnn_fixture();c['candidates'][4]['key']='G'
    with pytest.raises(ValueError):candidates(c)
    c=cnn_fixture();c['candidates'].pop()
    with pytest.raises(ValueError):candidates(c)


def test_missing_harmonic_masks_no_pseudoevidence():
    f=feature_rows(cnn_fixture(),None)
    for row in f:
        assert np.isnan(row['M2'][len(CNN_FIELDS):len(CNN_FIELDS)+len(HARMONIC_FIELDS)]).all()
        assert row['M2'][-len(HARMONIC_FIELDS):]==[0]*len(HARMONIC_FIELDS)


def test_reference_identity_extras_do_not_enter_features():
    c=cnn_fixture();h=harmonic_fixture();before=feature_rows(c,h)
    for x in (c,h):x.update(reference='F#m',title='known failure',artist='artist',split='sealed',human_key='A',label='B')
    assert feature_rows(c,h)==before
    assert not any(n in ' '.join(FEATURES['M2']) for n in ['reference','artist','split','path'])


def test_local_and_existing_loop_are_excluded():
    h=harmonic_fixture();before=feature_rows(cnn_fixture(),h)
    h['window_evidence']={'tonic_shares':{'B':1},'segments':[{'key':'B'}]}
    h['tonic_candidates']=[];h['final_key_candidates']=[]
    for rows in h['mode_candidates_by_tonic'].values():
        for row in rows:
            delta=.314;row['components']['loop']+=delta
            row['score']+=h['policy']['loop_weight']*delta
            row['centre_score']+=h['policy']['loop_weight']*delta
    for a,b in zip(before,feature_rows(cnn_fixture(),h)):
        np.testing.assert_allclose(a['M2'],b['M2'],atol=1e-12)


@pytest.mark.parametrize('semitones',range(12))
def test_structured_transposition_relative_features(semitones):
    c=cnn_fixture();h=harmonic_fixture();before=feature_rows(c,h)
    c['key']=shift(c['key'],semitones)
    for row in c['candidates']:row['key']=shift(row['key'],semitones)
    h['key']=shift(h['key'],semitones)
    h['mode_candidates_by_tonic']={shift(k,semitones):v for k,v in h['mode_candidates_by_tonic'].items()}
    after=feature_rows(c,h)
    for a,b in zip(before,after):
        assert shift(a['key'],semitones)==b['key'];np.testing.assert_allclose(a['M1'],b['M1']);np.testing.assert_allclose(a['M2'],b['M2'])


def test_group_folds_and_upstream_anchors():
    rows=toy_records(24);fixed={'group0','group1'}
    folds=freeze_folds(rows,5,fixed);ids=[]
    for fold in folds:
        assert not set(fold['train_groups'])&set(fold['evaluation_groups'])
        assert fixed.issubset(fold['train_groups']);ids+=fold['evaluation_ids']
    assert len(ids)==len(set(ids))==20


def test_components_connect_audio_artist_and_unselected_bridge():
    meta=[dict(id='a',group='1',group_tokens=['artist:x']),dict(id='bridge',group='2',group_tokens=['artist:x','work:y']),dict(id='b',group='3',group_tokens=['work:y']),dict(id='c',group='4',group_tokens=[])]
    roles={i:dict(audio_sha256='same' if i in ['b','c'] else 'different',group_id=i) for i in ['a','b','c']}
    groups,_=connect_groups(meta,roles);assert len(set(groups.values()))==1


def test_fold_local_scaler_save_load_and_ties(tmp_path):
    rows=toy_records();training=rows[:8];validation=rows[8:]
    model=fit_model(training,'M1',.1)
    assert set(model['preprocessing_fit_ids'])=={r['id'] for r in training}
    assert not set(model['preprocessing_fit_ids'])&{r['id'] for r in validation}
    X=np.asarray([f['M1'] for r in training for f in r['features']]);np.testing.assert_allclose(model['mean'],X.mean(axis=0))
    path=tmp_path/'model.json';atomic_json(path,model);loaded=load_model(path)
    assert rank(model,validation[0]['features'])==rank(loaded,validation[0]['features'])
    loaded['coefficient']=[0]*len(loaded['coefficient']);loaded['intercept']=0
    assert [r['key'] for r in rank(loaded,validation[0]['features'])]==['G','C','Em']


def test_cnn_exact_ties_preserve_frozen_order():
    c=cnn_fixture();c['candidates'][1]['score']=c['candidates'][0]['score']
    assert [r['key'] for r in candidates(c)[:3]]==['G','C','Em']
