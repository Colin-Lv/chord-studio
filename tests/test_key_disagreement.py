"""Statistics, data isolation and resolver decisions, not acoustic accuracy."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
import key_fusion as k
import key_disagreement as r
from prepare_key_resolver_data import freeze,grouped_split
from train_key_disagreement import read_cases,rule_decision
from test_key_fusion import example


def assignments(counts,unknown=0):
    labels=[key for key,n in counts.items() for _ in range(n)]
    return [dict(bar=i+1,key=key,stable=True) for i,key in enumerate(labels)]+[
        dict(bar=len(labels)+i+1,key='C',stable=False) for i in range(unknown)]


def test_unresolved_is_not_counter_evidence():
    metrics=k.stability_metrics(assignments({'C':64,'G':18,'F':2},42))
    assert metrics['valid_bars']==126 and metrics['decisive_bars']==84 and metrics['unresolved_bars']==42
    assert metrics['decisive_coverage']==pytest.approx(84/126)
    c=metrics['candidate_metrics']['C']
    assert c['candidate_decisive_share']==pytest.approx(64/84)
    assert c['stable_coverage']==pytest.approx(64/126)
    assert c['top_competitor_key']=='G' and c['top_competitor_stable_bars']==18
    assert c['top_competitor_share_valid']==pytest.approx(18/126)
    assert c['top_competitor_share_decisive']==pytest.approx(18/84)
    assert c['candidate_vs_competitor_gap']==pytest.approx(46/84)
    more=k.stability_metrics(assignments({'C':64,'G':18,'F':2},142))
    assert more['candidate_metrics']['C']['candidate_decisive_share']==c['candidate_decisive_share']
    assert more['candidate_metrics']['C']['top_competitor_share_decisive']==c['top_competitor_share_decisive']
    assert more['decisive_coverage']<metrics['decisive_coverage']


def test_no_decisive_evidence_is_undefined_not_zero_support():
    m=k.stability_metrics(assignments({},8))
    assert m['candidate_metrics']['C']['candidate_decisive_share'] is None
    assert m['decisive_coverage']==0 and m['unresolved_coverage']==1
    assert k.stability_metrics([])['decisive_coverage'] is None
    with pytest.raises(ValueError):k.stability_metrics([dict(bar=1,key='C',stable=True)]*2)


def test_primary_agreement_does_not_require_joint_vote():
    cnn,neutral,events=example('C G Am F','C','G')
    out=r.analyze(cnn,neutral,events)
    assert out['decision_path']=='agreement' and out['key']=='C'
    assert out['evidence']['functional_joint']['key']=='G'
    assert out['evidence_roles']['primary']==['madmom_cnn','raw_chord_functional']


def test_no_artifact_never_silently_promotes_harmonic():
    out=r.analyze(*example('C G Am F','G','C','C'))
    assert out['decision_path']=='ambiguous' and out['needs_review']
    assert 'no calibrated' in out['reason'][0]


def constant_model(e,cnn,harmonic):
    names=sorted(r.features(e))
    def head(logit):return dict(available=True,mean=[0]*len(names),scale=[1]*len(names),coef=[0]*len(names),intercept=logit)
    return dict(feature_names=names,heads=dict(cnn=head(cnn),harmonic=head(harmonic)),thresholds=dict(cnn=.75,harmonic=.75))


@pytest.mark.parametrize('a,b,path',[(3,-3,'disagreement_resolved_cnn'),(-3,3,'disagreement_resolved_harmonic'),(3,3,'ambiguous'),(-3,-3,'ambiguous')])
def test_two_calibrated_gates_have_explicit_ambiguous_state(a,b,path):
    inputs=example('C G Am F','G','C','C');e=k.analyze_key_evidence(*inputs)['evidence']
    out=r.analyze(*inputs,model=constant_model(e,a,b))
    assert out['decision_path']==path
    if path.endswith('harmonic'):
        assert out['key']=='C' and out['conflict']['cnn_overridden']
        assert out['candidates'][0]['key']=='C'
        assert 'min_bar_share' not in out['policy'] and 'min_bar_share' in out['legacy_policy']


def test_reference_title_and_dataset_are_not_features():
    e=k.analyze_key_evidence(*example('C G Am F'))['evidence'];before=r.features(e)
    e.update(reference='G',title='any',dataset='regression')
    assert r.features(e)==before
    assert all('reference' not in name and 'title' not in name for name in before)


def test_artist_work_groups_never_cross_splits_and_freeze(tmp_path):
    rows=[dict(id='1',group_tokens=['artist:A','work:X']),dict(id='2',group_tokens=['artist:B','work:X']),dict(id='3',group_tokens=['artist:B','work:Y'])]
    grouped_split(rows)
    assert len({r['group'] for r in rows})==1 and len({r['split'] for r in rows})==1
    path=tmp_path/'split.json';freeze(path,rows);freeze(path,rows)
    with pytest.raises(ValueError):freeze(path,[])


def test_training_reader_does_not_open_sealed_cases(tmp_path):
    split=tmp_path/'split.json';split.write_text(json.dumps(dict(rows=[dict(id='secret',selected=True,split='sealed_test')])) )
    (tmp_path/'secret').mkdir();(tmp_path/'secret/case.json').write_text('not JSON, must never be opened')
    assert read_cases(split,tmp_path,{'development','calibration'})==([],[])


@pytest.mark.parametrize('a,b,kind',[('C','G','perfect_fifth'),('C','Am','relative_major_minor'),('C','Cm','parallel_mode'),('G','Em','relative_major_minor'),('C','D','other')])
def test_key_relationships(a,b,kind):assert r.relation(a,b)==kind


def test_human_key_priority_and_malformed_artifact(tmp_path):
    out=r.analyze(*example('C G Am F'),requested_key='D')
    assert out['source']=='provided' and out['key']=='D'
    path=tmp_path/'model.json';path.write_text(json.dumps(dict(schema=r.SCHEMA,feature_schema=r.FEATURE_SCHEMA)))
    with pytest.raises(ValueError,match='contract'):r.load_model(path)


def test_cnn_weight_change_blocks_learned_override():
    inputs=example('C G Am F','G','C','C');e=k.analyze_key_evidence(*inputs)['evidence']
    model=constant_model(e,-3,3);model['cnn_model_sha256']=['different']
    out=r.analyze(*inputs,model=model)
    assert out['decision_path']=='ambiguous' and 'weight identity' in out['reason'][0]


@pytest.mark.parametrize('field,value',[('event_count',1),('root_count',1),('pitched_coverage',0),('margin',0),('complete_24',False)])
def test_rule_baseline_obeys_same_primary_quality_guard(field,value):
    e=k.analyze_key_evidence(*example('C G Am F','G','C','C'))['evidence']
    e['raw_chord_functional'][field]=value
    key,path,_=rule_decision(e,dict(coverage=0,dominance=0,gap=-1))
    assert key=='G' and path=='ambiguous'


def test_portable_logistic_scores_match_fitted_estimator():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    x=np.array([[0.,1.],[2.,4.],[1.,3.],[5.,8.],[7.,8.],[10.,2.]])
    y=np.array([0,0,0,1,1,1]);scaler=StandardScaler().fit(x)
    fitted=LogisticRegression(random_state=20261006).fit(scaler.transform(x),y)
    model=dict(feature_names=['a','b'],heads=dict(cnn=dict(available=True,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=fitted.coef_[0].tolist(),intercept=float(fitted.intercept_[0]))))
    for row,expected in zip(x,fitted.predict_proba(scaler.transform(x))[:,1]):
        assert r.gate_scores(model,dict(zip(['a','b'],row)))['cnn']==pytest.approx(expected,abs=1e-12)
