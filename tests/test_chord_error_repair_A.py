import json
import sys
from pathlib import Path

import mir_eval
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"app"))
from chord_semantics_v2 import parse_chord as old_parse
from chord_semantics_v3 import parse_chord, to_harte, require_new_contract
from reference_events_v1 import (normalize_reference, legacy_consistency, coverage, key_definition,
                                 edit_transaction, derive_views, save_reference_atomic)


def pitches(label):
    s=parse_chord(label)
    return {(i+s.root_pc)%12 for i,v in enumerate(s.chord_bitmap) if v}


@pytest.mark.parametrize("a,b",[("G9sus4","G:sus4(b7,9)"),("G7sus4","G:sus4(b7)"),
    ("G9sus4","G:7(9,4,*3)"),
    ("C9sus2/E","C:sus2(b7,9)/3"),("Cb9sus4/Eb","B:sus4(b7,9)/3"),
    ("D#m7b5","Eb:hdim7"),("D#dim7","Eb:dim7"),("Cadd9","C:maj(9)"),
    ("G9sus4(b9,*9)/D","G:sus4(b7,9,b9,*9)/5")])
def test_aliases_and_roundtrip(a,b):
    s=parse_chord(a)
    assert s.state=="chord" and not s.parse_error
    assert s.raw_label==a and s.identity==parse_chord(b).identity
    assert parse_chord(to_harte(a)).identity==s.identity
    assert s.schema=="chord_semantics_v3"


def test_conflict_pitches_are_independent_hand_checks():
    assert pitches("D#dim7")=={3,6,9,0}
    assert pitches("D#m7b5")=={3,6,9,1}
    assert pitches("B7/D#")=={11,3,6,9}
    assert pitches("G9sus4")=={7,0,2,5,9}
    assert old_parse("G9sus4").state=="X"
    with pytest.raises(ValueError):require_new_contract({"semantics_schema":"chord_semantics_v2"})


@pytest.mark.parametrize("s",[None,"", "N","X","C7","C9","C11","C13","Cmaj7","Cmaj9","Cadd9","C/E","C/G","D#m7b5","D#dim7","C:7(*9,#11)","H?"])
def test_unchanged_old_labels_and_standard_metric(s):
    a,b=old_parse(s),parse_chord(s)
    assert a.identity==b.identity and a.raw_label==b.raw_label and a.parse_error==b.parse_error
    if a.state in {"N","chord"}:
        assert mir_eval.chord.tetrads([a.harte],[a.harte]).tolist()==mir_eval.chord.tetrads([b.harte],[b.harte]).tolist()


def legacy():
    return {"key":"G","source":"audio.wav","bars":[{"bar":1,"start":0,"end":1,
      "spans":[{"start":0,"end":.12,"chord":"F#m","edited":True,"confidence":1},
               {"start":.12,"end":1,"chord":"G9sus4"}],
      "beats":[{"beat":1,"start":0,"end":1,"chord":"E"}]}],
      "events":[{"start":0,"end":1,"chord":"E"}]}


def normalize(d=None,**kwargs):
    return normalize_reference(d or legacy(),source_policy="legacy_studio_bars_spans",sample_rate=22050,source_identity={"sha256":"test"},**kwargs)


def test_explicit_source_and_stale_detection_and_real_short_event():
    d=legacy();a=legacy_consistency(d)
    assert a["spans_count"]==2 and a["events_count"]==1 and a["events_stale"]
    assert a["beat_conflicts"][0]["stored_label"]=="E"
    r=normalize(d)
    assert len(r["canonical_events"])==2 and r["canonical_events"][0]["end"]==.12
    assert r["bars"][0]["beats"][0]["chord"] is None
    assert r["bars"][0]["beats"][0]["segments"][0]["chord"]=="F#:min"
    assert r["bars"][0]["beats"][0]["summary_kind"]=="mixed"
    assert d==legacy()
    with pytest.raises(ValueError):normalize_reference(d,source_policy="guess",sample_rate=22050,source_identity={})


def test_subsample_artifact_preserved_raw_and_not_negative_duration():
    d={"key":"C","bars":[{"bar":1,"start":0,"end":.24,"spans":[
       {"start":0,"end":.12,"chord":"C"},{"start":.12,"end":.12000000000001,"chord":"X"},
       {"start":.12000000000001,"end":.24,"chord":"G"}],"beats":[]}]}
    r=normalize(d)
    assert len(r["canonical_events"])==2 and len(r["quantization_artifacts"])==1
    assert r["quantization_artifacts"][0]["raw_span"]["chord"]=="X"
    assert coverage(r)["labelled_seconds"]==.24


def test_unknown_fields_not_negatives_and_confidence_not_gold():
    r=normalize()
    a,b=r["canonical_events"]
    assert a["known_mask"]["root"] and not a["known_mask"]["seventh"] and not a["known_mask"]["bass"]
    assert b["known_mask"]["seventh"] and b["known_mask"]["ext9"] and not b["known_mask"]["ext11"]
    assert b["field_values"]["seventh"]==[10]
    assert not any(a["training_mask"].values())
    assert not any(coverage(r)["listened_confirmed_training_seconds"].values())
    with pytest.raises(ValueError):edit_transaction(r,[{"event_id":a["event_id"],"review_status":"listened_confirmed","confirmed_fields":["bass"]}],expected_revision=0,reviewer="test")


def test_copy_rule_and_inferred_are_preserved_as_separate_origins():
    d=legacy();d["auto_completion"]={"target_bars":[1,1]};d["progression_override"]={"progression":"6415"}
    d["bars"][0]["spans"][0]["dominant_seventh_inferred"]=True
    origins=normalize(d)["canonical_events"][0]["provenance"]["origins"]
    assert set(origins)=={"legacy_edited_unverified","auto_copy_possible_later_human_edits","user_rule_assisted","inherited_automatic_inference"}


@pytest.mark.parametrize("display,tonic",[("G",4),("C",9),("A",6),("E",1)])
def test_relative_minor_display_never_becomes_major_target(display,tonic):
    r=key_definition(display)
    assert r["tonic_pc"] is None and r["mode"] is None and not any(r["known_mask"].values())
    k=key_definition(display,{"tonic_pc":tonic,"mode":"minor"})
    assert k["display_key"]==display and k["tonic_pc"]==tonic and k["mode"]=="minor"


def test_conflict_mask_persists_until_explicit_field_adjudication():
    d=legacy();d["bars"][0]["spans"][0]["chord"]="D#m7b5"
    r=normalize(d,conflicts={0:["seventh"]});a=r["canonical_events"][0]
    assert not a["known_mask"]["seventh"] and a["field_values"]["quality"]=="diminished"
    no=edit_transaction(r,[{"event_id":a["event_id"],"review_status":"unresolved"}],expected_revision=0,reviewer="test")
    assert no["canonical_events"][0]["review"]["conflict_fields"]==["seventh"]
    yes=edit_transaction(r,[{"event_id":a["event_id"],"label":"D#dim7","review_status":"listened_confirmed","confirmed_fields":["seventh"]}],expected_revision=0,reviewer="test")
    assert yes["canonical_events"][0]["training_mask"]["seventh"] and yes["canonical_events"][0]["field_values"]["seventh"]==[9]
    assert r["canonical_events"][0]["raw_label"]=="D#m7b5"


def test_revision_transaction_and_atomic_save_conflicts(tmp_path):
    r=normalize();eid=r["canonical_events"][0]["event_id"]
    new=edit_transaction(r,[{"event_id":eid,"label":"C/E","review_status":"listened_confirmed","confirmed_fields":["root","bass"]}],expected_revision=0,reviewer="fixture_not_human")
    assert r["revision"]==0 and new["revision"]==1
    assert new["events"][0]["chord"]==new["bars"][0]["spans"][0]["chord"]=="C:maj/3"
    assert new["events"][0]["derived_revision"]==new["bars"][0]["beats"][0]["derived_revision"]==1
    with pytest.raises(ValueError):edit_transaction(new,[],expected_revision=0,reviewer="x")
    path=tmp_path/"saved.reference_events_v1.json"
    save_reference_atomic(path,r,expected_disk_revision=None)
    save_reference_atomic(path,new,expected_disk_revision=0)
    assert json.loads(path.read_text())["revision"]==1
    frozen=path.read_bytes()
    with pytest.raises(ValueError):save_reference_atomic(path,new,expected_disk_revision=0)
    assert path.read_bytes()==frozen and not list(tmp_path.glob('*.pending'))
    with pytest.raises(ValueError):save_reference_atomic(tmp_path/"legacy.json",r,expected_disk_revision=None)
    broken=json.loads(json.dumps(new));broken["events"][0]["chord"]="wrong"
    with pytest.raises(ValueError):edit_transaction(broken,[],expected_revision=1,reviewer="x")


@pytest.mark.parametrize("label,seventh,ext9",[("C",None,None),("C7",[10],None),
    ("Cmaj7",[11],None),("D#dim7",[9],None),("C:7(*b7,*9)",[],[])])
def test_separate_seventh_and_explicit_extension_absence(label,seventh,ext9):
    d=legacy();d['bars'][0]['spans'][0]['chord']=label
    e=normalize(d)['canonical_events'][0]
    assert e['field_values']['seventh']==seventh and e['field_values']['ext9']==ext9


def test_quality_target_is_independent_of_equivalent_base_quality_spelling():
    values=[]
    for label in ['G9sus4','G:sus4(b7,9)','G:7(9,4,*3)']:
        d=legacy();d['bars'][0]['spans'][0]['chord']=label
        values.append(normalize(d)['canonical_events'][0]['field_values'])
    assert values[0]==values[1]==values[2] and values[0]['quality']=='sus4'


def test_real_legacy_cases_preserve_sixteenth_events_and_conflicts():
    root=Path(__file__).resolve().parents[1]
    cases=root/'exports/manual_model_comparison_20261006/cases'
    if not all((cases/cid/'summary.json').is_file() for cid in ['01_yorushika','07_laonanhai','09_gurenge']):
        pytest.skip('Optional private legacy comparison corpus is not installed')
    source=lambda cid:json.loads((cases/cid/'summary.json').read_text(encoding='utf-8'))['manual_source']
    if not all((root/source(cid)).is_file() for cid in ['01_yorushika','07_laonanhai','09_gurenge']):
        pytest.skip('Optional private manual annotations are not installed')
    y=json.loads((root/source('01_yorushika')).read_text(encoding='utf-8'))
    r=normalize(y);fast=[e for e in r['canonical_events'] if e['bar']==97]
    assert len(fast)==16 and min(e['end']-e['start'] for e in fast)>.11
    old=json.loads((root/source('07_laonanhai')).read_text(encoding='utf-8'))
    assert legacy_consistency(old)['spans_count']==213 and legacy_consistency(old)['events_count']==262
    g=json.loads((root/source('09_gurenge')).read_text(encoding='utf-8'))
    selected=[s for b in g['bars'] for s in b['spans'] if s.get('chord')=='D#m7b5']
    assert len(selected)==7 and sum(s['end']-s['start'] for s in selected)==pytest.approx(12.41001892)


def test_concurrent_lock_and_invalid_reference_overlap(tmp_path):
    r=normalize();p=tmp_path/'save.reference_events_v1.json'
    lock=p.with_suffix(p.suffix+'.lock');lock.write_text('other_writer')
    with pytest.raises(ValueError,match='Concurrent'):save_reference_atomic(p,r,expected_disk_revision=None)
    assert not p.exists() and lock.read_text()=='other_writer'
    d=legacy();d['bars'][0]['spans'][1]['start']=.1
    with pytest.raises(ValueError,match='overlap'):normalize(d)
