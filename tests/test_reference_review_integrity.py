"""Independent expected musical facts and reference transaction regressions."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from reference_events_v1 import normalize_reference, edit_transaction, coverage, save_reference_atomic


def reference(label='C/E', identity='recording-a'):
    return normalize_reference({'key':'C','events':[{'start':0,'end':1,'chord':label}]},
        source_policy='declared_top_level_events',sample_rate=22050,source_identity={'sha256':identity})


def confirm(record, fields, **extra):
    change={'event_id':record['canonical_events'][0]['event_id'],
            'review_status':'listened_confirmed','confirmed_fields':fields,**extra}
    return edit_transaction(record,[change],expected_revision=record['revision'],reviewer='test_fixture')


@pytest.mark.parametrize('label,family',[
    ('C:sus2(11)','sus2'), ('C:sus4(#9)','sus4'), ('C:min7(#11)','minor'),
    ('C:7(*3,#9)','power'), ('C:7(#9)','major'), ('C:7(4,*3)','sus4'),
    ('G9sus4','sus4'), ('C:min7(b5,*5)','diminished'), ('C:7(#5,*5)','augmented'),
    ('C:(1,b3,5,#11)','minor'), ('C:sus4(9)','sus4'), ('C:7(*3,2)','sus2'),
])
def test_upper_extensions_do_not_define_base_quality(label, family):
    assert reference(label)['canonical_events'][0]['field_values']['quality']==family


def test_extension_bass_and_omission_targets_survive_base_quality_filter():
    e=reference('C:min7(#11)/G')['canonical_events'][0]
    assert e['field_values']['quality']=='minor'
    assert e['field_values']['seventh']==[10]
    assert e['field_values']['ext11']==[1]
    assert e['field_values']['bass']==7
    assert e['known_mask']['ext11'] and e['known_mask']['bass']
    assert e['semantics']['chord_bitmap'][6]==1
    assert reference('C:7(*9)')['canonical_events'][0]['field_values']['ext9']==[]


def test_sequential_confirmations_accumulate_unchanged_fields():
    r=reference();before=copy.deepcopy(r)
    root=confirm(r,['root']);both=confirm(root,['bass'])
    e=both['canonical_events'][0]
    assert e['training_mask']['root'] and e['training_mask']['bass']
    assert e['review']['confirmed_fields']==['root','bass']
    assert r==before


def test_changed_bass_revokes_only_changed_field():
    r=confirm(reference(),['root','bass','quality'])
    r=confirm(r,[],label='C/G');e=r['canonical_events'][0]
    assert e['training_mask']['root'] and e['training_mask']['quality']
    assert not e['training_mask']['bass'] and e['field_values']['bass']==7
    assert 'bass' not in e['review']['confirmed_fields']


def test_changed_seventh_preserves_same_root_and_quality():
    r=confirm(reference('C7'),['root','quality','seventh'])
    r=confirm(r,[],label='Cmaj7');e=r['canonical_events'][0]
    assert e['training_mask']['root'] and e['training_mask']['quality']
    assert not e['training_mask']['seventh'] and e['field_values']['seventh']==[11]


def test_changed_key_invalidates_key_confirmation_only():
    r=confirm(reference(),['root','tonic','mode'],confirmed_key={'tonic_pc':0,'mode':'major'})
    r=confirm(r,[],confirmed_key={'tonic_pc':9,'mode':'minor'});e=r['canonical_events'][0]
    assert e['training_mask']['root']
    assert not e['training_mask']['tonic'] and not e['training_mask']['mode']


@pytest.mark.parametrize('status',['unreviewed','unresolved','pending_conflict'])
def test_explicit_review_downgrade_revokes_confirmation(status):
    r=confirm(reference(),['root','bass'])
    r=edit_transaction(r,[{'event_id':'e00000','review_status':status}],expected_revision=1,reviewer='test_fixture')
    e=r['canonical_events'][0]
    assert not any(e['training_mask'].values()) and e['review']['confirmed_fields']==[]


def test_repeated_review_does_not_duplicate_duration_but_preserves_history():
    r=confirm(reference(),['root']);r=confirm(r,['root']);r=confirm(r,['root'])
    assert len(r['history'])==3 and r['revision']==3
    assert coverage(r)['origin_seconds_overlapping_categories_do_not_add']['explicit_listened_review']==1
    assert r['canonical_events'][0]['provenance']['origins'].count('explicit_listened_review')==1


def test_legacy_duplicate_origins_count_once_per_event():
    r=reference();r['canonical_events'][0]['provenance']['origins']=['same','same']
    assert coverage(r)['origin_seconds_overlapping_categories_do_not_add']['same']==1


@pytest.mark.parametrize('different',['source','clock','interval','origin'])
def test_save_rejects_foreign_reference_even_at_same_revision(tmp_path,different):
    a=reference();path=tmp_path/'saved.reference_events_v1.json'
    save_reference_atomic(path,a,expected_disk_revision=None);before=path.read_bytes()
    b=reference(identity='recording-b' if different=='source' else 'recording-a')
    if different=='clock':b['sample_rate']=44100
    if different=='interval':
        b['canonical_events'][0]['end_sample']=11025
        b['canonical_events'][0]['end']=.5
    if different=='origin':b['canonical_events'][0]['raw_span']['chord']='G'
    from reference_events_v1 import derive_views
    b=confirm(derive_views(b),['root'])
    with pytest.raises(ValueError,match='identity|binding'):
        save_reference_atomic(path,b,expected_disk_revision=0)
    assert path.read_bytes()==before
    assert not list(tmp_path.glob('*.pending')) and not list(tmp_path.glob('*.lock'))


def test_same_reference_revision_saved_and_restored(tmp_path):
    r=reference();path=tmp_path/'saved.reference_events_v1.json'
    save_reference_atomic(path,r,expected_disk_revision=None)
    r=confirm(r,['root']);save_reference_atomic(path,r,expected_disk_revision=0)
    disk=json.loads(path.read_text(encoding='utf-8'))
    disk=confirm(disk,['bass']);save_reference_atomic(path,disk,expected_disk_revision=1)
    assert json.loads(path.read_text(encoding='utf-8'))['canonical_events'][0]['training_mask']['root']


def test_old_field_semantics_requires_new_independent_normalization(tmp_path):
    r=reference();r.pop('field_semantics_schema',None)
    with pytest.raises(ValueError,match='field semantics'):
        confirm(r,['root'])
    with pytest.raises(ValueError,match='field semantics'):
        save_reference_atomic(tmp_path/'old.reference_events_v1.json',r,expected_disk_revision=None)
