import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from app.web.source_rows import canonical_rows,serialize_rows,lab_rows
from app.web.fusion_engine import segments_from


def test_roundtrip_preserves_submillisecond_chords_and_audio_end(tmp_path):
    end=10.123456789
    rows=segments_from(['C:maj','G:maj','C:maj'],[(0,1.0001),(1.0001,1.0002),(1.0002,end)])
    actual,text=serialize_rows(rows,end,complete=True)
    path=tmp_path/'result.lab';path.write_text(text)
    assert [r['end'] for r in actual]==[1.0001,1.0002,end]
    for row,read in zip(actual,lab_rows(path)):
        assert all(row[k]==read[k] for k in ('start','end','label','start_text','end_text'))


@pytest.mark.parametrize('bad',[
    [(0,0)],[(1,0)],[(0,float('nan'))],[(0,float('inf'))],
    [(-1,1)],[(0,1),(0.9,2)],[(0,1),(1.1,2)],[(0,2.0005)],
])
def test_bad_generated_timelines_fail_before_publication(bad):
    with pytest.raises(ValueError):
        canonical_rows([dict(start=a,end=b,label='C:maj') for a,b in bad],2,complete=True)


def test_numerical_noise_snapped_but_real_gaps_not_hidden():
    rows=canonical_rows([dict(start=0,end=.3,label='C:maj'),dict(start=.1+.2,end=1,label='G:maj')],1,True)
    assert rows[0]['end']==rows[1]['start']
    rows=segments_from(['C:maj','C:maj'],[(0,1),(1.0000005,2)])
    assert len(rows)==2 # A partial source's gap is preserved, not merged away.


def test_mismatched_cells_are_not_silently_truncated():
    with pytest.raises(ValueError):segments_from(['C:maj'],[(0,1),(1,2)])


@pytest.mark.parametrize('module',['stem_harmony','stem_bass_harmony','fusion_engine'])
def test_invalid_publish_leaves_previous_source_untouched(tmp_path,module):
    import importlib
    m=importlib.import_module('app.web.'+module)
    path=tmp_path/'analysis.json';before=json.dumps(dict(sources=[],duration=2,default_source='old'))
    path.write_text(before)
    rows=[dict(start=0,end=0,label='C:maj')]
    action=m.apply_to_analysis if module=='fusion_engine' else m.publish_result
    with pytest.raises(ValueError):action(tmp_path,rows,{},make_default=True)
    assert path.read_text()==before
    assert not list(tmp_path.glob('*.lab'))


def test_randomized_serialization_is_exact_and_complete(tmp_path):
    rng=np.random.default_rng(20261102)
    edges=np.r_[0,np.cumsum(rng.uniform(.000001,1,300))]
    rows=segments_from(['C:maj','D:min','G:7']*100,list(zip(edges,edges[1:])))
    canonical,text=serialize_rows(rows,float(edges[-1]),True)
    path=tmp_path/'r.lab';path.write_text(text)
    parsed=lab_rows(path)
    assert [(r['start'],r['end'],r['label']) for r in parsed]==[(r['start'],r['end'],r['label']) for r in canonical]


def test_display_distinguishes_short_ranges_and_carries_minutes():
    root=Path(__file__).resolve().parents[1]
    script="""const assert=require('node:assert/strict'), t=require('./frontend/timeline-time.js');
    assert.equal(t.range(56.825,56.83),'00:56.825–00:56.830');
    assert.equal(t.range(1.0001,1.0002),'00:01.0001–00:01.0002');
    assert.equal(t.time(59.999,2),'01:00.00');
    assert.equal(t.range(1,1),'无效时间区间');
    assert.equal(t.range(54.45,55.2),'00:54.45–00:55.20');
    """
    subprocess.run(['node','-e',script],cwd=root,check=True,capture_output=True)


def test_late_song_cell_gaps_are_not_hidden_by_relative_tolerance():
    from app.web.stem_harmony import validate_cells
    with pytest.raises(ValueError):
        validate_cells([dict(start=0,end=500),dict(start=500.001,end=600)],2,600)


@pytest.mark.parametrize('module',['stem_harmony','stem_bass_harmony','fusion_engine'])
def test_every_publisher_uses_identical_json_and_lab_times(tmp_path,module):
    import importlib
    m=importlib.import_module('app.web.'+module)
    (tmp_path/'analysis.json').write_text(json.dumps(dict(sources=[],duration=1.0002,default_source='old')))
    rows=[dict(start=0,end=1.0001,label='C:maj'),dict(start=1.0001,end=1.0002,label='G:maj')]
    action=m.apply_to_analysis if module=='fusion_engine' else m.publish_result
    action(tmp_path,rows,{},make_default=True)
    result=json.loads((tmp_path/'analysis.json').read_text())
    source=next(s for s in result['sources'] if s['id']==result['default_source'])
    parsed=lab_rows(tmp_path/source['path'])
    assert [(r['start'],r['end'],r['label']) for r in parsed]==[(r['start'],r['end'],r['label']) for r in source['rows']]


def test_full_decoder_lab_does_not_round_away_short_events(tmp_path):
    from app.web.full_decode import write_lab
    path=tmp_path/'full.lab'
    write_lab(path,[(0,1.0001,'C:maj'),(1.0001,1.0002,'G:maj')])
    assert lab_rows(path)[1]['end']-lab_rows(path)[1]['start']==pytest.approx(.0001)


def test_import_bounds_external_frame_padding_and_preserves_original(tmp_path):
    from app.web.source_rows import source_for_audio
    raw='0 1.0001 C:maj\n1.0001 1.1 G:maj\n'
    path=tmp_path/'external.lab';path.write_text(raw)
    source=source_for_audio('external','External',path,tmp_path,1.0002)
    assert path.read_text()==raw
    assert source['path']=='external_audio_bounds.lab'
    assert source['rows'][-1]['end']==1.0002
    assert source['audio_bounds']['original']['path']=='external.lab'
    assert lab_rows(tmp_path/source['path'])[-1]['end']==1.0002


def test_large_external_clock_mismatch_is_not_hidden(tmp_path):
    from app.web.source_rows import source_for_audio
    path=tmp_path/'external.lab';path.write_text('0 10 C:maj\n')
    with pytest.raises(ValueError):source_for_audio('external','External',path,tmp_path,1)
    assert not (tmp_path/'external_audio_bounds.lab').exists()
