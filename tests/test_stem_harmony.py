"""Regressions for audio clocks, harmonic semantics and non-destructive publication."""
import json
from pathlib import Path

import numpy as np
import pytest

from app.web import stem_harmony as h


def test_pool_integrates_irregular_cells_not_array_index():
    values = np.array([[1., 0.], [0., 1.]])
    bounds = np.array([[0., .1], [.1, 1.]])
    np.testing.assert_allclose(h.pool(values, bounds, 0, 1), [.1, .9])
    np.testing.assert_allclose(h.pool(values, bounds, .05, .15), [.5, .5])


@pytest.mark.parametrize('cells', [[dict(start=0,end=.4),dict(start=.5,end=1)],
                                  [dict(start=0,end=1)], [dict(start=0,end=.5),dict(start=.5,end=2)]])
def test_invalid_clock_rejected(cells):
    with pytest.raises(ValueError): h.validate_cells(cells, 2, 1)


def test_disabled_evidence_exactly_preserves_all_original_chords():
    rng = np.random.default_rng(14)
    for label in h.LABELS:
        prior = rng.dirichlet(np.ones(120))
        scores = h.score_chords(prior, label, np.zeros(12), np.zeros(12), 0, 0)
        assert h.LABELS[scores.argmax()] == label


def test_clear_minor_evidence_can_correct_major():
    tones = np.zeros(12);tones[[0,3,7]] = 1/3
    bass = np.zeros(12);bass[0] = 1
    scores = h.score_chords(np.ones(120)/120, 'C:maj', tones, bass, 1, 1)
    assert h.LABELS[scores.argmax()] == 'C:min'


def test_third_in_bass_does_not_force_root_change():
    tones = np.zeros(12);tones[[0,4,7]] = 1/3
    bass = np.zeros(12);bass[4] = 1
    scores = h.score_chords(np.ones(120)/120, 'C:maj', tones, bass, 1, 1)
    assert h.LABELS[scores.argmax()] == 'C:maj'


def test_seventh_supported_by_accompaniment():
    tones = np.zeros(12);tones[[0,4,7,11]] = .25
    bass = np.zeros(12);bass[0] = 1
    scores = h.score_chords(np.ones(120)/120, 'C:maj7', tones, bass, 1, 1)
    assert h.LABELS[scores.argmax()] == 'C:maj7'


def test_silent_separation_does_not_become_loud_evidence():
    evidence = dict(times=np.arange(22)*2048/44100, activations=np.ones((84,22)),
                    energy=np.ones(22)*1e-6, duration=1.)
    profile, activity = h.pitch_profile(evidence, 0, 1)
    assert activity == 0
    assert not profile.any()


def test_stem_rate_and_length_mismatch_is_rejected(tmp_path):
    import soundfile as sf
    base = tmp_path/'wav';base.mkdir()
    sf.write(base/'guitar.wav', np.ones(44100)*.1, 44100)
    sf.write(base/'piano.wav', np.ones(48000)*.1, 48000)
    with pytest.raises(ValueError, match='采样率'):
        h.build_views(tmp_path, dict(tracks=[dict(id='guitar'),dict(id='piano')]), 1, tmp_path/'out')


def test_two_stem_instrumental_supported_without_invented_bass(tmp_path):
    import soundfile as sf
    base = tmp_path/'wav';base.mkdir()
    sf.write(base/'instrumental.wav', np.ones(44100)*.1, 44100)
    views, identities, groups = h.build_views(tmp_path, dict(tracks=[dict(id='vocals'),dict(id='instrumental')]), 1, tmp_path/'out')
    assert set(views) == {'accompaniment'}
    assert groups['accompaniment'] == ['instrumental']


def test_publication_preserves_key_original_and_old_revisions(tmp_path):
    original = dict(sources=[dict(id='original', rows=[])],default_source='original',key_analysis={'key':'C'})
    (tmp_path/'analysis.json').write_text(json.dumps(original))
    rows = [dict(id=0,start=0,end=1,label='C:maj',candidates=[])]
    first = h.publish_result(tmp_path, rows, {}, False)
    rows[0]['label']='C:min'
    second = h.publish_result(tmp_path, rows, {}, False)
    result = json.loads((tmp_path/'analysis.json').read_text())
    assert result['key_analysis'] == original['key_analysis']
    assert result['default_source'] == 'original'
    assert {s['id'] for s in result['sources']} == {'original',first,second}
    assert (tmp_path/(first+'.lab')).read_text().strip().endswith('C:maj')


def test_pipeline_method_is_validated():
    from app.web.server import PipelineRequest
    assert PipelineRequest().harmony_method == 'stem_evidence_v2'
    with pytest.raises(ValueError): PipelineRequest(harmony_method='made_up')
