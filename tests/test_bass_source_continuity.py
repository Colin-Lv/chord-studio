import json
from types import SimpleNamespace
import pytest
from app.web import stem_bass_harmony as bass
from app.web import stems_worker
from app.web.storage import signature


def test_default_bass_preference_is_bound_to_audio_and_source(tmp_path):
    previous=dict(default_source='stem_bass_harmony_v2_abc',audio={'sha256':'same'},
                  stem_bass_harmony={'source_id':'stem_bass_harmony_v2_abc','model':'bs_roformer_sw'})
    path=tmp_path/'analysis.json';path.write_text(json.dumps(previous))
    store=SimpleNamespace(run_dir=lambda *args:tmp_path)
    project={'id':'p','current_run':'old','audio_identity':{'sha256':'same'}}
    assert bass.selected_bass_model(store,project)=='bs_roformer_sw'
    assert bass.selected_bass_model(store,dict(project,audio_identity={'sha256':'changed'})) is None
    previous['default_source']='retained10_full';path.write_text(json.dumps(previous))
    assert bass.selected_bass_model(store,project) is None


def test_reanalysis_recomputes_new_evidence_and_publishes_default(tmp_path,monkeypatch):
    audio=tmp_path/'audio.wav';audio.write_bytes(b'original audio')
    project={'id':'p','current_run':'old','audio':'audio.wav','audio_identity':signature(audio)}
    store=SimpleNamespace(project_dir=lambda *args:tmp_path)
    cached=tmp_path/'stems';cached.mkdir();(cached/'manifest.json').write_text('{}')
    fresh=tmp_path/'new-run';fresh.mkdir();seen={}
    monkeypatch.setattr(stems_worker,'stems_dir',lambda sha,model:cached)
    def recognize(run,base,out,publish):
        assert run==fresh and base==cached
        assert out==fresh/bass.VERSION/'bs_roformer_sw'
        seen['recomputed']=True
        return [{'start':0,'end':2,'label':'B:maj/C#'}],{'model':'bs_roformer_sw'}
    def publish(run,rows,meta,make_default=False):
        assert seen['recomputed'] and run==fresh and make_default
        assert rows[0]['label']=='B:maj/C#'
        return 'stem_bass_harmony_v2_new'
    monkeypatch.setattr(bass,'recognize',recognize);monkeypatch.setattr(bass,'publish_result',publish)
    result=bass.restore_selected_source(store,project,fresh,'bs_roformer_sw')
    assert result['recomputed'] and result['source_id']=='stem_bass_harmony_v2_new'


def test_missing_stems_cannot_silently_copy_old_labels(tmp_path,monkeypatch):
    audio=tmp_path/'audio.wav';audio.write_bytes(b'audio')
    project={'id':'p','audio':'audio.wav','audio_identity':signature(audio)}
    monkeypatch.setattr(stems_worker,'stems_dir',lambda *args:tmp_path/'absent')
    with pytest.raises(ValueError,match='缓存不可用'):
        bass.restore_selected_source(SimpleNamespace(project_dir=lambda *args:tmp_path),project,tmp_path/'new','bs_roformer_sw')
