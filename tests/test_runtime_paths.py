import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from app import runtime_paths as paths


@pytest.fixture
def installation(tmp_path, monkeypatch):
    app = tmp_path / '另一台电脑' / 'Chord Studio' / 'backend'
    app.mkdir(parents=True)
    monkeypatch.setattr(paths, 'APP_ROOT', app)
    for key in list(paths.os.environ):
        if key.startswith(('CHORD_', 'HARMONY_', 'STEMS_')):
            monkeypatch.delenv(key)
    (app / 'desktop_runtime.json').write_text(json.dumps({
        'data_root': '../user data', 'runtime_root': '../runtime',
        'assets_root': '../assets', 'model_python': '../runtime/core/python.exe',
        'legacy_root': '.',
    }), encoding='utf8')
    monkeypatch.chdir(tmp_path)
    return app


def test_installation_paths_do_not_depend_on_working_directory(installation):
    home = installation.parent
    assert paths.data_root() == home / 'user data'
    assert paths.python_path('model') == home / 'runtime/core/python.exe'
    assert paths.python_path('piano') == home / 'runtime/piano/Scripts/python.exe'
    assert paths.models_root() == home / 'user data/models'
    assert paths.soundfont_path() == home / 'assets/soundfonts/FluidR3_GM.sf2'


def test_custom_locations_and_child_environment(installation, monkeypatch, tmp_path):
    data = tmp_path / 'different disk' / 'music projects'
    model = tmp_path / 'optional models'
    monkeypatch.setenv('CHORD_DESKTOP_DATA', str(data))
    monkeypatch.setenv('CHORD_MODELS_ROOT', str(model))
    # Existing explicit upstream model overrides must not be overwritten.
    monkeypatch.setenv('STEMS_SEP_MODEL_DIR', str(tmp_path / 'six track'))
    paths.prepare_environment()
    env = paths.os.environ
    assert Path(env['CHORD_DESKTOP_DATA']) == data
    assert Path(env['CHORD_RECONSTRUCTION_MODELS']) == model / 'reconstruction'
    assert Path(env['STEMS_SEP_MODEL_DIR']) == tmp_path / 'six track'
    assert Path(env['CHORD_GPU_PYTHON']) == installation.parent / 'runtime/gpu/Scripts/python.exe'
    assert Path(env['HARMONY_LEGACY_ROOT']) == installation


def test_malformed_config_is_not_silently_replaced(installation):
    (installation/'desktop_runtime.json').write_text('{broken', encoding='utf8')
    with pytest.raises(json.JSONDecodeError):
        paths.data_root()


@pytest.mark.parametrize('cuda', [True, False])
def test_audio_environment_is_used_on_gpu_and_cpu(tmp_path, monkeypatch, cuda):
    from app.web import stems_worker as worker
    exe = tmp_path/'audio-python.exe'
    exe.touch()
    monkeypatch.setattr(worker, 'python_path', lambda role: exe)
    monkeypatch.setattr(worker, 'GPU_PYTHON', exe)
    calls = []
    def probe(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=b'1' if cuda else b'0')
    monkeypatch.setattr(worker.subprocess, 'run', probe)
    monkeypatch.setattr(worker, '_model_python', lambda: 'harmony-only-python')
    assert worker._runtime() == (str(exe), 'cuda' if cuda else 'cpu')
    assert len(calls) == 1


def test_unusable_audio_environment_falls_back_explicitly(tmp_path, monkeypatch):
    from app.web import stems_worker as worker
    missing = tmp_path/'missing-python.exe'
    monkeypatch.setattr(worker, 'python_path', lambda role: missing)
    monkeypatch.setattr(worker, 'GPU_PYTHON', missing)
    monkeypatch.setattr(worker, '_model_python', lambda: 'core-python')
    assert worker._runtime() == ('core-python', 'cpu')
