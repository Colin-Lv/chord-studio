"""Installation-relative resources shared by the desktop and audio workers.

Environment overrides win over desktop_runtime.json. Relative configuration paths
are anchored to the application, never to the caller's current working directory.
This module intentionally imports no audio/ML packages and creates no directories.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT).startswith('\\\\?\\'):
    APP_ROOT = Path(str(APP_ROOT)[4:])


def config():
    path = APP_ROOT / 'desktop_runtime.json'
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else {}


def configured_path(key, env=None, default=None):
    value = os.environ.get(env) if env else None
    if not value:
        value = config().get(key)
    if not value:
        value = default
    if value is None:
        return None
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else APP_ROOT / path).resolve()


def data_root():
    default = Path(os.environ.get('LOCALAPPDATA', Path.home() / '.local/share')) / 'Chord Studio/data'
    return configured_path('data_root', 'CHORD_DESKTOP_DATA', default)


def runtime_root():
    return configured_path('runtime_root', 'CHORD_RUNTIME_ROOT', APP_ROOT.parent / 'runtime')


def models_root():
    return configured_path('models_root', 'CHORD_MODELS_ROOT', data_root() / 'models')


def assets_root():
    return configured_path('assets_root', 'CHORD_ASSETS_ROOT', APP_ROOT.parent / 'assets')


def python_path(role):
    keys = {
        'web': ('web_python', 'CHORD_WEB_PYTHON'),
        'model': ('model_python', 'HARMONY_MODEL_PYTHON'),
        'gpu': ('stems_python', 'CHORD_GPU_PYTHON'),
        'piano': ('piano_python', 'CHORD_PIANO_PYTHON'),
        'reconstruction': ('reconstruction_python', 'CHORD_RECONSTRUCTION_PYTHON'),
        'madmom': ('madmom_python', 'CHORD_MADMOM_PYTHON'),
    }
    key, env = keys[role]
    default = Path(sys.executable) if role in {'web', 'model'} else runtime_root() / role / 'Scripts/python.exe'
    return configured_path(key, env, default)


def soundfont_path():
    return configured_path('soundfont', 'CHORD_SOUNDFONT', assets_root() / 'soundfonts/FluidR3_GM.sf2')


def fluidsynth_path():
    return configured_path('fluidsynth_dll', 'CHORD_FLUIDSYNTH_DLL',
                           runtime_root() / 'fluidsynth/fluidsynth-v2.6.2-win10-x64-cpp11/bin/libfluidsynth-3.dll')


def prepare_environment():
    """Resolve once at launch; child processes receive absolute resource locations."""
    values = {
        'CHORD_DESKTOP_DATA': data_root(), 'CHORD_RUNTIME_ROOT': runtime_root(),
        'CHORD_MODELS_ROOT': models_root(), 'CHORD_ASSETS_ROOT': assets_root(),
        'CHORD_SOUNDFONT': soundfont_path(), 'CHORD_FLUIDSYNTH_DLL': fluidsynth_path(),
        'HARMONY_LEGACY_ROOT': configured_path('legacy_root', 'HARMONY_LEGACY_ROOT', APP_ROOT),
        'STEMS_SEP_MODEL_DIR': models_root() / 'audio_separator',
        'STEMS_ROFORMER_DIR': models_root() / 'bsroformer',
        'STEMS_DRUMSEP_DIR': models_root() / 'drumsep',
        'CHORD_RECONSTRUCTION_MODELS': models_root() / 'reconstruction',
        # Existing installations retain their cache; new installers bind this to
        # the selected model directory rather than the Windows user profile.
        'TORCH_HOME': configured_path('torch_home', 'TORCH_HOME', Path.home() / '.cache/torch'),
    }
    for role, env in [('model','HARMONY_MODEL_PYTHON'), ('gpu','CHORD_GPU_PYTHON'),
                      ('piano','CHORD_PIANO_PYTHON'), ('reconstruction','CHORD_RECONSTRUCTION_PYTHON'),
                      ('madmom','CHORD_MADMOM_PYTHON')]:
        values[env] = python_path(role)
    for name, value in values.items():
        os.environ.setdefault(name, str(value))
    return values
