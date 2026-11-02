"""Install/repair a release. Project data is separate from replaceable program files."""
from pathlib import Path
from contextlib import contextmanager
import argparse
import ctypes
from ctypes import wintypes
import json
import os
import shutil
import sys
import uuid
from configure import configure, write
from downloads import inside, ready
from install_models import install as install_models
from install_models import selection
from install_tools import install as install_tools
from setup_runtime import install as install_runtime
from setup_runtime import dependencies

HERE = Path(__file__).resolve().parent
RESERVED = ('backend', 'runtime', 'assets', 'setup', '.setup-cache', '.backups')


def feature_plan(features, packs):
    """Resolve feature prerequisites even when default separation is unchecked."""
    selected = set(features)
    models = set(packs)
    if 'reconstruction' in models or 'reconstruction' in selected:
        selected.add('reconstruction')
        models.add('reconstruction')
    roles = dependencies(selected)
    # Transcribing a mixture starts by extracting its piano part. Standalone
    # piano input remains available, but must not be the only working mode.
    if 'piano' in roles: models.add('six_stems')
    if models and 'gpu' not in roles:
        selected.add('gpu')
        roles = dependencies(selected)
    return roles, sorted(models)


def validate_layout(home, data):
    home = Path(home).resolve(); data = Path(data).resolve()
    if home == Path(home.anchor) or home == Path.home().resolve():
        raise ValueError('请选择独立的 Chord Studio 安装文件夹')
    if home == data or home.is_relative_to(data):
        raise ValueError('项目目录不能包含程序目录')
    if any(data == home / part or data.is_relative_to(home / part) for part in RESERVED):
        raise ValueError('项目目录不能放在程序、缓存或运行环境文件夹内')
    owner = home / 'setup-owner.json'
    if home.exists() and any(home.iterdir()):
        if not owner.is_file():
            raise ValueError('所选目录包含其他文件。请选择空目录；旧版项目可保留在原位置。')
        saved = json.loads(owner.read_text(encoding='utf8'))
        if saved.get('application') != 'Chord Studio' or Path(saved['home']).resolve() != home:
            raise ValueError('安装目录的归属记录不匹配')
        if Path(saved['data_dir']).resolve() != data:
            raise ValueError('修复或升级时请保留原项目目录；安装程序不会自动搬移项目')
    return home, data


@contextmanager
def installation_lock(home, data):
    """Use the launcher's same named mutex, so no application can start mid-upgrade."""
    if os.name != 'nt': raise RuntimeError('此安装程序支持 Windows 10/11 x64')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    name = 'Local\\ChordStudio-' + str(data).replace('\\', '_').replace('/', '_').replace(':', '_').lower()
    handle = kernel.CreateMutexW(None, False, name)
    error = ctypes.get_last_error()
    if not handle: raise ctypes.WinError(error)
    try:
        if error == 183:
            raise RuntimeError('Chord Studio 或另一安装任务正在使用此项目目录。请关闭应用后重试，已有任务不会被强制中断。')
        with (home / '.setup.lock').open('a+b') as lock:
            import msvcrt
            lock.seek(0, 2)
            if not lock.tell(): lock.write(b'0'); lock.flush()
            lock.seek(0)
            try: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError: raise RuntimeError('已有安装任务正在运行') from None
            yield
    finally:
        kernel.CloseHandle(handle)


def verify_release(source):
    manifest = json.loads((source / 'release-files.json').read_text(encoding='utf8'))
    for item in manifest['files']:
        if not ready(inside(source, item['path']), item):
            raise ValueError('安装包文件不完整：' + item['path'])
    if not (source / 'backend/application-files.json').is_file():
        raise ValueError('缺少应用文件清单')
    return manifest


def commit_program(home, staged, source, data, version):
    """Keep the previous program as a rollback copy; never rename/delete data."""
    tag = uuid.uuid4().hex
    backup = home / '.backups' / tag
    backup.mkdir(parents=True)
    changed = []
    try:
        candidates = [('backend', staged), ('chord-studio.exe', source / 'chord-studio.exe')]
        for name, replacement in candidates:
            destination = home / name
            saved = backup / name
            had_old = destination.exists()
            if had_old: destination.rename(saved)
            changed.append((destination, saved, had_old))
            if name == 'backend': replacement.rename(destination)
            else: shutil.copyfile(replacement, destination)
        record = dict(schema=1, application='Chord Studio', version=version,
                      data_dir=str(data), backend='backend', web_python='runtime/core/Scripts/python.exe')
        write(home / 'installation.json', record)
    except BaseException:
        # Move the failed new program aside before restoring; no recursive delete.
        for destination, saved, had_old in reversed(changed):
            if destination.exists(): destination.rename(backup / ('failed-' + destination.name))
            if had_old: saved.rename(destination)
        raise


def install(home, data, source, uv, features, packs, wheelhouse=None, tool_cache=None):
    features, packs = feature_plan(features, packs)
    home, data = validate_layout(home, data)
    source = Path(source).resolve()
    if home == source or source.is_relative_to(home) or home.is_relative_to(source):
        raise ValueError('安装目录与解压后的安装包目录必须分开')
    manifest = verify_release(source)
    home.mkdir(parents=True, exist_ok=True)
    with installation_lock(home, data):
        write(home/'setup-owner.json', dict(application='Chord Studio', schema=1,
                                          home=str(home), data_dir=str(data)))
        roles = dependencies(features)
        if packs:
            selection(json.loads((HERE/'model-packs.json').read_text(encoding='utf8')), packs)
            if 'gpu' not in roles: raise ValueError('所选分轨模型需要勾选分轨运行环境')
        progress(dict(stage='start', message='正在准备所选功能，项目数据保留在独立目录'))
        install_runtime(home, features, uv, wheelhouse)
        install_tools(home, roles, progress, tool_cache)
        if packs: install_models(data/'models', packs, progress)
        stage = home / ('.setup-stage-' + uuid.uuid4().hex)
        shutil.copytree(source/'backend', stage)
        result = configure(home, data, backend_dir=stage, publish=False, version=manifest['version'])
        commit_program(home, stage, source, data, manifest['version'])
        # Retain installers for adding model packs or repairing the same installation.
        setup = home/'setup'; setup.mkdir(exist_ok=True)
        for item in manifest['files']:
            if item['path'].startswith('packaging/'):
                target = inside(setup, item['path'][len('packaging/'):])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(inside(source, item['path']), target)
        progress(dict(stage='complete', message='安装完成', home=str(home),
                      data_dir=str(data), model=result['model']))
    return result


def progress(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--uv', type=Path, required=True)
    p.add_argument('--features', nargs='*', default=['gpu', 'piano'])
    p.add_argument('--packs', nargs='*', default=['strings_v2', 'guitar_timbre'])
    p.add_argument('--wheelhouse', type=Path)
    p.add_argument('--tool-cache', type=Path)
    a = p.parse_args()
    try:
        install(a.home, a.data_dir, a.source, a.uv, a.features, a.packs, a.wheelhouse, a.tool_cache)
    except Exception as e:
        progress(dict(stage='failed', message=str(e)))
        raise
