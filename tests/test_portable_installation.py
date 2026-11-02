import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'packaging'))
spec = importlib.util.spec_from_file_location('portable_install', ROOT/'packaging/install.py')
installer = importlib.util.module_from_spec(spec); spec.loader.exec_module(installer)
import install_tools


def record(data):
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def test_piano_and_reconstruction_install_without_default_combo():
    roles, packs = installer.feature_plan(['piano'], [])
    assert {'core', 'madmom', 'piano', 'gpu'} == set(roles)
    assert packs == ['six_stems']
    roles, packs = installer.feature_plan([], ['reconstruction'])
    assert {'piano', 'gpu', 'reconstruction'} <= set(roles)
    assert set(packs) == {'six_stems', 'reconstruction'}
    assert installer.feature_plan([], []) == (['core', 'madmom'], [])


def test_never_adopts_nonempty_unowned_directory(tmp_path):
    home = tmp_path/'music'
    home.mkdir(); (home/'song.mid').write_bytes(b'private song')
    with pytest.raises(ValueError, match='其他文件'):
        installer.validate_layout(home, home/'projects')
    assert (home/'song.mid').read_bytes() == b'private song'


def test_project_directory_is_outside_replaceable_program(tmp_path):
    home = tmp_path/'studio'
    for data in [home, home/'runtime'/'music', home/'backend'/'projects', tmp_path]:
        with pytest.raises(ValueError): installer.validate_layout(home, data)
    assert installer.validate_layout(home, home/'user-data') == (home, home/'user-data')


def test_failed_program_commit_restores_old_program_and_keeps_projects(tmp_path, monkeypatch):
    home = tmp_path/'studio'; home.mkdir()
    (home/'backend').mkdir(); (home/'backend/app.py').write_text('old')
    (home/'chord-studio.exe').write_bytes(b'old native')
    (home/'installation.json').write_text('old configuration')
    data=home/'user-data';data.mkdir();(data/'song.wav').write_bytes(b'private audio')
    stage=home/'.stage';stage.mkdir();(stage/'app.py').write_text('new')
    source=tmp_path/'source';source.mkdir();(source/'chord-studio.exe').write_bytes(b'new native')
    def fail(*args, **kwargs):raise OSError('native exe unavailable')
    monkeypatch.setattr(installer.shutil,'copyfile',fail)
    with pytest.raises(OSError):installer.commit_program(home,stage,source,data,'1.1.0')
    assert (home/'backend/app.py').read_text() == 'old'
    assert (home/'chord-studio.exe').read_bytes() == b'old native'
    assert (home/'installation.json').read_text() == 'old configuration'
    assert (data/'song.wav').read_bytes() == b'private audio'


def test_tool_extraction_uses_verified_allowlist_only(tmp_path):
    archive=tmp_path/'tools.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('good.exe',b'expected binary')
        z.writestr('../../outside.txt',b'unrelated')
    item=dict(kind='zip',**record(archive.read_bytes()),files=[
        dict(source='good.exe',path='runtime/tool.exe',**record(b'expected binary'))])
    home=tmp_path/'home'
    install_tools.extract(item,archive,home)
    assert (home/'runtime/tool.exe').read_bytes() == b'expected binary'
    assert not (tmp_path/'outside.txt').exists()
    item['files'][0]['path']='../outside.txt'
    with pytest.raises(ValueError,match='超出'):install_tools.extract(item,archive,home)


def test_corrupt_tool_keeps_previous_binary(tmp_path):
    archive=tmp_path/'tools.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('tool.exe',b'incorrect')
    home=tmp_path/'home';home.mkdir();(home/'tool.exe').write_bytes(b'previous')
    item=dict(kind='zip',**record(archive.read_bytes()),files=[
        dict(source='tool.exe',path='tool.exe',**record(b'expected!'))])
    with pytest.raises(ValueError,match='校验失败'):install_tools.extract(item,archive,home)
    assert (home/'tool.exe').read_bytes() == b'previous'


@pytest.mark.skipif(os.name!='nt',reason='Windows launcher mutex')
def test_installer_and_launcher_share_single_directory_ownership(tmp_path):
    home=tmp_path/'home';home.mkdir();data=home/'projects'
    with installer.installation_lock(home,data):
        with pytest.raises(RuntimeError,match='正在使用'):
            with installer.installation_lock(home,data):pass
    with installer.installation_lock(home,data):pass
