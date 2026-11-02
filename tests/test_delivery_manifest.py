import importlib.util
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('delivery_build',ROOT/'packaging/build_backend.py')
build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)


def test_application_payload_includes_runtime_checks_and_excludes_user_data():
    paths={p.relative_to(ROOT).as_posix() for p in build.selected_files()}
    assert {'app/runtime_paths.py','app/web/job_process.py','frontend/task-queue.js',
            'app/web/vendor/MSST-LICENSE.txt','inference/model/example/cells.json',
            'inference/model/example/features.npz','inference/model/retained10.npz'} <= paths
    assert 'inference/runtime.local.json' not in paths
    assert 'app/web/models/bass_voice_v1.joblib' not in paths
    assert {'app/web/models/legacy_v1.joblib','app/web/models/legacy_v1.contract.json',
            'app/web/models/legacy_v2.joblib','app/web/models/legacy_v2.contract.json'} <= paths
    assert 'app/studio_assets/studio.html' in paths
    assert not any(p.startswith(('projects/','research/','.tmp/','frontend/midi-harmony-heart/')) for p in paths)
    assert not any(Path(p).suffix in {'.wav','.mp3','.m4a','.flac','.log'} for p in paths)
    assert not any('active-session' in p for p in paths)


def test_builder_refuses_to_overwrite_existing_installation(tmp_path):
    import pytest
    (tmp_path/'user-file.txt').write_text('keep me')
    with pytest.raises(ValueError,match='empty staging'):
        build.build(tmp_path)


def test_frozen_model_bytes_survive_source_checkout():
    import hashlib,json
    folder=ROOT/'inference/model'
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    for name,expected in manifest['files_sha256'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected,name
    migration=json.loads((ROOT/'app/web/models/strings_cache_migration.json').read_text(encoding='utf8'))
    for name,expected in migration['new_code'].items():
        assert hashlib.sha256((ROOT/'app/web'/name).read_bytes()).hexdigest()==expected,name
    assert (tmp_path/'user-file.txt').read_text()=='keep me'
