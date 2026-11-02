from pathlib import Path
import hashlib
import importlib.util
import io
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'packaging'))
from downloads import download,inside
from install_models import selection
import setup_runtime


class Response(io.BytesIO):
    def __init__(self,body,status=200,headers=None):
        super().__init__(body);self.status=status;self.headers=headers or {}


def item(body):
    return dict(url='https://example.test/model.ckpt',bytes=len(body),sha256=hashlib.sha256(body).hexdigest())


def test_resume_from_verified_range_then_reuse_without_network(tmp_path):
    target=tmp_path/'weights';body=b'abcdefgh'
    target.with_name('weights.partial').write_bytes(body[:3]);calls=[]
    def fetch(request,**kwargs):
        calls.append(request.get_header('Range'))
        return Response(body[3:],206,{'Content-Range':'bytes 3-7/8'})
    assert download(item(body),target,opener=fetch)
    assert calls==['bytes=3-'] and target.read_bytes()==body
    assert not download(item(body),target,opener=lambda *a,**k:pytest.fail('Cache hit used network'))


def test_server_without_resume_restarts_instead_of_appending(tmp_path):
    target=tmp_path/'weights';target.with_name('weights.partial').write_bytes(b'wrong')
    body=b'correct whole model'
    download(item(body),target,opener=lambda *a,**k:Response(body))
    assert target.read_bytes()==body


def test_bad_download_preserves_previous_model(tmp_path):
    target=tmp_path/'weights';target.write_bytes(b'previous version')
    with pytest.raises(ValueError,match='校验失败'):
        download(item(b'new model'),target,opener=lambda *a,**k:Response(b'bad model'),retries=1)
    assert target.read_bytes()==b'previous version'


def test_incomplete_transfer_retains_resumable_partial(tmp_path):
    target=tmp_path/'weights'
    with pytest.raises(ValueError,match='断点'):
        download(item(b'abcdef'),target,opener=lambda *a,**k:Response(b'abc'),retries=1)
    assert not target.exists() and target.with_name('weights.partial').read_bytes()==b'abc'


def test_wrong_range_does_not_corrupt_partial(tmp_path):
    target=tmp_path/'weights';target.with_name('weights.partial').write_bytes(b'abc')
    with pytest.raises(ValueError,match='续传范围'):
        download(item(b'abcdef'),target,opener=lambda *a,**k:Response(b'ef',206,{'Content-Range':'bytes 4-5/6'}),retries=1)
    assert target.with_name('weights.partial').read_bytes()==b'abc'


def test_pack_dependency_order_and_directory_boundaries(tmp_path):
    catalog={'packs':{'strings':{'requires':['six']},'guitar':{'requires':['six']},'six':{}}}
    assert selection(catalog,['strings','guitar'])==['six','strings','guitar']
    with pytest.raises(ValueError):selection(catalog,['unknown'])
    with pytest.raises(ValueError):inside(tmp_path,'../elsewhere')
    with pytest.raises(ValueError):inside(tmp_path,str(tmp_path.parent/'outside'))
    assert inside(tmp_path,'guitar/config.yaml')==tmp_path/'guitar/config.yaml'


def test_runtime_dependencies_share_gpu_once():
    assert setup_runtime.dependencies(['reconstruction'])==['core','madmom','gpu','piano','reconstruction']
    assert setup_runtime.dependencies([])==['core','madmom']
    with pytest.raises(ValueError):setup_runtime.dependencies(['unknown'])


@pytest.mark.skipif(sys.platform!='win32',reason='Windows concurrent realpath prefix')
def test_concurrent_parent_creation_keeps_valid_paths_inside(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    # Reproduced in the real installer while two model files shared a new parent.
    # Failures occurred within four iterations before canonical prefix handling.
    for i in range(50):
        root=tmp_path/str(i);root.mkdir()
        def create(name):
            target=inside(root,'new-parent/'+name)
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(b'x')
        with ThreadPoolExecutor(2) as pool:list(pool.map(create,['model.ckpt','config.yaml']))
        assert (root/'new-parent/model.ckpt').read_bytes()==b'x'


@pytest.mark.skipif(sys.platform!='win32',reason='Windows extended paths')
def test_extended_path_normalization_preserves_escape_rejection(tmp_path):
    root=Path('\\\\?\\'+str(tmp_path))
    assert inside(root,'models/weight')==tmp_path/'models/weight'
    with pytest.raises(ValueError):inside(root,'../escape')


def test_prebuilt_wheel_replaces_only_exact_version(tmp_path,monkeypatch):
    import json
    source=tmp_path/'packaging';(source/'requirements').mkdir(parents=True)
    cache=tmp_path/'cache';cache.mkdir();wheels=tmp_path/'wheels';wheels.mkdir()
    wheel=wheels/'pretty_midi-0.2.11-py3-none-any.whl';wheel.write_bytes(b'test wheel')
    record=dict(name=wheel.name,package='pretty-midi',url='https://example.test/'+wheel.name,
                bytes=wheel.stat().st_size,sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
    (source/'built-wheels.json').write_text(json.dumps({'files':[record]}))
    (source/'requirements/core.lock').write_text('pretty-midi==0.2.11.post0 \\\n    --hash=sha256:original\n')
    (source/'requirements/piano.lock').write_text('pretty-midi==0.2.11 \\\n    --hash=sha256:source\nnumpy==2.5.4\n')
    monkeypatch.setattr(setup_runtime,'HERE',source)
    assert 'pretty-midi==0.2.11.post0' in setup_runtime.wheel_lock('core',cache,wheels).read_text()
    actual=setup_runtime.wheel_lock('piano',cache,wheels).read_text()
    assert wheel.as_uri() in actual and record['sha256'] in actual
    assert 'numpy==2.5.4' in actual and '\n+' not in actual
