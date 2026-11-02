from pathlib import Path
import json
import pytest
from app.web.separation_cache import StageCache,identity


def recipe(**kw):
    return dict(stage='strings',pcm_sha256='input',checkpoint='weights',code='code',outputs=['other','strings'],**kw)


def files(tmp_path):
    result={}
    for name in ['strings','other']:
        p=tmp_path/(name+'.wav');p.write_bytes((name+'-fixture').encode());result[name]=p
    return result


def test_complete_stage_reuse_checks_bytes_and_preserves_order(tmp_path):
    cache=StageCache(tmp_path/'cache');r=recipe();source=files(tmp_path)
    assert cache.restore(r,tmp_path/'before') is None
    cache.save(r,source)
    restored=cache.restore(r,tmp_path/'after')
    assert list(restored)==['strings','other']
    assert all(p.read_bytes()==source[name].read_bytes() for name,p in restored.items())
    # Any input/weights/code/runtime change has its own stage identity.
    for key in ['pcm_sha256','checkpoint','code']:
        different=dict(r);different[key]='changed'
        assert cache.read(different) is None
    restored['strings'].write_bytes(b'corrupt')
    assert cache.read(r) is None


def test_interrupted_cache_write_never_publishes_partial_stage(tmp_path,monkeypatch):
    import app.web.separation_cache as module
    cache=StageCache(tmp_path/'cache');r=recipe();source=files(tmp_path)
    original=module.link_or_copy;count=0
    def interrupted(*args):
        nonlocal count
        count+=1
        if count==2:raise OSError('process stopped during stage publication')
        return original(*args)
    monkeypatch.setattr(module,'link_or_copy',interrupted)
    with pytest.raises(OSError):cache.save(r,source)
    assert cache.read(r) is None
    assert not (cache.root/identity(r)/'manifest.json').exists()
    monkeypatch.setattr(module,'link_or_copy',original)
    cache.save(r,source)
    assert cache.read(r)


def test_failed_refresh_retains_previous_completed_cache(tmp_path,monkeypatch):
    import app.web.separation_cache as module
    cache=StageCache(tmp_path/'cache');r=recipe();source=files(tmp_path);cache.save(r,source)
    before=(cache.root/identity(r)/'manifest.json').read_bytes()
    def interrupted(*args):raise OSError('cancelled')
    monkeypatch.setattr(module,'link_or_copy',interrupted)
    with pytest.raises(OSError):cache.save(r,source,replace=True)
    assert (cache.root/identity(r)/'manifest.json').read_bytes()==before
    assert cache.read(r)


def test_manifest_cannot_point_outside_cache(tmp_path):
    cache=StageCache(tmp_path/'cache');r=recipe();cache.save(r,files(tmp_path))
    path=cache.root/identity(r)/'manifest.json';m=json.loads(path.read_text())
    m['files']['strings']['file']='../../strings.wav'
    path.write_text(json.dumps(m))
    assert cache.read(r) is None


def test_legacy_recipe_alias_requires_exact_validated_new_code(tmp_path,monkeypatch):
    from app.web import strings_stems
    alias=tmp_path/'app/web/models/strings_cache_migration.json';alias.parent.mkdir(parents=True)
    alias.write_text(json.dumps(dict(old_code={'runner':'old'},new_code={'runner':'new','cache':'new-cache'})))
    monkeypatch.setattr(strings_stems,'ROOT',tmp_path)
    old=dict(runner='old',model='original-weights')
    new=dict(runner='new',cache='new-cache',model='original-weights')
    assert strings_stems.compatible_recipe(old,new)
    assert not strings_stems.compatible_recipe(old,dict(new,model='different-weights'))
    assert not strings_stems.compatible_recipe(old,dict(new,runner='future-change'))
