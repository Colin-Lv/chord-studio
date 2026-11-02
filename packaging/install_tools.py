"""Install only pinned files from verified upstream archives into the app directory."""
from pathlib import Path
import hashlib
import json
import os
import tarfile
import zipfile
import zlib
from downloads import download, inside, ready

HERE = Path(__file__).resolve().parent


def extract(item, archive, home):
    """No archive filenames control destinations; the checked-in manifest does."""
    archive = Path(archive)
    if not ready(archive, item):
        raise ValueError('工具安装包校验失败')
    z = zipfile.ZipFile(archive) if item['kind'] == 'zip' else None
    t = tarfile.open(archive) if item['kind'] == 'tar' else None
    blob = archive.read_bytes() if item['kind'] == 'resources' else None
    try:
        for record in item['files']:
            target = inside(home, record['path'])
            if ready(target, record):
                continue
            if z:
                info = z.getinfo(record['source'])
                if info.file_size != record['bytes']: raise ValueError('压缩包文件大小不符')
                data = z.read(info)
            elif t:
                info = t.getmember(record['source'])
                if not info.isfile() or info.size != record['bytes']: raise ValueError('压缩包文件类型或大小不符')
                data = t.extractfile(info).read()
            elif blob is not None:
                begin = record['offset']; end = begin + record['compressed_bytes']
                if begin < 0 or end > len(blob): raise ValueError('嵌入资源范围不符')
                data = blob[begin:end]
                if record['compression'] == 'zlib':
                    decoder = zlib.decompressobj()
                    data = decoder.decompress(data, record['bytes'] + 1)
                    if not decoder.eof or decoder.unused_data: raise ValueError('压缩资源不完整')
                elif record['compression'] != 'none': raise ValueError('未知资源压缩格式')
            else:
                raise ValueError('未知工具包类型')
            if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
                raise ValueError('工具文件校验失败：' + record['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            pending = target.with_name(target.name + '.pending')
            if pending.is_symlink(): raise ValueError('工具临时文件不能是符号链接')
            pending.write_bytes(data)
            os.replace(pending, target)
    finally:
        if z: z.close()
        if t: t.close()


def install(home, features, progress=print, cache=None):
    home = Path(home).resolve()
    cache = Path(cache) if cache else home / '.setup-cache/tools'
    catalog = json.loads((HERE/'tools.json').read_text(encoding='utf8'))
    selected = {'core'} | set(features)
    if 'reconstruction' in selected: selected.add('piano')
    for key, item in catalog['tools'].items():
        if item['feature'] not in selected: continue
        if all(ready(inside(home, f['path']), f) for f in item['files']):
            progress(dict(stage='tools', file=key, state='cached')); continue
        path = inside(cache, item['name'])
        download(item, path, progress)
        extract(item, path, home)
        progress(dict(stage='tools', file=key, state='ready'))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home', type=Path, required=True)
    p.add_argument('--features', nargs='*', default=[])
    p.add_argument('--cache', type=Path)
    a = p.parse_args()
    install(a.home, a.features, lambda v: print(json.dumps(v, ensure_ascii=False), flush=True), a.cache)
