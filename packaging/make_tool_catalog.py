"""Maintainer: pin upstream archives and the exact runtime files we consume."""
from pathlib import Path
import hashlib
import json
import tarfile
import zipfile

HERE = Path(__file__).resolve().parent


def make(cache, installed_vamp):
    cache = Path(cache)
    entries = {}
    def archive(key, name, url, kind, feature='core'):
        path = cache / name
        item = dict(name=name, url=url, bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    kind=kind, feature=feature, files=[])
        entries[key] = item
        return path, item['files']
    def member(files, source, target, data, **extra):
        files.append(dict(source=source, path=target, bytes=len(data),
                          sha256=hashlib.sha256(data).hexdigest(), **extra))
    path, files = archive('ffmpeg', 'ffmpeg-7.1-essentials_build.zip',
        'https://github.com/GyanD/codexffmpeg/releases/download/7.1/ffmpeg-7.1-essentials_build.zip', 'zip')
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            base = Path(name).name
            if base == 'ffmpeg.exe' or (name.count('/') == 1 and not name.endswith('/')):
                member(files, name, 'runtime/tools/ffmpeg/' + base, z.read(name))
    path, files = archive('sonic', 'sonic-annotator-1.7-win64.zip',
        'https://github.com/sonic-visualiser/sonic-annotator/releases/download/sonic-annotator-1.7/sonic-annotator-1.7-win64.zip', 'zip')
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith('/'):
                member(files, name, 'runtime/tools/sonic-annotator/' + name.split('/', 1)[1], z.read(name))
    path, files = archive('nnls', 'Vamp.Plugin.Pack.Installer.2.0.exe',
        'https://github.com/vamp-plugins/vamp-plugin-pack/releases/download/v2.0/Vamp.Plugin.Pack.Installer.2.0.exe', 'resources')
    # Read verified embedded resources; never run the upstream system-wide installer.
    binary = path.read_bytes()
    compressed = json.loads((cache/'nnls-resources.json').read_text())
    for f in sorted(Path(installed_vamp).glob('nnls-chroma*')):
        data = f.read_bytes(); offset = binary.find(data)
        extra = dict(offset=offset, compressed_bytes=len(data), compression='none')
        if offset < 0:
            record = next(r for r in compressed if r['name'] == f.name)
            extra = {key: record[key] for key in ('offset', 'compressed_bytes')}
            extra['compression'] = 'zlib'
        member(files, f.name, 'runtime/tools/vamp/' + f.name, data, **extra)
    path, files = archive('soundfont', 'fluid-soundfont_3.1.orig.tar.gz',
        'https://ftp.debian.org/debian/pool/main/f/fluid-soundfont/fluid-soundfont_3.1.orig.tar.gz', 'tar', 'piano')
    with tarfile.open(path) as t:
        for name in ['FluidR3_GM.sf2', 'COPYING', 'README']:
            source = 'fluid-soundfont-3.1/' + name
            member(files, source, 'assets/soundfonts/' + name, t.extractfile(source).read())
    path, files = archive('fluidsynth', 'fluidsynth-v2.6.2-win10-x64-cpp11.zip',
        'https://github.com/FluidSynth/fluidsynth/releases/download/v2.6.2/fluidsynth-v2.6.2-win10-x64-cpp11.zip', 'zip', 'reconstruction')
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith('/'):
                member(files, name, 'runtime/fluidsynth/' + name, z.read(name))
    result = dict(schema=1, tools=entries)
    (HERE/'tools.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    return {key: len(item['files']) for key, item in entries.items()}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--installed-vamp', type=Path, required=True)
    a = p.parse_args(); print(make(a.cache, a.installed_vamp))
