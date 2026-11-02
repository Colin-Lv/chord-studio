"""Build a clean release directory, including a manifest of every distributed file."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
from build_backend import build

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def release(destination, version='1.1.0'):
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('请使用空的发布目录')
    if destination == ROOT or ROOT.is_relative_to(destination):
        raise ValueError('发布目录不能包含源代码目录')
    launcher = ROOT / 'desktop/target/release/chord-studio.exe'
    if not launcher.is_file(): raise FileNotFoundError('请先构建桌面启动程序')
    build(destination/'backend')
    shutil.copyfile(launcher, destination/'chord-studio.exe')
    for f in HERE.rglob('*'):
        if f.is_file() and '__pycache__' not in f.parts and f.suffix in {'.py','.json','.ps1','.in','.lock','.constraints'}:
            target=destination/'packaging'/f.relative_to(HERE)
            target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(f,target)
    for name in ['Install.cmd','Install.vbs']:
        shutil.copyfile(HERE/name,destination/name)
    for name in ['README.md','THIRD_PARTY.md','LICENSE']:
        if (ROOT/name).is_file(): shutil.copyfile(ROOT/name,destination/name)
    for folder in ['docs/product','docs/screenshots']:
        for f in (ROOT/folder).glob('*'):
            if f.is_file() and f.suffix in {'.svg','.png'}:
                target=destination/folder/f.name
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
    files=[]
    for f in sorted(destination.rglob('*')):
        if f.is_file():
            files.append(dict(path=f.relative_to(destination).as_posix(),bytes=f.stat().st_size,
                              sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    manifest=dict(schema=1,version=version,files=files)
    (destination/'release-files.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    return dict(files=len(files),bytes=sum(f['bytes'] for f in files),version=version)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--version',default='1.1.0')
    a=p.parse_args();print(release(a.destination,a.version))
