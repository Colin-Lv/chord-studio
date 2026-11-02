"""Install isolated, pinned runtimes; share CUDA once across transcription tools."""
from pathlib import Path
import argparse
import json
import os
import re
import subprocess
import sys
from downloads import download, ready

HERE=Path(__file__).resolve().parent
VERSIONS={'core':'3.13.12','madmom':'3.9.25','gpu':'3.13.12','piano':'3.13.12','reconstruction':'3.13.12'}


def dependencies(features):
    selected={'core','madmom'}|set(features)
    if not selected<=set(VERSIONS):raise ValueError('未知运行环境')
    if 'reconstruction' in selected:selected.add('piano')
    if 'piano' in selected:selected.add('gpu')
    return [role for role in VERSIONS if role in selected]


def wheel_lock(role,cache,wheelhouse=None):
    """Replace only matching source-version entries with our prebuilt wheel hash."""
    catalog=json.loads((HERE/'built-wheels.json').read_text(encoding='utf8'))['files']
    source=(HERE/'requirements'/f'{role}.lock').read_text(encoding='utf8')
    blocks=re.split(r'(?m)(?=^[a-zA-Z0-9][^\n]*==)',source)
    output=[]
    for block in blocks:
        first=block.splitlines()[0] if block else ''
        match=re.match(r'^([\w.-]+)==([^\s\\]+)',first)
        name=match[1].lower().replace('_','-') if match else None
        wheel=next((w for w in catalog if w['package'].lower()==name and w['name'].split('-')[1]==match[2]),None) if match else None
        if wheel:
            target=cache/'wheels'/wheel['name']
            local=Path(wheelhouse)/wheel['name'] if wheelhouse else None
            if local and ready(local,wheel):target=local.resolve()
            else:download(wheel,target,progress)
            output.append(f"{match[1]} @ {target.resolve().as_uri()} \\\n    --hash=sha256:{wheel['sha256']}\n")
        else:output.append(block)
    target=cache/f'{role}.install.lock'
    target.write_text(''.join(output),encoding='utf8')
    return target


def progress(value):
    print(json.dumps(value,ensure_ascii=False),flush=True)


def install(home,features,uv,wheelhouse=None):
    if os.name!='nt':raise RuntimeError('此交付支持 Windows 10/11 x64')
    home=Path(home).resolve();runtime=home/'runtime';runtime.mkdir(parents=True,exist_ok=True)
    uv=Path(uv).resolve()
    if not uv.is_file():raise FileNotFoundError('安装工具未准备完整')
    roles=dependencies(features)
    cache=home/'.setup-cache';cache.mkdir(exist_ok=True)
    env=dict(os.environ,UV_PYTHON_INSTALL_DIR=str(runtime/'python'),UV_PYTHON_INSTALL_REGISTRY='false',
             UV_PYTHON_BIN_DIR=str(runtime/'bin'),UV_CACHE_DIR=str(cache/'uv'),UV_LINK_MODE='hardlink',
             PYTHONNOUSERSITE='1',PYTHONUTF8='1')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    def run(arguments):
        with (home/'runtime-install.log').open('a',encoding='utf8') as log:
            result=subprocess.run([str(a) for a in arguments],env=env,stdout=log,stderr=subprocess.STDOUT,
                                  timeout=7200,creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:raise RuntimeError('运行环境配置失败，详情见 runtime-install.log；再次安装会复用已下载内容。')
    progress(dict(stage='python',message='正在准备应用专用运行环境'))
    run([uv,'python','install',*sorted({VERSIONS[r] for r in roles}),'--no-registry','--no-bin'])
    for role in roles:
        target=runtime/role;python=target/'Scripts/python.exe'
        base=runtime/'python'/f'cpython-{VERSIONS[role]}-windows-x86_64-none/python.exe'
        if not python.is_file():run([uv,'venv','--python',base,target])
        progress(dict(stage=role,message='正在配置 '+role))
        lock=wheel_lock(role,cache,wheelhouse)
        if role=='gpu':run([uv,'pip','install','--python',python,'--no-deps','--require-hashes','-r',HERE/'requirements/gpu-cuda.in'])
        # --no-deps is intentional: every used package is explicitly locked, while
        # unused training/converter backends are not installed as side effects.
        run([uv,'pip','install','--python',python,'--no-deps','--require-hashes','--only-binary',':all:','-r',lock])
        if role in {'piano','reconstruction'}:
            shared=['../../../gpu/Lib/site-packages']
            if role=='reconstruction':shared.insert(0,'../../../piano/Lib/site-packages')
            (target/'Lib/site-packages/shared-runtime.pth').write_text('\n'.join(shared)+'\n',encoding='utf8')
        check={
            'core':'import fastapi,uvicorn,numpy,scipy,librosa,pretty_midi',
            'madmom':'import madmom,numpy,scipy',
            'gpu':'import torch,torchaudio;from audio_separator.separator import Separator;import demucs',
            'piano':'import torch,transkun,ncls,moduleconf;from transkun.Data import writeMidi',
            'reconstruction':'import torch,basic_pitch,transformers,pytorch_lightning',
        }[role]
        run([python,'-I','-c',check])
    record=runtime/'installed-runtimes.json'
    previous=json.loads(record.read_text()) if record.exists() else {'schema':1,'roles':{}}
    for role in roles:previous['roles'][role]={'python_version':VERSIONS[role],'python':f'{role}/Scripts/python.exe'}
    pending=record.with_suffix('.pending');pending.write_text(json.dumps(previous,indent=2),encoding='utf8');os.replace(pending,record)
    progress(dict(stage='complete',roles=roles,message='所选运行环境已准备完成'))
    return previous


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home',type=Path,required=True)
    p.add_argument('--uv',type=Path,required=True)
    p.add_argument('--features',nargs='*',default=['gpu'])
    p.add_argument('--wheelhouse',type=Path)
    a=p.parse_args();install(a.home,a.features,a.uv,a.wheelhouse)
