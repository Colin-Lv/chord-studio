"""Generate runtime bindings after installing into the user's chosen directory.

Only paths and their verified identities change. Acoustic code and weights are
checked against the original application manifest before configuration is written.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path, payload):
    pending=path.with_suffix('.pending')
    pending.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    pending.replace(path)


def configure(home, data, backend_dir=None, publish=True, version='1.1.1'):
    home=Path(home).resolve();data=Path(data).resolve()
    backend=Path(backend_dir).resolve() if backend_dir else home/'backend'
    runtime=home/'runtime'
    # Mandatory core dependencies are copied/provisioned into this installation.
    targets={
        'host_python':runtime/'core/Scripts/python.exe',
        'ffmpeg':runtime/'tools/ffmpeg/ffmpeg.exe',
        'sonic_annotator':runtime/'tools/sonic-annotator/sonic-annotator.exe',
        'nnls_plugin':runtime/'tools/vamp/nnls-chroma.dll',
        'nnls_description':runtime/'tools/vamp/nnls-chroma.n3',
        'nnls_categories':runtime/'tools/vamp/nnls-chroma.cat',
        'madmom_python':runtime/'madmom/Scripts/python.exe',
    }
    bound={name:dict(path=str(path),sha256=digest(path)) for name,path in targets.items()}
    bound['madmom_version']='0.16.1'
    weights=runtime/'madmom/Lib/site-packages/madmom/models'
    bound['madmom_models']={name:dict(path=str(weights/relative),sha256=digest(weights/relative))
                           for name,relative in {
                               'CNN':'chords/2016/chords_cnnfeat.pkl',
                               'CRF':'chords/2016/chords_cnncrf.pkl',
                               'DeepChroma':'chroma/2016/chroma_dnn.pkl',
                           }.items()}
    original=json.loads((backend/'application-files.json').read_text(encoding='utf8'))
    for entry in original['files']:
        if entry['path']=='inference/audio_manifest.json':
            continue  # Its runtime binding is generated per installation below.
        if digest(backend/entry['path'])!=entry['sha256']:
            raise ValueError('Application file changed before configuration: '+entry['path'])
    model=json.loads((backend/'inference/model/manifest.json').read_text(encoding='utf8'))
    frozen=digest(backend/'inference/model/retained10.npz')
    for name, expected in model['files_sha256'].items():
        if digest(backend/'inference/model'/name)!=expected:
            raise ValueError('Frozen model file mismatch: '+name)
    settings=dict(schema=2,application_version=version,web_python=str(targets['host_python']),model_python=str(targets['host_python']),
                  legacy_root=str(home/'backend'),data_root=str(data),runtime_root=str(runtime),
                  assets_root=str(home/'assets'),models_root=str(data/'models'),
                  torch_home=str(data/'models/torch'),
                  legacy_v1_model=str(home/'backend/app/web/models/legacy_v1.joblib'),
                  legacy_v2_model=str(home/'backend/app/web/models/legacy_v2.joblib'),
                  madmom_python=str(targets['madmom_python']),
                  model_version=model['bundle_version'],entry_version='v88',
                  inference_memory_budget_mib=512,weights_sha256=frozen,
                  frozen_utc=model['created_utc'])
    write(backend/'inference/runtime.local.json',bound)
    audio=json.loads((backend/'inference/audio_manifest.json').read_text(encoding='utf8'))
    audio['files_sha256']['runtime.local.json']=digest(backend/'inference/runtime.local.json')
    audio['installation_configuration']='Paths generated locally; acoustic code and weights unchanged.'
    write(backend/'inference/audio_manifest.json',audio)
    write(backend/'desktop_runtime.json',settings)
    if publish:
        write(home/'installation.json',dict(schema=1,data_dir=str(data),backend='backend',
                                          web_python='runtime/core/Scripts/python.exe'))
    return dict(model=settings['model_version'],weights_sha256=frozen,files_verified=len(original['files']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    a=p.parse_args();print(json.dumps(configure(a.home,a.data_dir)))
