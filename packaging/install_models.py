"""Install selected production model packs, reusing verified files and resuming downloads."""
from pathlib import Path
import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from downloads import download, inside, ready

HERE=Path(__file__).resolve().parent


def selection(catalog,names):
    result=[];visiting=set()
    def visit(name):
        if name in result:return
        if name in visiting:raise ValueError('模型包依赖形成循环')
        if name not in catalog['packs']:raise ValueError('未知模型包：'+name)
        visiting.add(name)
        for dependency in catalog['packs'][name].get('requires',[]):visit(dependency)
        visiting.remove(name);result.append(name)
    for name in names:visit(name)
    return result


def install(root,names,progress=print):
    catalog=json.loads((HERE/'model-packs.json').read_text(encoding='utf8'))
    selected=selection(catalog,names or catalog['default'])
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    # Serialize installers; readers and the application can still use old files.
    with (root/'.installation.lock').open('a+b') as lock:
        lock.seek(0,2)
        if not lock.tell():lock.write(b'0');lock.flush()
        lock.seek(0)
        if os.name=='nt':
            import msvcrt
            try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:raise RuntimeError('已有模型安装任务正在运行') from None
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        files={}
        for name in selected:
            for item in catalog['packs'][name]['files']:
                if item['path'] in files and files[item['path']]!=item:raise ValueError('模型包含有冲突的文件')
                files[item['path']]=item
        def one(item):
            target=inside(root,item['path'])
            if 'bundled' in item:
                source=inside(HERE,item['bundled'])
                if not ready(source,item):raise ValueError('随安装包提供的模型说明校验失败')
                if ready(target,item):return
                target.parent.mkdir(parents=True,exist_ok=True)
                pending=target.with_name(target.name+'.pending')
                if pending.is_symlink():raise ValueError('安装临时文件不能是符号链接')
                shutil.copyfile(source,pending);os.replace(pending,target)
            else:
                try:download(item,target,progress)
                except Exception as exc:
                    raise RuntimeError(f"{item['path']} 下载未完成：{exc}；重试会复用已验证文件与下载断点。") from exc
        with ThreadPoolExecutor(max_workers=2) as pool:
            for _ in pool.map(one,files.values()):pass
        marker=root/'installed-packs.json'
        previous=json.loads(marker.read_text(encoding='utf8')) if marker.is_file() else {'schema':1,'packs':{}}
        for name in selected:
            previous['packs'][name]={'label':catalog['packs'][name]['label'],
                                     'files':{f['path']:f['sha256'] for f in catalog['packs'][name]['files']}}
        pending=marker.with_suffix('.pending')
        pending.write_text(json.dumps(previous,ensure_ascii=False,indent=2),encoding='utf8')
        os.replace(pending,marker)
    return selected


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root',type=Path,required=True)
    p.add_argument('--packs',nargs='*')
    a=p.parse_args()
    def progress(value):print(json.dumps(value,ensure_ascii=False),flush=True)
    progress(dict(state='complete',packs=install(a.model_root,a.packs,progress)))
