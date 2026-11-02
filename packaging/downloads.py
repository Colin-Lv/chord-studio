"""Bounded, resumable HTTPS downloads. Publish only verified complete files."""
from pathlib import Path
import hashlib
import os
import re
import time
import urllib.request


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def canonical(path):
    value=str(Path(path).resolve())
    # Windows realpath can retain its extended-length prefix when another thread
    # creates a previously missing parent during resolution. Normalize *both*
    # paths before containment checks, without relaxing junction/symlink checks.
    if os.name=='nt':
        if value.startswith('\\\\?\\UNC\\'):value='\\\\'+value[8:]
        elif value.startswith('\\\\?\\') and len(value)>=7 and value[4].isalpha() and value[5:7]==':\\':value=value[4:]
    return Path(value)


def inside(root, relative):
    root=canonical(root);path=canonical(root/relative)
    if not path.is_relative_to(root) or path==root or Path(relative).is_absolute():
        raise ValueError(f'安装文件路径超出了所选目录：{relative}（目录 {root}；目标 {path}）')
    return path


def ready(path, item):
    path=Path(path)
    return path.is_file() and path.stat().st_size==item['bytes'] and digest(path)==item['sha256']


def download(item, target, progress=lambda value:None, opener=None, retries=3):
    target=Path(target)
    if not re.fullmatch('[0-9a-f]{64}',item['sha256']) or not isinstance(item['bytes'],int) or item['bytes']<=0:
        raise ValueError('下载清单缺少有效的文件摘要或大小')
    if not item['url'].startswith('https://'):raise ValueError('模型下载必须使用 HTTPS')
    if ready(target,item):
        progress(dict(file=target.name,state='cached',bytes=item['bytes']));return False
    target.parent.mkdir(parents=True,exist_ok=True)
    partial=target.with_name(target.name+'.partial')
    if partial.is_symlink():raise ValueError('下载临时文件不能是符号链接')
    open_url=opener or urllib.request.urlopen
    for attempt in range(retries):
        try:
            offset=partial.stat().st_size if partial.exists() else 0
            if offset>=item['bytes']:
                if ready(partial,item):os.replace(partial,target);return True
                partial.unlink();offset=0
            headers={'User-Agent':'Chord-Studio-Installer/1.1','Accept-Encoding':'identity'}
            if offset:headers['Range']=f'bytes={offset}-'
            request=urllib.request.Request(item['url'],headers=headers)
            with open_url(request,timeout=90) as response:
                status=response.status
                if status not in {200,206}:raise ValueError(f'下载服务返回了 {status}')
                if status==206:
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                    if not match or int(match[1])!=offset or int(match[3])!=item['bytes']:
                        raise ValueError('下载服务返回了不匹配的续传范围')
                else:offset=0
                received=offset;last=time.monotonic()
                with partial.open('ab' if offset else 'wb') as output:
                    while chunk:=response.read(1024*1024):
                        received+=len(chunk)
                        if received>item['bytes']:raise ValueError('下载文件超过清单大小')
                        output.write(chunk)
                        if time.monotonic()-last>1:
                            progress(dict(file=target.name,state='downloading',bytes=received,total=item['bytes']))
                            last=time.monotonic()
            if partial.stat().st_size!=item['bytes']:
                raise ValueError('下载尚未完整，已保存断点')
            if not ready(partial,item):
                # A corrupt resumed prefix must not be retried indefinitely.
                partial.unlink(missing_ok=True)
                raise ValueError('下载文件校验失败，原有模型未被覆盖')
            os.replace(partial,target)
            progress(dict(file=target.name,state='ready',bytes=item['bytes']))
            return True
        except Exception:
            if attempt+1==retries:raise
            time.sleep(min(attempt+1,3))
