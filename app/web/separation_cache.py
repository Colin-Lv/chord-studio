"""Completed, content-addressed inference stages; interrupted stages are never reused."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
from uuid import uuid4


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def identity(recipe):
    return hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def canonical(path):
    value=str(Path(path).resolve())
    if os.name=='nt':
        if value.startswith('\\\\?\\UNC\\'):value='\\\\'+value[8:]
        elif value.startswith('\\\\?\\') and len(value)>=7 and value[5:7]==':\\':value=value[4:]
    return Path(value)


def child(root,relative):
    root=canonical(root);path=canonical(root/relative)
    if path==root or not path.is_relative_to(root) or Path(relative).is_absolute():
        raise ValueError('分离缓存路径超出范围')
    return path


def link_or_copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.pending-'+uuid4().hex)
    try:
        try:os.link(source,temporary)
        except OSError:shutil.copyfile(source,temporary)
        os.replace(temporary,target)
    finally:
        temporary.unlink(missing_ok=True)


class StageCache:
    def __init__(self,root):
        self.root=canonical(root)

    def read(self,recipe):
        key=identity(recipe);base=self.root/key
        try:
            meta=json.loads((base/'manifest.json').read_text(encoding='utf8'))
            if meta.get('schema')!=1 or meta.get('recipe')!=recipe:return None
            names=meta['order']
            if not names or len(set(names))!=len(names) or set(names)!=set(recipe['outputs']):return None
            files={}
            for name in names:
                if not re.fullmatch('[a-z_]+',name):return None
                item=meta['files'][name];path=child(base,item['file'])
                if not path.is_file() or path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:return None
                files[name]=path
            return files
        except (OSError,ValueError,KeyError,TypeError):
            return None

    def restore(self,recipe,outdir):
        files=self.read(recipe)
        if files is None:return None
        out=Path(outdir)
        for name,source in files.items():link_or_copy(source,out/(name+'.wav'))
        return {name:out/(name+'.wav') for name in files}

    def save(self,recipe,files,replace=False):
        """Write a new generation, then atomically publish its complete manifest."""
        if set(files)!=set(recipe['outputs']):raise ValueError('缓存声部定义不匹配')
        if not replace and self.read(recipe) is not None:return
        key=identity(recipe);base=self.root/key;generation=uuid4().hex
        records={}
        for name,source in files.items():
            if not re.fullmatch('[a-z_]+',name):raise ValueError('缓存声部名称无效')
            relative=f'generations/{generation}/{name}.wav'
            target=child(base,relative);link_or_copy(source,target)
            records[name]=dict(file=relative,bytes=target.stat().st_size,sha256=digest(target))
        meta=dict(schema=1,recipe=recipe,order=list(files),files=records)
        pending=base/('manifest.pending-'+generation)
        pending.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
        os.replace(pending,base/'manifest.json')
