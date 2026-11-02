"""Release-maintainer tool: record only production model files and public sources.

The generated catalog contains no local absolute paths, audio, or credentials.
Do not run this against changed weights without validating their inference first.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
MIRROR='https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/'


def make(models):
    packs={}
    def pack(key,label,requires=()):
        packs[key]=dict(label=label,requires=list(requires),files=[],
                        terms='模型权重遵循其作者发布的条款；应用代码的许可不替代模型许可。')
        return packs[key]['files']
    def file(files,path,url):
        source=models/path
        with source.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        files.append(dict(path=path,url=url,bytes=source.stat().st_size,sha256=sha))
    demucs=json.loads((HERE/'demucs-models.json').read_text())
    pack('standard','标准四轨分离').extend(demucs[:1])
    pack('fine','精细四轨分离（四模型）').extend(demucs[1:])
    strings=json.loads((HERE/'model-metadata/strings.json').read_text())
    for version,name in [('strings_v2','提琴组 · 版本 2'),('strings_v1','提琴组 · 版本 1')]:
        files=pack(version,name,['six_stems'])
        for item in strings:
            basename=Path(item['path']).name
            if ('str2_' in basename) == (version=='strings_v2'):
                file(files,'bowed-strings/'+basename,item['url'])
                assert files[-1]['sha256']==item['sha256']
    for key,label,folder,names in [
        ('six_stems','六轨分离','audio_separator',['BS-Roformer-SW.ckpt','BS-Roformer-SW.yaml']),
        ('vocal_detail','主唱与背景和声','audio_separator',[
            'mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt',
            'mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956_config.yaml']),
        ('drum_detail','鼓组细分','drumsep',['MDX23C-DrumSep-aufr33-jarredou.ckpt','config_drumsep_mdx23c.yaml']),
        ('vocal_priority','人声优先分离','bsroformer',[
            'model_bs_roformer_ep_317_sdr_12.9755.ckpt','model_bs_roformer_ep_317_sdr_12.9755.yaml']),
    ]:
        files=pack(key,label)
        for name in names:file(files,folder+'/'+name,MIRROR+name)
    files=pack('guitar_timbre','木吉他／电吉他细分',['six_stems'])
    for kind in ['acoustic','electric']:
        for suffix in ['.ckpt','_config.yaml']:
            name=f'bs_mega_53stem_{kind}-guitar_mvsep{suffix}'
            record=json.loads((models/'guitar'/(name+'.download.json')).read_text())
            file(files,'guitar/'+name,record['url'])
            assert files[-1]['sha256']==record['sha256']
    files=pack('guitar_roles','旋律／伴奏吉他细分',['six_stems'])
    name='mbr_lead_rhythm_guitar_listra92_config.yaml'
    record=json.loads((models/'guitar'/(name+'.download.json')).read_text())
    file(files,'guitar/'+name,record['url'])
    file(files,'guitar/guitar_role_v1.ckpt',
         'https://github.com/Colin-Lv/chord-studio/releases/download/v1.1.0/guitar_role_v1.ckpt')
    packs['guitar_roles']['terms']+=' 角色模型为项目已有适配版本；上游架构和预训练模型来源见模型说明。'
    files=pack('reconstruction','整曲 MIDI 重构')
    (HERE/'model-metadata').mkdir(exist_ok=True)
    for folder,kind in [('yourmt3','spaces/'),('adt-str','')]:
        original=models/'reconstruction'/folder/'download-manifest.json'
        meta=json.loads(original.read_text())
        base=f"https://huggingface.co/{kind}{meta['repo']}/resolve/{meta['revision']}/"
        for item in meta['files']:
            file(files,'reconstruction/'+folder+'/'+item['path'],base+item['path'])
            assert files[-1]['sha256']==item['sha256']
        bundled=HERE/'model-metadata'/(folder+'.json')
        shutil.copyfile(original,bundled)
        files.append(dict(path=f'reconstruction/{folder}/download-manifest.json',
                          bundled='model-metadata/'+bundled.name,bytes=bundled.stat().st_size,
                          sha256=hashlib.sha256(bundled.read_bytes()).hexdigest()))
    for p in packs.values():p['download_bytes']=sum(f['bytes'] for f in p['files'])
    catalog=dict(schema=1,default=['strings_v2','guitar_timbre'],packs=packs)
    (HERE/'model-packs.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return {name:len(p['files']) for name,p in packs.items()}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,required=True)
    print(json.dumps(make(p.parse_args().models)))
