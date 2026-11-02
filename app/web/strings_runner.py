"""Approved bowed-string kernel, followed by six-stem separation of its complement."""
from pathlib import Path
import argparse,gc,hashlib,importlib.metadata,json,sys,time
import numpy as np
import soundfile as sf
import torch,yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from app.web.guitar_engine import ConfigLoader
from app.web.vendor.msst_model_utils import demix
from app.web.separation_cache import StageCache,digest,identity
from ml_collections import ConfigDict


def predict(x,config,checkpoint,device):
    from audio_separator.separator.uvr_lib_v5.roformer.bs_roformer import BSRoformer
    cfg=yaml.load(Path(config).read_text(encoding='utf8'),Loader=ConfigLoader)
    cfg['inference']['batch_size']=1
    model=BSRoformer(**cfg['model'])
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model=model.eval().requires_grad_(False).to(device)
    raw=demix(ConfigDict(cfg),model,x.T,device,model_type='bs_roformer',pbar=False)
    result={name:np.asarray(y,dtype='float32').T.copy() for name,y in raw.items()}
    for y in result.values():
        if y.shape!=x.shape or not np.isfinite(y).all():raise ValueError('分轨时长或数值异常')
    del model,raw;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return result


def separate(input_path,outdir,version,models,sw_models,stage_cache=None,force=False):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    x,sr=sf.read(input_path,dtype='float32',always_2d=True)
    if sr!=44100 or x.shape[1]!=2 or not len(x) or not np.isfinite(x).all():raise ValueError('需要 44.1 kHz 双声道音频')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    name='bs_bowed_str_gilliaaan' if version=='v1' else 'bs_bowed_str2_gilliaaan'
    start=time.perf_counter()
    cache=StageCache(stage_cache) if stage_cache else None
    hits={};keys={}
    def stage(label,signal,config,checkpoint,outputs,compute):
        recipe=dict(schema=1,stage=label,pcm_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(signal)).cast('B')).hexdigest(),
            shape=list(signal.shape),sample_rate=sr,outputs=sorted(outputs),
            config=digest(config),checkpoint=digest(checkpoint),
            code={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('separation_cache.py'),
                 Path(__file__).with_name('guitar_engine.py'),Path(__file__).parent/'vendor/msst_model_utils.py']},
            runtime={package:importlib.metadata.version(package) for package in ['torch','numpy','audio-separator']},
            device=torch.cuda.get_device_name(0) if device=='cuda' else 'cpu') if cache else None
        keys[label]=identity(recipe) if recipe else None
        restored=cache.restore(recipe,out) if cache and not force else None
        hits[label]=restored is not None
        if restored:
            print('CACHE '+label,flush=True)
            result={name:sf.read(path,dtype='float32',always_2d=True)[0] for name,path in restored.items()}
        else:
            result=compute()
        if set(result)!=set(outputs):raise ValueError('分离输出定义不符：'+label)
        for name,y in result.items():
            if y.shape!=signal.shape or not np.isfinite(y).all():raise ValueError('分离输出时长或数值异常：'+label)
            if not restored:sf.write(out/(name+'.wav'),y,sr,subtype='FLOAT')
        if cache and not restored:cache.save(recipe,{name:out/(name+'.wav') for name in result},replace=force)
        return result
    print('STAGE strings',flush=True)
    def strings_compute():
        target=predict(x,Path(models)/(name+'_config.yaml'),Path(models)/(name+'.ckpt'),device)
        if set(target)!={'strings','other'}:raise ValueError('提琴组输出定义不符')
        return dict(strings=target['strings'],strings_model_other=target['other'],remainder=x-target['strings'])
    target=stage('strings',x,Path(models)/(name+'_config.yaml'),Path(models)/(name+'.ckpt'),
                 {'strings','strings_model_other','remainder'},strings_compute)
    strings=target['strings'];remainder=target['remainder']
    string_seconds=time.perf_counter()-start
    del target
    print('STAGE remaining',flush=True)
    expected={'vocals','drums','bass','other','guitar','piano'}
    parts=stage('six_stems',remainder,Path(sw_models)/'BS-Roformer-SW.yaml',Path(sw_models)/'BS-Roformer-SW.ckpt',expected,
                lambda:predict(remainder,Path(sw_models)/'BS-Roformer-SW.yaml',Path(sw_models)/'BS-Roformer-SW.ckpt',device))
    reconstructed=strings+sum(parts.values())
    error=reconstructed-x
    info=dict(version=version,input='original mixture',second_stage_input='original minus strings',
        output_frames=len(x),sample_rate=sr,device=device,strings_seconds=string_seconds,
        total_seconds=time.perf_counter()-start,independent_normalization=False,stage_cache_hits=hits,stage_cache_keys=keys,
        sum_error_max=float(np.max(np.abs(error))),
        sum_error_relative_db=float(10*np.log10(max(float(np.mean(error.astype('float64')**2)),1e-20)/max(float(np.mean(x.astype('float64')**2)),1e-20))))
    (out/'inference.json').write_text(json.dumps(info,indent=2),encoding='utf8')
    return info


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--outdir',required=True)
    p.add_argument('--version',choices=['v1','v2'],required=True);p.add_argument('--models',required=True);p.add_argument('--sw-models',required=True)
    p.add_argument('--stage-cache');p.add_argument('--force',action='store_true')
    a=p.parse_args();separate(a.input,a.outdir,a.version,a.models,a.sw_models,a.stage_cache,a.force)
