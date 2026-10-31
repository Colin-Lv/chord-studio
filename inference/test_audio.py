"""Minimal local launcher for frozen v82 audio inference; never trains or retunes."""
from __future__ import annotations
import argparse,csv,json,subprocess,sys,uuid
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
VERSION='Chord Studio local test v88; frozen retained10 audio entry v82'
SCORE_DEFINITION=('Uncalibrated model score, not measured accuracy. N: N-head probability. '
 'Chord: conditional root/quality posterior times (1-N probability). Segment score is '
 'the duration-weighted mean of its audio cells. Slash-bass confidence is not included.')
LIMITATIONS=['Research model, not a default application deployment.',
 'Supports maj/min/7/maj7/min7/9/maj9/min9/sus2/sus4 plus N only.',
 '5,13,dim7 and other unsupported chords may be forced to a supported label; no reliable unknown detector.',
 'Reference no-chord intervals are often falsely assigned a chord in external evaluation.',
 'Slash bass, full musical modes, bars, sections and loops are not certified by this entry.',
 'Confidence is uncalibrated; high scores can be wrong. This is not an accuracy test.',
 'Uses the verified tools already installed on this Windows computer; not a standalone installer.',
 'Default length limit is300seconds. No new accuracy or long-recording guarantee.']
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
 with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def summarize(out):
 import numpy as np
 p=read(out/'prediction.json');status=read(out/'run_status.json');assert status['successful']
 cells=read(Path(p['cache_directory'])/'cells.json');segments=p['segments'];manifest=read(HERE/'model/manifest.json')
 with np.load(out/'posterior.npz',allow_pickle=False) as z:chord=z['chord'];n=z['N']
 assert chord.shape==(len(cells),120) and n.shape==(len(cells),)
 assert np.isfinite(chord).all() and np.isfinite(n).all()
 starts=np.array([c['start'] for c in cells]);ends=np.array([c['end'] for c in cells]);names=('C','C#','D','D#','E','F','F#','G','G#','A','A#','B');qualities=manifest['qualities'];rows=[]
 for s in segments:
  duration=np.maximum(0,np.minimum(ends,s['end'])-np.maximum(starts,s['start']));use=duration>1e-10;assert use.any();assert abs(duration.sum()-(s['end']-s['start']))<1e-7
  if s['chord']=='N':score=n
  else:
   root,quality=s['chord'].split('/')[0].split(':');idx=names.index(root)*10+qualities.index(quality);score=chord[:,idx]*(1-n)
  mean=float(duration@score/duration.sum());low=float(score[use].min());high=float(score[use].max());assert 0<=low<=mean+1e-12 and mean<=high+1e-12<=1+1e-12
  rows.append(dict(start_seconds=s['start'],end_seconds=s['end'],chord=s['chord'],confidence_score=mean,cell_score_min=low,cell_score_max=high))
 with (out/'chords.csv').open('x',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 result=dict(created_utc=datetime.now(timezone.utc).isoformat(),entry_version=VERSION,model_version=manifest['bundle_version'],weights_sha256=p['weights_sha256'],audio_path=p['input_audio'],audio_seconds=p['audio_seconds'],cache_reused=p['cache_reused'],confidence_definition=SCORE_DEFINITION,limitations=LIMITATIONS,segments=rows,new_fits=0,new_optimizer_updates=0,reference_labels_read=False,default_changed=False)
 write(out/'quick_result.json',result)
 lines=['Chord Studio 本地研究测试',f'入口版本：{VERSION}',f'模型版本：{result["model_version"]}',f'模型权重 SHA256：{result["weights_sha256"]}',f'音频：{result["audio_path"]}',f'时长：{result["audio_seconds"]:.3f} 秒；分段：{len(rows)}','', '置信分数是未经校准的模型分数，不是识别准确率，也不包含斜杠低音的可靠性。','支持 maj/min/7/maj7/min7/9/maj9/min9/sus2/sus4 和 N。','不保证识别5、13、dim7、其他词表外和弦；未知类型可能被归入已有类别。','无和弦区间被误判成和弦的问题仍明显；此入口不输出或认证小节、乐段、循环或完整调式。','', '开始秒\t结束秒\t和弦\t置信分数']
 lines.extend(f'{r["start_seconds"]:.3f}\t{r["end_seconds"]:.3f}\t{r["chord"]}\t{r["confidence_score"]:.4f}' for r in rows)
 (out/'结果说明.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 return result
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--audio',type=Path);ap.add_argument('--results-dir',type=Path,default=HERE/'results');ap.add_argument('--cache-dir',type=Path,default=HERE/'cache');a=ap.parse_args()
 raw=str(a.audio) if a.audio else input('请输入音频文件完整路径（可把文件拖入此窗口后按回车）：\n').strip().strip('"')
 if not raw:raise ValueError('No audio file was supplied')
 audio=Path(raw).expanduser().resolve(strict=True)
 if not audio.is_file():raise ValueError('Choose an audio file')
 a.results_dir.mkdir(parents=True,exist_ok=True);out=a.results_dir.resolve()/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);cache=a.cache_dir.resolve()
 print('开始本地推理；首次处理会生成声学缓存。结果目录：'+str(out),flush=True)
 proc=subprocess.run([sys.executable,'-I','-B',str(HERE/'audio_predict.py'),'--audio',str(audio),'--output-dir',str(out),'--cache-dir',str(cache)],cwd=HERE)
 if proc.returncode:raise RuntimeError('Inference failed; inspect preserved logs at '+str(out))
 result=summarize(out)
 print(json.dumps(dict(successful=True,output_directory=str(out),CSV=str(out/'chords.csv'),summary=str(out/'结果说明.txt'),model_version=result['model_version'],segments=len(result['segments']),cache_reused=result['cache_reused']),ensure_ascii=False),flush=True)
if __name__=='__main__':
 try:main()
 except (Exception,KeyboardInterrupt) as exc:print('测试未完成：'+str(exc),file=sys.stderr);sys.exit(1)
