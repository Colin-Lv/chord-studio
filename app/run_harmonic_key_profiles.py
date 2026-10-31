"""Prepare fixed profile evidence without joining any reference labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from key_profile_baseline import analyze_audio, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    split=json.loads(args.split.read_text(encoding='utf-8'))
    rows=[row for row in split['rows'] if row['harmonic_selected']]
    args.output.mkdir(parents=True,exist_ok=True)
    def one(row):
        audio=Path(row['audio'])
        index=json.loads((audio.parent.parent/'audio_index.json').read_text())
        expected=next(item['sha256'] for item in index['files'] if item['id']==row['id'])
        if digest(audio)!=expected:raise ValueError('Audio differs from official archive')
        result=analyze_audio(audio,args.output/(row['id']+'.json'))
        return dict(id=row['id'],split=row['harmonic_split'],seconds=result.get('elapsed_seconds'))
    completed=[]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs={executor.submit(one,row):row for row in rows}
        for job in as_completed(jobs):
            try:result=job.result()
            except Exception as exc:result=dict(id=jobs[job]['id'],error=repr(exc))
            completed.append(result)
            temp=args.output/'progress.json.part'
            temp.write_text(json.dumps(dict(total=len(rows),completed=completed),indent=2),encoding='utf-8')
            temp.replace(args.output/'progress.json')
            print(f'{len(completed)}/{len(rows)} {result}',flush=True)
    if any('error' in row for row in completed):raise RuntimeError('Profile batch has failures')


if __name__=='__main__':main()
