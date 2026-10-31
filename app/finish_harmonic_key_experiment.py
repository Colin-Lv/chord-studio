"""Wait for label-independent evidence, freeze on dev, then evaluate once.

This runner never examines sealed scores before a persisted code/policy freeze.
It also samples the acoustic process tree memory; no process is terminated.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--audio-pid',type=int,required=True)
    args=parser.parse_args();root=args.root;progress=root/'acoustics/progress.json'
    commands=[];samples=[];began=time.time()
    def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    def sample():
        try:
            processes=[psutil.Process(args.audio_pid)]+psutil.Process(args.audio_pid).children(recursive=True)
            rss=sum(process.memory_info().rss for process in processes if process.is_running())
            samples.append(dict(epoch=time.time(),rss_bytes=rss,process_count=len(processes),available_memory=psutil.virtual_memory().available))
        except psutil.Error:pass
        write(root/'resources.json',dict(samples=samples,sampling_interval_seconds=10,
            note='Sum of process RSS can double-count shared pages; sampling began after the first acoustic cases',
            peak_sampled_audio_tree_rss=max([row['rss_bytes'] for row in samples],default=0),
            monitor_started_epoch=began))
    def run(name,arguments):
        cmd=[sys.executable]+arguments;start=time.time()
        with (root/(name+'.log')).open('w',encoding='utf-8') as stream:
            process=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
            while process.poll() is None:sample();time.sleep(10)
        commands.append(dict(name=name,argv=cmd,started_epoch=start,seconds=time.time()-start,exit_code=process.returncode))
        write(root/'commands.json',commands)
        if process.returncode:raise RuntimeError(name+' failed; see its log')
    while True:
        sample()
        if progress.is_file():
            value=json.loads(progress.read_text(encoding='utf-8'))
            if len(value['completed'])==value['total']:
                if any('error' in row for row in value['completed']):raise RuntimeError('Acoustic failures; not opening sealed scores')
                break
        if not psutil.pid_exists(args.audio_pid):raise RuntimeError('Acoustic process stopped before completeness')
        time.sleep(10)
    profiles=json.loads((root/'profiles/progress.json').read_text(encoding='utf-8'))
    if len(profiles['completed'])!=profiles['total'] or any('error' in row for row in profiles['completed']):raise RuntimeError('Incomplete profiles')
    common=['--split',str(root/'split.json'),'--acoustics',str(root/'acoustics'),'--profiles',str(root/'profiles')]
    contract=root/'engine_contract.json'
    run('selection',['app/evaluate_harmonic_key_v2.py','select']+common+['--output',str(contract)])
    write(root/'sealed_opening.json',dict(contract_persisted=True,contract_file=str(contract),
        first_sealed_evaluation_started_epoch=time.time(),selection_log=str(root/'selection.log')))
    for phase in ['harmonic_dev','harmonic_calibration','harmonic_new_sealed']:
        run(phase,['app/evaluate_harmonic_key_v2.py','evaluate']+common+['--contract',str(contract),
            '--phase',phase,'--output',str(root/(phase+'.json'))])
    run('billboard',['app/evaluate_harmonic_billboard_v2.py','--split','benchmarks/reports/key_resolver_20261006/splits/billboard_split.json',
        '--contract',str(contract),'--old-report','benchmarks/reports/key_resolver_20261006/billboard/report.json',
        '--output',str(root/'billboard.json')])
    run('regressions',['app/run_harmonic_key_regressions.py','--contract',str(contract),'--output',str(root/'regressions')])
    write(root/'completed.json',dict(completed_epoch=time.time(),elapsed_since_monitor_seconds=time.time()-began,
        contract=str(contract),no_resolver_training=True,no_default_deployment=True))
    print('Frozen harmonic experiment complete',flush=True)


if __name__=='__main__':main()
