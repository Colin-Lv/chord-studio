"""Generate every frozen manifest recording once, without opening references.

Only id/audio columns enter jobs. One resident madmom39 CNN+CRF processor reads
complete 44100 Hz mono arrays, with at most two CPU threads and no score path.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

import numpy as np

from chord_recognizer import find_ffmpeg
from madmom_bridge import DEFAULT_MADMOM_PYTHON, file_identity


def load_audio_columns(manifest):
    """Project only declared IDs/audio paths; never follow annotation paths."""
    rows = []
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header.count('id') != 1 or header.count('audio') != 1:
            raise ValueError('Manifest must contain exactly one id and audio column')
        id_index, audio_index = header.index('id'), header.index('audio')
        for values in reader:
            if not values:
                continue
            identifier, audio = values[id_index], values[audio_index]
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', identifier):
                raise ValueError(f'Unsafe recording ID: {identifier!r}')
            source = (manifest.parent / audio).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            rows.append(dict(id=identifier, audio=str(source)))
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate recording IDs')
    if len({os.path.normcase(row['audio']) for row in rows}) != len(rows):
        raise ValueError('Duplicate audio paths')
    return rows


def decode_full_audio(audio):
    command = [find_ffmpeg(), '-hide_banner', '-loglevel', 'error',
               '-threads', '2', '-filter_threads', '2', '-i', str(audio),
               '-vn', '-ac', '1', '-ar', '44100', '-threads', '2',
               '-f', 'f32le', 'pipe:1']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', errors='replace'))
    signal = np.frombuffer(result.stdout, dtype=np.float32)
    if not signal.size or not np.isfinite(signal).all():
        raise ValueError('Full-source audio must be nonempty and finite')
    return signal


def write_json(path, payload):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', '-o', type=Path, required=True)
    parser.add_argument('--expected-count', type=int, required=True)
    parser.add_argument('--cohort', default='external_test', help='Recorded constant; no split column is read')
    parser.add_argument('--madmom-python', type=Path, default=DEFAULT_MADMOM_PYTHON)
    args = parser.parse_args()
    started = time.perf_counter()
    manifest, output = args.manifest.resolve(), args.output.resolve()
    rows = load_audio_columns(manifest)
    if len(rows) != args.expected_count or not rows:
        raise ValueError(f'Expected every one of {args.expected_count} recordings; found {len(rows)}')
    if not args.madmom_python.is_file():
        raise FileNotFoundError(args.madmom_python)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Use an empty output directory; existing LABs are never overwritten')
    (output / 'labs').mkdir()
    report = dict(status='planned_before_inference', created_at=datetime.now().astimezone().isoformat(),
                  source_manifest=str(manifest), manifest_fields_used=['id', 'audio'],
                  cohort=args.cohort, expected_count=args.expected_count, audio_only_inputs=rows,
                  sample_rate=44100, offset=0., requested_duration=None, threads=2,
                  references_opened=False, scoring_performed=False, repeated_inference=False,
                  frontend='madmom CNNChordFeatureProcessor + CRFChordRecognitionProcessor',
                  resampling='Complete source decoded by FFmpeg to mono 44100 Hz; no source time crop')
    write_json(output / 'protocol.json', report)
    jobs, audio_seconds = [], 0.
    try:
        with tempfile.TemporaryDirectory(prefix='.audio_only_', dir=output) as temporary:
            directory = Path(temporary)
            for index, row in enumerate(rows):
                audio = Path(row['audio'])
                identity = file_identity(audio)
                signal = decode_full_audio(audio)
                after = audio.stat()
                if (after.st_size, after.st_mtime_ns) != (identity['bytes'], identity['mtime_ns']):
                    raise RuntimeError(f'Audio changed during decoding: {audio}')
                npy = directory / (row['id'] + '.npy')
                np.save(npy, signal)
                audio_seconds += len(signal) / 44100
                jobs.append(dict(id=row['id'], split=args.cohort, npy=str(npy),
                                 lab=str(output / 'labs' / (row['id'] + '.lab')),
                                 audio_identity=identity))
                print(f'Prepared {index+1}/{len(rows)} {row["id"]}: {len(signal)/44100:.3f}s', flush=True)
                del signal
            job_path = output / 'audio_only_jobs.json'
            write_json(job_path, jobs)
            report.update(status='inference_started', audio_seconds=audio_seconds)
            write_json(output / 'protocol.json', report)
            environment = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                               MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', PYTHONIOENCODING='utf-8')
            worker = Path(__file__).with_name('guitarset_madmom_worker.py').resolve()
            result = subprocess.run([str(args.madmom_python), str(worker), str(job_path), '--skip-reuse-check'],
                                    env=environment)
            if result.returncode:
                raise RuntimeError(f'Resident worker exited with code {result.returncode}; partial LABs retained')
            for job in jobs:
                after = Path(job['audio_identity']['path']).stat()
                if (after.st_size, after.st_mtime_ns) != (job['audio_identity']['bytes'], job['audio_identity']['mtime_ns']):
                    raise RuntimeError(f'Audio changed during the batch: {job["id"]}')
            report.update(status='complete', completed_count=len(jobs), elapsed_seconds=time.perf_counter()-started)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.perf_counter()-started)
        raise
    finally:
        report['temporary_audio_arrays_retained'] = False
        write_json(output / 'protocol.json', report)
    print(f'Completed every {len(jobs)} recording; no reference was opened or scored.', flush=True)


if __name__ == '__main__':
    main()
