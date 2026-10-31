"""One frozen NNLS head evaluation against an explicit case manifest.

CSV columns: id,audio,upstream,reference,domain,reference_status.
This command cannot fit a classifier or tune any parameter.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from apply_nnls_extensions import apply_model
from evaluate_chords import read_reference, read_prediction, evaluate, validate_timeline
from madmom_bridge import file_identity
from train_multidomain_extensions import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('.tmp/nnls_seventh_features/notes'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'report.json'
    if target.exists():
        raise ValueError('Use a new report directory')
    artifact = joblib.load(args.model)
    with args.manifest.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    identities = [(row['id'], file_identity(Path(row['audio']))) for row in rows]
    if len({identity['sha256'] for _, identity in identities}) != len(rows) or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate recording/song in evaluation')
    for case_id, identity in identities:
        if case_id in artifact['training_ids'] or identity['sha256'] in artifact['training_audio_sha256']:
            raise ValueError('Training material cannot enter held-out evaluation')
    report = dict(protocol='frozen_nnls_extension_evaluation_no_fitting_or_selection',
                  metric_protocol='exact_duration_strict_missing_v2', model=file_identity(args.model),
                  manifest=file_identity(args.manifest), selected=artifact['selected'],
                  status='running', songs=[], reference_limitation='Read individual reference_status fields; local legacy test sets are not project-level blind tests')
    write_report(target, report)
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        for number, row in enumerate(rows):
            audio, upstream, reference = (Path(row[k]).resolve() for k in ('audio', 'upstream', 'reference'))
            output = args.output / (row['id'] + '.lab')
            # Reference is read only after the fixed audio-only prediction.
            provenance = apply_model(audio, upstream, args.model, output, args.cache)
            truth = read_prediction(reference) if reference.suffix == '.json' else read_reference(reference)
            truth = validate_timeline(truth, 'Reference')
            baseline = evaluate(truth, read_reference(upstream))
            predicted = evaluate(truth, read_reference(output))
            if abs(baseline['root_accuracy']-predicted['root_accuracy']) > 1e-9 or abs(baseline['triad_accuracy']-predicted['triad_accuracy']) > 1e-9:
                raise ValueError('Root/triad changed')
            result = dict(id=row['id'], domain=row['domain'], reference_status=row['reference_status'],
                          audio=provenance['audio'], upstream=provenance['upstream_lab'],
                          reference=file_identity(reference), prediction=provenance['output_lab'],
                          baseline=baseline, extension=predicted, changed_windows=provenance['change_count'])
            report['songs'].append(result)
            write_report(target, report)
            print(f"{number+1}/{len(rows)} {row['id']}: tetrad {baseline['tetrad_accuracy']:.4f}->{predicted['tetrad_accuracy']:.4f} changes={provenance['change_count']}", flush=True)
    report['summary'] = {}
    for domain in sorted({row['domain'] for row in rows}):
        songs = [row for row in report['songs'] if row['domain'] == domain]
        report['summary'][domain] = {stage: {metric: float(np.mean([r[stage][metric] for r in songs])) for metric in
                                            ('root_accuracy', 'triad_accuracy', 'tetrad_accuracy', 'tetrad_inv_accuracy', 'tetrad_wcsr', 'prediction_coverage')}
                                     for stage in ('baseline', 'extension')}
        report['summary'][domain]['songs'] = len(songs)
        report['summary'][domain]['changed_windows'] = sum(r['changed_windows'] for r in songs)
    report.update(status='complete', elapsed_seconds=time.perf_counter()-started)
    write_report(target, report)
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
