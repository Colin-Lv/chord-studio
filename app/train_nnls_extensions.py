"""Train/develop NNLS seventh heads; held-out evaluation is a separate action."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

import nnls_seventh_model as quality
from evaluate_chords import read_reference, validate_timeline, evaluate
from train_multidomain_extensions import signature, write_report


def load_manifest(path, lab_dir, domain, expected_roles):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    songs = []
    for row in rows:
        if row.get('split') not in expected_roles:
            raise ValueError(f'Unexpected role: {row.get("split")}')
        audio = (path.parent / row['audio']).resolve()
        ref = (path.parent / row['chord_reference']).resolve()
        lab = lab_dir / (row['id'] + '.lab')
        songs.append(dict(id=row['id'], domain=domain, audio=audio,
                          upstream=validate_timeline(read_reference(lab), 'Upstream'),
                          reference=validate_timeline(read_reference(ref), 'Reference'),
                          audio_identity=signature(audio), lab_identity=signature(lab), reference_identity=signature(ref)))
    return songs


def prepare_features(songs, cache, window_sizes=(0., .5), fast=True):
    import harmonic_note_features as notes
    import chord_extension_model as old
    import chord_studio as studio
    cache.mkdir(parents=True, exist_ok=True)
    implementation = [signature(notes.__file__), signature(quality.__file__)]
    if fast:
        implementation += [signature(old.__file__), signature(studio.__file__),
                           signature(Path(__file__).with_name('audio_features.py')),
                           signature(Path(__file__).with_name('acoustic_chord_model.py'))]
    for number, song in enumerate(songs):
        evidence, samples = None, None
        song['grids'] = {}
        note_identity = notes.extraction_identity(song['audio'])
        for window in window_sizes:
            rows, parents = quality.split_upstream(song['upstream'], window)
            identity = dict(audio=song['audio_identity']['sha256'], lab=song['lab_identity']['sha256'],
                            window=window, fast=fast, implementation=implementation, note_extractor=note_identity)
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            path = cache / (key + '.npz')
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    features = saved['features']
            else:
                if evidence is None:
                    evidence = notes.extract_note_evidence(song['audio'], cache_dir=cache / 'notes')
                feature_rows = quality.acoustic_windows(rows, evidence.metadata['duration_seconds'])
                features = notes.features_for_intervals(evidence, feature_rows)
                if fast:
                    if samples is None:
                        samples = studio.decode_audio(song['audio'])
                    old_features, _ = old.features_for_rows(samples, rows, feature_schema=old.FEATURE_SCHEMA_V2)
                    features = np.concatenate([features, old_features], axis=1)
                np.savez_compressed(path, features=features, identity=json.dumps(identity, sort_keys=True))
            expected = 245 if fast else 97
            if features.shape != (len(rows), expected) or not np.isfinite(features).all():
                raise ValueError(f'Invalid features: {song["id"]}')
            targets = quality.supervision(rows, song['reference'])
            song['grids'][window] = dict(rows=rows, parents=parents, features=features, targets=targets,
                                        seconds=np.asarray([r.end-r.start for r in rows]))
        print(f'NNLS features {number+1}/{len(songs)}: {song["id"]}', flush=True)
    return songs


def training_arrays(songs, window, view, balance):
    xs, ys, weights = [], [], []
    dimension = 97 if view == 'nnls' else 245
    for song in songs:
        grid = song['grids'][window]
        keep = grid['targets'] >= 0
        xs.append(grid['features'][keep, :dimension])
        ys.append(grid['targets'][keep])
        weights.append(grid['seconds'][keep])
    x, y, weight = np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)
    if balance == 'sqrt_class_duration':
        counts = np.bincount(y, weights=weight, minlength=3)
        weight = weight * np.sqrt(weight.sum()/np.maximum(counts[y], 1e-9))
    weight /= weight.mean()
    return x, y, weight


def predict_songs(songs, model, window, view):
    dimension = 97 if view == 'nnls' else 245
    return [quality.predict_probabilities(model, song['grids'][window]['features'][:, :dimension]) for song in songs]


def score_predictions(songs, probabilities, window, threshold, details=False):
    records = []
    for song, probability in zip(songs, probabilities):
        grid = song['grids'][window]
        predicted, changes = quality.decode_windows(song['upstream'], grid['rows'], grid['parents'], probability, threshold)
        score = evaluate(song['reference'], predicted)
        records.append(dict(id=song['id'], metrics=score, changed_windows=len(changes),
                            changed_seconds=sum(item['end']-item['start'] for item in changes)))
    summary = {key: float(np.mean([row['metrics'][key] for row in records])) for key in
               ('root_accuracy', 'triad_accuracy', 'tetrad_accuracy', 'tetrad_wcsr', 'tetrad_inv_accuracy', 'prediction_coverage')}
    summary['changed_seconds'] = sum(row['changed_seconds'] for row in records)
    return dict(summary=summary, songs=records if details else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--beatles-train', type=Path, default=Path('benchmarks/manifest.beatles_selected6_83_train.csv'))
    parser.add_argument('--beatles-labs', type=Path, default=Path('benchmarks/external_chords/madmom/beatles_selected6'))
    parser.add_argument('--guitar-study', type=Path, default=Path('benchmarks/reports/session_20260926/guitarset_extension_experiment'))
    parser.add_argument('--output', '-o', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('.tmp/nnls_seventh_features'))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / 'development_report.json'
    if report_path.exists():
        raise ValueError('Use a new experiment output directory')
    started = time.perf_counter()
    parameters = dict(max_iter=160, max_leaf_nodes=15, learning_rate=.06, min_samples_leaf=30,
                      l2_regularization=3., max_bins=63, early_stopping=False, random_state=20260926)
    report = dict(status='predeclared_before_development_scores', model_schema=quality.MODEL_SCHEMA,
        metric_protocol='exact_duration_strict_missing_v2; chord-only accuracy excludes reference N/X',
        grids=[0., .5], views=['nnls', 'nnls_fast'], domains=['beatles', 'beatles_guitar'],
        weighting=['duration', 'sqrt_class_duration'], thresholds=[1., .97, .9, .8, .7, .6, .5],
        parameters=parameters, minimum_target_support=.8,
        selection_rule='Require Beatles dev14 tetrad_accuracy improvement and Guitar val10 no reduction; maximize Beatles, then guitar, then fewer changed seconds; identity if no eligible model',
        held_out_labels_read=False, implementation=[signature(__file__), signature(quality.__file__)], candidates=[])
    write_report(report_path, report)
    b = load_manifest(args.beatles_train.resolve(), args.beatles_labs.resolve(), 'beatles', {'train'})
    gt = load_manifest((args.guitar_study / 'manifests/train40.csv').resolve(),
                       (args.guitar_study / 'labs').resolve(), 'guitar', {'train40'})
    gd = load_manifest((args.guitar_study / 'manifests/val10.csv').resolve(),
                       (args.guitar_study / 'labs').resolve(), 'guitar', {'val10'})
    all_songs = b + gt + gd
    if len({s['id'] for s in all_songs}) != len(all_songs) or len({s['audio_identity']['sha256'] for s in all_songs}) != len(all_songs):
        raise ValueError('Duplicate song or audio content')
    fit, dev = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20260926).split(np.arange(len(b)), groups=[s['id'] for s in b]))
    old_selection = json.loads((args.guitar_study / 'consensus_head/training_report.json').read_text(encoding='utf-8'))
    if [b[i]['id'] for i in dev] != old_selection['development_song_ids']['beatles']:
        raise ValueError('The fixed development split changed')
    report['selection_training_ids'] = dict(beatles=[b[i]['id'] for i in fit], guitar=[s['id'] for s in gt])
    report['development_ids'] = dict(beatles=[b[i]['id'] for i in dev], guitar=[s['id'] for s in gd])
    report['sources'] = [{key: value for key, value in s.items() if key.endswith('_identity') or key in ('id', 'domain')} for s in all_songs]
    write_report(report_path, report)
    with threadpool_limits(limits=args.threads):
        prepare_features(all_songs, args.cache)
        report['status'] = 'features_complete'
        write_report(report_path, report)
        if args.prepare_only:
            return
        validation = dict(beatles=[b[i] for i in dev], guitar=gd)
        report['baseline'] = {name: score_predictions(songs, [np.tile([1., 0., 0.], (len(s['grids'][0.]['rows']), 1)) for s in songs], 0., 1., True) for name, songs in validation.items()}
        winner, winning_key = None, None
        for window in report['grids']:
            for view in report['views']:
                for domain in report['domains']:
                    training = [b[i] for i in fit] + (gt if domain == 'beatles_guitar' else [])
                    for balance in report['weighting']:
                        x, y, weight = training_arrays(training, window, view, balance)
                        model = HistGradientBoostingClassifier(**parameters).fit(x, y, sample_weight=weight)
                        probabilities = {name: predict_songs(songs, model, window, view) for name, songs in validation.items()}
                        for threshold in report['thresholds']:
                            scores = {name: score_predictions(songs, probabilities[name], window, threshold)['summary'] for name, songs in validation.items()}
                            b_delta = scores['beatles']['tetrad_accuracy']-report['baseline']['beatles']['summary']['tetrad_accuracy']
                            g_delta = scores['guitar']['tetrad_accuracy']-report['baseline']['guitar']['summary']['tetrad_accuracy']
                            eligible = b_delta > 1e-12 and g_delta >= -1e-12
                            key = (scores['beatles']['tetrad_accuracy'], scores['guitar']['tetrad_accuracy'], -sum(v['changed_seconds'] for v in scores.values()))
                            candidate = dict(window=window, view=view, domain=domain, weighting=balance, threshold=threshold,
                                             scores=scores, beatles_delta=b_delta, guitar_delta=g_delta, eligible=eligible,
                                             target_counts=np.bincount(y, minlength=3).tolist())
                            report['candidates'].append(candidate)
                            if eligible and (winning_key is None or key > winning_key):
                                winner, winning_key = candidate, key
                                report['selected_development_details'] = {name: score_predictions(songs, probabilities[name], window, threshold, True) for name, songs in validation.items()}
                        report.update(status='development_selection', selected=winner)
                        write_report(report_path, report)
                        print(f'{window=} {view=} {domain=} {balance=} best={winning_key}', flush=True)
        report.update(status='selection_frozen', selected=winner)
        write_report(report_path, report)
        if winner is not None:
            from apply_nnls_extensions import feature_contract
            training = b + gt + gd if winner['domain'] == 'beatles_guitar' else b
            x, y, weight = training_arrays(training, winner['window'], winner['view'], winner['weighting'])
            model = HistGradientBoostingClassifier(**parameters).fit(x, y, sample_weight=weight)
            artifact = dict(model_schema=quality.MODEL_SCHEMA, model=model, selected=winner,
                            training_ids=[s['id'] for s in training], training_audio_sha256=[s['audio_identity']['sha256'] for s in training],
                            feature_contract=feature_contract(training[0]['audio'], winner['view']),
                            development_report=report, notes='Refit only after selection. No held-out labels read or scores used.')
            target = args.output / 'nnls_seventh.joblib'
            joblib.dump(artifact, target)
            report['artifact'] = signature(target)
        report.update(status='development_complete_model_frozen', elapsed_seconds=time.perf_counter()-started,
                      default_changed=False, held_out_labels_read=False)
        write_report(report_path, report)
        print(json.dumps(dict(selected=winner, elapsed_seconds=report['elapsed_seconds']), indent=2), flush=True)


if __name__ == '__main__':
    main()
