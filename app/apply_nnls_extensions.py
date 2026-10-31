"""Apply a frozen experimental NNLS seventh head to audio and a raw LAB."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import mir_eval
import numpy as np

import nnls_seventh_model as quality
import harmonic_note_features as notes
from apply_chord_extensions import atomic_write, read_upstream, same_file
from evaluate_chords import aligned_intervals, harte_chord
from madmom_bridge import file_identity


def feature_contract(audio, view):
    identity = notes.extraction_identity(audio)
    contract = {key: value for key, value in identity.items() if key in (
        'cache_schema', 'sample_rate', 'block_size', 'hop_length', 'right_padding_samples',
        'sonic_sha256', 'plugin_sha256', 'decoder_sha256', 'decoder_implementation_sha256',
        'bridge_implementation_sha256', 'transform_id', 'transform_sha256')}
    contract.update(note_feature_schema=notes.FEATURE_SCHEMA, note_features=list(notes.FEATURE_NAMES),
                    view=view, dimension=97 if view == 'nnls' else 245,
                    quality_implementation_sha256=notes.sha256_file(quality.__file__))
    if view == 'nnls_fast':
        import chord_extension_model as old
        contract['fast_schema'] = old.FEATURE_SCHEMA_V2
        contract['fast_code'] = {name: notes.sha256_file(Path(__file__).with_name(name)) for name in
                                 ('chord_extension_model.py', 'chord_studio.py', 'audio_features.py', 'acoustic_chord_model.py')}
    return contract


def inference_features(audio, upstream, selected, cache=None):
    windows, parents = quality.split_upstream(upstream, selected['window'])
    evidence = notes.extract_note_evidence(audio, cache_dir=cache)
    feature_rows = quality.acoustic_windows(windows, evidence.metadata['duration_seconds'])
    features = notes.features_for_intervals(evidence, feature_rows)
    if selected['view'] == 'nnls_fast':
        import chord_extension_model as old
        import chord_studio as studio
        fast, _ = old.features_for_rows(studio.decode_audio(audio), windows, feature_schema=old.FEATURE_SCHEMA_V2)
        features = np.concatenate([features, fast], axis=1)
    elif selected['view'] != 'nnls':
        raise ValueError('Unknown feature view')
    return windows, parents, features, evidence.metadata


def verify_semantics(upstream, result):
    for _, _, original, predicted in aligned_intervals(upstream, result):
        if original is None:
            if predicted is not None:
                raise ValueError('Prediction filled an upstream gap')
            continue
        if predicted is None:
            raise ValueError('Prediction lost upstream support')
        if original.chord == predicted.chord:
            continue
        before, after = harte_chord(original.chord), harte_chord(predicted.chord)
        if mir_eval.chord.root([before], [after])[0] != 1 or mir_eval.chord.triads([before], [after])[0] != 1:
            raise ValueError('Prediction changed an upstream root or triad')
        if '/' in original.chord:
            if '/' not in predicted.chord or mir_eval.chord.triads_inv([before], [after])[0] != 1:
                raise ValueError('Prediction changed explicit bass')


def apply_model(audio, upstream_path, model_path, output, cache=None):
    audio, upstream_path, model_path, output = (Path(p).resolve() for p in (audio, upstream_path, model_path, output))
    outputs = [output, output.with_name(output.name+'.provenance.json'), output.with_name(output.name+'.evidence.json')]
    sources = [('audio', audio), ('upstream_lab', upstream_path), ('model', model_path)]
    if any(same_file(source, target) for _, source in sources for target in outputs):
        raise ValueError('Outputs cannot overwrite inputs')
    signatures = {name: file_identity(path) for name, path in sources}
    artifact = joblib.load(model_path)  # User-selected trusted local artifact.
    if artifact.get('model_schema') != quality.MODEL_SCHEMA:
        raise ValueError('Wrong model family')
    selected = artifact['selected']
    contract = feature_contract(audio, selected['view'])
    if contract != artifact.get('feature_contract'):
        raise ValueError('Frozen artifact and current feature implementation/toolchain differ')
    raw, _, upstream, _ = read_upstream(upstream_path)
    windows, parents, features, note_metadata = inference_features(audio, upstream, selected, cache)
    probability = quality.predict_probabilities(artifact['model'], features)
    result, changes = quality.decode_windows(upstream, windows, parents, probability, selected['threshold'])
    verify_semantics(upstream, result)
    content = raw if not changes else ''.join(f'{r.start!r} {r.end!r} {r.chord}\n' for r in result).encode('utf-8')
    for name, path in sources:
        if file_identity(path)['sha256'] != signatures[name]['sha256']:
            raise ValueError(f'{name} changed during inference')
    evidence = dict(score_kind='uncalibrated_extension_evidence', changes=changes,
                    threshold=selected['threshold'], window=selected['window'], view=selected['view'])
    # Reuse the documented derived-LAB source schema; this does not make raw
    # cache claims or claim that all original interval counts were retained.
    provenance = dict(schema='fixed_root_seventh_application_v1', model_family=quality.MODEL_SCHEMA, **signatures,
                      feature_contract=contract, feature_schema=notes.FEATURE_SCHEMA,
                      threshold=selected['threshold'], change_count=len(changes),
                      upstream_intervals=len(upstream), output_intervals=len(result),
                      source_boundaries_root_triad_gaps_explicit_bass_preserved=True,
                      quality_subdivision_allowed=bool(selected['window']),
                      note_extraction=note_metadata, reference_labels_used=False,
                      output_lab=dict(path=str(output), sha256=hashlib.sha256(content).hexdigest(), bytes=len(content)))
    encode = lambda obj: (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    atomic_write(outputs[2], encode(evidence))
    atomic_write(outputs[1], encode(provenance))
    atomic_write(outputs[0], content)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', type=Path)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('.tmp/nnls_seventh_features/notes'))
    args = parser.parse_args()
    result = apply_model(args.audio, args.upstream, args.model, args.output, args.cache)
    print(f"{result['change_count']} changed windows; {result['output_lab']['path']}")


if __name__ == '__main__':
    main()
