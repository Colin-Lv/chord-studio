"""Prepare official GuitarSet comp excerpts as explicitly weak chord labels.

Does not change the default training corpus, trim audio, repair known timings,
or use annotations as model input. Two holdout protocols isolate different axes.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile
import zlib


KNOWN_TIMING_ERRORS = {'04_BN3-154-E_comp': .409, '04_Jazz1-200-B_comp': .309}
KNOWN_DUPLICATE_NOTE = '02_Funk2-119-G_comp'
EXPECTED_ANNOTATION_MD5 = 'b39b78e63d3446f2e54ddb7a54df9b10'
EXPECTED_AUDIO_MD5 = '275966d6610ac34999b58426beb119c3'
NAME_PATTERN = re.compile(r'(?P<performer>0[0-5])_(?P<style>BN|Funk|Jazz|Rock|SS)(?P<family>[123])-(?P<tempo>\d+)-(?P<key>[A-G][b#]?)_comp$')


def checksum(path: Path, algorithm='sha256') -> str:
    value = hashlib.new(algorithm)
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def select_chord_annotations(jam: dict) -> tuple[dict, dict]:
    chords = [(index, annotation) for index, annotation in enumerate(jam['annotations']) if annotation['namespace'] == 'chord']
    performed = [(index, annotation) for index, annotation in chords if 'sheet-informed' in str(annotation.get('annotation_metadata', {}).get('annotation_rules', '')).lower()]
    if len(chords) != 2 or len(performed) != 1:
        raise ValueError('Expected two chord namespaces, exactly one explicitly sheet-informed performed annotation')
    index, played = performed[0]
    instructed_index, instructed = next((i, a) for i, a in chords if i != index)
    return ({'annotation_index': instructed_index, **instructed}, {'annotation_index': index, **played})


def split_for(performer: str, family: str, protocol: str) -> str:
    if protocol == 'performer':
        return 'train' if performer in {'00', '01', '02', '03'} else 'validation' if performer == '04' else 'test'
    if protocol == 'progression':
        return {'1': 'train', '2': 'validation', '3': 'test'}[family]
    raise ValueError('Unknown split protocol')


def write_lab(path: Path, annotation: dict, duration: float) -> None:
    rows, last_end = [], 0.
    for observation in annotation['data']:
        start, length = float(observation['time']), float(observation['duration'])
        end = start + length
        if not math.isfinite(start + end) or start < 0 or length <= 0 or start < last_end - 1e-8 or end > duration + 1e-6:
            raise ValueError(f'Invalid original chord interval in {path.name}')
        rows.append(f"{start!r}\t{end!r}\t{observation['value']}\n")
        last_end = end
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(rows), encoding='utf-8')


def extract_verified(archive: Path, destination: Path, suffix: str, expected: str) -> int:
    if checksum(archive, 'md5') != expected:
        raise ValueError(f'Archive checksum failed: {archive}')
    destination.mkdir(parents=True, exist_ok=True)
    names = set()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if not member.filename.endswith(suffix) or member.filename.startswith('__MACOSX/'):
                continue
            name = Path(member.filename).name
            if name in names:
                raise ValueError('Duplicate archive basename')
            names.add(name)
            # Flatten to a controlled destination; never extract archive paths.
            target = destination / name
            matches = target.exists() and target.stat().st_size == member.file_size
            if matches:
                crc = 0
                with target.open('rb') as existing:
                    for chunk in iter(lambda: existing.read(1024 * 1024), b''):
                        crc = zlib.crc32(chunk, crc)
                matches = crc == member.CRC
            if not matches:
                target.write_bytes(source.read(member))
    return len(names)


def write_manifest(path: Path, rows: list[dict]) -> None:
    fields = ['id', 'audio', 'chord_reference', 'instructed_reference', 'beat_reference', 'annotation_metadata', 'reference_kind', 'performer', 'composition_id', 'progression_family', 'style', 'split', 'audio_ready']
    with path.open('w', newline='', encoding='utf-8-sig') as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('benchmarks/raw/guitarset_v1.1.0'))
    parser.add_argument('--benchmark-root', type=Path, default=Path('benchmarks'))
    args = parser.parse_args()
    root, raw = args.benchmark_root.resolve(), args.raw.resolve()
    annotations, audio_directory = raw / 'annotations', raw / 'audio_mono-mic'
    annotation_count = extract_verified(raw / 'annotation.zip', annotations, '.jams', EXPECTED_ANNOTATION_MD5)
    audio_count = extract_verified(raw / 'audio_mono-mic.zip', audio_directory, '.wav', EXPECTED_AUDIO_MD5) if (raw / 'audio_mono-mic.zip').exists() else 0
    if annotation_count != 360 or audio_count not in {0, 360}:
        raise ValueError('Official GuitarSet archive should contain 360 matching recordings')
    manifests = root / 'manifests/guitarset_v1.1.0'
    references = root / 'references/guitarset_v1.1.0'
    manifests.mkdir(parents=True, exist_ok=True)
    rows, excluded, warnings = [], [], []
    for source in sorted(annotations.glob('*_comp.jams')):
        match = NAME_PATTERN.fullmatch(source.stem)
        if match is None:
            raise ValueError(f'Unrecognized official recording name: {source.stem}')
        info = match.groupdict()
        jam = json.loads(source.read_text(encoding='utf-8'))
        duration = float(jam['file_metadata']['duration'])
        instructed, performed = select_chord_annotations(jam)
        audio = audio_directory / (source.stem + '_mic.wav')
        metadata_path = references / 'metadata' / (source.stem + '.json')
        performed_path = references / 'performed' / (source.stem + '.lab')
        instructed_path = references / 'instructed' / (source.stem + '.lab')
        beat_path = references / 'beats' / (source.stem + '.beats')
        write_lab(performed_path, performed, duration)
        write_lab(instructed_path, instructed, duration)
        beats = [a for a in jam['annotations'] if a['namespace'] == 'beat_position']
        if len(beats) != 1:
            raise ValueError('Ambiguous beat annotation')
        beat_path.parent.mkdir(parents=True, exist_ok=True)
        beat_path.write_text(''.join(f"{float(o['time'])!r}\t{int(o['value']['position'])}\n" for o in beats[0]['data']), encoding='utf-8')
        metadata = {'recording_id': source.stem, 'source_jams': str(source), 'source_jams_sha256': checksum(source), 'source_audio': str(audio), 'audio_available': audio.exists(), 'original_file_metadata': jam['file_metadata'], 'duration_seconds': duration, 'audio_offset': 0, 'source_license': 'CC-BY-4.0', 'label_kind': 'weak_sheet_informed_symbolic_derivative', 'independent_manual_chord_ground_truth': False, 'instructed': instructed, 'performed': performed, 'beat_annotation': beats[0], 'beat_source': 'known click-track tempo/meter, not freely performed beat annotations', 'known_timing_error': None, 'default_eligible': source.stem not in KNOWN_TIMING_ERRORS, 'references_must_not_be_model_inputs': True}
        if audio.exists():
            import soundfile as sf
            sound = sf.info(audio)
            metadata.update(audio_sha256=checksum(audio), audio_duration_seconds=sound.duration, sample_rate=sound.samplerate, channels=sound.channels, audio_minus_annotation_duration_seconds=sound.duration-duration)
            if abs(sound.duration - duration) > .05:
                warnings.append({'id': source.stem, 'kind': 'audio_annotation_duration_difference', 'seconds': sound.duration-duration})
        if source.stem in KNOWN_TIMING_ERRORS:
            metadata['known_timing_error'] = {'reported_annotation_delay_seconds': KNOWN_TIMING_ERRORS[source.stem], 'issue': 'https://github.com/marl/GuitarSet/issues/5', 'scope': 'reported note-alignment error; entire comp excerpt conservatively excluded', 'correction_applied': False}
        if source.stem == KNOWN_DUPLICATE_NOTE:
            metadata['duplicate_note_advisory'] = 'https://github.com/marl/GuitarSet/issues/4'
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        row = {'id': 'guitarset_' + source.stem, 'recording_id': source.stem, 'audio': str(audio.relative_to(root)), 'chord_reference': str(performed_path.relative_to(root)), 'instructed_reference': str(instructed_path.relative_to(root)), 'beat_reference': str(beat_path.relative_to(root)), 'annotation_metadata': str(metadata_path.relative_to(root)), 'reference_kind': 'weak_performed_sheet_informed', 'performer': info['performer'], 'composition_id': source.stem.split('_')[1], 'progression_family': info['family'], 'style': info['style'], 'audio_ready': audio.exists()}
        if not metadata['default_eligible']:
            excluded.append({**row, 'reason': metadata['known_timing_error']})
        else:
            rows.append(row)
    if audio_count and any(not row['audio_ready'] for row in rows + excluded):
        raise ValueError('Verified microphone archive did not match every JAMS basename')
    diagnostic = []
    for index, style in enumerate(['BN', 'Funk', 'Jazz', 'Rock', 'SS']):
        for offset, performer in enumerate(['00', '05']):
            family = str((index + offset) % 3 + 1)
            choice = next(row for row in rows if row['style'] == style and row['performer'] == performer and row['progression_family'] == family)
            diagnostic.append({**choice, 'split': 'diagnostic_no_fitting'})
    write_manifest(manifests / 'diagnostic10.csv', diagnostic)
    diagnostic_ids = {row['id'] for row in diagnostic}
    regular = [row for row in rows if row['id'] not in diagnostic_ids]
    protocol_counts = {}
    for protocol in ['performer', 'progression']:
        for split in ['train', 'validation', 'test']:
            selected = [{**row, 'split': split} for row in regular if split_for(row['performer'], row['progression_family'], protocol) == split]
            write_manifest(manifests / f'{protocol}_{split}.csv', selected)
            protocol_counts[f'{protocol}_{split}'] = len(selected)
    write_manifest(manifests / 'all_eligible_comp.csv', [{**row, 'split': 'diagnostic_no_fitting' if row['id'] in diagnostic_ids else 'unassigned'} for row in rows])
    report = {'dataset': 'GuitarSet 1.1.0', 'official_record': 'https://zenodo.org/records/3371780', 'official_repository': 'https://github.com/marl/GuitarSet', 'dataset_license': 'CC-BY-4.0', 'repository_code_license': 'MIT', 'annotation_file_count': annotation_count, 'audio_file_count': audio_count, 'comp_count': len(rows)+len(excluded), 'eligible_comp_count': len(rows), 'audio_ready_comp_count': sum(row['audio_ready'] for row in rows), 'solo_excluded_from_chord_diagnostics': True, 'performed_labels_are_independent_manual_truth': False, 'original_times_and_labels_preserved': True, 'default_training_corpus_modified': False, 'protocol_counts': protocol_counts, 'split_protocols': {'performer': {'train': ['00','01','02','03'], 'validation': ['04'], 'test': ['05'], 'limitation': 'Lead sheets/progression families shared across splits'}, 'progression': {'train': ['1'], 'validation': ['2'], 'test': ['3'], 'families': {'1': '12-bar blues', '2': 'Autumn Leaves', '3': 'Pachelbel Canon'}, 'limitation': 'Performers shared across splits; only three progression families'}}, 'diagnostic_selection': 'Fixed 5 styles x performers 00/05, family=(style_index+performer_index)%3+1, first sorted lead; no predictions used; no fitting/tuning permitted', 'diagnostic_ids': [row['id'] for row in diagnostic], 'excluded': excluded, 'warnings': warnings, 'annotation_archive_sha256': checksum(raw/'annotation.zip')}
    report.update(diagnostic_reserved_count=len(diagnostic), regular_split_pool_count=len(regular), diagnostic_excluded_from_all_regular_splits=True)
    (manifests / 'import_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key not in ['excluded','diagnostic_ids','warnings']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
