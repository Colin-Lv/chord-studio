"""Summarize a completed frozen expert evaluation and its existing role policy."""
import argparse
import bisect
import collections
import csv
import json
import statistics
from pathlib import Path

from evaluate_song_structure import locate_annotations

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('report', type=Path)
parser.add_argument('--benchmark-root', type=Path, default=Path('benchmarks'))
args = parser.parse_args()
directory = args.report.parent
report = json.loads(args.report.read_text(encoding='utf-8'))
if report['remaining'] or report['failure_count'] or report['skipped']:
    raise RuntimeError('Full fixed manifest has not finished successfully')
root = args.benchmark_root.resolve()
role_mapping = {'intro': 'intro', 'outro': 'outro', 'verse': 'verse', 'chorus': 'chorus', 'bridge': 'bridge', 'verse_(instrumental)': 'instrumental', 'chorus_(instrumental)': 'instrumental'}
manifest = {row['id']: row for row in csv.DictReader(Path(report['manifest']).open(encoding='utf-8-sig'))}
rows, policy_rows, phase_rows, confusion = [], [], [], {}
reference_label_seconds = collections.Counter()
for item in report['tracks']:
    track_id = item['id']
    rows.append({'id': track_id, 'boundary_f1_0.5s': item['boundary_metrics']['0.5']['f1'], 'boundary_f1_3s': item['boundary_metrics']['3.0']['f1'], 'beat_f1_70ms': item['beat_metrics_70ms']['f1'], 'downbeat_f1_70ms': item['downbeat_metrics_70ms']['f1'], 'mapped_role_accuracy': item['role_metrics']['duration_weighted_accuracy'], 'cache_reused': item['provenance']['cache_reused'], 'runtime_seconds': item['runtime_seconds'], 'decoded_duration_seconds': item['provenance']['duration']})
    payload_path = directory / (args.report.stem + '_payloads') / report['model'] / (track_id + '.json')
    payload = json.loads(payload_path.read_text(encoding='utf-8'))
    raw = json.loads(Path(item['provenance']['result']).read_text(encoding='utf-8'))
    beat_path = root / manifest[track_id]['beat_reference']
    beat_rows = [line.split() for line in beat_path.read_text(encoding='utf-8-sig').splitlines() if line.strip() and not line.startswith('#')]
    times = [float(line[0]) for line in beat_rows]
    phase_counts = collections.Counter()
    for timestamp in raw['downbeats']:
        index = bisect.bisect_left(times, timestamp)
        options = [value for value in [index - 1, index] if 0 <= value < len(times)]
        if not options:
            phase_counts['unmatched'] += 1
            continue
        nearest = min(options, key=lambda value: abs(times[value] - timestamp))
        phase_counts[beat_rows[nearest][1] if len(beat_rows[nearest]) > 1 and abs(times[nearest] - timestamp) <= .07 else 'unmatched'] += 1
    phase_rows.append({'id': track_id, 'reference_position_of_predicted_downbeats_within_70ms': dict(phase_counts), 'predicted_downbeat_count': len(raw['downbeats'])})
    annotations = [line.split(maxsplit=2) for line in locate_annotations(root, track_id).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    total = accepted = correct = 0.
    sections = payload['structure']['sections']
    for start, end, label in annotations:
        reference_label_seconds[label] += float(end) - float(start)
        role = role_mapping.get(label)
        if role is None:
            continue
        start, end = float(start), float(end)
        total += end - start
        for segment in raw['segments']:
            overlap = max(0., min(end, segment['end']) - max(start, segment['start']))
            if overlap:
                predicted = 'instrumental' if segment['label'] == 'inst' else segment['label']
                confusion.setdefault(role, {})[predicted] = confusion.get(role, {}).get(predicted, 0.) + overlap
        for section in sections:
            if section['role'] == 'unknown':
                continue
            overlap = max(0., min(end, section['end']) - max(start, section['start']))
            accepted += overlap
            if section['role'] == role:
                correct += overlap
    policy_rows.append({'id': track_id, 'evaluated_seconds': total, 'accepted_seconds': accepted, 'correct_accepted_seconds': correct, 'accepted_coverage': accepted / total if total else None, 'accuracy_among_accepted': correct / accepted if accepted else None, 'section_count': len(sections), 'accepted_section_count': sum(s['role'] != 'unknown' for s in sections)})

with (directory / (args.report.stem + '_per_song.csv')).open('w', newline='', encoding='utf-8-sig') as target:
    writer = csv.DictWriter(target, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
total = sum(r['evaluated_seconds'] for r in policy_rows)
accepted = sum(r['accepted_seconds'] for r in policy_rows)
correct = sum(r['correct_accepted_seconds'] for r in policy_rows)
policy = {'protocol': 'Fixed pre-existing display threshold 0.55; no threshold tuning on eval17', 'note': 'Selective coverage/accuracy among mapped roles only, with no silence. This evaluates the frozen suggestion policy, not calibrated probabilities or full vocabulary equivalence.', 'threshold': .55, 'track_count': len(policy_rows), 'accepted_coverage_duration_weighted': accepted / total, 'accuracy_among_accepted_duration_weighted': correct / accepted, 'macro_accepted_coverage': statistics.mean(r['accepted_coverage'] for r in policy_rows), 'macro_accuracy_among_accepted': statistics.mean(r['accuracy_among_accepted'] for r in policy_rows if r['accuracy_among_accepted'] is not None), 'tracks': policy_rows}
policy['reference_vocabulary'] = {'seconds_by_label': dict(reference_label_seconds), 'mapped_seconds': total, 'all_annotated_seconds': sum(reference_label_seconds.values()), 'mapped_fraction_of_all_annotations': total / sum(reference_label_seconds.values()), 'mapped_fraction_of_non_silence_annotations': total / (sum(reference_label_seconds.values()) - reference_label_seconds.get('silence', 0)), 'excluded_labels': sorted(set(reference_label_seconds) - set(role_mapping)), 'literal_chorus_reference_seconds': reference_label_seconds.get('chorus', 0)}
(directory / (args.report.stem + '_role_policy.json')).write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding='utf-8')
(directory / (args.report.stem + '_role_confusion.json')).write_text(json.dumps({'protocol': 'Descriptive raw-role errors on fixed eval17; not used to fit parameters', 'values': 'overlapping seconds; rows mapped reference roles, columns raw predicted roles', 'matrix': confusion}, ensure_ascii=False, indent=2), encoding='utf-8')
(directory / (args.report.stem + '_phase_diagnostics.json')).write_text(json.dumps({'protocol': 'Reference-position error analysis only; no phase correction applied to predictions or scores', 'note': 'Each predicted downbeat is associated with its nearest annotated beat only if within the existing 70ms tolerance; counts by reference beat position help distinguish timing from bar-phase mistakes.', 'tracks': phase_rows}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in report.items() if k.startswith('macro')}, ensure_ascii=False))
print(json.dumps({k: v for k, v in policy.items() if k != 'tracks'}, ensure_ascii=False))
