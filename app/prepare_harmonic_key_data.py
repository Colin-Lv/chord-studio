"""Freeze new harmonic-only groups before any new acoustic prediction.

All previously selected GiantSteps+ tracks, including their former sealed set,
are known regressions. Rebuild components on every eligible track and exclude
the whole component whenever it contains any old selected track.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import zipfile

from evaluate_global_key_fusion import signature
from prepare_key_resolver_data import freeze, grouped_split

SEED = 2026100602


def canonical(text):
    text = unicodedata.normalize('NFKC', str(text)).casefold().strip()
    return re.sub(r'\s+', ' ', text)


def isolated_rows(rows):
    eligible = [copy.deepcopy(row) for row in rows if row['eligible']]
    grouped_split(eligible)
    known_groups = {row['group'] for row in eligible if row['selected']}
    for row in eligible:
        row['known_component'] = row['group'] in known_groups
        row['harmonic_selected'] = not row['known_component']
        value = hashlib.sha256(f'{SEED}:{row["group"]}'.encode()).hexdigest()
        portion = int(value[:8], 16) / 2**32
        row['harmonic_split'] = ('harmonic_dev' if portion < .6 else
                                 'harmonic_calibration' if portion < .8 else 'harmonic_new_sealed')
    return eligible


def prepare(old_split, metadata_zip, output):
    old = json.loads(Path(old_split).read_text(encoding='utf-8'))
    rows = copy.deepcopy(old['rows'])
    metadata = {}
    with zipfile.ZipFile(metadata_zip) as archive:
        for name in archive.namelist():
            if not name.startswith('meta/') or not name.endswith('.json'):
                continue
            try:
                item = json.loads(archive.read(name))
                metadata[str(item['id'])] = item
            except json.JSONDecodeError:
                continue
    for row in rows:
        item = metadata.get(row['id'], {})
        for artist in item.get('artists', []) + item.get('remixers', []):
            name = canonical(artist.get('name', ''))
            if name:
                row['group_tokens'].append('artist_name:' + name)
        # Record names normally omit version qualifiers. Conservative stripping
        # also links common original/radio/extended/instrumental versions.
        for token in list(row['group_tokens']):
            if token.startswith('work:'):
                work = canonical(token[5:])
                base = re.sub(r'\s*\((?:original|radio|extended|instrumental|album|club)(?: mix| edit| version)?\)\s*$', '', work)
                row['group_tokens'].append('canonical_work:' + base)
            elif token.startswith('artist_text:'):
                row['group_tokens'].append('artist_name:' + canonical(token[12:]))
    prepared = isolated_rows(rows)
    selected = [row for row in prepared if row['harmonic_selected']]
    stats = {phase: dict(tracks=sum(row['harmonic_split'] == phase for row in selected),
                         groups=len({row['group'] for row in selected if row['harmonic_split'] == phase}),
                         major=sum(row['harmonic_split'] == phase and not row['reference'].endswith('m') for row in selected),
                         minor=sum(row['harmonic_split'] == phase and row['reference'].endswith('m') for row in selected))
             for phase in ['harmonic_dev', 'harmonic_calibration', 'harmonic_new_sealed']}
    result = dict(schema='harmonic_group_split_v2', seed=SEED, old_split=signature(old_split),
                  metadata=signature(metadata_zip), eligible=len(prepared), old_known_tracks=sum(row['selected'] for row in prepared),
                  excluded_old_component_tracks=sum(row['known_component'] for row in prepared), selected=len(selected),
                  grouping='eligible connected artist/remixer IDs + artist names + canonical work + track IDs',
                  group_intersection_with_old=0, partitions=stats, rows=prepared,
                  selection='all eligible tracks in components without an old selected track; hash partition before inference',
                  limitations=['Metadata aliases and unlisted recording versions cannot be exhaustively verified',
                               'Group separation from resolver study does not establish upstream CNN pretraining independence'])
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    freeze(output, result)
    print(json.dumps({key: result[key] for key in ['eligible', 'old_known_tracks', 'excluded_old_component_tracks', 'selected', 'partitions']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-split', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.old_split, args.metadata, args.output)
