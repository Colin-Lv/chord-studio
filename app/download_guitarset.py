"""Download official GuitarSet 1.1.0 archives with checksums and resumable partials."""
import argparse
import datetime
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=Path('benchmarks/raw/guitarset_v1.1.0'))
parser.add_argument('--deadline', help='Optional local ISO date-time after which partial downloads are preserved and stopped')
parser.add_argument('--annotations-only', action='store_true')
args = parser.parse_args()
directory = args.output_dir
directory.mkdir(parents=True, exist_ok=True)
deadline = datetime.datetime.fromisoformat(args.deadline) if args.deadline else datetime.datetime.max
records = []
for name, expected in [('annotation.zip', 'b39b78e63d3446f2e54ddb7a54df9b10'), ('audio_mono-mic.zip', '275966d6610ac34999b58426beb119c3')]:
    if args.annotations_only and name != 'annotation.zip':
        continue
    target = directory / name
    partial = directory / (name + '.part')
    url = f'https://zenodo.org/records/3371780/files/{name}?download=1'
    record = {'name': name, 'url': url, 'expected_md5': expected, 'status': 'pending'}
    records.append(record)
    for attempt in range(1, 5):
        if datetime.datetime.now() >= deadline:
            record.update(status='partial_deadline', bytes=partial.stat().st_size if partial.exists() else 0)
            break
        try:
            if not target.exists():
                size = partial.stat().st_size if partial.exists() else 0
                request = urllib.request.Request(url, headers={'User-Agent': 'ChordStudio dataset verification', **({'Range': f'bytes={size}-'} if size else {})})
                with urllib.request.urlopen(request, timeout=25) as response:
                    if response.status != 206:
                        size = 0
                    record.update(resolved_url=response.url, response_status=response.status)
                    last_log = time.monotonic()
                    with partial.open('ab' if size else 'wb') as output:
                        while True:
                            if datetime.datetime.now() >= deadline:
                                raise TimeoutError('Dataset download deadline reached; partial preserved')
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                            size += len(chunk)
                            if time.monotonic() - last_log > 15:
                                print(f'{name}: {size / 1e6:.1f} MB', flush=True)
                                last_log = time.monotonic()
                candidate = partial
            else:
                candidate = target
            md5, sha = hashlib.md5(), hashlib.sha256()
            with candidate.open('rb') as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    md5.update(chunk); sha.update(chunk)
            if md5.hexdigest() != expected:
                raise ValueError(f'MD5 mismatch {md5.hexdigest()} != {expected}')
            if candidate != target:
                partial.replace(target)
            record.update(status='verified', bytes=target.stat().st_size, md5=md5.hexdigest(), sha256=sha.hexdigest(), completed_local=datetime.datetime.now().isoformat())
            print(json.dumps(record), flush=True)
            break
        except Exception as exc:
            record.update(status='partial_or_failed', error=str(exc), attempt=attempt, bytes=partial.stat().st_size if partial.exists() else 0)
            print(json.dumps(record), flush=True)
            if isinstance(exc, ValueError):
                break
    (directory / 'download_provenance.json').write_text(json.dumps({'official_record': 'https://zenodo.org/records/3371780', 'files': records}, ensure_ascii=False, indent=2), encoding='utf-8')
