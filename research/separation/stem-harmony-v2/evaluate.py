"""Exact interval-duration scoring; missing output counts as wrong, never disappears."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_lab(path):
    return [dict(start=float(a), end=float(b), label=c) for a, b, c in
            (line.split() for line in Path(path).read_text().splitlines() if line.strip())]


def score(rows, truth):
    total = sum(t['end']-t['start'] for t in truth)
    exact = root = covered = 0.
    for ref in truth:
        for row in rows:
            overlap = max(0., min(ref['end'], row['end'])-max(ref['start'], row['start']))
            covered += overlap
            exact += overlap * (ref['label'] == row['label'])
            root += overlap * (ref['label'].split(':')[0] == row['label'].split(':')[0])
    rb = [r['start'] for i, r in enumerate(truth) if i and r['label'] != truth[i-1]['label']]
    pb = [r['start'] for i, r in enumerate(rows) if i and r['label'] != rows[i-1]['label']
          and truth[0]['start'] < r['start'] < truth[-1]['end']]
    unused = set(range(len(rb))); hits = 0
    for t in pb:
        options = [i for i in unused if abs(rb[i]-t) <= .5]
        if options:
            unused.remove(min(options, key=lambda i: abs(rb[i]-t)));hits += 1
    precision, recall = hits/max(1,len(pb)), hits/max(1,len(rb))
    return dict(exact=exact/total, root=root/total, coverage=covered/total, seconds=total,
                segments=len(rows), boundary_precision=precision, boundary_recall=recall,
                boundary_f1=2*precision*recall/max(1e-9,precision+recall), tolerance_seconds=.5)


if __name__ == '__main__':
    truth = read_lab(HERE.parents[1] / 'golden/reference_golden.lab')
    candidate = json.loads((HERE / 'xuni/result.json').read_text(encoding='utf8'))
    result = {name: score(rows, truth) for name, rows in [('released_full', candidate['baseline']), ('stem_v2', candidate['rows'])]}
    diagnostics = json.loads((HERE/'xuni/diagnostics.json').read_text(encoding='utf8'))
    control = [dict(start=r['start'],end=r['end'],label=r['baseline']) for r in diagnostics['segments']]
    result['complete_coverage_without_stem_evidence'] = score(control, truth)
    (HERE / 'xuni_scores.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
