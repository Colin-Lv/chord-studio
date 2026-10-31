"""Experimental reference-free triad/seventh proposals; no pipeline promotion.

The suffix head is global, not root-conditioned. A sum of log marginals is
an uncalibrated ranking heuristic, never a joint probability or bass estimate.
"""
from __future__ import annotations
import numpy as np
from largevocab_posteriors_v2 import HEAD_LABELS

SCHEMA = "posterior_triad_seventh_proposals_v1"


def propose(triad, seventh, *, budget=3, coverage=1.):
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("budget must be a positive integer")
    if not np.isfinite(coverage) or not 0 <= coverage <= 1:
        raise ValueError("coverage must be between zero and one")
    if coverage == 0:
        return []
    arrays = []
    for values, size in ((triad, 73), (seventh, 4)):
        values = np.asarray(values, dtype=float)
        if (values.shape != (size,) or not np.isfinite(values).all()
                or np.any(values < 0) or np.any(values > 1)
                or not np.isclose(values.sum(), 1., rtol=0, atol=2e-5)):
            raise ValueError("Expected finite normalized original-head probabilities")
        arrays.append(values)
    triad, seventh = arrays
    candidates = []
    for ti, body in enumerate(HEAD_LABELS["triad"]):
        for si, degree in enumerate(HEAD_LABELS["seventh"]):
            if ti == 0 and si != 0:
                continue
            score = float(np.log(max(triad[ti], 1e-30)) + np.log(max(seventh[si], 1e-30)))
            label = body if degree == "none" else body + "(" + degree + ")"
            candidates.append(dict(label=label, score=score, triad_index=ti, seventh_index=si,
                                   acoustic_coverage=float(coverage)))
    candidates.sort(key=lambda c: (-c["score"], c["triad_index"], c["seventh_index"]))
    return candidates[:budget]


def split_supported_intervals(raw, external):
    """Split at source endpoints without filling raw gaps or changing labels."""
    from evaluate_chords import TimedChord
    previous = -float('inf')
    for row in raw:
        if (not np.isfinite([row.start, row.end]).all() or row.start < 0
                or row.end <= row.start or row.start < previous):
            raise ValueError("Raw intervals must be ordered, disjoint and finite")
        previous = row.end
    for rows in external:
        for row in rows:
            if (not np.isfinite([row.start, row.end]).all() or row.start < 0
                    or row.end <= row.start):
                raise ValueError("External intervals must be finite and positive")
    points = sorted({v for rows in external for row in rows for v in (row.start, row.end)})
    output = []
    for row in raw:
        cuts = [row.start, *[v for v in points if row.start < v < row.end], row.end]
        output.extend(TimedChord(a, b, row.chord) for a, b in zip(cuts, cuts[1:]) if b > a)
    return output


def export_candidates(evidence, anchors, *, expected_audio_sha256, budget=3, split_sources=()):
    """An opt-in inspection artifact, never selected predictions or references."""
    from largevocab_posteriors_v2 import aggregate_posteriors, validate_evidence
    validate_evidence(evidence)
    source = evidence.metadata['identity'].get('source_audio', {})
    if not expected_audio_sha256 or source.get('sha256') != expected_audio_sha256:
        raise ValueError("Posterior belongs to a different declared audio SHA")
    anchors = split_supported_intervals(anchors, split_sources)
    means = aggregate_posteriors(evidence, [[r.start, r.end] for r in anchors],
                                anchor_source=dict(kind='caller_supplied_timing', reference_inputs=False))
    rows = []
    for i, row in enumerate(anchors):
        rows.append(dict(start=row.start, end=row.end, acoustic_coverage=float(means.coverage[i]),
            candidates=propose(means.probabilities['triad'][i], means.probabilities['seventh'][i],
                               budget=budget, coverage=means.coverage[i])))
    return dict(schema=SCHEMA, status='experimental_candidates_not_predictions',
        source_identity=evidence.metadata['identity'], budget=budget, reference_inputs=False,
        anchor_audio_identity_verified=False,
        warning='Timing LAB pairing is caller supplied. Global suffix marginals are not root-conditioned; '
                'scores are uncalibrated heuristics. Bass and higher extensions are not estimated. '
                'No thresholds or recognizer output are replaced.', events=rows)


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    from candidate_data_v2 import read_source
    from largevocab_posteriors_v2 import load_posteriors
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--posterior', type=Path, required=True)
    parser.add_argument('--anchors', type=Path, required=True)
    parser.add_argument('--expected-audio-sha256', required=True)
    parser.add_argument('--split-source', type=Path, action='append', default=[])
    parser.add_argument('--budget', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite a candidate artifact')
    evidence = load_posteriors(args.posterior)
    raw = read_source(args.anchors, anchor=True)[0]
    external = [read_source(path)[0] for path in args.split_source]
    result = export_candidates(evidence, raw, expected_audio_sha256=args.expected_audio_sha256,
                               budget=args.budget, split_sources=external)
    paths = [args.posterior, args.anchors, *args.split_source]
    result['inputs'] = [dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                        for path in paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids races with another experiment.
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), intervals=len(result['events']), budget=args.budget)))


if __name__ == '__main__':
    main()
