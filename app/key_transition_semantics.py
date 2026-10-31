"""Parsed chord semantics shared by the isolated production repair experiment.

These are adjacent harmonic arrivals, NOT phrase-final cadence detection.
No thresholds for choosing the global key live here.
"""
import math


def triad_quality(chord):
    return {(4, 7): 'major', (3, 7): 'minor', (3, 6): 'diminished',
            (4, 8): 'augmented'}.get((chord.third, chord.fifth))


def major_minor_flag(chord):
    quality = triad_quality(chord)
    return True if quality == 'minor' else False if quality == 'major' else None


def tonic_triad(chord, tonic, mode):
    if mode not in ('major', 'minor'):
        raise ValueError('Expected major or minor mode')
    return chord.root == tonic and triad_quality(chord) == mode


def dominant_arrivals(chords, tonic, mode):
    """Conventional V(7) -> I/i, preserving time gaps and target triad quality.

    Inversions are retained: this makes no claim of a root-position cadence.
    An altered dominant or suspended third is not silently treated as V(7).
    """
    if mode not in ('major', 'minor'):
        raise ValueError('Expected major or minor mode')
    events = []
    for prev, cur in zip(chords, chords[1:]):
        times = (prev.start, prev.end, cur.start, cur.end)
        if not all(math.isfinite(t) for t in times):
            raise ValueError('Non-finite chord time')
        if prev.end <= prev.start or cur.end <= cur.start:
            raise ValueError('Non-positive chord duration')
        if abs(prev.end - cur.start) > 1e-5:
            continue
        if prev.root is None or cur.root is None:
            continue
        if ((prev.root - tonic) % 12 == 7 and triad_quality(prev) == 'major'
                and prev.seventh in (None, 10) and tonic_triad(cur, tonic, mode)):
            events.append(dict(time=cur.start, previous=prev.label, target=cur.label,
                               scope='adjacent_harmony_only'))
    return events
