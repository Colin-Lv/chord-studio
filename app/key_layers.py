"""Two extra layers for the production key decision, self-contained.

A. degree / diatonic accounting + soft penalties (Mixolydian b7, minor dominant,
   tonic minor inside a major candidate, weak tonic, too much out-of-key material)
B. pop-idiom match on the song's most-used scale degrees (456 / 451 / 6415 / 4536 /
   4361 / 251 / canon / minor loops) with a major-frame bonus and a V-start penalty

Works on key_from_chords Chord objects so nothing else has to change.
"""
from __future__ import annotations

from collections import defaultdict

NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]
MINOR_STEPS = [0, 2, 3, 5, 7, 8, 10]
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# canonical loops as 0-based degrees with the triad quality of each degree
PATTERNS = {
    "456": ([3, 4, 5], "major", 1.00),
    "451": ([3, 4, 0], "major", 1.00),
    "6415": ([5, 3, 0, 4], "major", 0.98),
    "4536": ([3, 4, 2, 5], "major", 0.95),
    "4361": ([3, 2, 5, 0], "major", 0.90),
    "251": ([1, 4, 0], "major", 0.90),
    "15634145": ([0, 4, 5, 2, 3, 0, 3, 4], "major", 0.95),
    "671": ([5, 6, 0], "minor", 0.90),
    "6451": ([5, 3, 0, 4], "minor", 0.85),
    "176": ([0, 6, 5], "minor", 0.80),
}
MAJOR_MINOR_DEGREE = {0: False, 1: True, 2: True, 3: False, 4: False, 5: True, 6: True}
MINOR_MINOR_DEGREE = {0: True, 1: True, 2: False, 3: True, 4: True, 5: False, 6: False}
START_PRIOR = {0: 1.00, 3: 0.85, 5: 0.85, 1: 0.85, 4: 0.35, 2: 0.70, 6: 0.70}


def _degree(root: int | None, tonic: int, mode: str):
    if root is None:
        return None, None
    steps = MAJOR_STEPS if mode == "major" else MINOR_STEPS
    interval = (root - tonic) % 12
    if interval in steps:
        return steps.index(interval), 0
    for index, step in enumerate(steps):
        if (step + 1) % 12 == interval:
            return index, 1
        if (step - 1) % 12 == interval:
            return index, -1
    return None, None


def _pcs(chord):
    from key_from_chords import chord_pcs
    try:
        return [int(pc) % 12 for pc, _ in chord_pcs(chord)]
    except Exception:
        return [chord.root % 12] if chord.root is not None else []


def score(chords, tonic: int, mode: str) -> dict:
    key = NAMES[tonic] + ("m" if mode == "minor" else "")
    total = sum(c.dur for c in chords) or 1.0
    scale = {(tonic + s) % 12 for s in (MAJOR_STEPS if mode == "major" else MINOR_STEPS)}
    signature = defaultdict(float)
    diatonic_time = tonic_time = 0.0
    outside_time = 0.0
    for chord in chords:
        pcs = _pcs(chord)
        inside = [pc for pc in pcs if pc in scale]
        outside_time += (len(pcs) - len(inside)) / max(1, len(pcs)) * chord.dur
        if pcs and all(pc in scale for pc in pcs):
            diatonic_time += chord.dur
        if chord.root == tonic and (chord.kind in ("min",) if mode == "minor" else chord.kind not in ("min",)):
            tonic_time += chord.dur
        degree, accidental = _degree(chord.root, tonic, mode)
        if degree is not None:
            signature[(degree, accidental, chord.kind in ("min",))] += chord.dur

    shares = {k: v / total for k, v in signature.items()}
    diatonic_share = diatonic_time / total
    tonic_share = tonic_time / total

    def pc_time(pc):
        return sum(c.dur for c in chords if pc in _pcs(c)) / total

    leading = pc_time((tonic + 11) % 12)
    flat7 = pc_time((tonic + 10) % 12)
    minor_dominant = sum(c.dur for c in chords
                         if c.root == (tonic + 7) % 12 and c.kind == "min") / total
    tonic_minor = sum(c.dur for c in chords
                      if c.root == tonic and c.kind == "min" and mode == "major") / total
    cadences = _cadences(chords, tonic)

    penalties = {}
    if tonic_share < 0.08 and cadences == 0:
        penalties["weak_tonic"] = -0.15
    if mode == "major" and leading < 0.03 and flat7 > 0.05:
        penalties["flat7_mixolydian"] = -0.20
    if minor_dominant > 0.03:
        penalties["minor_dominant"] = -0.10
    if mode == "major" and tonic_minor > 0.05:
        penalties["tonic_minor"] = -0.15
    if diatonic_share < 0.75:
        penalties["out_of_key_share"] = -round(min(0.30, ((0.75 - diatonic_share) / 0.05) * 0.05), 3)

    # B. idiom match on the most-used degrees
    ranked = sorted(shares.items(), key=lambda kv: -kv[1])
    start_degree = next((_degree(c.root, tonic, mode)[0] for c in chords if c.root is not None), None)
    start_prior = START_PRIOR.get(start_degree, 0.7) if start_degree is not None else 0.7
    best = dict(id=None, frame=None, coverage=0.0, prior=0.0)
    for pid, (degrees, frame, prior) in PATTERNS.items():
        table = MINOR_MINOR_DEGREE if frame == "minor" else MAJOR_MINOR_DEGREE
        template = {(d, 0, table[d]) for d in degrees}
        n = len(degrees)
        if len(ranked) < n:
            continue
        top = {item for item, _ in ranked[:n]}
        if top == template:
            coverage = sum(share for _, share in ranked[:n])
            if prior * coverage > best["prior"] * best["coverage"]:
                best = dict(id=pid, frame=frame, coverage=coverage, prior=prior)

    minor_cadence = _cadences(chords, (tonic + 9) % 12)
    frame_bonus = 0.0
    if mode == "major" and best["frame"] == "major" and best["coverage"] >= 0.40 and minor_cadence < 2:
        frame_bonus = 1.20
    idiom = best["coverage"] * best["prior"] * start_prior + frame_bonus

    return dict(key=key, penalties=penalties, penalty=round(sum(penalties.values()), 4),
                diatonic_share=round(diatonic_share, 4), tonic_share=round(tonic_share, 4),
                leading=round(leading, 4), flat7=round(flat7, 4),
                idiom=round(idiom, 4), idiom_id=best["id"], idiom_frame=best["frame"],
                idiom_coverage=round(best["coverage"], 4), start_prior=start_prior,
                frame_bonus=frame_bonus, cadences=cadences, minor_cadences=minor_cadence)


def _cadences(chords, tonic: int) -> int:
    count = 0
    for prev, cur in zip(chords, chords[1:]):
        if prev.root is None or cur.root is None:
            continue
        if cur.root % 12 == tonic and (prev.root - cur.root) % 12 == 7 and prev.kind != "min":
            count += 1
    return count


def adjust(cnn_key, chord_key, chords, cnn_total: float = 0.0, chord_total: float = 0.0,
           margin: float = 0.5) -> dict:
    """Decide between the audio key and the chord key with both layers."""
    from key_from_chords import parse_rows
    parsed, _ = parse_rows([(c.start, c.end, c.label) for c in chords])
    a = score(parsed, *_pc_mode(chord_key, NAMES))
    b = score(parsed, *_pc_mode(cnn_key, NAMES))
    chord_score = chord_total + a["idiom"] + a["penalty"]
    cnn_score = cnn_total + b["idiom"] + b["penalty"]
    return dict(use_chord=chord_score - cnn_score >= 0.10, chord_score=round(chord_score, 3),
                cnn_score=round(cnn_score, 3), chord_layers=a, cnn_layers=b)


def _pc_mode(label):
    body = str(label).strip()
    mode = "minor" if body.endswith("m") and not body.endswith("maj") else "major"
    root = body[:-1] if mode == "minor" else body
    pc = NOTE_PC[root[0]]
    for ch in root[1:]:
        pc += 1 if ch == "#" else -1
    return pc % 12, mode
