"""Decide whether the chord evidence should replace the audio CNN key.

Two guards, both validated in research:

* `guard_456` — a pure IV-V-vi loop has no tonic-major duration, so the mode margin
  saturates and the relative minor wins (12/12 synthetic loops). If >=75% of the
  sounding time is some major key's {IV, V, vi} and the minor reading has no
  authentic V(major)->i(minor) cadence, prefer the major.
* `decide` — replace the CNN key with the chord key only when the chord key explains
  the chord tones clearly better (coverage gap >= 3 pp) and the chord evidence itself
  is strong (coverage >= 85%, |margin| >= 0.5).
"""
from __future__ import annotations

NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
MAJOR_STEPS = {0, 2, 4, 5, 7, 9, 11}
MINOR_STEPS = {0, 2, 3, 5, 7, 8, 10}


def key_pc_mode(label):
    if not label:
        return None
    body = label.strip()
    mode = "minor" if body.endswith("m") and not body.endswith("maj") else "major"
    root = body[:-1] if mode == "minor" else body
    if not root or root[0] not in NOTE_PC:
        return None
    pc = NOTE_PC[root[0]]
    for ch in root[1:]:
        pc += 1 if ch == "#" else -1
    return pc % 12, mode


def _pc_weights(chord):
    """Pitch classes with the same weighting the research analysis used (chord_pcs)."""
    from key_from_chords import chord_pcs
    try:
        return [(int(pc) % 12, float(weight)) for pc, weight in chord_pcs(chord)]
    except Exception:
        if getattr(chord, "root", None) is None:
            return []
        return [(chord.root % 12, 1.0)]


def scale_of(label):
    parsed = key_pc_mode(label)
    if not parsed:
        return None
    pc, mode = parsed
    steps = MAJOR_STEPS if mode == "major" else MINOR_STEPS
    return {(pc + step) % 12 for step in steps}


def diatonic_coverage(chords, key_label) -> float:
    scale = scale_of(key_label)
    if not scale:
        return 0.0
    inside = total = 0.0
    for chord in chords:
        for pc, weight in _pc_weights(chord):
            total += weight * chord.dur
            if pc in scale:
                inside += weight * chord.dur
    return 100.0 * inside / total if total else 0.0


def _inventory(chords):
    total = sum(c.dur for c in chords) or 1.0
    shares = {}
    for chord in chords:
        if chord.root is None:
            continue
        minor = "min" in chord.label
        key = (chord.root % 12, minor)
        shares[key] = shares.get(key, 0.0) + chord.dur / total
    return shares


def _authentic_cadences(chords, minor_tonic: int) -> int:
    count = 0
    for prev, cur in zip(chords, chords[1:]):
        if prev.root is None or cur.root is None:
            continue
        if ((prev.root - cur.root) % 12 == 7 and cur.root % 12 == minor_tonic
                and "min" in cur.label and "min" not in prev.label):
            count += 1
    return count


def guard_456(chords, predicted_key: str, min_coverage: float = 0.75):
    shares = _inventory(chords)
    if not shares:
        return predicted_key, dict(applied=False, reason="empty inventory", coverage=0.0)
    best_tonic, best_cover = None, 0.0
    for tonic in range(12):
        template = {(tonic + 5) % 12: False, (tonic + 7) % 12: False, (tonic + 9) % 12: True}
        covered = sum(share for (root, minor), share in shares.items()
                      if root in template and template[root] == minor)
        if covered > best_cover:
            best_tonic, best_cover = tonic, covered
    if best_cover < min_coverage:
        return predicted_key, dict(applied=False, reason="not a 4-5-6 idiom", coverage=round(best_cover, 3))
    minor_tonic = (best_tonic + 9) % 12
    cadences = _authentic_cadences(chords, minor_tonic)
    if cadences:
        return predicted_key, dict(applied=False, reason=f"{cadences} authentic V-i cadence(s)",
                                   coverage=round(best_cover, 3))
    from key_from_chords import key_name
    return key_name(best_tonic, "major"), dict(applied=True, reason="4-5-6 idiom without V-i",
                                               coverage=round(best_cover, 3))


def decide(cnn_key: str, chord_key: str, margin: float, chords,
           min_gap: float = 3.0, min_coverage: float = 90.0, min_margin: float = 0.5):
    """Audio key vs chord key, with the two extra layers (degree/idiom) as evidence."""
    from key_layers import score as layer_score, _pc_mode
    coverage_cnn = diatonic_coverage(chords, cnn_key)
    coverage_chord = diatonic_coverage(chords, chord_key)
    chord_layers = layer_score(chords, *_pc_mode(chord_key))
    cnn_layers = layer_score(chords, *_pc_mode(cnn_key))

    coverage_ok = (chord_key != cnn_key and coverage_chord - coverage_cnn >= min_gap
                   and coverage_chord >= min_coverage and abs(margin) >= min_margin)
    # the layers can veto an override that needs too much explaining, and can force
    # one when the audio key leaves many chords unexplained while the chord key does not
    veto = chord_layers["penalty"] <= -0.25
    force = (chord_key != cnn_key and cnn_layers["penalty"] <= -0.25
             and chord_layers["penalty"] > cnn_layers["penalty"] + 0.10)
    use_chord = (coverage_ok and not veto) or force
    if use_chord and coverage_ok and not veto:
        reason = "chord evidence explains the chords better"
    elif use_chord and force:
        reason = "audio key leaves out-of-key material the chord key explains"
    elif coverage_ok and veto:
        reason = "chord key would need too much explaining; keep the audio key"
    else:
        reason = "keep the audio key (evidence not decisive)"
    return dict(use_chord=use_chord, cnn_key=cnn_key, chord_key=chord_key,
                margin=margin, coverage_cnn=round(coverage_cnn, 2),
                coverage_chord=round(coverage_chord, 2),
                chord_layers=chord_layers, cnn_layers=cnn_layers,
                reason=reason)
