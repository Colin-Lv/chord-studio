from functools import lru_cache
import re
import mir_eval

@lru_cache(maxsize=4096)
def harte_chord(label: str) -> str:
    """Normalize project display labels for the official mir_eval metrics."""
    label = str(label).strip().replace("♭", "b").replace("♯", "#")
    if not label or label.upper() == "N":
        return "N"
    if label.upper() == "X":
        return "X"
    body, slash, bass = label.partition("/")
    match = re.fullmatch(r"([A-G][b#]*)(.*)", body)
    if not match:
        return "X"
    root, suffix = match.groups()
    aliases = {"": "maj", "m": "min", "m7": "min7", "m6": "min6", "6": "maj6",
               "m9": "min9", "m11": "min11", "m13": "min13", "m7b5": "hdim7",
               "mMaj7": "minmaj7", "mmaj7": "minmaj7", "add9": "maj(9)",
               "madd9": "min(9)", "sus": "sus4", "7sus4": "sus4(b7)",
               "sus47": "sus4(b7)", "9sus4": "sus4(b7,9)", "sus49": "sus4(b7,9)",
               "7sus2": "sus2(b7)", "maj11": "maj9(11)"}
    quality = suffix[1:] if suffix.startswith(":") else aliases.get(suffix, suffix)
    if slash and re.fullmatch(r"[A-G][b#]*", bass):
        interval = (mir_eval.chord.pitch_class_to_semitone(bass)
                    - mir_eval.chord.pitch_class_to_semitone(root)) % 12
        bass = ("1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7")[interval]
    normalized = root + ":" + quality + ("/" + bass if slash else "")
    try:
        mir_eval.chord.encode(normalized)
    except (mir_eval.chord.InvalidChordException, KeyError):
        return "X"
    return normalized

def chord_attributes(label):
    normalized = harte_chord(label)
    if normalized in {"N", "X"}:
        return None
    root, tones, _ = mir_eval.chord.encode(normalized)
    # This head deliberately has no authority over suspended/diminished chords.
    if not tones[0] or not tones[7] or bool(tones[3]) == bool(tones[4]):
        return None
    if tones[10] and tones[11]:
        return None
    return int(root), int(bool(tones[3])), 1 if tones[10] else 2 if tones[11] else 0
