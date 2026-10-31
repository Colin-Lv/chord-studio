"""Fixed root-only output protection; no candidate reselection or references."""
import mir_eval

from evaluate_chords import TimedChord, harte_chord


GUARD_SCHEMA = "raw_root_preserving_candidate_guard_v1"
GUARD_RULE = ("Keep the already-selected full-head label only when its root equals raw; "
              "otherwise return raw's entire label. Raw N/X remain unchanged and "
              "chord-to-N/X changes are rejected. Same-root triad changes are allowed. "
              "No other candidate is reconsidered.")


def preserve_raw_root(raw, selected):
    if len(raw) != len(selected):
        raise ValueError("Root guard requires one prediction per original raw interval")
    output = []
    reverted, retained = 0, 0
    for original, prediction in zip(raw, selected):
        if abs(original.start - prediction.start) > 5.1e-10 or abs(original.end - prediction.end) > 5.1e-10:
            raise ValueError("Root guard cannot align or retime mismatched intervals")
        baseline, candidate = harte_chord(original.chord), harte_chord(prediction.chord)
        allowed = (baseline not in {"N", "X"} and candidate not in {"N", "X"}
                   and mir_eval.chord.root([baseline], [candidate])[0] == 1)
        label = prediction.chord if allowed else original.chord
        reverted += int(not allowed and prediction.chord != original.chord)
        retained += int(allowed and prediction.chord != original.chord)
        output.append(TimedChord(original.start, original.end, label))
    return output, dict(schema=GUARD_SCHEMA, rule=GUARD_RULE, reverted_intervals=reverted,
                        retained_label_changes=retained, interval_count=len(raw),
                        original_interval_support_preserved=True)
