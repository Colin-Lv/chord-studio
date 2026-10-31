"""Verify the key-pinning fix: with the worker pinned to the frozen stabilised timeline, the
published key must equal the key the 'stable' column already measured, for all 147 cases.

Recomputes the worker's key path from the on-disk stable timeline of every case (no re-run of
the acoustic frontend), so the fix is verified from artifacts rather than assumed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend")
sys.path.insert(0, str(BACKEND / "app" / "web"))
sys.path.insert(0, str(BACKEND / "app"))

HERE = Path(__file__).resolve().parent
NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def key_to_id(label):
    if not label:
        return None
    body = str(label).strip()
    mode = "minor" if body.lower().endswith("minor") or (
        body.endswith("m") and not body.endswith("maj")) else "major"
    root = body.replace(" Major", "").replace(" minor", "").replace(" major", "").strip()
    if root.endswith("m"):
        root = root[:-1]
    if not root or root[0] not in NOTE_PC:
        return None
    pc = NOTE_PC[root[0]]
    for ch in root[1:]:
        pc += 1 if ch == "#" else -1
    return [pc % 12, mode]


def key_from_lab(path, acoustic_key):
    from key_from_chords import infer as infer_key, key_name, parse_rows
    from key_decision import decide, guard_456
    from source_rows import lab_rows

    if acoustic_key is None:
        return None
    rows = lab_rows(path)
    parsed, _ = parse_rows([dict(start=r["start"], end=r["end"], label=r["label"]) for r in rows])
    if len(parsed) < 3:
        return acoustic_key
    ev = infer_key(parsed)
    pc, mode = ev["pred"]
    raw = key_name(pc, mode)
    guarded, _ = guard_456(parsed, raw)
    d = decide(acoustic_key, guarded, float(ev["pred_pair"]["margin"]), parsed)
    return d["chord_key"] if d["use_chord"] else acoustic_key


def main():
    same = mismatch = 0
    hits = 0
    scope = 0
    rows = []
    for result_path in sorted((HERE / "cases").glob("*/result.json")):
        data = json.loads(result_path.read_text(encoding="utf-8"))
        case = data["case"]
        stable_lab = HERE / "cases" / case / "run" / "retained10_stable.lab"
        if not stable_lab.is_file() or not data.get("reference_id"):
            continue
        acoustic = data.get("madmom_key")
        recomputed = key_from_lab(stable_lab, acoustic)
        recorded = data["stable"].get("final_key")
        scope += 1
        if recomputed == recorded:
            same += 1
        else:
            mismatch += 1
            rows.append((case, recorded, recomputed))
        if key_to_id(recomputed) == data["reference_id"]:
            hits += 1
    print(f"cases recomputed {scope}")
    print(f"recomputed key == recorded stable key: {same}/{scope}  (mismatch {mismatch})")
    for case, rec, got in rows[:10]:
        print(f"   {case}: recorded {rec} recomputed {got}")
    print(f"pinned-key accuracy {hits}/{scope} = {hits / scope:.3f}")
    out = dict(cases=scope, equal=same, mismatch=mismatch, pinned_hits=hits,
               pinned_accuracy=hits / scope if scope else 0.0)
    (HERE / "verify_key_pinning.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0 if mismatch == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
