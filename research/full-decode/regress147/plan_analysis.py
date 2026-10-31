"""Where does the published key lose against the acoustic key? Numbers for planning.

Reads the recorded 147-case results (acoustic key, the decide() output, the reference) and
cross-tabulates: acoustic-only correct, chord-override-only correct, both wrong. Also prints
the coverage/margin distribution of the cases the chord override changed.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def key_id(label):
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


def main():
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((HERE / "cases").glob("*/result.json"))]
    scope = [r for r in rows if r.get("reference_id") and r["stable"].get("final_key")]
    both = acc_only = pub_only = neither = 0
    overridden = []
    for r in scope:
        ref = r["reference_id"]
        acc = r.get("madmom_key_id")
        pub = key_id(r["stable"]["final_key"])
        a_ok = bool(acc) and list(acc) == list(ref)
        p_ok = bool(pub) and list(pub) == list(ref)
        both += a_ok and p_ok
        acc_only += a_ok and not p_ok
        pub_only += p_ok and not a_ok
        neither += not a_ok and not p_ok
        d = r["stable"].get("decision") or {}
        if d.get("use_chord"):
            overridden.append(r)
    print(f"cases {len(scope)}")
    print(f"  both correct        {both}")
    print(f"  acoustic only       {acc_only}   <- chord override made these worse")
    print(f"  chord override only {pub_only}   <- override fixed these")
    print(f"  neither             {neither}")
    print(f"  acoustic  {both + acc_only}/{len(scope)} = {(both + acc_only) / len(scope):.3f}")
    print(f"  published {both + pub_only}/{len(scope)} = {(both + pub_only) / len(scope):.3f}")
    print(f"  override fired in {len(overridden)} cases")

    def stats(name, key):
        vals = [float((r["stable"].get("decision") or {}).get(key) or 0) for r in overridden]
        if vals:
            print(f"  {name:16s} n={len(vals)} median {statistics.median(vals):.1f} "
                  f"min {min(vals):.1f} max {max(vals):.1f}")

    for key in ("coverage_cnn", "coverage_chord", "margin"):
        stats(key, key)
    gaps = [float((r["stable"].get("decision") or {}).get("coverage_chord") or 0)
            - float((r["stable"].get("decision") or {}).get("coverage_cnn") or 0)
            for r in overridden]
    good = [g for g, r in zip(gaps, overridden)
            if key_id(r["stable"]["final_key"]) == r["reference_id"]]
    bad = [g for g, r in zip(gaps, overridden)
           if key_id(r["stable"]["final_key"]) != r["reference_id"]]
    print(f"  override {len(overridden)}: fixed {len(good)} broke {len(bad)}")
    if good:
        print(f"    gap when fixed: median {statistics.median(good):.1f} {sorted(round(g,1) for g in good)}")
    if bad:
        print(f"    gap when broke: median {statistics.median(bad):.1f} {sorted(round(g,1) for g in bad)}")
    reasons = {}
    for r in overridden:
        why = (r["stable"].get("decision") or {}).get("reason", "?")
        reasons[why] = reasons.get(why, 0) + 1
    print("  reasons:", reasons)


if __name__ == "__main__":
    main()
