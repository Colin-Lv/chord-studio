"""Ablation on the recorded 147-case results: what if `decide()` could not force an override?

`decide` has two exits: coverage_ok and not veto, or `force` (which bypasses coverage/margin).
KEY-005 records that the force branch can fire while the veto is also true. This script
recomputes the published key with force disabled, using only the evidence already recorded
per case, and reports the delta against the acoustic key and the current published key.

Input domain: production chords (retained10_stable) -> the same decision logic the shipped
worker calls. The frozen raw.lab replay (110/147) is a different input domain and is NOT mixed
in here.
"""
from __future__ import annotations

import json
from pathlib import Path

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


def main():
    rows = []
    for path in sorted((HERE / "cases").glob("*/result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("reference_id") and data["stable"].get("final_key"):
            rows.append(data)
    n = len(rows)
    tab = dict(cnn=0, published=0, force_off=0)
    by_reason = {}
    for r in rows:
        ref = r["reference_id"]
        d = r["stable"].get("decision") or {}
        cnn = d.get("cnn_key")
        chord = d.get("chord_key")
        used = bool(d.get("use_chord"))
        pub = r["stable"].get("final_key")
        cov_c, cov_k = float(d.get("coverage_cnn") or 0), float(d.get("coverage_chord") or 0)
        margin = float(d.get("margin") or 0)
        veto = float((d.get("chord_layers") or {}).get("penalty") or 0) <= -0.25
        coverage_ok = (chord != cnn and cov_k - cov_c >= 3.0 and cov_k >= 90.0
                       and abs(margin) >= 0.5)
        off = chord if (coverage_ok and not veto) else cnn
        tab["cnn"] += key_to_id(cnn) == list(ref)
        tab["published"] += key_to_id(pub) == list(ref)
        tab["force_off"] += key_to_id(off) == list(ref)
        if used:
            why = "force" if str(d.get("reason", "")).startswith("audio key leaves") else "coverage"
            b = by_reason.setdefault(why, dict(fired=0, ok=0))
            b["fired"] += 1
            b["ok"] += key_to_id(pub) == list(ref)
    print(f"cases {n}")
    for k in ("cnn", "published", "force_off"):
        print(f"  {k:10s} {tab[k]:3d}/{n} = {tab[k] / n:.3f}")
    print(f"  delta published -> force_off: {tab['force_off'] - tab['published']:+d}")
    print("overrides by exit:")
    for why, b in by_reason.items():
        print(f"  {why:9s} fired {b['fired']:2d}, published key correct in {b['ok']}/{b['fired']}")
    out = dict(cases=n, **tab, by_reason=by_reason)
    (HERE / "force_ablation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
