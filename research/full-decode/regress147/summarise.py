"""Summarise the 147-case regression: acoustic key, stable-key, span-key, and segment churn.

Every case was run through the production modules (inference -> key -> beats -> stabilize ->
span -> key_from_chords/key_decision); the only difference between the two columns is the
chord source used for the harmonic key evidence.
"""
from __future__ import annotations

import json
from collections import Counter
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
        rows.append(data)
    total_cases = len(json.loads((HERE / "cases.json").read_text(encoding="utf-8")))
    print(f"cases expected {total_cases}, results present {len(rows)}")

    scope = [r for r in rows if r.get("reference_id")]
    print(f"usable (reference present) {len(scope)}")
    print("reference modes:", dict(Counter(r["reference_id"][1] for r in scope)))

    def hit(k, ref):
        return bool(k) and list(k) == list(ref)

    madmom = sum(hit(r.get("madmom_key_id"), r["reference_id"]) for r in scope)
    stable = sum(hit(key_to_id(r["stable"].get("final_key")), r["reference_id"]) for r in scope)
    span = sum(hit(key_to_id(r["span"].get("final_key")), r["reference_id"]) for r in scope)
    n = len(scope)
    print(f"\nacoustic (madmom CNN) key     {madmom:3d}/{n} = {madmom / n:.3f}")
    print(f"final key @ retained10_stable {stable:3d}/{n} = {stable / n:.3f}")
    print(f"final key @ retained10_span   {span:3d}/{n} = {span / n:.3f}")

    fixed, broke, switched = [], [], []
    for r in scope:
        s_ok = hit(key_to_id(r["stable"].get("final_key")), r["reference_id"])
        p_ok = hit(key_to_id(r["span"].get("final_key")), r["reference_id"])
        s_key = key_to_id(r["stable"].get("final_key"))
        p_key = key_to_id(r["span"].get("final_key"))
        if (s_key is None) != (p_key is None) or (s_key and p_key and s_key != p_key):
            switched.append(r["case"])
        if not s_ok and p_ok:
            fixed.append(r["case"])
        if s_ok and not p_ok:
            broke.append((r["case"], r["stable"].get("final_key"), r["span"].get("final_key"),
                          r["reference"]))
    print(f"\nstable -> span key changed in {len(switched)} cases")
    print(f"fixed  (stable wrong, span right): {len(fixed)} {fixed}")
    print(f"broke  (stable right, span wrong): {len(broke)} {[b[0] for b in broke]}")
    for case, sk, pk, ref in broke:
        print(f"   {case}: stable {sk} -> span {pk} (ref {ref})")

    seg = [(r["stable"]["rows"], r["span"]["rows"]) for r in rows]
    ratios = sorted(b / a for a, b in seg if a)
    print(f"\nsegments stable mean {sum(a for a, _ in seg) / len(seg):.1f} -> "
          f"span mean {sum(b for _, b in seg) / len(seg):.1f}  "
          f"median ratio {ratios[len(ratios) // 2]:.3f}  "
          f"min {ratios[0]:.3f} max {ratios[-1]:.3f}")
    degenerate = [(r["case"], r["stable"]["rows"], r["span"]["rows"]) for r in rows
                  if r["span"]["rows"] <= 2]
    print(f"span collapsed to <=2 segments: {len(degenerate)} {degenerate[:6]}")

    # Post-fix behaviour: the worker now takes the harmonic key evidence from the frozen
    # stabilised timeline, so the published key no longer depends on the display source.
    import verify_key_pinning as V
    pinned = 0
    equal = 0
    for r in scope:
        stable_lab = HERE / "cases" / r["case"] / "run" / "retained10_stable.lab"
        if not stable_lab.is_file():
            continue
        key = V.key_from_lab(stable_lab, r.get("madmom_key"))
        equal += int(key == r["stable"].get("final_key"))
        pinned += int(bool(key) and key_to_id(key) == r["reference_id"])
    print(f"\npinned key (evidence from retained10_stable): {pinned}/{n} = {pinned / n:.3f}")
    print(f"  recomputation equals the recorded stable key in {equal}/{n} cases")

    verdict = "PASS" if pinned >= stable else "FAIL"
    print(f"\nno-degradation check after pinning: {pinned} vs stable {stable} -> {verdict}")
    print(f"  (before pinning, key from the display source: span {span} vs stable {stable}, "
          f"{len(fixed)} fixed / {len(broke)} broke)")
    out = dict(cases=len(rows), scope=n, madmom=madmom, stable=stable, span=span,
               pinned=pinned, pinned_equals_stable=equal,
               fixed=fixed, broke=[b[0] for b in broke], switched=len(switched),
               segments_stable_mean=sum(a for a, _ in seg) / len(seg),
               segments_span_mean=sum(b for _, b in seg) / len(seg),
               median_segment_ratio=ratios[len(ratios) // 2],
               degenerate=degenerate, verdict=verdict)
    (HERE / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                      encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
