"""Score the seven analyses against the verified bar map.

Protocol (from the brief): sample each condition's timeline every 0.25 s and compare
with the bar-map truth on the same grid.  exact = same root and quality,
root = same root only.

The app publishes five timeline views of the same analysis:
    retained10          base chord timeline, full coverage
    retained10_stable   stabilised timeline (the app's published/key evidence source)
    retained10_span     bar-pooled
    retained10_full     complete-segment decode == default_source (what the app shows)
    raw_madmom          unsmoothed
Only retained10_full is the default; the others are switchable.  retained10_span /
retained10_full can cover less than the whole file (they refuse partial edge
segments), so a missing row is "no published chord", not a chord decision.  We
therefore report, for every view: exact/root over all in-bar frames, coverage %, and
the split between uncovered frames and genuinely wrong chord labels.
"""
from __future__ import annotations

import json
from pathlib import Path

from cd_common import BAR, RESULTS, STEP, TRUTH_BARS, WORK, key_analysis

VIEWS = ["retained10", "retained10_stable", "retained10_span", "retained10_full",
         "raw_madmom"]
CONDS = [
    ("01_mix", "full mix", True),
    ("02_truth_harm", "truth harmonic = piano+bass", True),
    ("03_sep_harm", "separated harmonic = SW piano+bass", True),
    ("04_sep_harm4", "separated 4-part harmonic = SW p+b+g+other", True),
    ("05_mix_repeat", "full mix, identical repeat run", False),
    ("06_sep_harm_natural", "control: separated p+b at natural level", False),
    ("07_truth_harm4", "control: truth p+b+guitar", False),
]
N_FRAMES = int(len(TRUTH_BARS) * BAR / STEP + 1e-9)   # 146.28 -> 146 full frames; use 147? see below


def canon(label: str):
    if not label or label == "N":
        return ("N", "N")
    if ":" in label:
        root, quality = label.split(":", 1)
        return (root.strip(), "min" if quality.strip() == "min" else "maj")
    root = label.strip()
    if root.endswith("m") and len(root) > 1:
        return (root[:-1], "min")
    return (root, "maj")


def truth_at(index: int):
    return canon(TRUTH_BARS[index])


def frames():
    out = []
    i = 0
    while i * STEP < len(TRUTH_BARS) * BAR - 1e-9:
        out.append(round(i * STEP, 4))
        i += 1
    return out


def label_at(rows, t):
    for row in rows:
        if row["start"] <= t < row["end"]:
            return row["label"]
    return None


def score_view(rows, times):
    exact = root = covered = 0
    uncovered = wrong = 0
    confusions = {}
    for t in times:
        idx = int(t / BAR + 1e-9)
        tr = truth_at(idx)
        label = label_at(rows, t)
        if label is None:
            uncovered += 1
            confusions[f"{TRUTH_BARS[idx]}->(no output)"] = \
                confusions.get(f"{TRUTH_BARS[idx]}->(no output)", 0) + 1
            continue
        covered += 1
        if canon(label) == tr:
            exact += 1
        else:
            wrong += 1
            key = f"{TRUTH_BARS[idx]}->{label}"
            confusions[key] = confusions.get(key, 0) + 1
        if canon(label)[0] == tr[0]:
            root += 1
    n = len(times)
    return {
        "frames": n,
        "exact_pct": round(100 * exact / n, 2),
        "root_pct": round(100 * root / n, 2),
        "exact_pct_covered": round(100 * exact / covered, 2) if covered else None,
        "covered_pct": round(100 * covered / n, 2),
        "uncovered_frames": uncovered,
        "wrong_frames": wrong,
        "confusions": dict(sorted(confusions.items(), key=lambda kv: -kv[1])),
    }


def main() -> int:
    times = frames()
    print(f"{len(times)} frames of {STEP}s, last frame {times[-1]:.2f}s, "
          f"bar = {BAR:.4f}s")
    payload = {"bar_seconds": BAR, "step": STEP, "truth_bars": TRUTH_BARS,
               "frames": len(times), "conditions": {}, "noise": {}, "keys": {},
               "grid_quality": {}}

    for name, desc, _primary in CONDS:
        path = RESULTS / f"{name}.json"
        if not path.is_file():
            print(f"!! {name}: no result file")
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        views = {s["id"]: s["rows"] for s in result.get("sources", [])}
        entry = {"desc": desc, "views": {}}
        for view in VIEWS:
            if view in views:
                entry["views"][view] = score_view(views[view], times)
        key_info = key_analysis(result)
        cands = key_info.get("candidates") or []
        cmaj = next((c for c in cands if c.get("key") == "C"), None)
        entry["key"] = {
            "top_key": cands[0]["key"] if cands else None,
            "top_score": round(float(cands[0]["score"]), 4) if cands else None,
            "C_major_score": round(float(cmaj["score"]), 4) if cmaj else 0.0,
            "C_major_rank": (cands.index(cmaj) + 1) if cmaj else None,
            "runner_up": (cands[1]["key"], round(float(cands[1]["score"]), 4))
            if len(cands) > 1 else None,
        }
        grid = (result.get("structure") or {}).get("grid_quality") or {}
        bars = result.get("bars") or []
        entry["grid_quality"] = {
            "bar_count": grid.get("bar_count"),
            "first_bar": (bars[0]["start"], bars[0]["end"]) if bars else None,
            "leading_ungridded_seconds": grid.get("leading_ungridded_seconds"),
        }
        payload["conditions"][name] = entry
        d = entry["views"].get("retained10_full")
        st = entry["views"].get("retained10_stable")
        print(f"{name:20s} default exact {d['exact_pct']:5.2f} (cov {d['covered_pct']:5.1f}, "
              f"wrong {d['wrong_frames']}) | stable exact {st['exact_pct']:5.2f} "
              f"(wrong {st['wrong_frames']}) | key {entry['key']['top_key']} "
              f"{entry['key']['top_score']}")

    # noise: the two identical mix runs
    noise = {}
    for view in ("retained10_full", "retained10_stable", "raw_madmom"):
        a = payload["conditions"]["01_mix"]["views"].get(view)
        b = payload["conditions"]["05_mix_repeat"]["views"].get(view)
        if a and b:
            noise[view] = {
                "exact_run1": a["exact_pct"], "exact_run2": b["exact_pct"],
                "delta_pp": round(abs(a["exact_pct"] - b["exact_pct"]), 2),
                "confusions_run1": a["confusions"],
                "confusions_run2": b["confusions"],
            }
    payload["noise"] = noise
    print("noise:", json.dumps(noise, indent=1)[:400])
    (WORK / "score.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", WORK / "score.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
