#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""key_from_chords.py -- harmonic global-key inference from a chord sequence.

Symbolic, CPU-only, no third-party dependencies.

  python key_from_chords.py --eval                 # POP909 (909 songs) + 3 songs
  python key_from_chords.py --eval --limit 120     # reproducible sample
  python key_from_chords.py --eval --source raw    # richer archive labels
  python key_from_chords.py --song <analysis.json>
  python key_from_chords.py --all-songs

Model
-----
Stage 1 -- relative PAIR selection (12 pairs: {X major, X-3 minor}), because
           relative keys share a diatonic set and are cheap to confuse.
           Pair score = max over the two members of a z-scored fusion of
             * vote_ks      Krumhansl-Kessler profile on chord-tone PC vector
             * hf           harmonic-function degree template (I/IV/V core)
             * pcset_deg    degree -> expected pitch-class-set profile
             * vote_temp    Temperley profile on the same PC vector
Stage 2 -- MODE adjudication inside the chosen pair, an explicit additive
           margin:  m = 1.0 * (z_ks[major] - z_ks[minor]) + 2.5 * tonic_dur
           m > 0 -> major.  `tonic_dur` is the duration share of major triads
           on the major tonic plus major triads on the relative-minor root,
           minus the minor-triad counterparts (i.e. "which of the two tonic
           chords does the piece actually sit on").
           Cadence / first / last / bass cues are implemented and reported as
           evidence, but the POP909 ablation shows they *hurt* (see REPORT.md).

The final 24-key score is  pair_z[p] +/- m_p/2, so every pair is split by its
own mode margin while the pair ordering (stage 1) is preserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

# --------------------------------------------------------------------------
# 0. paths / constants
# --------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
POP909_LAB_ROOT = r"D:\ChordStudio\Dialogue\benchmarks\imported\pop909"
POP909_RAW_ROOT = r"D:\ChordStudio\_archive\2026-10-30\benchmarks\raw\pop909\POP909"
MANIFEST = r"D:\ChordStudio\Dialogue\benchmarks\manifest.pop909_synth.csv"
THREE_SONGS = [
    r"D:\ChordStudio\data\projects\aa5a41797bb343e590abcf4d8b27ae54\runs\11fa80454ad94da3a81efa19c1cab64c\analysis.json",  # 虚拟
    r"D:\ChordStudio\data\projects\0bc0304e01ed40d8baac6b76575b25b6\runs\82f77b7a39764ae2a7e2111db5443c39\analysis.json",  # 坏女孩
    r"D:\ChordStudio\data\projects\e18b69d0edef4c5e91f6594520a816cb\runs\e14e27daf9d1408fa782bf601073fcff\analysis.json",  # 沧海一粟
]

SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
NAME2PC = {}
for _i, _n in enumerate(SHARP):
    NAME2PC[_n] = _i
for _i, _n in enumerate(FLAT):
    NAME2PC[_n] = _i
NAME2PC.update({"B#": 0, "Cb": 11, "E#": 5, "Fb": 4})

# Krumhansl-Kessler (1982) probe-tone profiles, tonic-normalised.
KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# Temperley (2007) "modified" profiles: the minor table is the major table with
# the mode-defining degrees swapped, as in the published version.
TEMP_MAJOR = [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]
TEMP_MINOR = [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]

# Harmonic-function templates: chord ROOT (semitones above tonic) -> (weight,
# expected triad kind).  I/IV/V are the core, vi/ii/iii decrease, borrowed
# degrees get small weights.  Popular minor leans on bVII, so it is not
# harmonic-minor-pure.
MAJOR_DEGREES = {
    0: (1.00, "maj"),   # I
    7: (0.92, "maj"),   # V
    5: (0.80, "maj"),   # IV
    9: (0.62, "min"),   # vi
    2: (0.55, "min"),   # ii
    4: (0.38, "min"),   # iii
    11: (0.28, "dim"),  # vii
    10: (0.16, "maj"),  # bVII
    3: (0.12, "maj"),   # bIII
    8: (0.10, "maj"),   # bVI
    1: (0.04, "maj"),   # bII
    6: (0.04, "dim"),   # #IV
}
MINOR_DEGREES = {
    0: (1.00, "min"),   # i
    7: (0.90, "maj"),   # V (harmonic); v is also common
    10: (0.62, "maj"),  # VII
    3: (0.60, "maj"),   # III
    8: (0.58, "maj"),   # VI
    5: (0.52, "min"),   # iv
    2: (0.30, "dim"),   # ii
    11: (0.32, "dim"),  # vii
    9: (0.08, "maj"),   # VI (melodic)
    4: (0.06, "maj"),   # #III
    1: (0.04, "maj"),   # bII
    6: (0.04, "dim"),   # #iv
}

# chord-tone weighting for the PC histogram
W_ROOT, W_THIRD, W_FIFTH, W_SEVENTH, W_COLOR, W_BASS = 1.00, 0.85, 0.55, 0.35, 0.20, 0.25

# stage-1 fusion (z-score weights).  Selected on a weight grid; the surface is
# flat (+/-0.005 exact over a wide neighbourhood).
FUSION_WEIGHTS = {
    "vote_ks": 1.0,
    "hf": 1.4,
    "pcset_deg": 0.3,
    "vote_temp": 0.2,
    "vote_ks_root": 0.0,
}

# stage-2 mode margin weights.  `ks_lean` is the profile lean between the two
# relative keys; `tonic_dur` is the tonic-chord duration share difference.
MODE_WEIGHTS = {
    "ks_lean": 1.0,
    "tonic_dur": 2.5,
    # the cues below are computed + reported but zero-weighted by default
    # (see REPORT.md, ablation: each one costs accuracy on POP909)
    "cadence": 0.0,
    "cadence_strict": 0.0,
    "final": 0.0,
    "final_strict": 0.0,
    "first": 0.0,
    "dominant_res": 0.0,
    "bass": 0.0,
}


# --------------------------------------------------------------------------
# 1. chord parsing
# --------------------------------------------------------------------------

# suffix -> (third semitones, fifth semitones, seventh semitones, triad kind)
_QUAL_TABLE = {
    "": (4, 7, None, "maj"),
    "maj": (4, 7, None, "maj"),
    "maj7": (4, 7, 11, "maj"),
    "maj9": (4, 7, 11, "maj"),
    "maj11": (4, 7, 11, "maj"),
    "maj13": (4, 7, 11, "maj"),
    "maj6": (4, 7, 9, "maj"),
    "6": (4, 7, 9, "maj"),
    "7": (4, 7, 10, "dom"),
    "9": (4, 7, 10, "dom"),
    "11": (4, 7, 10, "dom"),
    "13": (4, 7, 10, "dom"),
    "m": (3, 7, None, "min"),
    "min": (3, 7, None, "min"),
    "m7": (3, 7, 10, "min"),
    "min7": (3, 7, 10, "min"),
    "m9": (3, 7, 10, "min"),
    "m11": (3, 7, 10, "min"),
    "m6": (3, 7, 9, "min"),
    "min6": (3, 7, 9, "min"),
    "mmaj7": (3, 7, 11, "min"),
    "minmaj7": (3, 7, 11, "min"),
    "mmaj9": (3, 7, 11, "min"),
    "m7b5": (3, 6, 10, "dim"),
    "hdim7": (3, 6, 10, "dim"),
    "dim": (3, 6, None, "dim"),
    "dim7": (3, 6, 9, "dim"),
    "aug": (4, 8, None, "aug"),
    "5": (None, 7, None, "power"),
    "sus2": (None, 7, None, "sus"),
    "sus4": (None, 7, None, "sus"),
    "sus": (None, 7, None, "sus"),
    "7sus4": (None, 7, 10, "sus"),
}
_QUAL_KEYS_BY_LEN = sorted(_QUAL_TABLE, key=len, reverse=True)

_BASS_OFFSET = {"3": 4, "b3": 3, "#3": 5, "5": 7, "b5": 6, "#5": 8,
                "7": 11, "b7": 10, "9": 2, "b9": 1, "#9": 3,
                "4": 5, "2": 2, "6": 9, "b6": 8}

_EXT_OFFSET = {"2": 2, "4": 5, "6": 9, "7": 10, "b7": 10, "9": 2,
               "b9": 1, "#9": 3, "11": 5, "13": 9, "b13": 8}


class Chord(object):
    __slots__ = ("start", "end", "label", "root", "kind", "third", "fifth",
                 "seventh", "bass", "colors")

    def __init__(self, start, end, label):
        self.start = start
        self.end = end
        self.label = label
        self.root = None
        self.kind = "other"
        self.third = None
        self.fifth = None
        self.seventh = None
        self.bass = None
        self.colors = []

    @property
    def dur(self):
        d = self.end - self.start
        return d if d > 0 else 0.0

    def __repr__(self):
        return "Chord(%.2f,%.2f,%s)" % (self.start, self.end, self.label)


def parse_chord_label(label):
    """Parse one chord label.  Returns None for N / empty / unparseable."""
    if label is None:
        return None
    s = str(label).strip()
    if not s or s.upper() == "N":
        return None
    s = s.replace("\u266f", "#").replace("\u266d", "b").replace(" ", "")
    parts = s.split("/", 1)
    head = parts[0]
    bass_s = parts[1] if len(parts) > 1 else None

    m = re.match(r"^([A-G][#b]?)", head)
    if not m:
        return None
    root = NAME2PC.get(m.group(1))
    if root is None:
        return None
    rest = head[m.end():]

    # parenthesised extensions: Cmaj(9) -> Cmaj, extension 9 as a colour tone
    ext_tokens = []
    for grp in re.findall(r"\(([^)]*)\)", rest):
        ext_tokens.extend([t.strip() for t in grp.split(",") if t.strip()])
    rest = re.sub(r"\([^)]*\)", "", rest)

    if ":" in rest:                       # raw archive format  B:maj
        rest = rest.split(":", 1)[1]
    rest = rest.strip().replace("M", "maj").replace("min", "m")

    qual = rest if rest in _QUAL_TABLE else None
    if qual is None:
        for cand in _QUAL_KEYS_BY_LEN:
            if cand and rest.startswith(cand):
                qual = cand
                break
        if qual is None and rest == "":
            qual = ""
    if qual is None:
        return None
    third, fifth, seventh, kind = _QUAL_TABLE[qual]

    c = Chord(0.0, 0.0, s)
    c.root = root
    c.kind = kind
    c.third = third
    c.fifth = fifth
    c.seventh = seventh
    if kind == "sus":
        c.colors = [(2 if qual.startswith("sus2") else 5, 0.35)]
    for tok in ext_tokens:
        off = _EXT_OFFSET.get(tok)
        if off is not None:
            c.colors.append((off, 0.20))
    if bass_s:
        b = bass_s.strip()
        if b in _BASS_OFFSET:
            c.bass = (root + _BASS_OFFSET[b]) % 12
        elif b in NAME2PC:
            c.bass = NAME2PC[b]
    return c


def parse_rows(rows):
    """rows: iterable of (start, end, label) / dict rows -> (chords, stats)."""
    out = []
    n_tok = n_bad = 0
    for r in rows:
        if isinstance(r, dict):
            start, end, label = float(r["start"]), float(r["end"]), r.get("label", "")
        else:
            start, end, label = float(r[0]), float(r[1]), r[2]
        n_tok += 1
        c = parse_chord_label(label)
        if c is None:
            if str(label).strip().upper() not in ("N", ""):
                n_bad += 1
            continue
        c.start, c.end = start, end
        out.append(c)
    return out, {"tokens": n_tok, "unparsed": n_bad}


# --------------------------------------------------------------------------
# 2. features
# --------------------------------------------------------------------------

def chord_pcs(c):
    """[(pc, weight)] for the sounding pitch classes of a chord."""
    pcs = [(c.root, W_ROOT)]
    if c.third is not None:
        pcs.append(((c.root + c.third) % 12, W_THIRD))
    if c.fifth is not None:
        pcs.append(((c.root + c.fifth) % 12, W_FIFTH))
    if c.seventh is not None:
        pcs.append(((c.root + c.seventh) % 12, W_SEVENTH))
    for off, w in c.colors:
        pcs.append(((c.root + off) % 12, w))
    if c.bass is not None and c.bass != c.root:
        pcs.append((c.bass, W_BASS))
    return pcs


def build_features(chords):
    pc_vec = [0.0] * 12
    root_vec = [0.0] * 12
    total = 0.0
    for c in chords:
        d = c.dur
        if d <= 0:
            continue
        total += d
        root_vec[c.root] += d
        for pc, w in chord_pcs(c):
            pc_vec[pc] += d * w
    units = [(c.root, c.kind, c.dur, c.bass) for c in chords if c.dur > 0]
    return {"chords": chords, "pc": pc_vec, "root": root_vec, "dur": total,
            "units": units}


# --------------------------------------------------------------------------
# 3. scorers (24 scores, index = pc*2 + (0 major / 1 minor))
# --------------------------------------------------------------------------

def key_id(pc, mode):
    return pc * 2 + (0 if mode == "major" else 1)


def id_to_key(k):
    return k // 2, ("major" if k % 2 == 0 else "minor")


def _pearson(a, b):
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def _rot(prof, tonic):
    return [prof[(p - tonic) % 12] for p in range(12)]


def score_vote(vec, major_prof, minor_prof):
    out = [0.0] * 24
    for t in range(12):
        out[key_id(t, "major")] = _pearson(vec, _rot(major_prof, t))
        out[key_id(t, "minor")] = _pearson(vec, _rot(minor_prof, t))
    return out


def score_vote_ks(f):
    return score_vote(f["pc"], KS_MAJOR, KS_MINOR)


def score_vote_temp(f):
    return score_vote(f["pc"], TEMP_MAJOR, TEMP_MINOR)


def score_vote_ks_root(f):
    """Ablation: only chord roots contribute to the PC vector."""
    return score_vote(f["root"], KS_MAJOR, KS_MINOR)


def _triad_pcs(kind):
    if kind == "maj":
        return (0, 4, 7)
    if kind == "min":
        return (0, 3, 7)
    if kind == "dim":
        return (0, 3, 6)
    if kind == "aug":
        return (0, 4, 8)
    return (0, 5, 7)


def _degree_profile(deg_table):
    """Degree template -> expected pitch-class-set profile."""
    prof = [0.0] * 12
    tone_w = {0: 1.0, 3: 0.8, 4: 0.8, 6: 0.6, 7: 0.6, 8: 0.6}
    for rel, (w, kind) in deg_table.items():
        for iv in _triad_pcs(kind):
            prof[(rel + iv) % 12] += w * tone_w.get(iv, 0.6)
    s = sum(prof)
    return [x / s * 12.0 for x in prof] if s else prof


DEG_PROF_MAJOR = _degree_profile(MAJOR_DEGREES)
DEG_PROF_MINOR = _degree_profile(MINOR_DEGREES)


def score_pcset_deg(f):
    return score_vote(f["pc"], DEG_PROF_MAJOR, DEG_PROF_MINOR)


def _kind_factor(observed, expected):
    if observed == expected:
        return 1.0
    if observed in ("sus", "power"):
        return 0.85
    if observed == "dom" and expected in ("maj", "min"):
        return 0.95
    if observed in ("maj", "min") and expected == "dom":
        return 0.9
    if observed in ("maj", "min") and expected in ("maj", "min"):
        return 0.45          # right root, wrong mode -> still root evidence
    if expected == "dim" or observed == "dim":
        return 0.7
    return 0.6


def _hf_one(f, tonic, mode, deg_table):
    total = f["dur"]
    if total <= 0:
        return 0.0
    steps = (0, 2, 4, 5, 7, 9, 11) if mode == "major" else (0, 2, 3, 5, 7, 8, 10)
    scale = set((tonic + iv) % 12 for iv in steps)
    acc = 0.0
    for c in f["chords"]:
        d = c.dur
        if d <= 0:
            continue
        rel = (c.root - tonic) % 12
        w, exp = deg_table.get(rel, (0.02, None))
        kf = _kind_factor(c.kind, exp) if exp else 0.5
        pcs = chord_pcs(c)
        in_scale = sum(1 for pc, _w in pcs if pc in scale) / max(1, len(pcs))
        acc += d * (w * kf + 0.06 * in_scale)
    return acc / total


def score_hf(f):
    out = [0.0] * 24
    for t in range(12):
        out[key_id(t, "major")] = _hf_one(f, t, "major", MAJOR_DEGREES)
        out[key_id(t, "minor")] = _hf_one(f, t, "minor", MINOR_DEGREES)
    return out


COMPUTED_SCORERS = {
    "vote_ks": score_vote_ks,
    "vote_temp": score_vote_temp,
    "vote_ks_root": score_vote_ks_root,
    "pcset_deg": score_pcset_deg,
    "hf": score_hf,
}


# --------------------------------------------------------------------------
# 4. fusion / normalisation
# --------------------------------------------------------------------------

def zscore(scores):
    n = len(scores)
    m = sum(scores) / n
    sd = (sum((x - m) ** 2 for x in scores) / n) ** 0.5
    if sd == 0:
        return [0.0] * n
    return [(x - m) / sd for x in scores]


def fuse(score_dict, weights):
    zs = {k: zscore(v) for k, v in score_dict.items()}
    out = [0.0] * 24
    for k, w in weights.items():
        if not w:
            continue
        for i in range(24):
            out[i] += w * zs[k][i]
    return out


def pair_of_key(k):
    """Relative pair id = pc of the major member."""
    pc, mode = id_to_key(k)
    return pc if mode == "major" else (pc + 3) % 12


# --------------------------------------------------------------------------
# 5. stage 2 -- relative-key mode adjudication
# --------------------------------------------------------------------------

def relative_evidence(units, major_pc):
    """Structural evidence for `major_pc` major vs its relative minor.

    units: [(root, kind, duration, bass_pc|None)]
    Returns a dict with signed cue values (+ = major) and the additive margin
    contributions.  All cues are O(1)-scaled so weights are readable.
    """
    tM = major_pc % 12
    tm = (tM - 3) % 12
    total = sum(u[2] for u in units)
    ev = {"major_pc": tM, "minor_pc": tm, "total_dur": total,
          "cues": {}, "counts": {}, "dur_share": {}, "contrib": {}}
    if total <= 0:
        return ev

    d = defaultdict(float)
    final = first = None
    for root, kind, dur, bass in units:
        if dur <= 0:
            continue
        d[(root, kind)] += dur
        if first is None:
            first = (root, kind)
        final = (root, kind)

    def dur_of(root, kinds):
        return sum(v for (r, k), v in d.items() if r == root and k in kinds)

    d_tonic_maj = dur_of(tM, ("maj", "dom"))
    d_tonic_min = dur_of(tM, ("min",))
    d_relmaj = dur_of(tm, ("maj", "dom"))
    d_relmin = dur_of(tm, ("min",))

    def chord_cue(c):
        if c is None:
            return 0.0
        if c[0] == tM:
            return 1.0 if c[1] in ("maj", "dom") else -0.55
        if c[0] == tm:
            return -0.85 if c[1] == "min" else 0.45
        return 0.0

    def chord_cue_strict(c):
        if c is None:
            return 0.0
        if c[0] == tM:
            return 1.0 if c[1] in ("maj", "dom") else -0.6
        if c[0] == tm:
            return -1.0 if c[1] == "min" else 0.0
        return 0.0

    counts = Counter()
    for (r1, k1, _d1, _b1), (r2, k2, _d2, _b2) in zip(units, units[1:]):
        strong1 = k1 in ("maj", "dom")
        if r1 == (tM + 7) % 12 and r2 == tM and strong1 and k2 in ("maj", "dom"):
            counts["auth_major"] += 1          # V -> I
        if r1 == (tm + 7) % 12 and r2 == tm and strong1 and k2 == "min":
            counts["auth_minor"] += 1          # V -> i
        if r1 == (tM + 7) % 12 and r2 == tm and strong1 and k2 == "min":
            counts["decept_major"] += 1        # V -> vi
        if r1 == (tm + 7) % 12 and r2 == tM and strong1 and k2 in ("maj", "dom"):
            counts["decept_minor"] += 1        # V -> III
        if r1 == (tM + 5) % 12 and r2 == tM and strong1 and k2 in ("maj", "dom"):
            counts["plag_major"] += 1          # IV -> I
        if r1 == (tm + 5) % 12 and r2 == tm and k1 == "min" and k2 == "min":
            counts["plag_minor"] += 1          # iv -> i
        if r1 == (tM + 10) % 12 and r2 == tM and strong1 and k2 in ("maj", "dom"):
            counts["bVII_major"] += 1          # bVII -> I

    n_cad = sum(counts.values())
    cad_signed = (counts["auth_major"] * 1.0 + counts["decept_minor"] * 0.6
                  + counts["plag_major"] * 0.35 + counts["bVII_major"] * 0.25
                  - counts["auth_minor"] * 1.0 - counts["decept_major"] * 0.6
                  - counts["plag_minor"] * 0.35)
    c_cad = cad_signed / max(1, n_cad)
    n_auth = counts["auth_major"] + counts["auth_minor"]
    c_cad_s = ((counts["auth_major"] - counts["auth_minor"]) / n_auth) if n_auth else 0.0
    c_domres = (((counts["auth_major"] - counts["decept_major"]) /
                 (counts["auth_major"] + counts["decept_major"]))
                if (counts["auth_major"] + counts["decept_major"]) else 0.0)

    with_bass = [u for u in units if u[3] is not None and u[2] > 0]
    c_bass = 0.0
    if with_bass:
        tb = sum(u[2] for u in with_bass)
        c_bass = (sum(u[2] for u in with_bass if u[3] == tM) -
                  sum(u[2] for u in with_bass if u[3] == tm)) / tb

    c_td = (d_tonic_maj - d_tonic_min - d_relmin + d_relmaj) / total

    cues = {
        "tonic_dur": c_td,
        "final": chord_cue(final),
        "final_strict": chord_cue_strict(final),
        "first": chord_cue(first),
        "cadence": c_cad,
        "cadence_strict": c_cad_s,
        "dominant_res": c_domres,
        "bass": c_bass,
    }
    ev["cues"] = cues
    ev["counts"] = {k: v for k, v in counts.items() if v}
    ev["dur_share"] = {
        "tonic_major": d_tonic_maj / total,
        "tonic_minor": d_tonic_min / total,
        "rel_major": d_relmaj / total,
        "rel_minor": d_relmin / total,
    }
    ev["final_chord"] = final
    ev["first_chord"] = first
    ev["major_tonic_chords"] = sorted(
        set(r for (r, k), v in d.items() if r == tM and v > 0))
    return ev


def mode_margin(ev, ks_lean, weights=None):
    """Additive, explainable margin (z units).  >0 => major."""
    w = weights or MODE_WEIGHTS
    contrib = {"ks_lean": w.get("ks_lean", 1.0) * ks_lean}
    for k, v in ev["cues"].items():
        ww = w.get(k, 0.0)
        if ww:
            contrib[k] = ww * v
    return sum(contrib.values()), contrib


# --------------------------------------------------------------------------
# 6. inference
# --------------------------------------------------------------------------

def infer(chords):
    f = build_features(chords)
    raw_scores = {}
    for name, fn in COMPUTED_SCORERS.items():
        raw_scores[name] = fn(f)
    zs = {k: zscore(v) for k, v in raw_scores.items()}
    base = fuse(raw_scores, FUSION_WEIGHTS)          # 24-key fused score
    base_z = zscore(base)

    pairs = []
    for p in range(12):
        kmaj = key_id(p, "major")
        kmin = key_id((p - 3) % 12, "minor")
        # Pair salience: each profile family contributes the score of its best
        # member of the pair.  A pair stays in contention when *any* family can
        # explain the piece inside it, which is what makes the pair decision
        # robust to the relative-key ambiguity that hurts a plain 24-way rank.
        pair_z = sum(w * max(zs[k][kmaj], zs[k][kmin])
                     for k, w in FUSION_WEIGHTS.items())
        ev = relative_evidence(f["units"], p)
        ks_lean = zs["vote_ks"][kmaj] - zs["vote_ks"][kmin]
        margin, contrib = mode_margin(ev, ks_lean)
        pairs.append({
            "pair_pc": p, "pair_z": pair_z,
            "kmaj": kmaj, "kmin": kmin,
            "margin": margin, "contrib": contrib,
            "evidence": ev,
            "mode": "major" if margin > 0 else "minor",
        })

    # Order: pair salience first, then the mode each pair's margin selects.
    # (A global pair_z +/- margin/2 scalar lets a low-salience pair with a big
    #  margin jump the queue; that cost 0.7 pt of POP909 accuracy, so the pair
    #  decision is kept as the outer sort key and is not overridden.)
    pairs.sort(key=lambda q: -q["pair_z"])
    cands = []
    for rank, p in enumerate(pairs):
        winner = p["kmaj"] if p["margin"] > 0 else p["kmin"]
        loser = p["kmin"] if p["margin"] > 0 else p["kmaj"]
        for k in (winner, loser):
            cands.append({
                "key_id": k, "pc": id_to_key(k)[0], "mode": id_to_key(k)[1],
                "score": p["pair_z"], "rank": rank,
                "is_pair_winner": k == winner,
                "pair_pc": p["pair_pc"], "pair_z": p["pair_z"],
                "margin": p["margin"], "mode_margin": p["margin"] if k == winner else -p["margin"],
            })
    pred = (cands[0]["pc"], cands[0]["mode"])
    best_pair = pairs[0]
    return {
        "features": f,
        "scores_raw": raw_scores,
        "scores_z": zs,
        "stage1": base_z,
        "pairs": pairs,
        "candidates": cands,
        "pred": pred,
        "pred_pair": best_pair,
    }


# --------------------------------------------------------------------------
# 7. data loading
# --------------------------------------------------------------------------

def load_lab(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 3:
                rows.append((float(p[0]), float(p[1]), p[2]))
    return rows


def load_raw(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            p = line.rstrip("\n").rstrip("\r").split("\t")
            if len(p) >= 3:
                rows.append((float(p[0]), float(p[1]), p[2]))
    return rows


def load_manifest():
    with open(MANIFEST, "r", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        k = r["key"].strip()
        mode = "minor" if k.endswith("m") else "major"
        pc = NAME2PC.get(k[:-1] if k.endswith("m") else k)
        if pc is None:
            continue
        out.append({"id": r["id"], "num": r["id"].split("_")[1], "key": k,
                    "pc": pc, "mode": mode})
    return out


# --------------------------------------------------------------------------
# 8. metrics
# --------------------------------------------------------------------------

def is_relative(pred, gt):
    (p, pm), (g, gm) = pred, gt
    return pm != gm and abs((p - g) % 12) in (3, 9)


def fifth_dist(pred, gt):
    return (pred[0] - gt[0]) % 12 in (5, 7)


def key_name(pc, mode, style="sharp"):
    arr = FLAT if style == "flat" else SHARP
    return arr[pc] + ("" if mode == "major" else "m")


def evaluate(preds, gts):
    n = len(gts)
    exact = sum(1 for p, g in zip(preds, gts) if p == g)
    rootpc = sum(1 for p, g in zip(preds, gts) if p[0] == g[0])
    mode = sum(1 for p, g in zip(preds, gts) if p[1] == g[1])
    rel = sum(1 for p, g in zip(preds, gts) if is_relative(p, g))
    fif = sum(1 for p, g in zip(preds, gts) if fifth_dist(p, g))
    rootok = [(p, g) for p, g in zip(preds, gts) if p[0] == g[0]]
    mode_given_root = (sum(1 for p, g in rootok if p[1] == g[1]) / len(rootok)) if rootok else 0.0

    def acc_mode(m):
        idx = [i for i, g in enumerate(gts) if g[1] == m]
        return (sum(1 for i in idx if preds[i] == gts[i]) / len(idx)) if idx else 0.0

    pair_ok = sum(1 for p, g in zip(preds, gts)
                  if pair_of_key(key_id(*p)) == pair_of_key(key_id(*g)))
    mode_ok_in_pair = 0
    for p, g in zip(preds, gts):
        if pair_of_key(key_id(*p)) == pair_of_key(key_id(*g)) and p[1] == g[1]:
            mode_ok_in_pair += 1
    return {
        "n": n,
        "exact": exact / n if n else 0.0,
        "exact_count": exact,
        "root_pc": rootpc / n if n else 0.0,
        "mode": mode / n if n else 0.0,
        "mode_given_root": mode_given_root,
        "pair_acc": pair_ok / n if n else 0.0,
        "mode_given_pair": (mode_ok_in_pair / pair_ok) if pair_ok else 0.0,
        "relative_rate": rel / n if n else 0.0,
        "fifth_rate": fif / n if n else 0.0,
        "other_rate": (n - exact - rel - fif) / n if n else 0.0,
        "relative_or_exact": (exact + rel) / n if n else 0.0,
        "acc_major_gt": acc_mode("major"),
        "acc_minor_gt": acc_mode("minor"),
    }


def naive_first_chord(chords):
    for c in chords:
        if c.dur <= 0:
            continue
        if c.kind in ("maj", "dom", "aug", "sus", "power"):
            mode = "major"
        else:
            mode = "minor"
        return c.root, mode
    return None


def naive_longest_chord(chords):
    acc = defaultdict(float)
    for c in chords:
        if c.dur > 0:
            acc[c.root] += c.dur
    if not acc:
        return None
    root = max(acc, key=acc.get)
    maj = sum(c.dur for c in chords if c.root == root and c.kind in ("maj", "dom"))
    mn = sum(c.dur for c in chords if c.root == root and c.kind == "min")
    return root, ("major" if maj >= mn else "minor")


# --------------------------------------------------------------------------
# 9. evaluation driver
# --------------------------------------------------------------------------

SYSTEMS = [
    ("naive_first", "naive: first chord root + quality"),
    ("naive_longest", "naive: longest chord root + quality"),
    ("vote_ks", "KS profile on chord-tone PC vector"),
    ("vote_temp", "Temperley profile on chord-tone PC vector"),
    ("vote_ks_root", "KS profile on chord ROOTS only (ablation)"),
    ("pcset_deg", "degree -> expected PC-set profile"),
    ("hf", "harmonic-function chord likelihood"),
    ("fused_rank", "fused 24-way rank, no pair logic"),
    ("stage1", "relative-pair salience, no mode adjudication"),
    ("harmonic_key", "stage 1 + relative-mode adjudication (FINAL)"),
]


def _load_song_rows(rec, source):
    if source == "lab":
        path = os.path.join(POP909_LAB_ROOT, rec["id"], "chords.lab")
        return (load_lab(path) if os.path.exists(path) else []), path
    path = os.path.join(POP909_RAW_ROOT, rec["num"], "chord_midi.txt")
    return (load_raw(path) if os.path.exists(path) else []), path


def run_eval(limit=None, seed=0, source="lab", out_path=None):
    manifest = load_manifest()
    if limit and limit < len(manifest):
        manifest = random.Random(seed).sample(manifest, limit)
        manifest.sort(key=lambda r: r["id"])

    preds = {name: [] for name, _ in SYSTEMS}
    gts = []
    per_song = []
    coverage = {"songs": 0, "songs_with_chords": 0, "tokens": 0,
                "unparsed": 0, "parsed_dur": 0.0, "total_dur": 0.0}
    adj_stats = Counter()
    t0 = time.time()

    for rec in manifest:
        rows, path = _load_song_rows(rec, source)
        chords, st = parse_rows(rows)
        coverage["tokens"] += st["tokens"]
        coverage["unparsed"] += st["unparsed"]
        coverage["songs"] += 1
        coverage["total_dur"] += sum(max(0.0, r[1] - r[0]) for r in rows)
        coverage["parsed_dur"] += sum(c.dur for c in chords)
        gt = (rec["pc"], rec["mode"])
        gts.append(gt)
        if not chords:
            for name in preds:
                preds[name].append((0, "major"))
            continue
        coverage["songs_with_chords"] += 1

        preds["naive_first"].append(naive_first_chord(chords) or (0, "major"))
        preds["naive_longest"].append(naive_longest_chord(chords) or (0, "major"))

        res = infer(chords)
        for name in ("vote_ks", "vote_temp", "vote_ks_root", "pcset_deg", "hf"):
            sc = res["scores_raw"][name]
            preds[name].append(id_to_key(max(range(24), key=lambda k: sc[k])))
        # fused 24-way rank, no pair decomposition
        fz = zscore(res["stage1"])
        preds["fused_rank"].append(id_to_key(max(range(24), key=lambda k: fz[k])))
        # stage-1 only: best relative pair by salience, best member by fused z
        bst = max(res["pairs"], key=lambda q: q["pair_z"])
        k1, k2 = bst["kmaj"], bst["kmin"]
        fz_all = res["stage1"]
        preds["stage1"].append(id_to_key(k1 if fz_all[k1] >= fz_all[k2] else k2))
        preds["harmonic_key"].append(res["pred"])

        bp = res["pred_pair"]
        if pair_of_key(key_id(*res["pred"])) == pair_of_key(key_id(*gt)):
            adj_stats["pair_correct"] += 1
        else:
            adj_stats["pair_wrong"] += 1
        if abs(bp["margin"]) > 0.5:
            adj_stats["decisive_margin"] += 1

        per_song.append({
            "id": rec["id"], "gt": rec["key"], "gt_id": list(gt),
            "pred": key_name(*res["pred"]), "pred_id": list(res["pred"]),
            "pair_ok": pair_of_key(key_id(*res["pred"])) == pair_of_key(key_id(*gt)),
            "margin": round(bp["margin"], 4),
            "pair_major": key_name(bp["pair_pc"], "major"),
            "pair_minor": key_name((bp["pair_pc"] - 3) % 12, "minor"),
            "top5": [[key_name(c["pc"], c["mode"]), round(c["score"], 4)]
                     for c in res["candidates"][:5]],
        })

    metrics = {name: evaluate(preds[name], gts) for name, _ in SYSTEMS}
    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": round(time.time() - t0, 2),
        "config": {"source": source, "limit": limit, "seed": seed,
                   "n_songs": len(gts), "fusion": FUSION_WEIGHTS,
                   "mode_weights": MODE_WEIGHTS},
        "coverage": {
            "songs": coverage["songs"],
            "songs_with_chords": coverage["songs_with_chords"],
            "song_coverage": coverage["songs_with_chords"] / max(1, coverage["songs"]),
            "tokens": coverage["tokens"],
            "unparsed_tokens": coverage["unparsed"],
            "token_coverage": 1 - coverage["unparsed"] / max(1, coverage["tokens"]),
            "duration_coverage": coverage["parsed_dur"] / max(1e-9, coverage["total_dur"]),
        },
        "systems": metrics,
        "pair_stats": dict(adj_stats),
        "per_song": per_song,
    }
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=1)
    return results


def print_eval(results):
    print("=" * 86)
    print("POP909 harmonic-key evaluation   n=%d   source=%s   %.1fs"
          % (results["config"]["n_songs"], results["config"]["source"],
             results["runtime_seconds"]))
    cov = results["coverage"]
    print("coverage: songs %.1f%% (%d/%d)  tokens %.2f%% (unparsed=%d)  duration %.2f%%"
          % (100 * cov["song_coverage"], cov["songs_with_chords"], cov["songs"],
             100 * cov["token_coverage"], cov["unparsed_tokens"],
             100 * cov["duration_coverage"]))
    print("-" * 86)
    print("%-14s %6s %6s %6s %6s %6s %6s %6s %6s" %
          ("system", "exact", "root", "pair", "mode|p", "rel", "fifth", "other", "M/m"))
    for name, _desc in SYSTEMS:
        m = results["systems"][name]
        print("%-14s %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.1f%%  %.2f/%.2f"
              % (name, 100 * m["exact"], 100 * m["root_pc"], 100 * m["pair_acc"],
                 100 * m["mode_given_pair"], 100 * m["relative_rate"],
                 100 * m["fifth_rate"], 100 * m["other_rate"],
                 m["acc_major_gt"], m["acc_minor_gt"]))
    print("-" * 86)
    print("pair stats:", results["pair_stats"])
    err = Counter()
    for r in results["per_song"]:
        g, p = tuple(r["gt_id"]), tuple(r["pred_id"])
        if g == p:
            continue
        kind = "relative" if is_relative(p, g) else ("fifth" if fifth_dist(p, g) else "other")
        err[(kind, r["gt"], r["pred"])] += 1
    print("top errors:", err.most_common(10))


# --------------------------------------------------------------------------
# 10. single song
# --------------------------------------------------------------------------

def _pick_source(doc):
    srcs = doc.get("sources") or []
    if not srcs:
        return None
    want = doc.get("default_source")
    for s in srcs:
        if s.get("id") == want:
            return s
    for s in srcs:
        if s.get("name") == want:
            return s
    return max(srcs, key=lambda s: len(s.get("rows") or []))


def _cnn_list(doc):
    ka = doc.get("key_analysis") or {}
    out = []
    for c in (ka.get("candidates") or []):
        k = c.get("key", "")
        pc = NAME2PC.get(k[:-1] if k.endswith("m") else k)
        if pc is None:
            continue
        out.append({"key": k, "pc": pc,
                    "mode": "minor" if k.endswith("m") else "major",
                    "score": c.get("score")})
    out.sort(key=lambda x: -(x["score"] or 0))
    return out


def _chord_label(c):
    if not c:
        return None
    return key_name(c[0], "major" if c[1] in ("maj", "dom") else "minor") + \
        ("" if c[1] in ("maj", "dom", "min") else ":" + c[1])


def analyze_song(path, source=None, verbose=True):
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if source:
        src = next((s for s in doc.get("sources", []) if s.get("id") == source), None)
    else:
        src = _pick_source(doc)
    chords, st = parse_rows((src or {}).get("rows") or [])
    res = infer(chords) if chords else None
    cnn = _cnn_list(doc)

    out = {
        "file": path,
        "title": doc.get("title"),
        "source": (src or {}).get("id") or (src or {}).get("name"),
        "n_chords": len(chords),
        "parse": st,
        "coverage_dur": round(sum(c.dur for c in chords), 1),
        "cnn": cnn[:6],
    }
    if res is None:
        out.update(pred=None, verdict="no_chords")
        if verbose:
            print("no parseable chords:", path)
        return out, None

    pc, mode = res["pred"]
    out["pred"] = key_name(pc, mode)
    out["pred_id"] = [pc, mode]
    out["candidates"] = [
        {"key": key_name(c["pc"], c["mode"]),
         "pair": key_name(c["pair_pc"], "major"),
         "pair_z": round(c["pair_z"], 4),
         "mode_margin": round(c["mode_margin"], 4),
         "winner": bool(c["is_pair_winner"]),
         "rank": c["rank"]}
        for c in res["candidates"][:10]]
    bp = res["pred_pair"]
    ev = bp["evidence"]
    out["adjudication"] = {
        "applied": True,
        "pair_major": key_name(bp["pair_pc"], "major"),
        "pair_minor": key_name((bp["pair_pc"] - 3) % 12, "minor"),
        "margin": round(bp["margin"], 4),
        "contrib": {k: round(v, 4) for k, v in bp["contrib"].items()},
        "cues": {k: round(v, 4) for k, v in ev["cues"].items()},
        "counts": ev["counts"],
        "dur_share": {k: round(v, 4) for k, v in ev["dur_share"].items()},
        "final_chord": _chord_label(ev["final_chord"]),
        "first_chord": _chord_label(ev["first_chord"]),
        "winner": key_name(*id_to_key(bp["kmaj"] if bp["margin"] > 0 else bp["kmin"])),
    }

    if cnn:
        c0 = cnn[0]
        if (c0["pc"], c0["mode"]) == (pc, mode):
            verdict = "agree"
        elif c0["pc"] == pc:
            verdict = "mode_flip"
        elif abs((c0["pc"] - pc) % 12) in (3, 9):
            verdict = "relative_override"
        elif abs((c0["pc"] - pc) % 12) in (5, 7):
            verdict = "fifth_override"
        else:
            verdict = "other_override"
        out["verdict"] = verdict
        out["cnn_top"] = c0["key"]
    if verbose:
        print_song(out)
    return out, res


def print_song(out):
    print("=" * 86)
    print(out["title"])
    print("source=%s  chords=%d  coverage=%.1fs  parse_bad=%d"
          % (out["source"], out["n_chords"], out["coverage_dur"],
             out["parse"]["unparsed"]))
    print("CNN   : " + " | ".join("%s %.3f" % (c["key"], c["score"] or 0)
                                 for c in out["cnn"][:4]))
    print("OURS  : " + " | ".join("%s [pair %.2f, margin %+.2f]"
                                 % (c["key"], c["pair_z"], c["mode_margin"])
                                 for c in out["candidates"][:4]))
    print("verdict: %-17s pred=%s  (CNN top: %s)"
          % (out["verdict"], out["pred"], out.get("cnn_top")))
    a = out["adjudication"]
    print("pair %s / %s   margin=%+.3f -> %s"
          % (a["pair_major"], a["pair_minor"], a["margin"], a["winner"]))
    print("  contributions: " + "  ".join("%s=%+.3f" % (k, v)
                                          for k, v in a["contrib"].items()))
    cues = {k: v for k, v in a["cues"].items() if abs(v) > 1e-9}
    print("  evidence     : " + "  ".join("%s=%+.2f" % (k, v)
                                          for k, v in cues.items()))
    if a["counts"]:
        print("  cadences     : " + str(a["counts"]))
    print("  duration     : " + " ".join("%s=%.2f" % (k, v)
                                         for k, v in a["dur_share"].items()))
    print("  first=%s  final=%s" % (a["first_chord"], a["final_chord"]))
    print()


# --------------------------------------------------------------------------
# 11. CLI
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Chord sequence -> global key")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", choices=["lab", "raw"], default="lab")
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    ap.add_argument("--song", default=None)
    ap.add_argument("--song-source", default=None)
    ap.add_argument("--all-songs", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-songs-in-results", action="store_true")
    args = ap.parse_args(argv)

    if args.song:
        out, _ = analyze_song(args.song, source=args.song_source,
                              verbose=not args.json)
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=1))
        return 0

    if args.all_songs:
        allout = [analyze_song(p)[0] for p in THREE_SONGS if os.path.exists(p)]
        with open(os.path.join(HERE, "songs.json"), "w", encoding="utf-8") as fh:
            json.dump(allout, fh, ensure_ascii=False, indent=1)
        return 0

    if args.eval:
        res = run_eval(limit=args.limit, seed=args.seed, source=args.source,
                       out_path=args.out)
        print_eval(res)
        if not args.no_songs_in_results:
            songs = [analyze_song(p, verbose=False)[0] for p in THREE_SONGS
                     if os.path.exists(p)]
            res["songs"] = songs
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(res, fh, ensure_ascii=False, indent=1)
            print()
            for s in songs:
                print_song(s)
        print("wrote", args.out)
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
