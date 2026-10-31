#!/usr/bin/env python3
"""Build a chord-boundary-labelled MIDI dataset from POP909 annotations.

For every selected song the script

1. reads the *real* harmonic annotation of the song (``chord_midi.txt`` /
   ``chords.lab``) together with the *real* beat/bar grid (``beat_midi.txt`` /
   ``beats.beats``) and the key (``key_audio.txt``; inferred when absent);
2. cuts a 30-60 s window that starts on a real downbeat, contains no "no chord"
   region, starts on a non-dominant chord and has at most ~20 % out-of-key
   chords (the harmonic-sanity filter from the task description);
3. quantises every chord change to the beat grid.  POP909 chord boundaries sit
   exactly on beats (max residual 5e-7 s over all 124 805 annotated segments),
   so the quantisation is lossless;
4. synthesises three accompaniment textures from the chord tones only:
   ``block`` (root + upper voices struck together on every beat),
   ``arpeggio`` (root on the beat, one chord tone on the off-beat) and
   ``strum`` (bass note then the upper voices spread over <= 25 ms);
5. optionally adds a melody voice on 30 % of the samples.  Melody pitches are
   restricted to chord tones plus in-key notes that are not a semitone away
   from any chord tone, and its onsets are placed off the accompaniment grid;
6. writes ``midi/<id>.mid`` and ``labels/<id>.events.json`` with the positive
   boundaries (real chord changes) and the negative points (bar / half-bar /
   beat positions and melody onsets that lie inside one chord).

The script never invents a harmonic progression: labels always come from the
POP909 annotation of the song.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mido

HERE = Path(__file__).resolve().parent
TICKS_PER_BEAT = 960

# --------------------------------------------------------------------------
# POP909 discovery
# --------------------------------------------------------------------------
CANDIDATE_ROOTS = [
    Path(r"D:\ChordStudio\dialogue\benchmarks\external_chords\pop909"),
    Path(r"D:\ChordStudio\Dialogue\benchmarks\external_chords\pop909"),
    Path(r"D:\ChordStudio\_archive\2026-10-30\benchmarks\raw\pop909\POP909"),
    Path(r"D:\ChordStudio\Dialogue\benchmarks\imported\pop909"),
    Path(r"D:\ChordStudio\dialogue\benchmarks\imported\pop909"),
]


def discover_root(explicit: str | None = None) -> Path:
    roots = [Path(explicit)] if explicit else list(CANDIDATE_ROOTS)
    tried = []
    for root in roots:
        tried.append(str(root))
        if not root.is_dir():
            continue
        songs = [d for d in sorted(root.iterdir()) if d.is_dir() and song_files(d)]
        if songs:
            return root
    raise SystemExit("POP909 not found. Tried:\n  " + "\n  ".join(tried))


def song_files(d: Path) -> dict | None:
    """Return the file layout of one POP909 song directory, if usable."""
    beat = None
    for name in ("beat_midi.txt", "beats.beats"):
        if (d / name).is_file():
            beat = d / name
            break
    chord = None
    for name in ("chord_midi.txt", "chords.lab", "chord_audio.txt"):
        if (d / name).is_file():
            chord = d / name
            break
    if beat is None or chord is None:
        return None
    key = None
    for name in ("key_midi.txt", "key_audio.txt"):
        if (d / name).is_file():
            key = d / name
            break
    return {"beat": beat, "chord": chord, "key": key}


# --------------------------------------------------------------------------
# Music constants and chord parsing
# --------------------------------------------------------------------------
NOTE_PC = {
    "C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5, "F": 5, "F#": 6, "Gb": 6, "G": 7,
    "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}
PC_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PC_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

QUALITY_INTERVALS = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "sus4(b7)": (0, 5, 7, 10),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "minmaj7": (0, 3, 7, 11),
    "dim7": (0, 3, 6, 9),
    "hdim7": (0, 3, 6, 10),
    "maj6": (0, 4, 7, 9),
    "min6": (0, 3, 7, 9),
    "9": (0, 4, 7, 10, 14),
    "maj9": (0, 4, 7, 11, 14),
    "min9": (0, 3, 7, 10, 14),
}
# Harte-style aliases (only used when the repackaged ``chords.lab`` is read)
QUALITY_ALIASES = {
    "M": "maj", "m": "min", "": "maj", "maj": "maj", "min": "min",
    "m7": "min7", "M7": "maj7", "7": "7", "dim": "dim", "aug": "aug",
    "sus2": "sus2", "sus4": "sus4", "sus": "sus4", "dim7": "dim7",
    "hdim7": "hdim7", "minmaj7": "minmaj7", "mM7": "minmaj7",
    "maj6": "maj6", "min6": "min6", "69": "maj6", "9": "9",
    "maj9": "maj9", "min9": "min9", "mmaj7": "minmaj7",
}
BASS_INTERVALS = {
    "1": 0, "2": 2, "b3": 3, "3": 4, "4": 5, "5": 7, "b5": 6, "6": 9,
    "b7": 10, "7": 11, "b9": 13, "9": 14, "#5": 8, "b6": 8,
}
MAJOR_STEPS = {0: "I", 1: "bII", 2: "II", 3: "bIII", 4: "III", 5: "IV",
               6: "#IV", 7: "V", 8: "bVI", 9: "VI", 10: "bVII", 11: "VII"}
MINOR_STEPS = {0: "I", 1: "bII", 2: "II", 3: "III", 4: "#III", 5: "IV",
               6: "#IV", 7: "V", 8: "VI", 9: "#VI", 10: "VII", 11: "#VII"}
SCALE_PCS = {
    "maj": {0, 2, 4, 5, 7, 9, 11},
    "min": {0, 2, 3, 5, 7, 8, 10},
}
# GM programs, in rotation order
INSTRUMENTS = [
    (0, "Acoustic Grand Piano"),
    (25, "Acoustic Guitar (steel)"),
    (24, "Acoustic Guitar (nylon)"),
    (4, "Electric Piano 1"),
    (48, "String Ensemble 1"),
]
BLOCK_PROGRAMS = [0, 4, 48, 24, 25]
ARP_PROGRAMS = [0, 4, 24, 25, 48]
STRUM_PROGRAMS = [25, 24, 0]
MELODY_PROGRAMS = [(0, "Acoustic Grand Piano"), (4, "Electric Piano 1"),
                   (48, "String Ensemble 1")]
TEXTURES = ("block", "arpeggio", "strum")


def parse_note_name(name: str) -> int:
    name = name.strip()
    if name in NOTE_PC:
        return NOTE_PC[name]
    if len(name) >= 2 and name[1] in "#b" and name[:2] in NOTE_PC:
        return NOTE_PC[name[:2]]
    raise KeyError(name)


def parse_chord_label(label: str):
    """Return (root_pc, quality, pcs, bass_pc, root_name, bass_name)."""
    raw = label.strip()
    if not raw or raw.upper() in {"N", "X"}:
        return None
    if ":" in raw:
        root_s, rest = raw.split(":", 1)
        root = parse_note_name(root_s)
    else:
        i = 1
        if len(raw) > 1 and raw[1] in "#b":
            i = 2
        root_s = raw[:i]
        root = parse_note_name(root_s)
        rest = raw[i:]
    quality, _, bass = rest.partition("/")
    quality = quality.strip()
    if quality in QUALITY_INTERVALS:
        intervals = QUALITY_INTERVALS[quality]
    elif quality in QUALITY_ALIASES:
        quality = QUALITY_ALIASES[quality]
        intervals = QUALITY_INTERVALS.get(quality, (0, 4, 7))
    else:
        # unknown extension: keep the triad family, drop the extension
        low = quality.lower()
        if low.startswith("m") and not low.startswith("maj"):
            quality, intervals = "min", QUALITY_INTERVALS["min"]
        elif low.startswith("dim"):
            quality, intervals = "dim", QUALITY_INTERVALS["dim"]
        elif low.startswith("aug"):
            quality, intervals = "aug", QUALITY_INTERVALS["aug"]
        elif low.startswith("sus"):
            quality, intervals = "sus4", QUALITY_INTERVALS["sus4"]
        else:
            quality, intervals = "maj", QUALITY_INTERVALS["maj"]
    pcs = tuple(sorted({(root + iv) % 12 for iv in intervals}))
    bass_pc = root
    if bass:
        key = bass.strip()
        if key in BASS_INTERVALS:
            bass_pc = (root + BASS_INTERVALS[key]) % 12
        else:
            try:
                bass_pc = parse_note_name(key)
            except KeyError:
                bass_pc = root
    return root, quality, pcs, bass_pc, root_s, pc_name(bass_pc)


def parse_key_label(label: str):
    raw = label.strip()
    if ":" in raw:
        tonic_s, mode_s = raw.split(":", 1)
    else:
        m = raw[-1].lower() in {"m", "a", "i"} and len(raw) > 1
        tonic_s, mode_s = (raw[:-1], "min") if m else (raw, "maj")
    tonic = parse_note_name(tonic_s)
    mode = "min" if mode_s.lower().startswith("m") and not mode_s.lower().startswith("maj") else "maj"
    return tonic, mode, tonic_s


def pc_name(pc: int, flat: bool = False) -> str:
    return (PC_FLAT if flat else PC_SHARP)[pc % 12]


def roman_numeral(root_pc: int, quality: str, tonic: int, mode: str) -> str:
    rel = (root_pc - tonic) % 12
    table = MAJOR_STEPS if mode == "maj" else MINOR_STEPS
    numeral = table[rel]
    if quality in ("maj", ""):
        case, suffix = True, ""
    elif quality == "maj7":
        case, suffix = True, "maj7"
    elif quality == "maj6":
        case, suffix = True, "6"
    elif quality == "7":
        case, suffix = True, "7"
    elif quality == "9":
        case, suffix = True, "9"
    elif quality == "maj9":
        case, suffix = True, "maj9"
    elif quality == "aug":
        case, suffix = True, "+"
    elif quality == "sus2":
        case, suffix = True, "sus2"
    elif quality == "sus4":
        case, suffix = True, "sus4"
    elif quality == "sus4(b7)":
        case, suffix = True, "7sus4"
    elif quality == "min":
        case, suffix = False, ""
    elif quality == "min7":
        case, suffix = False, "7"
    elif quality == "min9":
        case, suffix = False, "9"
    elif quality == "min6":
        case, suffix = False, "6"
    elif quality == "minmaj7":
        case, suffix = False, "maj7"
    elif quality == "dim":
        case, suffix = False, "dim"
    elif quality == "dim7":
        case, suffix = False, "dim7"
    elif quality == "hdim7":
        case, suffix = False, "hdim7"
    else:
        case, suffix = True, quality
    body = numeral.upper() if case else numeral.lower()
    if numeral[0] in "b#":
        body = numeral[0] + (numeral[1:].upper() if case else numeral[1:].lower())
    return body + suffix


# --------------------------------------------------------------------------
# Song parsing
# --------------------------------------------------------------------------
@dataclass
class Chord:
    start: int          # beat index
    end: int            # beat index (exclusive)
    label: str
    root: int
    quality: str
    pcs: tuple
    bass: int
    roman: str = ""
    root_name: str = ""
    bass_name: str = ""


@dataclass
class Song:
    sid: str
    path: Path
    layout: dict
    key_name: str
    tonic: int
    mode: str
    bpm: float
    beat_times: list[float]
    downbeats: list[int]
    chords: list[Chord]
    n_beats: int
    bpb: int

    def chord_at(self, beat: int) -> Chord | None:
        for c in self.chords:
            if c.start <= beat < c.end:
                return c
        return None


def read_lines(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(line.split())
    return rows


def nearest_beat(times: list[float], t: float, tol: float) -> int | None:
    import bisect
    i = bisect.bisect_left(times, t)
    best, best_d = None, 1e9
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(times):
            d = abs(times[j] - t)
            if d < best_d:
                best, best_d = j, d
    if best is None or best_d > tol:
        return None
    return best


def parse_beat_grid(path: Path) -> tuple[list[float], list[int], list[int]]:
    """Return (beat times, downbeat indices, half-bar pulse indices)."""
    rows = read_lines(path)
    times: list[float] = []
    down: list[int] = []
    pulse: list[int] = []
    name = path.name.lower()
    for i, r in enumerate(rows):
        t = float(r[0])
        times.append(t)
        if name.startswith("beats"):
            in_bar = int(float(r[1]))
            if in_bar == 1:
                down.append(i)
        else:
            if len(r) >= 3 and float(r[2]) > 0.5:
                down.append(i)
            if len(r) >= 2 and float(r[1]) > 0.5:
                pulse.append(i)
    if not down and times:
        down = [0]
    if not pulse:
        pulse = down
    return times, down, pulse


def parse_key_file(path: Path | None) -> tuple[str, int, str] | None:
    if path is None or not path.is_file():
        return None
    best, best_len = None, -1.0
    for r in read_lines(path):
        if len(r) < 3:
            continue
        try:
            dur = float(r[1]) - float(r[0])
        except ValueError:
            continue
        if dur > best_len:
            best_len, best = dur, r[2]
    if best is None:
        return None
    tonic, mode, name = parse_key_label(best)
    return name, tonic, mode


def infer_key(chords: list[Chord], beat_times: list[float]) -> tuple[str, int, str]:
    """Krumhansl-style key inference used only when key_audio.txt is missing."""
    maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    hist = [0.0] * 12
    for c in chords:
        w = max(1, c.end - c.start)
        hist[c.root] += w
        for pc in c.pcs:
            hist[pc] += 0.35 * w
    best = (-1e9, 0, 0, "C")
    for tonic in range(12):
        rot = [hist[(tonic + k) % 12] for k in range(12)]
        for mode, prof in (("maj", maj_profile), ("min", min_profile)):
            s = sum(a * b for a, b in zip(rot, prof))
            if s > best[0]:
                best = (s, tonic, mode, pc_name(tonic))
    _, tonic, mode, name = best
    return name, tonic, mode


def load_song(song_dir: Path, flatten_seconds: float = 0.0) -> Song | None:
    layout = song_files(song_dir)
    if layout is None:
        return None
    times, downbeats, pulse = parse_beat_grid(layout["beat"])
    if len(times) < 8:
        return None
    diffs = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    beat_dur = statistics.median(diffs)
    if beat_dur <= 0.15 or beat_dur > 2.0:
        return None
    bpm = 60.0 / beat_dur

    raw = []
    for r in read_lines(layout["chord"]):
        if len(r) < 3:
            continue
        try:
            t0, t1 = float(r[0]), float(r[1])
        except ValueError:
            continue
        raw.append((t0, t1, r[2]))
    if not raw:
        return None

    tol = max(0.02, 0.25 * beat_dur)
    chords: list[Chord] = []
    for t0, t1, label in raw:
        b0 = nearest_beat(times, t0, tol)
        b1 = nearest_beat(times, t1, tol)
        if b0 is None or b1 is None or b1 <= b0:
            continue
        parsed = parse_chord_label(label)
        if parsed is None:
            continue
        root, quality, pcs, bass, root_name, bass_name = parsed
        prev = chords[-1] if chords else None
        # merge repeated annotations of the same chord (incl. enharmonic spellings)
        if prev and prev.end == b0 and (prev.root, prev.quality, prev.bass) == (root, quality, bass):
            prev.end = b1
            continue
        chords.append(Chord(b0, b1, label, root, quality, pcs, bass,
                            root_name=root_name, bass_name=bass_name))
    if len(chords) < 4:
        return None

    key = parse_key_file(layout["key"])
    if key is None:
        key_name, tonic, mode = infer_key(chords, times)
    else:
        key_name, tonic, mode = key

    for c in chords:
        c.roman = roman_numeral(c.root, c.quality, tonic, mode)

    bpb = 4
    if len(downbeats) >= 3:
        steps = [downbeats[i + 1] - downbeats[i] for i in range(len(downbeats) - 1)]
        bpb = int(statistics.mode(steps)) if steps else 4
        if bpb not in (2, 3, 4, 5, 6, 7):
            bpb = 4
    return Song(song_dir.name, song_dir, layout, key_name, tonic, mode, bpm,
                times, downbeats, chords, len(times), bpb)


# --------------------------------------------------------------------------
# Window planning (harmonic sanity filter)
# --------------------------------------------------------------------------
def window_segments(song: Song, start: int, end: int) -> list[Chord]:
    out = []
    for c in song.chords:
        if c.end <= start or c.start >= end:
            continue
        out.append(Chord(max(c.start, start), min(c.end, end), c.label, c.root,
                         c.quality, c.pcs, c.bass, c.roman,
                         root_name=c.root_name, bass_name=c.bass_name))
    return out


def explainable_chromatic(root: int, tonic: int, mode: str) -> bool:
    """Is a chord whose root is outside the key explainable by a simple rule?

    Accepted as *explainable*: mixture chords from the parallel mode
    (bII / bIII / bVI / bVII in major, III / VI / VII in minor) and secondary
    dominants (root a perfect fifth above a diatonic degree).  Everything else
    is treated as an unexplained chromatic chord and is filtered out.
    """
    rel = (root - tonic) % 12
    scale = SCALE_PCS[mode]
    if rel in scale:
        return True
    if mode == "maj":
        if rel in {1, 3, 8, 10}:          # bII (Neapolitan), bIII, bVI, bVII
            return True
    else:
        if rel in {4, 9, 11}:             # harmonic / melodic minor colours
            return True
    return ((rel - 7) % 12) in scale      # secondary dominant (V/x)


def score_window(song: Song, start: int, end: int, segs: list[Chord],
                 target: float = 42.0) -> float:
    if not segs:
        return -1e9
    if segs[0].start != start or segs[-1].end != end:
        return -1e9
    # full coverage: every beat must belong to a chord segment
    pos = start
    for c in segs:
        if c.start != pos:
            return -1e9
        pos = c.end
    if pos != end:
        return -1e9
    scale = {(song.tonic + k) % 12 for k in SCALE_PCS[song.mode]}
    roots = {c.root for c in segs}
    chromatic = [c for c in segs if c.root not in scale]
    chrom_ratio = len(chromatic) / len(segs)
    first = segs[0].roman
    first_base = first.lstrip("b#")[:2] if first.lstrip("b#")[:2] in ("II", "IV") else first.lstrip("b#")[:1]
    starts_on_v = first.lstrip("b#").startswith("V") and not first.lstrip("b#").startswith("VI") and not first.lstrip("b#").startswith("VII")
    dur = (end - start) * 60.0 / song.bpm
    score = 3.0 * min(len(roots), 6) - 4.0 * chrom_ratio
    score -= 6.0 if starts_on_v else 0.0
    score += 2.0 if segs[0].start == start else 0.0
    score -= 2.0 * max(0, 6 - len(segs))
    score -= 0.8 * abs(dur - target)
    return score


def plan_song(song: Song, min_dur: float = 30.0, max_dur: float = 58.0,
              target: float = 42.0, max_chromatic: float = 0.30) -> dict | None:
    if not song.downbeats:
        return None
    if len(song.downbeats) >= 2:
        bar_dur = statistics.median(
            [song.downbeats[i + 1] - song.downbeats[i] for i in range(len(song.downbeats) - 1)])
    else:
        bar_dur = 4
    bar_dur_sec = bar_dur * 60.0 / song.bpm
    nb_min = max(2, int(math.ceil(min_dur / bar_dur_sec)))
    nb_max = max(nb_min, int(math.floor(max_dur / bar_dur_sec)))
    if nb_max - nb_min > 24:
        nb_max = nb_min + 24
    best = None
    for i, start in enumerate(song.downbeats):
        if start >= song.n_beats - 4:
            break
        for nb in range(nb_min, nb_max + 1):
            j = i + nb
            if j >= len(song.downbeats):
                break
            end = song.downbeats[j]
            dur = (end - start) * 60.0 / song.bpm
            if dur < min_dur or dur > max_dur:
                continue
            segs = window_segments(song, start, end)
            if len(segs) < 4:
                continue
            if len({c.root for c in segs}) < 3:
                continue
            scale = {(song.tonic + k) % 12 for k in SCALE_PCS[song.mode]}
            chrom = [c for c in segs if c.root not in scale]
            if len(chrom) > 2 or len(chrom) / len(segs) > 0.20:
                continue
            if any(not explainable_chromatic(c.root, song.tonic, song.mode) for c in chrom):
                continue
            # chromatic diminished / augmented chords are not explained by the
            # simple mixture / secondary-dominant rule, so drop those windows
            if any(c.quality in ("dim", "dim7", "hdim7", "aug") for c in chrom):
                continue
            first = segs[0].roman.lstrip("b#")
            if first.startswith("V") and not first.startswith("VI") and not first.startswith("VII"):
                continue
            sc = score_window(song, start, end, segs, target)
            if best is None or sc > best["score"]:
                best = {"start": start, "end": end, "score": sc, "segments": segs,
                        "duration": dur, "bars": nb, "bpb": int(round(bar_dur)),
                        "n_chromatic": len(chrom),
                        "chromatic": [c.label for c in chrom]}
    return best


# --------------------------------------------------------------------------
# Arrangement building
# --------------------------------------------------------------------------
@dataclass
class Note:
    pitch: int
    start: float
    dur: float
    vel: int


def choose_octave(pc: int, target: float, lo: int = 55, hi: int = 79) -> int:
    cands = [p for p in range(lo, hi + 1) if p % 12 == pc]
    if not cands:
        cands = [pc + 60]
    return min(cands, key=lambda p: abs(p - target))


def voice_chord(chord: Chord, prev: list[int], prev_bass: int | None) -> tuple[int, list[int]]:
    """Return (bass pitch, upper pitches) with simple voice leading."""
    bass_target = prev_bass if prev_bass is not None else 41
    bass = choose_octave(chord.bass, bass_target, lo=36, hi=50)
    upper = []
    for idx, pc in enumerate(chord.pcs):
        target = prev[idx] if idx < len(prev) else 64 + (pc - chord.root) % 12 * 0.5
        upper.append(choose_octave(pc, target))
    upper.sort()
    return bass, upper


def chord_tones_to_pcs(chord: Chord) -> set[int]:
    return set(chord.pcs)


def allowed_melody_pcs(chord: Chord, scale_pcs: set[int]) -> set[int]:
    allowed = set(chord.pcs)
    for pc in scale_pcs:
        if pc in allowed:
            continue
        if any((pc - c) % 12 in (1, 11) for c in chord.pcs):
            continue
        allowed.add(pc)
    return allowed


def build_accompaniment(segs: list[Chord], start: int, end: int, texture: str,
                        rng: random.Random, bpm: float = 90.0) -> list[Note]:
    notes: list[Note] = []
    prev: list[int] = []
    prev_bass = None
    # cache one voicing per chord segment so re-attacks keep the same voicing
    voicings: dict[int, tuple[int, list[int]]] = {}
    for idx, c in enumerate(segs):
        bass, upper = voice_chord(c, prev, prev_bass)
        voicings[idx] = (bass, upper)
        prev, prev_bass = upper, bass

    def seg_index(beat: int) -> int:
        for i, c in enumerate(segs):
            if c.start <= beat < c.end:
                return i
        return len(segs) - 1

    for beat in range(start, end):
        c = segs[seg_index(beat)]
        bass, upper = voicings[seg_index(beat)]
        is_change = c.start == beat
        is_bar = beat in set()  # placeholder; refined by caller-independent grid
        vel = 82 if is_change else 68
        if texture == "block":
            for k, pitch in enumerate([bass] + upper):
                notes.append(Note(pitch, beat, 0.86, int(vel + (6 if k == 0 else 0))))
        elif texture == "arpeggio":
            notes.append(Note(bass, beat, 0.92, vel + 8))
            tones = upper if upper else [bass + 12]
            tone = tones[(beat - start) % len(tones)]
            notes.append(Note(tone, beat + 0.5, 0.42, vel - 4))
        elif texture == "strum":
            # spread the strings over <= 26 ms (7 ms per string); a chord change
            # always uses a down-strum so its bass note is exactly on the beat
            spread_ms = [0.0, 7.0, 14.0, 20.0, 26.0]
            seq = [bass] + upper
            if not is_change and rng.random() < 0.5:   # up-strums on repeats
                seq = list(reversed(seq))
            for k, pitch in enumerate(seq):
                off_ms = 0.0 if k == 0 else spread_ms[min(k, len(spread_ms) - 1)]
                notes.append(Note(pitch, beat + off_ms * bpm / 60000.0, 0.80,
                                  int(vel - 4 + k * 2)))
        else:
            raise ValueError(texture)
    return notes


def build_melody(segs: list[Chord], start: int, end: int, tonic: int, mode: str,
                 rng: random.Random) -> list[Note]:
    scale_pcs = {(tonic + k) % 12 for k in SCALE_PCS[mode]}
    onsets: list[tuple[float, Chord]] = []
    for beat in range(start, end):
        c = next((s for s in segs if s.start <= beat < s.end), segs[-1])
        # deliberately off the accompaniment grid: mostly the "and" of the beat
        if rng.random() < 0.55:
            onsets.append((beat + 0.5, c))
        if rng.random() < 0.18:
            onsets.append((float(beat), c))
    if not onsets:
        return []
    onsets.sort(key=lambda x: x[0])
    notes: list[Note] = []
    prev_pitch = rng.choice([67, 69, 72, 74])
    for i, (t, c) in enumerate(onsets):
        allowed = allowed_melody_pcs(c, scale_pcs)
        on_beat = abs(t - round(t)) < 1e-6
        pool_pcs = set(c.pcs) if on_beat else allowed
        cands = [p for p in range(60, 85) if p % 12 in pool_pcs]
        if not cands:
            cands = [p for p in range(60, 85) if p % 12 in c.pcs]
        scored = sorted(cands, key=lambda p: abs(p - prev_pitch) + rng.uniform(0, 3.5))
        pitch = scored[0]
        if pitch == prev_pitch and rng.random() < 0.6 and len(scored) > 1:
            pitch = scored[1]
        nxt = onsets[i + 1][0] if i + 1 < len(onsets) else t + 1.0
        dur = min(1.1, max(0.22, nxt - t - 0.03))
        notes.append(Note(pitch, t, dur, rng.randint(78, 100)))
        prev_pitch = pitch
    return notes


# --------------------------------------------------------------------------
# MIDI writing
# --------------------------------------------------------------------------
MAJOR_BY_FIFTHS = {-7: "Cb", -6: "Gb", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb",
                   -1: "F", 0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B",
                   6: "F#", 7: "C#"}
MINOR_BY_FIFTHS = {0: "Am", -7: "Abm", -6: "Ebm", -5: "Bbm", -4: "Fm", -3: "Cm",
                   -2: "Gm", -1: "Dm", 1: "Em", 2: "Bm", 3: "F#m", 4: "C#m",
                   5: "G#m", 6: "D#m", 7: "A#m"}


def key_signature_name(tonic: int, mode: str) -> str:
    if mode == "maj":
        f = (tonic * 7) % 12
        f = f - 12 if f > 6 else f
        return MAJOR_BY_FIFTHS.get(f, "C")
    rel = (tonic + 3) % 12
    f = (rel * 7) % 12
    f = f - 12 if f > 6 else f
    return MINOR_BY_FIFTHS.get(f, "Am")


def tempo_from_bpm(bpm: float) -> tuple[int, float]:
    us = max(200000, min(1200000, int(round(60_000_000.0 / bpm))))
    return us, 60_000_000.0 / us


def write_midi(path: Path, tracks: list[tuple[str, int, int, list[Note]]],
               bpm: float, time_sig: tuple[int, int], key_sig: str) -> float:
    """tracks: (name, channel, program, notes).  Returns effective bpm."""
    us, eff_bpm = tempo_from_bpm(bpm)
    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT, type=1)
    meta = mido.MidiTrack()
    mid.tracks.append(meta)
    meta.append(mido.MetaMessage("track_name", name="META", time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=us, time=0))
    try:
        ks = mido.MetaMessage("key_signature", key=key_sig, time=0)
        meta.append(ks)
    except (ValueError, KeyError):
        pass
    meta.append(mido.MetaMessage("time_signature", numerator=time_sig[0],
                                 denominator=time_sig[1], time=0))
    for name, channel, program, notes in tracks:
        tr = mido.MidiTrack()
        mid.tracks.append(tr)
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        tr.append(mido.Message("program_change", program=program,
                               channel=channel, time=0))
        events = []
        for n in notes:
            t0 = int(round(n.start * TICKS_PER_BEAT))
            t1 = int(round((n.start + n.dur) * TICKS_PER_BEAT))
            t1 = max(t1, t0 + 8)
            events.append((t0, 1, n.pitch, max(1, min(127, n.vel))))
            events.append((t1, 0, n.pitch, 0))
        events.sort(key=lambda e: (e[0], e[1]))
        last = 0
        for tick, kind, pitch, vel in events:
            delta = tick - last
            last = tick
            if kind:
                tr.append(mido.Message("note_on", note=pitch, velocity=vel,
                                       time=delta, channel=channel))
            else:
                tr.append(mido.Message("note_off", note=pitch, velocity=0,
                                       time=delta, channel=channel))
    path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(path))
    return eff_bpm


# --------------------------------------------------------------------------
# Sample building
# --------------------------------------------------------------------------
def pick_instrument(texture: str, song_index: int, texture_index: int) -> tuple[int, str]:
    slot = (song_index * 3 + texture_index) % 5
    if texture == "strum":
        program = STRUM_PROGRAMS[slot % len(STRUM_PROGRAMS)]
    elif texture == "arpeggio":
        program = ARP_PROGRAMS[slot % len(ARP_PROGRAMS)]
    else:
        program = BLOCK_PROGRAMS[slot % len(BLOCK_PROGRAMS)]
    name = dict(INSTRUMENTS)[program]
    return program, name


def build_sample(song: Song, plan: dict, texture: str, texture_index: int,
                 song_index: int, rng: random.Random, out_dir: Path,
                 melody_frac: float = 0.30) -> dict:
    start, end = plan["start"], plan["end"]
    segs = [Chord(c.start - start, c.end - start, c.label, c.root, c.quality,
                  c.pcs, c.bass, c.roman, root_name=c.root_name,
                  bass_name=c.bass_name) for c in plan["segments"]]
    span = end - start
    program, program_name = pick_instrument(texture, song_index, texture_index)
    with_melody = rng.random() < melody_frac
    sample_id = f"pop909-{song.sid}-{texture}" + ("-mel" if with_melody else "")
    if with_melody:
        mel_options = [p for p in MELODY_PROGRAMS if p[0] != program] or MELODY_PROGRAMS
        mel_program, mel_name = rng.choice(mel_options)
    else:
        mel_program, mel_name = None, None

    notes = build_accompaniment(segs, 0, span, texture, rng, bpm=song.bpm)
    melody: list[Note] = []
    if with_melody:
        melody = build_melody(segs, 0, span, song.tonic, song.mode, rng)

    us, eff_bpm = tempo_from_bpm(song.bpm)
    sec = 60.0 / eff_bpm
    time_sig = (plan["bpb"] if plan["bpb"] in (2, 3, 4, 6) else 4, 4)
    tracks = [("CHORDS", 0, program, notes)]
    if melody:
        tracks.append(("MELODY", 1, mel_program, melody))
    midi_path = out_dir / "midi" / f"{sample_id}.mid"
    write_midi(midi_path, tracks, song.bpm, time_sig, key_signature_name(song.tonic, song.mode))

    # --- labels
    downbeat_rel = {song.downbeats[i] - start for i in range(len(song.downbeats))
                    if 0 <= song.downbeats[i] - start < span}
    bar_starts = sorted(downbeat_rel)
    bar_pos = {}
    for bi, b in enumerate(bar_starts):
        nxt = bar_starts[bi + 1] if bi + 1 < len(bar_starts) else span
        for k in range(b, nxt):
            bar_pos[k] = (b, k - b)
    attack_beats = {round(n.start * 2) / 2 for n in notes}
    chords_out = []
    boundaries = []
    negatives = []
    for i, c in enumerate(segs):
        cs = {
            "label": c.label,
            "start": round(c.start * sec, 4),
            "end": round(c.end * sec, 4),
            "start_beat": c.start,
            "end_beat": c.end,
            "roman": c.roman,
            "root_pc": c.root,
            "root_name": c.root_name or pc_name(c.root),
            "quality": c.quality,
            "pcs": list(c.pcs),
            "bass_pc": c.bass,
            "bass_name": c.bass_name or pc_name(c.bass),
        }
        chords_out.append(cs)
        if c.start > 0:
            boundaries.append({"time": round(c.start * sec, 4),
                               "beat": c.start,
                               "from": segs[i - 1].label,
                               "to": c.label,
                               "roman": c.roman})
        # negative points: grid positions inside the chord
        for b in range(c.start + 1, c.end):
            if b not in bar_pos:
                continue
            bar_start, idx = bar_pos[b]
            if idx == 0:
                # a bar line inside a chord is the strongest negative
                kind = "bar"
            elif idx % 2 == 0:
                kind = "half_bar"
            else:
                kind = "beat"
            negatives.append({
                "time": round(b * sec, 4),
                "beat": b,
                "kind": kind,
                "chord": c.label,
                "has_onset": b in attack_beats,
            })
    for n in melody:
        negatives.append({
            "time": round(n.start * sec, 4),
            "beat": round(n.start, 3),
            "kind": "melody",
            "chord": next((c.label for c in segs if c.start <= n.start < c.end), segs[-1].label),
            "has_onset": True,
        })
    negatives.sort(key=lambda x: x["time"])

    roman_seq = [c.roman for c in segs]
    labels = {
        "sample_id": sample_id,
        "source_song": song.sid,
        "source_dir": str(song.path),
        "source_chord_file": song.layout["chord"].name,
        "source_beat_file": song.layout["beat"].name,
        "source_key_file": song.layout["key"].name if song.layout["key"] else None,
        "key": f"{song.key_name}:{song.mode}",
        "key_label": song.key_name,
        "mode": song.mode,
        "tonic_pc": song.tonic,
        "scale_pcs": sorted({(song.tonic + k) % 12 for k in SCALE_PCS[song.mode]}),
        "bpm": round(eff_bpm, 4),
        "time_signature": f"{time_sig[0]}/{time_sig[1]}",
        "texture": texture,
        "instrument": {"program": program, "name": program_name},
        "melody": with_melody and bool(melody),
        "melody_instrument": ({"program": mel_program, "name": mel_name}
                              if with_melody and melody else None),
        "duration": round(span * sec, 4),
        "beats": span,
        "window_start_beat_in_song": start,
        "window_end_beat_in_song": end,
        "midi": f"midi/{sample_id}.mid",
        "audio": f"audio/{sample_id}.wav",
        "chords": chords_out,
        "boundaries": boundaries,
        "boundaries_sec": [b["time"] for b in boundaries],
        "negative_points": negatives,
        "negative_sec": [n["time"] for n in negatives],
        "progression_roman": roman_seq,
        "progression_labels": [c.label for c in segs],
        "n_chromatic_chords": plan.get("n_chromatic", 0),
        "chromatic_chords": plan.get("chromatic", []),
        "stats": {
            "n_chords": len(segs),
            "n_boundaries": len(boundaries),
            "n_negatives": len(negatives),
            "n_negative_grid": sum(1 for n in negatives if n["kind"] != "melody"),
            "n_negative_melody": sum(1 for n in negatives if n["kind"] == "melody"),
            "n_negative_with_onset": sum(1 for n in negatives if n["has_onset"]),
            "n_notes_chord": len(notes),
            "n_notes_melody": len(melody),
        },
    }
    label_path = out_dir / "labels" / f"{sample_id}.events.json"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    return {
        "sample_id": sample_id,
        "source_song": song.sid,
        "midi": labels["midi"],
        "labels": f"labels/{sample_id}.events.json",
        "audio": labels["audio"],
        "texture": texture,
        "instrument": program_name,
        "instrument_program": program,
        "melody": labels["melody"],
        "key": labels["key"],
        "bpm": labels["bpm"],
        "duration": labels["duration"],
        "n_boundaries": len(boundaries),
        "n_negatives": len(negatives),
        "n_negatives_bar": sum(1 for n in negatives if n["kind"] == "bar"),
        "n_negatives_halfbar": sum(1 for n in negatives if n["kind"] == "half_bar"),
        "n_negatives_beat": sum(1 for n in negatives if n["kind"] == "beat"),
        "n_negatives_melody": sum(1 for n in negatives if n["kind"] == "melody"),
        "n_chromatic": plan.get("n_chromatic", 0),
        "progression_roman": roman_seq,
    }


# --------------------------------------------------------------------------
# Song selection
# --------------------------------------------------------------------------
def tempo_bin(bpm: float) -> str:
    if bpm < 75:
        return "slow(<75)"
    if bpm <= 100:
        return "mid(75-100)"
    return "fast(>100)"


def _balanced_groups(plans: list[dict], n: int, used_tonic: dict[int, int],
                     used_ids: set[str]) -> list[dict]:
    """Pick up to n songs, stratified by (mode, tempo bin) with tonic diversity."""
    pool = [p for p in plans if p["sid"] not in used_ids]
    groups: dict[tuple[str, str], list[dict]] = {}
    for p in pool:
        groups.setdefault((p["mode"], tempo_bin(p["bpm"])), []).append(p)
    if not groups or n <= 0:
        return []
    for g in groups.values():
        g.sort(key=lambda p: -p["score"])
    keys = sorted(groups)
    per, rem = divmod(n, len(keys))
    chosen: list[dict] = []
    for gi, key in enumerate(keys):
        bucket = groups[key]
        quota = per + (1 if gi < rem else 0)
        picked = 0
        while bucket and picked < quota:
            best_i, best_v = None, None
            for i, p in enumerate(bucket):
                v = p["score"] - 3.0 * used_tonic.get(p["tonic"], 0)
                if best_v is None or v > best_v:
                    best_i, best_v = i, v
            p = bucket.pop(best_i)
            chosen.append(p)
            used_ids.add(p["sid"])
            used_tonic[p["tonic"]] = used_tonic.get(p["tonic"], 0) + 1
            picked += 1
    return chosen


def select_songs(plans: list[dict], n_songs: int, n_chromatic: int = 8) -> list[dict]:
    """Stratified selection with a small quota of windows that contain one or
    two explainable chromatic chords (secondary dominants / modal mixture)."""
    used_tonic: dict[int, int] = {}
    used_ids: set[str] = set()
    diatonic = [p for p in plans if p.get("n_chromatic", 0) == 0]
    chromatic = [p for p in plans if 0 < p.get("n_chromatic", 0) <= 2]
    chosen = _balanced_groups(diatonic, n_songs - n_chromatic, used_tonic, used_ids)
    chosen += _balanced_groups(chromatic, n_songs - len(chosen), used_tonic, used_ids)
    if len(chosen) < n_songs:
        rest = sorted([p for p in plans if p["sid"] not in used_ids],
                      key=lambda p: -p["score"])
        for p in rest:
            if len(chosen) >= n_songs:
                break
            chosen.append(p)
            used_ids.add(p["sid"])
    chosen.sort(key=lambda p: p["sid"])
    return chosen[:n_songs]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pop909-root", default=None)
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--songs", type=int, default=48)
    ap.add_argument("--chromatic-songs", type=int, default=8,
                    help="how many of the selected songs should contain 1-2 explainable "
                         "chromatic chords (secondary dominants / modal mixture)")
    ap.add_argument("--melody-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-dur", type=float, default=30.0)
    ap.add_argument("--max-dur", type=float, default=58.0)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N songs")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = discover_root(args.pop909_root)
    print(f"[pop909] root = {root}", flush=True)

    t0 = time.time()
    song_dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and song_files(d)]
    print(f"[pop909] {len(song_dirs)} song directories", flush=True)
    plans = []
    layouts: dict[str, str] = {}
    for d in song_dirs:
        song = load_song(d)
        if song is None:
            continue
        layouts[f"{song.layout['beat'].name}+{song.layout['chord'].name}"] = \
            layouts.get(f"{song.layout['beat'].name}+{song.layout['chord'].name}", 0) + 1
        plan = plan_song(song, args.min_dur, args.max_dur,
                         target=(34 + (int(song.sid) % 5) * 4) if song.sid.isdigit() else 42)
        if plan is None:
            continue
        plans.append({
            "sid": song.sid,
            "dir": str(song.path),
            "key": f"{song.key_name}:{song.mode}",
            "key_label": song.key_name,
            "tonic": song.tonic,
            "mode": song.mode,
            "bpm": round(song.bpm, 2),
            "duration": round(plan["duration"], 2),
            "bars": plan["bars"],
            "start": plan["start"],
            "end": plan["end"],
            "score": plan["score"],
            "segments": plan["segments"],
            "bpb": plan["bpb"],
            "n_chromatic": plan.get("n_chromatic", 0),
            "chromatic": plan.get("chromatic", []),
            "song": song,
        })
    print(f"[plan] {len(plans)} songs have a usable {args.min_dur:.0f}-{args.max_dur:.0f}s window "
          f"({time.time() - t0:.1f}s)", flush=True)
    print(f"[plan] layouts: {layouts}", flush=True)
    if not plans:
        raise SystemExit("no usable windows found")

    selected = select_songs(plans, args.songs, args.chromatic_songs)
    if args.limit:
        selected = selected[:args.limit]
    print(f"[plan] selected {len(selected)} songs", flush=True)

    rng = random.Random(args.seed)
    samples = []
    for si, p in enumerate(selected):
        song, plan = p["song"], p
        if args.plan_only:
            print(f"  {song.sid} {p['key']:>7} {p['bpm']:5.1f}bpm bars={p['bars']:2d} "
                  f"{p['duration']:5.1f}s chords={len(plan['segments']):2d} "
                  f"{' '.join(c.roman for c in plan['segments'])[:90]}")
            continue
        for ti, texture in enumerate(TEXTURES):
            samples.append(build_sample(song, plan, texture, ti, si, rng, out,
                                        args.melody_frac))
        print(f"  [{si + 1}/{len(selected)}] {song.sid} {p['key']} {p['bpm']:.1f}bpm "
              f"{p['duration']:.1f}s -> 3 samples", flush=True)

    if args.plan_only:
        return

    stats = {
        "pop909_root": str(root),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "songs": len(selected),
        "samples": len(samples),
        "total_audio_sec": round(sum(s["duration"] for s in samples), 2),
        "texture_counts": {t: sum(1 for s in samples if s["texture"] == t) for t in TEXTURES},
        "instrument_counts": {},
        "melody_samples": sum(1 for s in samples if s["melody"]),
        "boundaries": sum(s["n_boundaries"] for s in samples),
        "negatives": sum(s["n_negatives"] for s in samples),
        "negative_kinds": {
            "bar": sum(s["n_negatives_bar"] for s in samples),
            "half_bar": sum(s["n_negatives_halfbar"] for s in samples),
            "beat": sum(s["n_negatives_beat"] for s in samples),
            "melody": sum(s["n_negatives_melody"] for s in samples),
        },
        "songs_with_chromatic_chords": sum(1 for p in selected if p["n_chromatic"] > 0),
        "chromatic_chords": sum(s["n_chromatic"] for s in samples),
        "key_counts": {},
        "tempo_bins": {},
        "duration": {"min": round(min(s["duration"] for s in samples), 2),
                     "max": round(max(s["duration"] for s in samples), 2),
                     "mean": round(statistics.mean([s["duration"] for s in samples]), 2)},
    }
    for s in samples:
        stats["instrument_counts"][s["instrument"]] = stats["instrument_counts"].get(s["instrument"], 0) + 1
        stats["key_counts"][s["key"]] = stats["key_counts"].get(s["key"], 0) + 1
        b = tempo_bin(s["bpm"])
        stats["tempo_bins"][b] = stats["tempo_bins"].get(b, 0) + 1
    manifest = {"meta": stats, "samples": samples}
    (out / "dataset.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    (out / "plan_selected.json").write_text(json.dumps(
        [{k: v for k, v in p.items() if k not in ("segments", "song")} for p in selected],
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"[done] {len(samples)} samples in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
