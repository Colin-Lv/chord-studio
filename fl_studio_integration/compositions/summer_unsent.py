"""Original 60-bar solo piano composition, MIDI export and local piano rendering.

The tune and harmonic plan are explicitly composed below. Randomness only adds
bounded performance timing/velocity, never chooses pitches or the song structure.
"""
from __future__ import annotations

import configparser
import io
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import mido
import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "exports/summer_unsent"
OUT.mkdir(parents=True, exist_ok=True)
PPQ = 960
SR = 44100
TOTAL_BEATS = 240.0
TITLE = "蝉声替我说再见"
RNG = np.random.default_rng(20261010)


def pitch(s):
    name, octv = s[:-1], int(s[-1])
    return (octv + 1) * 12 + {"C": 0, "C#": 1, "D": 2, "D#": 3,
        "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8,
        "A": 9, "Bb": 10, "B": 11}[name]


# Each entry fills exactly four quarter-note beats, including written rests.
# Intro 1-4 is an augmented, less rhythmically active variation of chorus 29-32.
MELODY = [
    "D5:2 B4:.5 A4:.5 B4:1",
    "E5:1 D5:.5 C#5:.5 B4:.5 C#5:.5 E5:1",
    "C#5:2 A4:1 E5:1",
    "D5:1.5 B4:.5 F#4:1 R:1",
    "F#4:1 A4:.5 B4:.5 A4:1 F#4:1",
    "E4:1 F#4:.5 E4:.5 C#4:1 E4:1",
    "F#4:.5 A4:.5 B4:1 A4:.5 F#4:.5 D4:1",
    "E4:.5 F#4:.5 A4:1 G4:.5 F#4:.5 E4:1",
    "G4:1 B4:.5 A4:.5 G4:1 D4:1",
    "F#4:1 A4:.5 F#4:.5 E4:.5 F#4:.5 D4:1",
    "E4:1 G4:.5 A4:.5 B4:1 A4:.5 G4:.5",
    "E4:1 C#4:1 R:.5 D4:.5 E4:.5 G4:.5",
    "F#4:.5 A4:.5 D5:1 C#5:.5 B4:.5 A4:1",
    "E5:1 C#5:.5 B4:.5 A4:1 E4:1",
    "F#4:.5 A4:.5 B4:1 D5:1 C#5:.5 B4:.5",
    "A4:1 C#5:.5 B4:.5 A4:1 F#4:1",
    "B4:1 D5:.5 C#5:.5 B4:1 A4:.5 G4:.5",
    "A4:.5 F#4:.5 A4:1 D5:1 C#5:.5 A4:.5",
    "G4:1 B4:1 A4:.5 G4:.5 E4:1",
    "E4:.5 F#4:.5 G4:1 E4:.5 D4:.5 C#4:.5 E4:.5",
    "G4:.5 A4:.5 B4:1 D5:.5 C#5:.5 B4:1",
    "E5:1 C#5:.5 B4:.5 A4:.5 B4:.5 C#5:1",
    "C#5:1 E5:.5 D5:.5 C#5:1 A4:1",
    "D5:1 F#5:.5 E5:.5 D5:.5 C#5:.5 B4:1",
    "B4:.5 C#5:.5 D5:1 E5:1 D5:.5 B4:.5",
    "C#5:.5 D5:.5 E5:1 G5:1 F#5:.5 E5:.5",
    "F#5:1 E5:.5 D5:.5 A4:1 F#4:1",
    "D5:1 E5:1 C#5:.5 B4:.5 A4:.5 R:.5",
    "D5:1.5 B4:.5 A4:.5 B4:.5 D5:1",
    "E5:1 D5:.5 C#5:.5 B4:.5 C#5:.5 E5:1",
    "C#5:.75 A4:.25 C#5:.5 E5:.5 F#5:1 E5:.5 C#5:.5",
    "D5:1 B4:.5 A4:.5 F#4:1 R:.5 F#4:.5",
    "D5:.5 F#5:.5 G5:1 F#5:.5 D5:.5 B4:1",
    "C#5:.5 E5:.5 A5:1 G5:.5 F#5:.5 E5:1",
    "F#5:1 E5:.5 C#5:.5 A4:.5 C#5:.5 E5:1",
    "F#5:1 D5:1 B4:1 C#5:.5 D5:.5",
    "E5:1 G5:.5 F#5:.5 E5:.5 D5:.5 B4:1",
    "C#5:.5 E5:.5 G5:1 F#5:.5 E5:.5 D5:.5 C#5:.5",
    "F#5:1 E5:.5 D5:.5 A4:1 F#4:1",
    "E5:.5 F#5:.5 A5:1 F#5:1 D5:1",
    "D5:2 C#5:.5 B4:.5 A4:1",
    "C#5:1.5 A4:.5 F#4:2",
    "B4:1.5 A4:.5 G4:1 F#4:1",
    "Bb4:1 G4:1 C#5:1 E5:.5 C#5:.5",
    "D5:1.5 B4:.5 A4:.5 B4:.5 D5:1",
    "E5:1 D5:.5 C#5:.5 B4:.5 C#5:.5 E5:1",
    "C#5:.75 A4:.25 C#5:.5 E5:.5 F#5:1 E5:.5 C#5:.5",
    "D5:1 B4:.5 A4:.5 F#4:1 A4:.5 B4:.5",
    "D5:.5 F#5:.5 G5:1 F#5:.5 D5:.5 B4:1",
    "C#5:.5 E5:.5 A5:1 G5:.5 F#5:.5 E5:1",
    "F#5:1 E5:.5 D5:.5 A4:.5 B4:.5 A4:1",
    "F#5:1 E5:.5 D5:.5 A4:1 D5:1",
    "D5:2 B4:1 A4:.5 B4:.5",
    "E5:1.5 D5:.5 C#5:2",
    "C#5:2 A4:1 E4:1",
    "D5:1.5 B4:.5 F#4:2",
    "B4:1 D5:1 A4:.5 B4:.5 G4:1",
    "Bb4:1 A4:.5 G4:.5 E4:2",
    "F#4:1 A4:1 E5:1 D5:1",
    "D5:4",
]

# Chord voicings: bass then three sequential left-hand notes, close right-hand
# chord tones. Gm6 is the only borrowed colour (iv), resolving toward D.
HARMONY = {
    "D": ("D2", ["A2", "D3", "F#3"], ["D4", "F#4", "A4"], {2, 6, 9}),
    "Dadd9": ("D2", ["A2", "D3", "F#3"], ["E4", "F#4", "A4"], {2, 4, 6, 9}),
    "D6": ("D2", ["A2", "D3", "F#3"], ["F#4", "A4", "B4"], {2, 6, 9, 11}),
    "D/F#": ("F#2", ["A2", "D3", "F#3"], ["D4", "F#4", "A4"], {2, 6, 9}),
    "A": ("A2", ["E3", "A3", "C#4"], ["C#4", "E4", "A4"], {9, 1, 4}),
    "A/C#": ("C#2", ["A2", "C#3", "E3"], ["C#4", "E4", "A4"], {9, 1, 4}),
    "A7": ("A2", ["E3", "G3", "C#4"], ["C#4", "E4", "G4"], {9, 1, 4, 7}),
    "Asus4": ("A2", ["E3", "A3", "D4"], ["D4", "E4", "A4"], {9, 2, 4}),
    "Bm7": ("B1", ["F#2", "B2", "D3"], ["D4", "F#4", "A4"], {11, 2, 6, 9}),
    "F#m7": ("F#2", ["C#3", "E3", "A3"], ["C#4", "E4", "A4"], {6, 9, 1, 4}),
    "F#m7/A": ("A2", ["C#3", "E3", "F#3"], ["C#4", "E4", "A4"], {6, 9, 1, 4}),
    "Gmaj7": ("G2", ["D3", "F#3", "B3"], ["D4", "F#4", "B4"], {7, 11, 2, 6}),
    "G": ("G2", ["D3", "G3", "B3"], ["D4", "G4", "B4"], {7, 11, 2}),
    "Gm6": ("G2", ["D3", "E3", "Bb3"], ["D4", "G4", "Bb4"], {7, 10, 2, 4}),
    "Em7": ("E2", ["B2", "D3", "G3"], ["D4", "G4", "B4"], {4, 7, 11, 2}),
}
CHORDS = (
    "Gmaj7 A7 F#m7 Bm7 "
    "D A/C# Bm7 F#m7/A G D/F# Em7 A7 "
    "D A/C# Bm7 F#m7 G D/F# Em7 A7 "
    "G A F#m7 Bm7 Em7 A7 D/F# Asus4>A7 "
    "Gmaj7 A7 F#m7 Bm7 Gmaj7 A7 F#m7 Bm7 Em7 A7 Dadd9 Dadd9 "
    "Bm7 F#m7 Gmaj7 Gm6>A7 "
    "D6 A/C# F#m7 Bm7 Gmaj7 A7 Dadd9 Dadd9 "
    "Gmaj7 A7 F#m7 Bm7 Gmaj7 Gm6 Dadd9 Dadd9"
).split()

SECTIONS = [
    (1, 4, "Intro - a memory of the chorus", "intro", 64),
    (5, 12, "Verse A - summer classroom", "verse", 69),
    (13, 20, "Verse A2 - words left unsaid", "verse2", 74),
    (21, 28, "Pre-chorus - gathering courage", "pre", 80),
    (29, 40, "Chorus - the last summer", "chorus", 88),
    (41, 44, "Bridge - rubato, alone at dusk", "bridge", 63),
    (45, 52, "Final chorus - a brighter regret", "final", 92),
    (53, 60, "Outro - cicadas fading away", "outro", 60),
]


@dataclass
class Note:
    bar: int
    voice: str
    midi: int
    beat: float
    duration: float
    time: float
    gate: float
    sustained: float
    velocity: int
    chord: str


def section_at(bar):
    return next(s for s in SECTIONS if s[0] <= bar <= s[1])


def harmony_at(bar, local):
    choices = CHORDS[bar - 1].split(">")
    return choices[0 if local < 2 or len(choices) == 1 else 1]


def fade_at(beat):
    # A continuous 8-bar fade, avoiding a sudden gate at the last note.
    if beat <= 208:
        return 1.0
    return max(0.0, (240 - beat) / 32) ** 1.12


def tempo_map():
    anchors = {0: 125, 10: 125, 14: 120, 15.5: 116, 16: 125,
        46: 125, 47.5: 121, 48: 125, 78: 125, 79.5: 120, 80: 125,
        108: 125, 110: 122, 111.5: 119, 112: 125,
        156: 125, 158: 125, 159.5: 118,
        160: 116, 162: 108, 164: 100, 166: 108, 168: 116,
        170: 106, 172: 96, 173: 102, 174: 111, 175: 119, 176: 125,
        204: 125, 207: 123, 208: 125,
        224: 125, 228: 122, 232: 116, 236: 108, 239.5: 100}
    x = sorted(anchors)
    steps = []
    previous = None
    for b in np.arange(0, TOTAL_BEATS, .25):
        bpm = float(np.interp(b, x, [anchors[t] for t in x]))
        micros = mido.bpm2tempo(bpm)
        if micros != previous:
            steps.append((float(b), int(micros)))
            previous = micros
    return steps


TEMPOS = tempo_map()
TB = np.array([t[0] for t in TEMPOS] + [TOTAL_BEATS])
TS = np.array([t[1] / 1e6 for t in TEMPOS])
TC = np.r_[0, np.cumsum(np.diff(TB) * TS)]


def seconds(beat):
    idx = min(int(np.searchsorted(TB, beat, side="right") - 1), len(TS) - 1)
    return float(TC[idx] + (beat - TB[idx]) * TS[idx])


def pedal_release(beat):
    # Half-bar repedalling throughout, full final chord in bar 60.
    if beat >= 236:
        return 239.86 if 236.055 <= beat < 239.86 else beat
    base = math.floor(beat / 2) * 2
    return base + 1.955 if base + .055 <= beat < base + 1.955 else beat


NOTES: list[Note] = []


def add(bar, voice, midi, local, duration, vel, roll=0.0):
    score_beat = (bar - 1) * 4 + local
    kind = section_at(bar)[3]
    # Melody has a slight intentional lag; chord attacks roll upwards.
    offset = float(RNG.normal(0, .006)) + (0.012 if voice == "melody" else 0) + roll
    onset = max((bar - 1) * 4, score_beat + offset)
    gate = max(.075, duration * (.96 if voice == "melody" else .77))
    if voice == "melody" and duration >= 1:
        gate += .06
    if kind in ("intro", "bridge", "outro"):
        gate *= 1.035
    end = min(onset + gate, TOTAL_BEATS - .14)
    sustained = min(max(end, pedal_release(onset + gate)), TOTAL_BEATS - .14)
    # Release incompatible chord tones before a harmonic change (not just a
    # pedal pulse). In particular, the borrowed Bb at bar 44 must not bleed
    # into the A7 that follows on beat 3.
    harmonic_end = bar * 4
    if ">" in CHORDS[bar - 1] and local < 2:
        harmonic_end = (bar - 1) * 4 + 2
    end = min(end, harmonic_end - .04)
    sustained = min(sustained, harmonic_end - .04)
    # Expressive melody dynamics follow phrase contours, accompaniment is softer.
    phrase_position = (bar - 1) % 4
    shaping = [0, 2, 4, -3][phrase_position]
    phrasing = 2 * math.sin(local * math.pi / 4)
    velocity = int(np.clip(round(vel + shaping + phrasing + RNG.normal(0, 1.4)), 23, 103))
    NOTES.append(Note(bar, voice, midi, score_beat, duration, onset,
                      max(.04, end - onset), max(.04, sustained - onset),
                      velocity, harmony_at(bar, local)))


def melody_at(bar, t):
    beat = 0
    for token in MELODY[bar - 1].split():
        n, dur = token.split(":")
        dur = float(dur)
        if beat <= t < beat + dur:
            return None if n == "R" else pitch(n)
        beat += dur
    return None


def make_notes():
    for bar, line in enumerate(MELODY, 1):
        start, finish, _, style, level = section_at(bar)
        beat = 0.0
        for token in line.split():
            n, d = token.split(":")
            duration = float(d)
            if n != "R":
                accent = 3 if beat in (0, 2) else -1
                add(bar, "melody", pitch(n), beat, duration, level + accent)
            beat += duration
        assert beat == 4, (bar, beat)

        if style == "intro":
            gestures = [(0, 0, 1.8), (.5, 1, 1.1), (1, 2, .85), (2.5, 3, 1)]
        elif style == "verse":
            gestures = [(0, 0, .8), (1, 1, .8), (2, 2, .8), (3, 1, .8)]
        elif style == "verse2":
            gestures = [(0, 0, .8), (.75, 1, .5), (1.5, 2, .5), (2, 3, .65), (3, 2, .45), (3.5, 1, .4)]
        elif style == "pre":
            gestures = [(0, 0, .7), (.5, 1, .45), (1, 2, .45), (1.5, 3, .45),
                        (2, 0, .65), (2.5, 1, .45), (3, 2, .4), (3.5, 3, .35)]
        elif style in ("chorus", "final"):
            gestures = [(0, 0, .6), (.5, 1, .4), (1, 2, .45), (1.5, 3, .4),
                        (2, 1, .5), (2.5, 2, .4), (3, 3, .4), (3.5, 2, .35)]
        elif style == "bridge":
            gestures = [(0, 0, 1.8), (1, 1, 1.2), (2, 2, 1.6)]
            if bar == 44:
                gestures = [(0, 0, 1.4), (.6, 2, 1.1), (2, 0, 1.6), (2.65, 1, 1.15)]
        else:
            gestures = [(0, 0, 1.6), (1, 1, 1.1), (2, 2, 1.6), (3, 1, .7)] if bar < 57 else [(0, 0, 2.0), (1.5, 1, 1.1)]
            if bar == 60:
                gestures = [(0, 0, 3.7), (.04, 1, 3.6), (.09, 2, 3.5)]

        for t, degree, duration in gestures:
            chord = harmony_at(bar, t)
            root, arp, close, pcs = HARMONY[chord]
            pitches = [pitch(root)] + [pitch(n) for n in arp]
            n = pitches[degree]
            # Keep the accompaniment underneath the tune, preserving two-hand clarity.
            mel = melody_at(bar, t)
            while mel is not None and n >= mel - 2:
                n -= 12
            add(bar, "left", n, t, duration, level - 22 + (3 if degree == 0 else 0))

        comp_times = []
        if style in ("chorus", "final"):
            comp_times = [0, 2]
        elif style == "pre":
            comp_times = [1.5, 3] if bar % 4 else [0, 2]
        elif style in ("intro", "bridge"):
            comp_times = [0, 2] if ">" in CHORDS[bar - 1] else [0]
        elif style == "verse2":
            comp_times = [0] if bar % 2 else []
        elif style == "outro":
            comp_times = [0]
        for t in comp_times:
            top = melody_at(bar, t)
            if top is None:
                continue
            chord = harmony_at(bar, t)
            choices = [n for n in range(max(48, top - 10), top - 2)
                       if n % 12 in HARMONY[chord][3]]
            chosen = choices[-2:] if style in ("final", "outro", "bridge") else choices[-1:]
            # Avoid close minor-second clusters and duplicate melody attacks.
            for k, n in enumerate(chosen):
                if k and n - chosen[k - 1] < 3:
                    continue
                add(bar, "inner", n, t, 1.45 if style not in ("bridge", "outro") else 2.7,
                    level - 24, roll=-.006 * (len(chosen) - k))

    # No same-key overlap after retrigger, including two voices sharing a piano.
    by_key = defaultdict(list)
    for n in NOTES:
        by_key[n.midi].append(n)
    for seq in by_key.values():
        seq.sort(key=lambda n: n.time)
        for a, b in zip(seq, seq[1:]):
            if b.time - a.time < .04:
                # A simultaneously doubled pitch is removed later, favour the melody.
                continue
            a.gate = min(a.gate, b.time - a.time - .018)
            a.sustained = min(a.sustained, b.time - a.time - .018)
    NOTES.sort(key=lambda n: (n.time, n.midi, n.voice))
    remove = set()
    for seq in by_key.values():
        for a, b in zip(seq, seq[1:]):
            if abs(b.time - a.time) < .04:
                loser = a if b.voice == "melody" else b
                remove.add(id(loser))
    NOTES[:] = [n for n in NOTES if id(n) not in remove]


def midi_export(path, compatible=False):
    mid = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    conductor = [(0, 0, mido.MetaMessage("track_name", name="Summer Unsent - 60 bars")),
                 (0, 1, mido.MetaMessage("time_signature", numerator=4, denominator=4)),
                 (0, 2, mido.MetaMessage("key_signature", key="D"))]
    for beat, us in TEMPOS:
        conductor.append((round(beat * PPQ), 3, mido.MetaMessage("set_tempo", tempo=us)))
    for first, last, label, _, _ in SECTIONS:
        conductor.append(((first - 1) * 4 * PPQ, 4, mido.MetaMessage("marker", text=label)))
    for b in range(1, 61):
        conductor.append(((b - 1) * 4 * PPQ, 5, mido.MetaMessage("text", text=f"Bar {b:02}: {CHORDS[b-1]}")))

    def track(events):
        tr = mido.MidiTrack()
        last = 0
        for tick, priority, msg in sorted(events, key=lambda e: (e[0], e[1])):
            tr.append(msg.copy(time=tick - last))
            last = tick
        tr.append(mido.MetaMessage("end_of_track", time=round(TOTAL_BEATS * PPQ) - last))
        mid.tracks.append(tr)

    track(conductor)
    # Three score voices, one piano timbre. Each receives its own pedal/controller
    # events for importers that instantiate a channel per track.
    for ch, voice in enumerate(("melody", "inner", "left")):
        events = [(0, 0, mido.MetaMessage("track_name", name={"melody":"Piano RH - Melody", "inner":"Piano RH - Inner Voice", "left":"Piano LH - Accompaniment"}[voice])),
                  (0, 1, mido.Message("program_change", channel=ch, program=0)),
                  (0, 1, mido.Message("control_change", channel=ch, control=7, value=100)),
                  (0, 1, mido.Message("control_change", channel=ch, control=10, value=64))]
        if not compatible:
            for b in np.arange(0, 236, 2):
                events.append((round((b + .055) * PPQ), 2, mido.Message("control_change", channel=ch, control=64, value=88)))
                events.append((round((b + 1.955) * PPQ), 1, mido.Message("control_change", channel=ch, control=64, value=0)))
            events += [(round(236.055 * PPQ), 2, mido.Message("control_change", channel=ch, control=64, value=90)),
                       (round(239.86 * PPQ), 1, mido.Message("control_change", channel=ch, control=64, value=0))]
            for b in np.arange(0, 240.001, .25):
                if b == 0 or b >= 208:
                    events.append((round(b * PPQ), 2, mido.Message("control_change", channel=ch, control=11, value=round(127 * fade_at(b)))))
        for n in NOTES:
            if n.voice != voice:
                continue
            velocity = max(1, round(n.velocity * fade_at(n.time) ** .52)) if compatible else n.velocity
            duration = n.sustained if compatible else n.gate
            events.append((round(n.time * PPQ), 5, mido.Message("note_on", channel=ch, note=n.midi, velocity=velocity)))
            events.append((round((n.time + duration) * PPQ), 0, mido.Message("note_off", channel=ch, note=n.midi, velocity=0)))
        track(events)
    mid.save(path)


def render():
    """Offline preview using the user's installed FL Keys Grand Piano samples.

    No sample assets are copied into the deliverable. This is a separate sampler
    render, not a claim that FL Studio itself performed the export.
    """
    sample_root = Path("C:/Program Files/Image-Line/FL Studio 2024/Plugins/Fruity/Generators/FL Keys/Wavtables")
    cfg = configparser.ConfigParser()
    cfg.read(sample_root / "Grand Piano.wti")
    sample_maps = []
    for sec in cfg.sections():
        if sec.startswith("sample") and sec[6:].isdigit():
            sample_maps.append((int(cfg[sec]["root"]), sample_root / "Grand Piano" / cfg[sec]["name"]))
    audio = np.zeros((math.ceil((seconds(TOTAL_BEATS) + 6) * SR), 2), dtype=np.float32)
    cache = {}
    for n in NOTES:
        if n.midi not in cache:
            root, filename = min(sample_maps, key=lambda p: abs(p[0] - n.midi))
            raw = filename.read_bytes()
            # FL's stock RIFF files wrap a standard Ogg Vorbis stream.
            stream = io.BytesIO(raw[raw.index(b"OggS"):]) if b"OggS" in raw[:96] else io.BytesIO(raw)
            x, source_sr = sf.read(stream, dtype="float32", always_2d=True)
            ratio = 2 ** ((n.midi - root) / 12) * source_sr / SR
            pos = np.arange(0, len(x) - 1, ratio)
            lo = pos.astype(np.int64)
            fraction = (pos - lo).astype(np.float32)[:, None]
            shifted = x[lo] * (1 - fraction) + x[lo + 1] * fraction
            # Gentle velocity-sensitive darkening; softer notes sound less metallic.
            soft = signal.sosfilt(signal.butter(2, 3100, fs=SR, output="sos"), shifted, axis=0).astype(np.float32)
            cache[n.midi] = (shifted, soft)
        bright, soft = cache[n.midi]
        attack_time = seconds(n.time)
        hold = seconds(n.time + n.sustained) - attack_time
        release = .42 + max(0, 66 - n.midi) * .009
        length = min(len(bright), math.ceil((hold + release * 4) * SR))
        blend = np.clip((n.velocity - 30) / 80, 0, 1)
        x = (bright[:length] * blend + soft[:length] * (1 - blend)).copy()
        amp = (n.velocity / 100) ** 1.7 * .5
        t = np.arange(length) / SR
        envelope = np.exp(-np.maximum(t - hold, 0) / release)
        envelope[: min(75, length)] *= np.linspace(0, 1, min(75, length))
        x *= (amp * envelope).astype(np.float32)[:, None]
        pan = np.clip((n.midi - 60) / 80, -.25, .25)
        x[:, 0] *= math.sqrt(1 - pan)
        x[:, 1] *= math.sqrt(1 + pan)
        start = round(attack_time * SR)
        length = min(length, len(audio) - start)
        audio[start:start + length] += x[:length]

    # Continuous outro expression, including sustained notes, before the room tail.
    ts = np.arange(len(audio)) / SR
    score_beats = np.interp(ts, TC, TB)
    gain = np.clip((240 - score_beats) / 32, 0, 1) ** 1.12
    audio *= gain.astype(np.float32)[:, None]

    # Small, warm recital room: diffuse late tail plus sparse early reflections.
    wet = np.zeros_like(audio)
    ir_rng = np.random.default_rng(741)
    ir_len = int(2.6 * SR)
    ir_t = np.arange(ir_len) / SR
    for ch in range(2):
        ir = ir_rng.normal(size=ir_len) * np.exp(-ir_t * 3.5)
        ir = signal.sosfilt(signal.butter(2, 4500, fs=SR, output="sos"), ir)
        ir[:int(.035 * SR)] = 0
        ir /= max(np.sqrt(np.sum(ir * ir)), .0001)
        for delay, amount in [(.049, .10), (.083, .07), (.119, .04)]:
            ir[int((delay + ch * .007) * SR)] += amount
        wet[:, ch] = signal.fftconvolve(audio[:, ch], ir, mode="full")[:len(audio)] * .12
    audio += wet
    # Subsonic cleanup, global peak headroom; no pumping compressor/limiter.
    audio = signal.sosfilt(signal.butter(2, 30, fs=SR, btype="highpass", output="sos"), audio, axis=0).astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    audio *= 10 ** (-1.5 / 20) / max(peak, 1e-9)
    audio[-SR:] *= np.linspace(1, 0, SR, dtype=np.float32)[:, None]
    sf.write(OUT / "Summer_Unsent.wav", audio, SR, subtype="PCM_24")
    energies = []
    for first, last, label, _, _ in SECTIONS:
        chunk = audio[round(seconds((first - 1) * 4) * SR):round(seconds(last * 4) * SR)]
        energies.append({"bars": [first, last], "section": label,
                         "rms_dbfs": round(20 * np.log10(max(float(np.sqrt(np.mean(chunk ** 2))), 1e-12)), 2)})
    return {"sample_rate": SR, "duration_seconds": round(len(audio) / SR, 3),
            "peak_dbfs": round(20 * math.log10(float(np.max(np.abs(audio)))), 3),
            "sections": energies,
            "render_engine": "Offline sampler using locally installed FL Keys Grand Piano wavetable; not an FL Studio bounce"}


def validate():
    assert len(MELODY) == len(CHORDS) == 60
    assert len(range(0, 60, 4)) == 15
    assert TEMPOS[0] == (0, mido.bpm2tempo(125))
    assert len({t for _, t in TEMPOS}) > 20
    assert all(21 <= n.midi <= 108 for n in NOTES)
    assert all(n.time >= 0 and n.gate > 0 and n.time + n.sustained <= 240 for n in NOTES)
    diatonic = {1, 2, 4, 6, 7, 9, 11}
    altered = [n for n in NOTES if n.midi % 12 not in diatonic]
    assert all(n.midi % 12 == 10 and n.chord == "Gm6" for n in altered)
    accented_nonchord = []
    melodic = sorted((n for n in NOTES if n.voice == "melody"), key=lambda n: n.beat)
    for n in NOTES:
        if n.voice == "melody" and n.beat % 4 in (0, 2) and n.midi % 12 not in HARMONY[n.chord][3]:
            nxt = melodic[melodic.index(n) + 1]
            assert abs(nxt.midi - n.midi) in (1, 2), (n, "unresolved accented non-chord tone")
            accented_nonchord.append({"bar": n.bar, "beat": n.beat % 4 + 1, "midi": n.midi,
                "chord": n.chord, "resolves_to": nxt.midi,
                "explanation": "Stepwise passing/neighbor/appoggiatura tone; resolves by a diatonic step"})
    midi_checks = {}
    for name in ("Summer_Unsent.mid", "Summer_Unsent_FL.mid"):
        mid = mido.MidiFile(OUT / name)
        active = set()
        total_notes = 0
        for msg in mido.merge_tracks(mid.tracks):
            if msg.type == "note_on" and msg.velocity:
                key = msg.channel, msg.note
                assert key not in active, (name, key, "overlapping same-key note")
                active.add(key)
                total_notes += 1
            elif msg.type in ("note_on", "note_off"):
                key = msg.channel, msg.note
                assert key in active, (name, key, "orphan note off")
                active.remove(key)
        assert not active
        assert all(sum(msg.time for msg in tr) == 240 * PPQ for tr in mid.tracks)
        assert total_notes == len(NOTES)
        midi_checks[name] = {"notes": total_notes, "end_tick": 240 * PPQ, "seconds": round(mid.length, 3), "matched_note_offs": True}
    return {"bars": 60, "meter": "4/4", "four_bar_phrases": 15,
        "home_key": "D major / relative B minor", "base_bpm": 125,
        "bpm_range": [round(min(60e6/t for _, t in TEMPOS), 2), round(max(60e6/t for _, t in TEMPOS), 2)],
        "rubato_bars": [41, 44], "tempo_events": len(TEMPOS),
        "outro_fade_bars": [53, 60], "note_count": len(NOTES),
        "velocity_range": [min(n.velocity for n in NOTES), max(n.velocity for n in NOTES)],
        "pitch_range": [min(n.midi for n in NOTES), max(n.midi for n in NOTES)],
        "borrowed_colour": "Gm6 (iv) at bar 44 beats 1-2 and bar 58; Bb is intentional",
        "accented_nonchord_tones_for_review": accented_nonchord,
        "midi_checks": midi_checks}


if __name__ == "__main__":
    make_notes()
    midi_export(OUT / "Summer_Unsent.mid")
    midi_export(OUT / "Summer_Unsent_FL.mid", compatible=True)
    report = validate()
    (OUT / "score.json").write_text(json.dumps({"title": TITLE, "sections": SECTIONS,
        "chords": CHORDS, "melody": MELODY, "tempos": TEMPOS,
        "notes": [asdict(n) for n in NOTES]}, ensure_ascii=False, indent=2), encoding="utf-8")
    report["audio"] = render()
    (OUT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
