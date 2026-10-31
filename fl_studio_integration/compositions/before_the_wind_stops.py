"""Melody-first J-pop instrumental. Original, explicitly written phrases.

Offline demo: locally installed FL Keys / FPC samples, synthesized finger bass.
No GUI automation, generative-music service, or mutation of the open FL project.
"""
from __future__ import annotations

import configparser
import io
import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import mido
import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "exports/before_the_wind_stops"
OUT.mkdir(parents=True, exist_ok=True)
TITLE = "在风停下以前"
SR, PPQ, BARS = 44100, 960, 44
END = BARS * 4
FL = Path("C:/Program Files/Image-Line/FL Studio 2024")
RNG = np.random.default_rng(314159)


def pitch(s):
    return 12 * (int(s[-1]) + 1) + {"C": 0, "C#": 1, "D": 2,
        "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8,
        "A": 9, "Bb": 10, "B": 11}[s[:-1]]


# The chorus is composed first. A two-bar a/b/a'/c design; the first falling
# E-D-B gesture is answered by the short-short-long A-B-D gesture.
# R is a real melodic rest, not a request for accompaniment to fill every gap.
CHORUS = [
    "E5:1.5 D5:.5 B4:1 R:.5 B4:.5",
    "A4:.5 B4:.5 D5:2 R:1",
    "D5:.5 D5:.5 F#5:1.5 E5:.5 D5:1",
    "B4:2 R:1 B4:.5 D5:.5",
    "E5:1.5 D5:.5 G5:1.5 R:.5",
    "F#5:1.5 E5:.5 D5:1 R:1",
    "B4:.5 A4:.5 G4:1 B4:1 A4:.5 B4:.5",
    "G4:2.5 R:1.5",
]
INTRO = [
    "E5:1.5 D5:.5 B4:1 R:1",
    "A4:.5 B4:.5 D5:2 R:1",
    "E5:1 D5:.5 B4:.5 A4:1 R:1",
    "G4:1.5 R:2 D4:.5",
]
VERSE = [
    "G4:.5 G4:.5 B4:1.5 A4:.5 R:1",
    "G4:.5 F#4:1.5 R:2",
    "G4:.5 G4:.5 A4:.5 B4:1.5 R:1",
    "D4:.5 F#4:.5 F#4:1.5 R:1.5",
    "E4:.5 G4:.5 G4:1 A4:.5 G4:.5 R:1",
    "D4:.5 G4:.5 A4:1 B4:1 R:1",
    "A4:1.5 G4:.5 E4:1 R:.5 F#4:.5",
    "G4:1 F#4:1 D4:.5 R:1.5",
]
PRE = [
    "E4:.5 G4:.5 A4:1 B4:.5 C5:1 R:.5",
    "F#4:.5 A4:.5 B4:1 D5:1 R:1",
    "B4:.5 D5:1 B4:.5 A4:1 R:1",
    "G4:1 A4:1 R:1 B4:.5 D5:.5",
]
BRIDGE = [
    "B4:1.5 A4:.5 G4:1 R:1",
    "F#4:1 D4:.5 F#4:.5 A4:1 R:1",
    "G4:.5 A4:.5 B4:1 E5:1.5 R:.5",
    "Eb5:1 D5:1 C5:1 R:.5 D5:.5",
]
FINAL = CHORUS[:4] + [
    "E5:1 D5:.5 E5:.5 G5:1 A5:1",
    "~A5:1.5 G5:.5 F#5:1 D5:.5 R:.5",
    "E5:1 D5:1 B4:1 A4:.5 B4:.5",
    "G4:2.5 R:1.5",
]
OUTRO = [
    "E5:1.5 D5:.5 B4:1 R:1",
    "A4:.5 B4:.5 D5:1.5 R:1.5",
    "Eb5:1.5 D5:.5 C5:1 R:1",
    "B4:1 A4:.5 G4:2.5",
]
MELODY = INTRO + VERSE + PRE + CHORUS + BRIDGE + CHORUS[:4] + FINAL + OUTRO

# Chords support a previously written melody. Voicings deliberately stay below
# C4; the melody has its own register. No parallel chord-tone melody generator.
HARMONY = {
    "G": ("G2", "D3 G3 B3"),
    "G/B": ("B2", "D3 G3 B3"),
    "D/F#": ("F#2", "D3 F#3 A3"),
    "D": ("D2", "F#3 A3 D4"),
    "Dsus": ("D2", "G3 A3 D4"),
    "Em": ("E2", "E3 G3 B3"),
    "Bm": ("B1", "F#3 A3 D4"),
    "Cmaj7": ("C2", "E3 G3 B3"),
    "C": ("C2", "E3 G3 C4"),
    "Am7": ("A2", "E3 G3 C4"),
    "Cm": ("C2", "Eb3 G3 C4"),
}
HOOK_CHORDS = ["Cmaj7", "D", "Bm", "Em", "C", "D", "G", "G"]
CHORDS = (["Cmaj7", "D", "Em", "G"]
    + ["G", "D/F#", "Em", "Bm", "Cmaj7", "G/B", "Am7", "D"]
    + ["C", "D", "Bm", "Dsus"] + HOOK_CHORDS
    + ["Em", "Bm", "Cmaj7", "Cm"] + HOOK_CHORDS[:4]
    + HOOK_CHORDS + ["Cmaj7", "D", "Cm", "G"])

SECTIONS = [
    (1, 4, "Intro", "前奏：主题预示", "vary", 73),
    (5, 12, "Verse", "主歌：说话般的短句", "new", 80),
    (13, 16, "Pre", "预副歌：向上，随后留白", "fragment+sequence", 86),
    (17, 24, "Chorus", "副歌：完整主题", "repeat", 95),
    (25, 28, "Bridge", "间奏：下行与小调色彩", "vary", 78),
    (29, 32, "Thin", "落ちサビ：撤去节奏组", "recap", 82),
    (33, 40, "Final", "末副歌：主题回归与换尾", "recap+vary", 98),
    (41, 44, "Outro", "尾声：主题逐渐收拢", "fragment", 70),
]
TEMPO_POINTS = [(0, 108), (16, 112), (60, 110), (62, 104), (63, 112),
    (64, 114), (96, 104), (104, 100), (108, 97), (111, 94),
    (112, 108), (128, 114), (160, 108), (164, 104),
    (168, 98), (170, 94), (172, 90), (174, 82)]
TB = np.array([x[0] for x in TEMPO_POINTS] + [END], float)
BPMS = np.array([x[1] for x in TEMPO_POINTS], float)
TC = np.r_[0, np.cumsum(np.diff(TB) * 60 / BPMS)]


def seconds(beat):
    return float(np.interp(beat, TB, TC))


def section(bar):
    return next(s for s in SECTIONS if s[0] <= bar <= s[1])


@dataclass
class Note:
    voice: str
    beat: float
    duration: float
    midi: int
    velocity: int
    performed: float = 0
    gate: float = 0


NOTES = []


def add(voice, bar, offset, p, duration, velocity):
    NOTES.append(Note(voice, (bar - 1) * 4 + offset, duration,
                      pitch(p) if isinstance(p, str) else p, int(velocity)))


def melody():
    for bar, line in enumerate(MELODY, 1):
        t = 0
        level = section(bar)[5]
        for token in line.split():
            name, value = token.split(":")
            d = float(value)
            if name.startswith("~"):
                assert NOTES[-1].midi == pitch(name[1:])
                assert NOTES[-1].beat + NOTES[-1].duration == (bar - 1) * 4 + t
                NOTES[-1].duration += d
            elif name != "R":
                # Direction, articulation and phrase position shape accents.
                accent = 4 if d >= 1.5 else (-5 if d == .5 else 0)
                add("lead", bar, t, name, d, level + accent)
            t += d
        assert t == 4, (bar, t)


def accompaniment():
    for bar, chord in enumerate(CHORDS, 1):
        kind = section(bar)[2]
        base, spelling = HARMONY[chord]
        root = pitch(base)
        voicing = [pitch(x) for x in spelling.split()]
        # Drop any competing upper note into the inner register. D4 must not
        # collide with the verse's D4; all chord tops stay at or below B3.
        voicing = sorted(n - 12 if n >= 60 else n for n in voicing)
        level = {"Intro": 47, "Verse": 49, "Pre": 54, "Chorus": 57,
                 "Bridge": 46, "Thin": 47, "Final": 61, "Outro": 43}[kind]
        if kind in ("Intro", "Bridge", "Thin", "Outro"):
            # One quiet, slightly spread chord; its attack does not follow
            # every melodic note. Occasional second attack only in phrase gaps.
            attacks = [(0, 2.7)]
            if kind == "Thin" and bar % 2 == 1:
                attacks = [(0, 2.2), (2.5, 1.05)]
            if kind == "Outro":
                level -= (bar - 41) * 4
        elif kind == "Verse":
            attacks = [(0, 1.55), (2.5, 1.15)] if bar % 2 else [(0, 2.3)]
        elif kind == "Pre":
            attacks = [(0, 1.3), (1.5, .7), (3, .65)]
            if bar == 16:
                attacks = [(0, 1.55)]  # purposeful break before the hook pickup
        else:
            # A syncopated two/three-hit comp, never a continuous arpeggio.
            attacks = [(0, 1.2), (1.5, .9), (3, .65)] if bar % 2 else [(0, 1.65), (2.5, .85)]
            if bar in (24, 40):
                attacks = [(0, 2.3)]
        for t, duration in attacks:
            for i, note in enumerate(voicing):
                add("piano", bar, t + i * .013, note, duration,
                    level + (2 if i == len(voicing) - 1 else -2))

        if kind in ("Intro", "Bridge", "Thin", "Outro"):
            # Piano bass supports the exposed sections, not an electronic drone.
            add("piano", bar, 0, root + (12 if root < 36 else 0), 2.8, level + 4)
        else:
            bass_root = root + 12 if root < 35 else root
            if kind == "Verse":
                rhythm = [(0, 2.1), (2.5, 1.1)] if bar % 2 else [(0, 3.1)]
            elif kind == "Pre":
                rhythm = [(0, 1.4), (1.5, 1), (3, .7)] if bar != 16 else [(0, 1.6)]
            else:
                rhythm = [(0, 1.15), (1.5, .75), (2.5, .65), (3.5, .4)]
                if bar in (24, 40):
                    rhythm = [(0, 2.25)]
            for i, (t, d) in enumerate(rhythm):
                # Stay on the root except the last weak-beat fifth. No busy fills.
                p = bass_root + (7 if i == 3 else 0)
                add("bass", bar, t, p, d, 78 if kind in ("Chorus", "Final") else 66)


DRUM_FILES = {
    36: "Kicks/FPC 2 Kick.wav",
    38: "Snares/FPC Snare 4.wav",
    37: "Snares/FPC Rim.wav",
    42: "Hats/Grv CH 01.wav",
    46: "Hats/Grv OH 01.wav",
    49: "Cymbals/Grv Crash 01.wav",
}


def drums():
    for bar in range(9, 41):
        kind = section(bar)[2]
        if kind not in ("Verse", "Pre", "Chorus", "Final"):
            continue
        full = kind in ("Chorus", "Final")
        kicks = [0, 1.5, 2.5] if full and bar % 2 else ([0, 2] if full else [0])
        for t in kicks:
            add("drums", bar, t, 36, .12, 76 if full else 56)
        for t in (1, 3):
            if bar == 16 and t >= 2:
                continue
            add("drums", bar, t, 38 if full else 37, .12, 67 if full else 46)
        hats = np.arange(0, 4, .5) if full else np.arange(.5, 4, 1)
        for t in hats:
            if (bar == 16 and t >= 2) or (bar in (24, 40) and t >= 2.5):
                continue
            v = (42 if t % 1 == 0 else 30) if full else 26
            add("drums", bar, float(t), 42, .12, v)
        if bar in (17, 33):
            add("drums", bar, 0, 49, .2, 42)
        if bar in (20, 36):
            add("drums", bar, 3.5, 46, .2, 36)


def perform():
    for n in NOTES:
        # Tiny deviations articulate a steady band pulse; no large random drift.
        offset = {"lead": .014, "piano": -.006, "bass": -.008, "drums": 0}[n.voice]
        if n.voice == "drums" and n.midi in (38, 37):
            offset = .018
        if n.voice == "lead" and n.duration >= 1.5:
            offset += .01
        n.performed = max(0, n.beat + offset + float(RNG.normal(0, .006)))
        n.velocity = int(np.clip(n.velocity + RNG.integers(-2, 3), 1, 115))
        factor = {"lead": .95, "piano": 1, "bass": .94, "drums": 1}[n.voice]
        n.gate = min(n.duration * factor, END - n.performed)
    by_key = defaultdict(list)
    for n in NOTES:
        by_key[n.voice, n.midi].append(n)
    for ns in by_key.values():
        ns.sort(key=lambda n: n.performed)
        for a, b in zip(ns, ns[1:]):
            a.gate = min(a.gate, b.performed - a.performed - .012)
    NOTES.sort(key=lambda n: (n.performed, n.midi))


CHANNELS = {"lead": 0, "piano": 1, "bass": 2, "drums": 9}


def midi_export(name, voices, expressive=True):
    mid = mido.MidiFile(type=1, ticks_per_beat=PPQ)

    def append_track(events):
        track = mido.MidiTrack()
        last = 0
        for tick, order, msg in sorted(events, key=lambda e: (e[0], e[1])):
            track.append(msg.copy(time=tick - last))
            last = tick
        track.append(mido.MetaMessage("end_of_track", time=END * PPQ - last))
        mid.tracks.append(track)

    conductor = [(0, 0, mido.MetaMessage("track_name", name="Before The Wind Stops")),
        (0, 1, mido.MetaMessage("time_signature", numerator=4, denominator=4)),
        (0, 2, mido.MetaMessage("key_signature", key="G"))]
    for beat, bpm in TEMPO_POINTS:
        conductor.append((round(beat * PPQ), 3, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm))))
    for first, last, name_, *_ in SECTIONS:
        conductor.append(((first - 1) * 4 * PPQ, 4, mido.MetaMessage("marker", text=name_)))
    for i, ch in enumerate(CHORDS):
        conductor.append((i * 4 * PPQ, 5, mido.MetaMessage("text", text=f"{i+1}: {ch}")))
    append_track(conductor)
    for voice in voices:
        ch = CHANNELS[voice]
        events = [(0, 0, mido.MetaMessage("track_name", name=voice)),
            (0, 1, mido.Message("program_change", channel=ch, program=33 if voice == "bass" else 0))]
        for n in NOTES:
            if n.voice != voice:
                continue
            t = n.performed if expressive else n.beat
            d = n.gate if expressive else n.duration
            events += [(round(t * PPQ), 5, mido.Message("note_on", channel=ch, note=n.midi, velocity=n.velocity)),
                (round((t + d) * PPQ), 0, mido.Message("note_off", channel=ch, note=n.midi))]
        append_track(events)
    mid.save(OUT / name)


def read_audio(path):
    raw = path.read_bytes()
    stream = io.BytesIO(raw[raw.index(b"OggS"):]) if b"OggS" in raw[:96] else io.BytesIO(raw)
    x, rate = sf.read(stream, dtype="float32", always_2d=True)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    if rate != SR:
        divisor = math.gcd(rate, SR)
        x = signal.resample_poly(x, SR // divisor, rate // divisor, axis=0)
    return x.astype(np.float32)


def render():
    samples_root = FL / "Plugins/Fruity/Generators/FL Keys/Wavtables"
    cfg = configparser.ConfigParser()
    cfg.read(samples_root / "Grand Piano.wti")
    mapping = [(int(cfg[s]["root"]), samples_root / "Grand Piano" / cfg[s]["name"])
        for s in cfg.sections() if s.startswith("sample") and s[6:].isdigit()]
    length = math.ceil((seconds(END) + 4) * SR)
    stems = {v: np.zeros((length, 2), np.float32) for v in CHANNELS}
    piano_cache, drum_cache = {}, {}
    for n in NOTES:
        start = round(seconds(n.performed) * SR)
        hold = seconds(n.performed + n.gate) - seconds(n.performed)
        if n.voice in ("lead", "piano"):
            if n.midi not in piano_cache:
                root, file = min(mapping, key=lambda p: abs(p[0] - n.midi))
                raw = read_audio(file)
                pos = np.arange(0, len(raw) - 1, 2 ** ((n.midi - root) / 12))
                low = pos.astype(int)
                frac = (pos - low).astype(np.float32)[:, None]
                bright = raw[low] * (1 - frac) + raw[low + 1] * frac
                dark = signal.sosfilt(signal.butter(2, 2600, fs=SR, output="sos"), bright, axis=0).astype(np.float32)
                piano_cache[n.midi] = bright, dark
            bright, dark = piano_cache[n.midi]
            release = .13 if n.voice == "lead" else .30
            if n.beat >= 172:
                release = .65
            count = min(len(bright), math.ceil((hold + release * 6) * SR))
            mix = np.clip((n.velocity - 32) / 78, .08, .95)
            x = (bright[:count] * mix + dark[:count] * (1 - mix)).copy()
            t = np.arange(count) / SR
            env = np.exp(-np.maximum(0, t - hold) / release)
            env[:64] *= np.linspace(0, 1, 64)
            amp = (n.velocity / 100) ** 1.6 * (.80 if n.voice == "lead" else .35)
            x *= (env * amp).astype(np.float32)[:, None]
            # Lead close to centre; accompaniment offset slightly left.
            pan = .035 if n.voice == "lead" else -.16
            x *= np.array([math.sqrt(1 - pan), math.sqrt(1 + pan)], np.float32)
        elif n.voice == "bass":
            t = np.arange(math.ceil((hold + .16) * SR)) / SR
            f = 440 * 2 ** ((n.midi - 69) / 12)
            # Rounded plucked electric-bass approximation, with no busy upper line.
            mono = (np.sin(2 * np.pi * f * t) + .30 * np.sin(4 * np.pi * f * t) * np.exp(-t * 3)
                + .13 * np.sin(6 * np.pi * f * t) * np.exp(-t * 6))
            env = (1 - np.exp(-t / .003)) * np.exp(-t * 1.1) * np.exp(-np.maximum(t - hold, 0) / .033)
            mono *= env * .09 * (n.velocity / 95) ** 1.4
            x = np.repeat(mono.astype(np.float32)[:, None], 2, axis=1)
        else:
            if n.midi not in drum_cache:
                raw = read_audio(FL / "Data/Patches/Packs/Drums" / DRUM_FILES[n.midi])
                drum_cache[n.midi] = raw / max(.1, float(np.max(np.abs(raw))))
            gain = {36: .18, 38: .13, 37: .10, 42: .08, 46: .07, 49: .09}[n.midi]
            x = drum_cache[n.midi] * gain * (n.velocity / 85) ** 1.5
        count = min(len(x), length - start)
        stems[n.voice][start:start + count] += x[:count]

    # Sparse early reflections, a quiet short room. The lead remains dry enough
    # to hear the rests and note releases; no wall of diffuse pedal/reverb.
    for voice in ("lead", "piano"):
        dry = stems[voice].copy()
        wet_scale = .065 if voice == "lead" else .10
        for delay, attenuation in ((.041, 1), (.079, .7), (.137, .45), (.221, .22)):
            shift = round(delay * SR)
            stems[voice][shift:] += dry[:-shift, ::-1] * wet_scale * attenuation
    for voice in stems:
        stems[voice] = signal.sosfilt(signal.butter(2, 32, fs=SR, btype="highpass", output="sos"),
                                      stems[voice], axis=0).astype(np.float32)
        stems[voice][-SR:] *= np.linspace(1, 0, SR, dtype=np.float32)[:, None]

    mix = sum(stems.values())
    master_gain = 10 ** (-1.5 / 20) / float(np.max(np.abs(mix)))
    for v in stems:
        stems[v] *= master_gain
    mix *= master_gain
    sf.write(OUT / "Before_The_Wind_Stops.wav", mix, SR, subtype="PCM_24")
    sf.write(OUT / "Melody_Only.wav", stems["lead"], SR, subtype="PCM_24")
    # A short immediate comparison avoids making the listener search for the hook.
    a, b = round(seconds(63) * SR), round(seconds(96) * SR)
    excerpt = mix[a:b].copy()
    excerpt[-int(.16 * SR):] *= np.linspace(1, 0, int(.16 * SR), dtype=np.float32)[:, None]
    sf.write(OUT / "Chorus_Preview.wav", excerpt, SR, subtype="PCM_24")
    energies = []
    for first, last, name, *_ in SECTIONS:
        lo, hi = round(seconds((first - 1) * 4) * SR), round(seconds(last * 4) * SR)
        level = lambda x: round(20 * np.log10(max(1e-10, float(np.sqrt(np.mean(x[lo:hi] ** 2))))), 2)
        energies.append({"section": name, "start_seconds": round(seconds((first - 1) * 4), 2),
            "mix_rms_dbfs": level(mix), "lead_rms_dbfs": level(stems["lead"]),
            "piano_rms_dbfs": level(stems["piano"]),
            "bass_rms_dbfs": level(stems["bass"]), "drums_rms_dbfs": level(stems["drums"])})
    return {"duration_seconds": round(length / SR, 3), "peak_dbfs": -1.5,
        "sections": energies, "render": "Offline local sampler; not an FL Studio bounce"}


def validate():
    assert len(MELODY) == len(CHORDS) == BARS
    assert all(n.gate > 0 and n.performed + n.gate <= END + 1e-8 for n in NOTES)
    assert max(n.midi for n in NOTES if n.voice == "piano") <= 59
    lead = sorted([n for n in NOTES if n.voice == "lead"], key=lambda n: n.beat)
    assert sum(n.midi == pitch("A5") for n in lead) == 1
    assert any(n.beat % 4 + n.duration > 4 for n in lead)
    # Core four bars return exactly; the last chorus deliberately changes only
    # the second half. Symbolic checks do not certify aesthetic quality.
    assert MELODY[16:20] == MELODY[28:32] == MELODY[32:36]
    diatonic = {0, 2, 4, 6, 7, 9, 11}
    assert all(n.midi % 12 in diatonic or (n.midi % 12 == 3 and int(n.beat // 4) + 1 in (28, 43))
        for n in NOTES if n.voice != "drums")
    report = []
    for first, last, name, *_ in SECTIONS:
        ns = [n for n in lead if (first - 1) * 4 <= n.beat < last * 4]
        report.append({"section": name, "melody_notes": len(ns),
            "notes_per_bar": round(len(ns) / (last - first + 1), 2),
            "written_rest_percent": round(100 * (1 - sum(n.duration for n in ns) / ((last-first+1)*4)), 1),
            "range_midi": [min(n.midi for n in ns), max(n.midi for n in ns)]})
    midi_results = {}
    for file in OUT.glob("*.mid"):
        mid = mido.MidiFile(file)
        active = set()
        for msg in mido.merge_tracks(mid.tracks):
            if msg.type == "note_on" and msg.velocity:
                key = msg.channel, msg.note
                assert key not in active, (file, key)
                active.add(key)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                active.remove((msg.channel, msg.note))
        assert not active
        assert all(sum(m.time for m in tr) == END * PPQ for tr in mid.tracks)
        midi_results[file.name] = {"duration_seconds": round(mid.length, 3), "matched_note_offs": True}
    return {"bars": BARS, "key": "G major", "main_tempo": 112,
        "note_counts": dict(Counter(n.voice for n in NOTES)), "melody": report,
        "midi": midi_results, "note": "Structural and file checks; not a claim of listener approval."}


def documentation(report):
    spec = {"title": TITLE, "skills_source": "https://github.com/jtydhr88/music-composition-skills",
        "skills_commit": "7adca0c321db602deb3c94d79b04037cf339e5d2",
        "skills_used": ["mc-melody", "mc-development", "mc-style-jpop", "mc-texture-layering"],
        "material": {"motif": "两小节：长音之后下行二度、三度，接两个短音上行至长音；尾端留出一拍换气。",
            "character": "想说出口，又把后半句轻轻收回。", "stated_in": ["Chorus"], "method": "加工处理"},
        "melody_design": {"lead": "钢琴扮演歌唱声部", "central_octave": "G4-G5",
            "climax": "第37小节末至38小节，唯一A5长音跨小节保持",
            "contour": "主歌窄幅说话，预副歌上行，副歌舒展后下行，末遍延长最高点",
            "late_climax_reason": "保留末副歌换尾的唯一高点；有意晚于默认的四分之三位置"},
        "form": [{"bars": [a,b], "id": c, "name": d, "development": e,
            "note": {"Verse": "与副歌对照的新主题，保留短短长的语气节奏。",
                "Thin": "原样再现前四小节，仅撤去节奏组。",
                "Final": "动力再现，前四小节不变，后四小节换尾，唯一最高音跨小节延长。"}.get(c, d)}
            for a,b,c,d,e,_ in SECTIONS],
        "texture": {"foreground": "钢琴旋律，D4-A5，较干，力度明显高于伴奏",
            "middle": "钢琴和声，每小节一至三次和弦攻击，最高B3",
            "background": "根音贝斯与轻鼓；前奏、间奏、落ちサビ和尾声退出",
            "conflict_solution": "钢琴伴奏下移音区、减少攻击次数，不在旋律长音下跑连续琶音"},
        "harmony_notes": ["副歌Cmaj7-D-Bm-Em，随后C-D-G-G。",
            "第6小节G-F#为D和弦上的4-3解决。",
            "第28、43小节Cm借自同主小调，Eb是明确安排的借用色彩；D是经过/九度色彩。",
            "末副歌E5落在G上作为六度倚音，级进解决到D5；A5为C上的六度并延留到D的五度。"],
        "melody_bars": MELODY, "chords": CHORDS, "tempo_points": TEMPO_POINTS,
        "notes": [asdict(n) for n in NOTES]}
    (OUT / "composition.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [TITLE, "原创 J-pop 器乐小品 | G 大调 | 4/4 | 44 小节 | 主速 112 BPM", "",
        "试听顺序：Chorus_Preview.mp3（副歌片段）→ Before_The_Wind_Stops.mp3（完整版）→ Melody_Only.mp3（裸旋律）。",
        "这次先写8小节副歌，再倒推主歌、预副歌；不是按每小节和弦抽取音符。",
        "核心变化：旋律留白、重复与换尾；伴奏限制在低音区；段落用节奏组进出区分。", "",
        "段落时间："]
    for s in report["audio"]["sections"]:
        sec = int(s["start_seconds"])
        lines.append(f"{sec//60:02d}:{sec%60:02d}  {s['section']}")
    lines += ["", "文件：", "Before_The_Wind_Stops.mid：含速度变化、段落标记、分轨和演奏时值。",
        "Melody_Only.mid：独立主旋律轨。Score_Grid.mid：便于编辑的整齐谱面时值。",
        "WAV 为44.1kHz、24bit立体声；MP3用于快速试听。", "",
        "FL Studio 导入：在新工程中打开完整MIDI并保留所有轨道及速度信息。",
        "lead和piano轨使用钢琴；bass轨使用指弹贝斯；drums轨使用GM鼓组。",
        "鼓轨：36底鼓、37边击、38军鼓、42闭镲、46开镲、49吊镲。",
        "钢琴应保持旋律轨明显高于伴奏轨，勿给所有音符统一力度。",
        "FL是否保留MIDI速度事件取决于导入选项；音频试听已包含所有速度变化。",
        "当前输出为离线采样渲染，并非FL Studio原生导出的工程；未改动你当前打开的工程。", "",
        "音色来自本机已有的FL Keys Grand Piano与FPC/Grv鼓采样；贝斯为脚本合成。",
        "应用的skills：mc-melody、mc-development、mc-style-jpop、mc-texture-layering。",
        "来源：https://github.com/jtydhr88/music-composition-skills", "",
        "MIDI与音频检查只验证文件、休止、重复和声部分离等结构，不能替代听感判断。"]
    (OUT / "作品说明.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    melody()
    accompaniment()
    drums()
    perform()
    midi_export("Before_The_Wind_Stops.mid", list(CHANNELS))
    midi_export("Melody_Only.mid", ["lead"])
    midi_export("Score_Grid.mid", list(CHANNELS), expressive=False)
    report = validate()
    print("Symbolic checks passed; rendering audio.", flush=True)
    report["audio"] = render()
    ffmpeg = "C:/Users/33573/AppData/Roaming/Python/Python313/Scripts/ffmpeg.EXE"
    for name in ("Before_The_Wind_Stops", "Melody_Only", "Chorus_Preview"):
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(OUT / (name + ".wav")),
            "-codec:a", "libmp3lame", "-q:a", "2", "-metadata", "title=" + TITLE, str(OUT / (name + ".mp3"))], check=True)
    documentation(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
