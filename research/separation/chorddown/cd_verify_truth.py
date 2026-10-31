"""Verify the ground-truth bar map against the built MIDI and the rendered audio.

The task brief said bars were C x4, Am x4, F x4, G x4.  The build script says
PROGRESSION = [C, Am, F, G] with `chord = PROGRESSION[bar % 4]`, i.e. ONE bar per
chord.  This script decides it from the material itself: MIDI pitches per bar and
a chroma check on truth/piano.wav, truth/bass.wav and mix.wav.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

from cd_common import BAR, SYNTH, WORK

PITCH_CLASS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
VOICINGS = {
    "C": [48, 52, 55, 60],
    "Am": [45, 48, 52, 57],
    "F": [41, 45, 48, 53],
    "G": [43, 47, 50, 55],
}
PC_SETS = {name: sorted({p % 12 for p in pitches}) for name, pitches in VOICINGS.items()}


def midi_per_bar():
    out = {}
    for part in ("piano", "bass", "guitar"):
        pm = pretty_midi.PrettyMIDI(str(SYNTH / "parts" / f"{part}.mid"))
        inst = pm.instruments[0]
        bars = {}
        for note in inst.notes:
            # MIDI stores tempo as integer microseconds/quarter, so written onsets
            # sit ~2e-6 s early; snap into the bar with a 1 ms tolerance.
            k = int((note.start + 1e-3) / BAR)
            bars.setdefault(k, []).append(note.pitch)
        out[part] = {k: sorted(set(v)) for k, v in bars.items()}
    return out


def name_chord(pcs: set) -> str:
    best, best_hit = "?", -1
    for name, ref in PC_SETS.items():
        hit = len(pcs & set(ref)) - len(pcs - set(ref))
        if hit > best_hit:
            best, best_hit = name, hit
    return best


def chroma(path: Path, sr_target: int = 48000):
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    audio = audio.mean(axis=1)
    if sr != sr_target:
        import scipy.signal as sig
        g = np.gcd(sr, sr_target)
        audio = sig.resample_poly(audio, sr_target // g, sr // g)
    return audio, sr_target


def bar_chroma(audio: np.ndarray, sr: int, bar: int, n_fft: int = 8192,
               hop: int = 2048) -> np.ndarray:
    start = int(bar * BAR * sr)
    end = int(min((bar + 1) * BAR * sr, audio.size))
    seg = audio[start:end]
    if seg.size < n_fft:
        return np.zeros(12)
    window = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    valid = (freqs > 55) & (freqs < 2000)
    chroma = np.zeros(12)
    for offset in range(0, seg.size - n_fft, hop):
        frame = seg[offset:offset + n_fft] * window
        mag = np.abs(np.fft.rfft(frame))[valid]
        f = freqs[valid]
        pc = np.round(12 * np.log2(f / 440.0) + 69).astype(int) % 12
        for k in range(12):
            chroma[k] += mag[pc == k].sum()
    total = chroma.sum()
    return chroma / total if total else chroma


def best_chord(chroma: np.ndarray) -> str:
    best, best_score = "?", -9e9
    for name, ref in PC_SETS.items():
        score = chroma[list(ref)].sum() - chroma[list(set(range(12)) - set(ref))].sum()
        if score > best_score:
            best, best_score = name, score
    return best


def main() -> int:
    midi = midi_per_bar()
    print("== MIDI pitch classes per bar (one chord per bar?)")
    rows = []
    for bar in range(16):
        pitches = midi["piano"].get(bar, [])
        pcs = {p % 12 for p in pitches}
        bass_pitches = midi["bass"].get(bar, [])
        guess = name_chord(pcs)
        rows.append({
            "bar": bar,
            "t0": round(bar * BAR, 4),
            "piano_pitches": pitches,
            "piano_pcs": sorted(pcs),
            "bass_pitches": bass_pitches,
            "piano_chord": guess,
            "bass_root": (PITCH_CLASS[min(bass_pitches) % 12]
                          if bass_pitches else "?"),
        })
        print(f"  bar {bar:2d} t={bar * BAR:7.3f}  piano={pitches}  "
              f"-> {guess:3s}   bass={bass_pitches} root={rows[-1]['bass_root']}")

    audio_sources = {
        "mix": SYNTH / "mix.wav",
        "truth_piano": SYNTH / "truth" / "piano.wav",
        "truth_bass": SYNTH / "truth" / "bass.wav",
    }
    chroma_guess = {}
    for name, path in audio_sources.items():
        audio, sr = chroma(path)
        guesses = [best_chord(bar_chroma(audio, sr, bar)) for bar in range(16)]
        chroma_guess[name] = guesses
        print(f"== audio chroma per bar: {name}")
        print("   " + " ".join(f"{g:3s}" for g in guesses))

    verdict = {
        "midi_progression": [row["piano_chord"] for row in rows],
        "bass_roots": [row["bass_root"] for row in rows],
        "mix_chroma": chroma_guess["mix"],
        "truth_piano_chroma": chroma_guess["truth_piano"],
        "truth_bass_chroma": chroma_guess["truth_bass"],
        "bar_seconds": BAR,
        "per_bar": rows,
    }
    agree = sum(1 for a, b in zip(verdict["midi_progression"],
                                 chroma_guess["truth_piano"]) if a == b)
    print(f"\nMIDI vs truth-piano chroma agreement: {agree}/16 bars")
    print("MIDI progression:", verdict["midi_progression"])
    (WORK / "truth_check.json").write_text(json.dumps(verdict, indent=2),
                                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
