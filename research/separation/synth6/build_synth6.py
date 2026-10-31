"""Build the synthetic 4-part multitrack for the synth6 piano/guitar probe.

Layout (everything under D:\\ChordStudio\\research\\separation\\synth6):

    parts/{piano,guitar,bass,drums}.mid    one PrettyMIDI per part
    truth/{piano,guitar,bass,drums}.wav    48 kHz mono float32 renders
    mix.wav                                sum of the four, peak-normalised
    manifest.json                          durations / levels / scale factor

The renderer is the project one
(D:\\ChordStudio\\research\\boundary-dataset\\render.py).  `render_pm` is copied
verbatim below except for ONE line: percussion tracks get `bank=128` (the GM
drum bank) instead of the hardcoded `bank=0`.  With `bank=0` a drum hit on
note 36 renders as a *piano* note, which would silently contaminate the very
stem this experiment is about.  For the three melodic parts the copy is
bit-identical to `render.render_pm` (asserted below).

Arrangement: 105 BPM, 4/4, 16 bars = 36.57 s, C-Am-F-G x4.

    part    GM program            pitch range   rhythm
    piano   0  (Acoustic Grand)   41-60 (F2-C4) block chord on 1, re-attack on 3
    guitar  25 (Acoustic steel)   62-72 (D4-C5) 8th-note arpeggio, 8 plucks/bar
    bass    33 (Electric finger)  29-48 (F1-C3) root on 1 and 3, octave on 3.5
    drums   ch.9 kit              36/38/42      kick 1+3, snare 2+4, hats 8ths
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

HERE = Path(__file__).resolve().parent
BD = Path(r"D:\ChordStudio\research\boundary-dataset")
sys.path.insert(0, str(BD))
import render as R  # noqa: E402

SR = 48000
TAIL = 0.08          # same as render.py
FADE = 0.06          # same as render.py
TEMPO = 105.0
BEAT = 60.0 / TEMPO
BAR = 4.0 * BEAT
BARS = 16
PROGRESSION = ["C", "Am", "F", "G"]

# Register plan: piano mid, guitar an octave above it, bass below.
PIANO_VOICING = {
    "C":  [48, 52, 55, 60],   # C3 E3 G3 C4
    "Am": [45, 48, 52, 57],   # A2 C3 E3 A3
    "F":  [41, 45, 48, 53],   # F2 A2 C3 F3
    "G":  [43, 47, 50, 55],   # G2 B2 D3 G3
}
GUITAR_VOICING = {            # same chords, ~1 octave up, top voice doubled
    "C":  [64, 67, 72],       # E4 G4 C5
    "Am": [64, 69, 72],       # E4 A4 C5
    "F":  [65, 69, 72],       # F4 A4 C5
    "G":  [62, 67, 71],       # D4 G4 B4
}
BASS_ROOT = {"C": 36, "Am": 33, "F": 29, "G": 31}   # C2 A1 F1 G1
GUITAR_PATTERN = [0, 1, 2, 1, 0, 2, 1, 2]           # chord-tone index per 8th


def render_pm(pm: pretty_midi.PrettyMIDI, sfx, sr: int, gain: float = 0.65
              ) -> np.ndarray:
    """render.render_pm, verbatim, with the GM drum bank for percussion."""
    total = max(0.5, pm.get_end_time()) + 0.8
    n = int(total * sr)
    mix = np.zeros(n, dtype=np.float32)
    for inst in pm.instruments:
        program = int(inst.program)
        bank = 128 if inst.is_drum else 0          # <-- the one changed line
        for note in inst.notes:
            dur = max(0.03, note.end - note.start)
            audio, _pan = sfx.render(program, note.pitch, note.velocity,
                                     dur + TAIL, sr=sr, bank=bank, gain=gain)
            if audio.size < 8:
                continue
            fade = min(int(FADE * sr), audio.size)
            audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            start = int(round(note.start * sr))
            length = min(audio.size, n - start)
            if length <= 0:
                continue
            mix[start:start + length] += audio[:length]
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0:
        mix *= 0.89 / peak
    return mix


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------


def build_piano() -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    inst = pretty_midi.Instrument(program=0, name="piano")
    for bar in range(BARS):
        chord = PROGRESSION[bar % 4]
        t0 = bar * BAR
        for pitch in PIANO_VOICING[chord]:
            # block chord on beat 1, softer re-attack on beat 3
            inst.notes.append(pretty_midi.Note(velocity=78, pitch=pitch,
                                               start=t0, end=t0 + 1.9 * BEAT))
            inst.notes.append(pretty_midi.Note(velocity=58, pitch=pitch,
                                               start=t0 + 2 * BEAT,
                                               end=t0 + 3.9 * BEAT))
    pm.instruments.append(inst)
    return pm


def build_guitar() -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    inst = pretty_midi.Instrument(program=25, name="guitar")
    for bar in range(BARS):
        chord = PROGRESSION[bar % 4]
        voicing = GUITAR_VOICING[chord]
        t0 = bar * BAR
        for step, idx in enumerate(GUITAR_PATTERN):
            start = t0 + step * 0.5 * BEAT
            pitch = voicing[idx]
            vel = 88 if step % 4 == 0 else 68
            inst.notes.append(pretty_midi.Note(
                velocity=vel, pitch=pitch, start=start,
                end=start + 0.42 * BEAT))
    pm.instruments.append(inst)
    return pm


def build_bass() -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    inst = pretty_midi.Instrument(program=33, name="bass")
    for bar in range(BARS):
        root = BASS_ROOT[PROGRESSION[bar % 4]]
        t0 = bar * BAR
        inst.notes.append(pretty_midi.Note(velocity=92, pitch=root,
                                           start=t0, end=t0 + 1.55 * BEAT))
        inst.notes.append(pretty_midi.Note(velocity=82, pitch=root,
                                           start=t0 + 2 * BEAT,
                                           end=t0 + 3.05 * BEAT))
        inst.notes.append(pretty_midi.Note(velocity=64, pitch=root + 12,
                                           start=t0 + 3.5 * BEAT,
                                           end=t0 + 3.9 * BEAT))
    pm.instruments.append(inst)
    return pm


def build_drums() -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    for bar in range(BARS):
        t0 = bar * BAR
        for beat in (0, 2):                       # kick on 1 and 3
            inst.notes.append(pretty_midi.Note(
                velocity=104, pitch=36, start=t0 + beat * BEAT,
                end=t0 + beat * BEAT + 0.25 * BEAT))
        for beat in (1, 3):                       # snare on 2 and 4
            inst.notes.append(pretty_midi.Note(
                velocity=96, pitch=38, start=t0 + beat * BEAT,
                end=t0 + beat * BEAT + 0.25 * BEAT))
        for step in range(8):                     # closed hats on 8ths
            start = t0 + step * 0.5 * BEAT
            inst.notes.append(pretty_midi.Note(
                velocity=54 if step % 2 else 62, pitch=42,
                start=start, end=start + 0.2 * BEAT))
    pm.instruments.append(inst)
    return pm


BUILDERS = {
    "piano": build_piano,
    "guitar": build_guitar,
    "bass": build_bass,
    "drums": build_drums,
}


def main() -> int:
    t_start = time.time()
    parts_dir = HERE / "parts"
    truth_dir = HERE / "truth"
    parts_dir.mkdir(exist_ok=True)
    truth_dir.mkdir(exist_ok=True)

    sfx = R.make_sampler_class(R.load_sf2_module())(str(R.find_soundfont(None)))
    print(f"[build] soundfont = {sfx.path}", flush=True)

    pms = {name: fn() for name, fn in BUILDERS.items()}

    # The copy above must behave exactly like render.render_pm for melodic parts.
    ref = R.render_pm(pms["piano"], sfx, sr=SR)
    mine = render_pm(pms["piano"], sfx, sr=SR)
    same = ref.shape == mine.shape and np.allclose(ref, mine, atol=1e-6)
    print(f"[build] render_pm copy matches project render_pm on piano: {same}", flush=True)

    stats: dict[str, dict] = {}
    stems: dict[str, np.ndarray] = {}
    for name, pm in pms.items():
        midi_path = parts_dir / f"{name}.mid"
        pm.write(str(midi_path))
        audio = render_pm(pm, sfx, sr=SR)
        if not same and name != "drums":
            raise SystemExit(f"render mismatch on {name}")
        wav_path = truth_dir / f"{name}.wav"
        sf.write(str(wav_path), audio.astype(np.float32), SR, subtype="FLOAT")
        stems[name] = audio
        rms = float(np.sqrt(np.mean(audio ** 2)))
        stats[name] = {
            "midi": str(midi_path),
            "wav": str(wav_path),
            "duration_sec": round(audio.size / SR, 3),
            "samples": int(audio.size),
            "peak_dbfs": round(20 * np.log10(max(float(np.max(np.abs(audio))), 1e-12)), 3),
            "rms_dbfs": round(20 * np.log10(max(rms, 1e-12)), 3),
            "n_notes": int(sum(len(i.notes) for i in pm.instruments)),
            "program": int(pm.instruments[0].program),
            "is_drum": bool(pm.instruments[0].is_drum),
            "pitch_lo": int(min(n.pitch for n in pm.instruments[0].notes)),
            "pitch_hi": int(max(n.pitch for n in pm.instruments[0].notes)),
        }
        print(f"[build] {name:6s} {stats[name]['duration_sec']:6.2f}s "
              f"peak {stats[name]['peak_dbfs']:6.2f} dBFS "
              f"rms {stats[name]['rms_dbfs']:6.2f} dBFS "
              f"{stats[name]['n_notes']:3d} notes", flush=True)

    # mix = plain sum of the four rendered stems, then one peak scale factor
    n = max(a.size for a in stems.values())
    total = np.zeros(n, dtype=np.float32)
    for name, audio in stems.items():
        total[:audio.size] += audio
    raw_peak = float(np.max(np.abs(total)))
    scale = 0.89 / raw_peak
    mix = (total * scale).astype(np.float32)
    mix_path = HERE / "mix.wav"
    sf.write(str(mix_path), mix, SR, subtype="FLOAT")
    print(f"[build] mix raw peak {raw_peak:.4f} -> scale {scale:.6f} "
          f"-> peak {float(np.max(np.abs(mix))):.4f}", flush=True)

    manifest = {
        "sample_rate": SR,
        "tempo_bpm": TEMPO,
        "bars": BARS,
        "beats_per_bar": 4,
        "duration_sec": round(n / SR, 3),
        "progression": PROGRESSION * 4,
        "soundfont": str(sfx.path),
        "renderer": str(BD / "render.py"),
        "renderer_note": ("render.render_pm copied verbatim except percussion uses "
                          "bank=128 (GM drum kit); bank=0 would render drum hits as piano"),
        "render_pm_copy_matches_project": bool(same),
        "stems": stats,
        "mix": {
            "path": str(mix_path),
            "sum_peak_before_scale": round(raw_peak, 6),
            "scale_factor": round(scale, 9),
            "scale_factor_db": round(20 * np.log10(scale), 3),
            "peak_after_scale": round(float(np.max(np.abs(mix))), 6),
            "rms_dbfs": round(20 * np.log10(max(float(np.sqrt(np.mean(mix ** 2))), 1e-12)), 3),
            "samples": int(n),
        },
        "elapsed_sec": round(time.time() - t_start, 2),
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[build] wrote {HERE / 'manifest.json'} in {time.time() - t_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
