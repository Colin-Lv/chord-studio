#!/usr/bin/env python3
"""Render the boundary dataset MIDI to 48 kHz mono WAV.

The renderer reuses the SoundFont sampler that already powers
``research/midi-gen/render/render_sf2.py`` (``sf2.SoundFont`` + its
FluidR3_GM.sf2).  ``render_sf2.py`` itself maps *track names* to GM programs
(``CHORDS``/``PAD``/``MELODY`` ...) and always writes 44.1 kHz stereo, so the
instrument rotation of this dataset would collapse to piano.  This script
therefore calls the same sampler directly with each track's GM program and
writes 48 kHz mono, as requested:

    python render.py --dataset . --engine sf2      # in-process (default)
    python render.py --dataset . --engine cli      # subprocess render_sf2.py

``--engine cli`` literally shells out to
``research/midi-gen/render/render_sf2.py --no-stems`` and then down-mixes /
resamples the produced ``mix.wav`` to 48 kHz mono.

Two adjustments are applied to the sampler so that the dataset is clean:
  * every SoundFont zone release is capped at 0.6 s (the FluidR3 steel guitar
    preset declares a 13-25 s release, which would make rendering 20x slower
    and smear every chord into the next one);
  * a 60 ms fade-out is applied to each note tail, because the piano / guitar /
    strings presets of FluidR3 declare a 0 s release and would click.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

HERE = Path(__file__).resolve().parent
RENDER_DIR = Path(r"D:\ChordStudio\research\midi-gen\render")
SF2_CANDIDATES = [
    Path(r"D:\ChordStudio\research\midi-gen\render\FluidR3_GM.sf2"),
    Path(r"D:\ChordStudio\assets\soundfonts\FluidR3_GM.sf2"),
]
SR = 48000
TAIL = 0.08          # extra render time per note (natural release before fade)
FADE = 0.06          # fade-out applied to the note tail
MAX_RELEASE = 0.60   # cap the SoundFont release


def find_soundfont(explicit: str | None = None) -> Path:
    for p in ([Path(explicit)] if explicit else SF2_CANDIDATES):
        if p.is_file():
            return p
    raise SystemExit("SoundFont not found, tried:\n  " + "\n  ".join(map(str, SF2_CANDIDATES)))


def load_sf2_module():
    if not (RENDER_DIR / "sf2.py").is_file():
        raise SystemExit(f"sf2.py not found in {RENDER_DIR}")
    if str(RENDER_DIR) not in sys.path:
        sys.path.insert(0, str(RENDER_DIR))
    import sf2  # noqa: E402
    return sf2


def make_sampler_class(sf2):
    class DatasetSoundFont(sf2.SoundFont):
        """Sampler with cached preset zones and a bounded release tail."""

        def __init__(self, path):
            super().__init__(path)
            self._zones: dict[tuple[int, int], list] = {}

        def preset_zones(self, bank: int, program: int):
            key = (bank, program)
            if key not in self._zones:
                zones = super().preset_zones(bank, program)
                for z in zones:
                    z.release = min(z.release, MAX_RELEASE)
                self._zones[key] = zones
            return self._zones[key]

    return DatasetSoundFont


def render_pm(pm: pretty_midi.PrettyMIDI, sfx, sr: int = SR, gain: float = 0.65
              ) -> np.ndarray:
    total = max(0.5, pm.get_end_time()) + 0.8
    n = int(total * sr)
    mix = np.zeros(n, dtype=np.float32)
    for inst in pm.instruments:
        program = int(inst.program)
        for note in inst.notes:
            dur = max(0.03, note.end - note.start)
            audio, _pan = sfx.render(program, note.pitch, note.velocity,
                                     dur + TAIL, sr=sr, bank=0, gain=gain)
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


def render_cli(midi: Path, out_wav: Path, soundfont: Path, tmp_dir: Path) -> dict:
    """Render through the unmodified render_sf2.py CLI, then post-process."""
    cmd = [sys.executable, str(RENDER_DIR / "render_sf2.py"),
           "--midi", str(midi), "--out", str(tmp_dir),
           "--soundfont", str(soundfont), "--no-stems"]
    subprocess.run(cmd, check=True, capture_output=True)
    data, sr = sf.read(str(tmp_dir / "mix.wav"), always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(SR, sr)
        mono = resample_poly(mono, SR // g, sr // g).astype(np.float32)
    peak = float(np.max(np.abs(mono))) or 0.0
    if peak > 0:
        mono *= 0.89 / peak
    sf.write(str(out_wav), mono, SR, subtype="PCM_16")
    return {"sample_rate": SR, "channels": 1, "duration": round(len(mono) / SR, 3)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(HERE))
    ap.add_argument("--soundfont", default=None)
    ap.add_argument("--engine", choices=("sf2", "cli"), default="sf2")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma separated sample ids")
    ap.add_argument("--force", action="store_true", help="re-render existing files")
    ap.add_argument("--gain", type=float, default=0.65)
    args = ap.parse_args(argv)

    ds = Path(args.dataset)
    manifest = json.loads((ds / "dataset.json").read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        samples = [s for s in samples if s["sample_id"] in wanted]
    if args.limit:
        samples = samples[:args.limit]

    sf_path = find_soundfont(args.soundfont)
    audio_dir = ds / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    print(f"[render] soundfont = {sf_path}", flush=True)
    print(f"[render] engine = {args.engine}, {len(samples)} samples, {SR} Hz mono", flush=True)

    sfx = None
    if args.engine == "sf2":
        sf2 = load_sf2_module()
        sfx = make_sampler_class(sf2)(str(sf_path))
        print(f"[render] soundfont loaded in {sfx.path.stat().st_size / 1e6:.0f} MB", flush=True)

    results = []
    t0 = time.time()
    for i, s in enumerate(samples):
        midi = ds / s["midi"]
        out = audio_dir / f"{s['sample_id']}.wav"
        if out.is_file() and not args.force:
            info = sf.info(str(out))
            results.append({"sample_id": s["sample_id"], "file": str(out),
                            "sample_rate": info.samplerate, "channels": info.channels,
                            "duration": round(info.duration, 3), "cached": True})
            continue
        t = time.time()
        if args.engine == "cli":
            tmp = ds / ".render_tmp"
            tmp.mkdir(exist_ok=True)
            meta = render_cli(midi, out, sf_path, tmp)
        else:
            pm = pretty_midi.PrettyMIDI(str(midi))
            mix = render_pm(pm, sfx, gain=args.gain)
            sf.write(str(out), mix, SR, subtype="PCM_16")
            meta = {"sample_rate": SR, "channels": 1,
                    "duration": round(len(mix) / SR, 3)}
        dt = time.time() - t
        results.append({"sample_id": s["sample_id"], "file": str(out), **meta,
                        "render_sec": round(dt, 2)})
        print(f"  [{i + 1}/{len(samples)}] {s['sample_id']} {meta['duration']:.1f}s "
              f"in {dt:.2f}s", flush=True)

    total = sum(r["duration"] for r in results)
    summary = {
        "engine": args.engine,
        "soundfont": str(sf_path),
        "sample_rate": SR,
        "channels": 1,
        "rendered": len(results),
        "total_audio_sec": round(total, 2),
        "elapsed_sec": round(time.time() - t0, 2),
        "files": results,
    }
    (ds / "render_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print(f"[done] {len(results)} wav, {total / 60:.1f} min audio, "
          f"{summary['elapsed_sec']:.1f}s wall -> {audio_dir}")


if __name__ == "__main__":
    main()
