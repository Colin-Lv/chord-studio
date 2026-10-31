"""Build the 60-channel chord patch set: CENS + CQT + NNLS note evidence (3 registers).

Channels (12 each, 64 frames per chord):
  0-11   CENS chroma            (smoothed, robust to timbre)
 12-23   CQT chroma             (no smoothing, keeps 7th/9th detail)
 24-35   NNLS low  register     (MIDI 21-47: bass)
 36-47   NNLS mid  register     (MIDI 48-71: chord voicing)
 48-59   NNLS high register     (MIDI 72-104: upper voicing / melody)

The NNLS activations are the official Vamp NNLS-Chroma semitone spectrum: 84 nonnegative
note saliences per frame at 2048/44100 s hop. Each frame is L1-normalised over all 84
notes before summing a register, so a quiet register keeps its honest weight.

Also records texture / melody / domain per chord so results can be split by condition.
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

BACKEND = r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app"
SONIC = r"D:\ChordStudio\Dialogue\tools\sonic-annotator-1.7-win64\sonic-annotator-win64\sonic-annotator.exe"
PLUGIN = r"C:\Program Files\Vamp Plugins\nnls-chroma.dll"
DATA = Path(r"D:\ChordStudio\research\boundary-dataset")
EXT = Path(r"D:\ChordStudio\research\extension-dataset")
AUDIO_FIXED = DATA / "audio_fixed"
OUT = Path(r"D:\ChordStudio\research\train-v2\data")
OUT_NAME = os.environ.get("OUT_NAME", "chords_full.npz")
CACHE = OUT / "nnls_cache"
QUALITIES = ["maj", "min", "7", "maj7", "min7", "9", "maj9", "min9", "sus2", "sus4"]
FRAMES = 64
MIDI = np.arange(21, 105)
REGISTERS = ((21, 48), (48, 72), (72, 105))


def resample(patch, frames=FRAMES):
    if patch.shape[1] == 0:
        return np.zeros((patch.shape[0], frames), dtype=np.float32)
    if patch.shape[1] == frames:
        return patch.astype(np.float32)
    idx = np.linspace(0, patch.shape[1] - 1, frames)
    return np.stack([np.interp(idx, np.arange(patch.shape[1]), row) for row in patch],
                    axis=0).astype(np.float32)


def nnls_profile(activations, times, start, end):
    """(3, 12, T) register chroma for the frames inside [start, end)."""
    norm = activations / np.maximum(activations.sum(axis=0, keepdims=True), 1e-9)
    out = np.zeros((len(REGISTERS), 12, activations.shape[1]), dtype=np.float32)
    for r, (lo, hi) in enumerate(REGISTERS):
        mask = (MIDI >= lo) & (MIDI < hi)
        pc = MIDI[mask] % 12
        np.add.at(out[r], pc, norm[mask])
    return out


def process(job):
    import librosa

    sys.path.insert(0, BACKEND)
    import harmonic_note_features as H

    base = Path(job["base"])
    labels = json.loads((base / job["labels"]).read_text(encoding="utf-8"))
    audio = base / job["audio"]
    y, _ = librosa.load(str(audio), sr=22050, mono=True)
    y, _ = librosa.effects.hpss(y)
    cens = librosa.feature.chroma_cens(y=y, sr=22050, hop_length=512)
    cens = cens / (cens.sum(axis=0, keepdims=True) + 1e-9)
    cqt = np.abs(librosa.cqt(y, sr=22050, hop_length=512, n_bins=84, bins_per_octave=12))
    chroma = np.zeros((12, cqt.shape[1]), dtype=np.float32)
    for pc in range(12):
        chroma[pc] = cqt[pc::12].sum(axis=0)
    chroma = np.log1p(chroma)
    chroma = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-9)
    times = np.arange(cens.shape[1]) * 512 / 22050
    del y, cqt

    evidence = H.extract_note_evidence(audio, cache_dir=CACHE, sonic_annotator=SONIC,
                                       plugin_path=PLUGIN)
    reg = nnls_profile(evidence.activations, evidence.times, 0, 0)   # all frames

    rows = []
    for chord in labels["chords"]:
        quality = chord["quality"]
        if quality not in QUALITIES:
            continue
        sel = (times >= chord["start"]) & (times < chord["end"])
        if not sel.any():
            sel = np.zeros(len(times), dtype=bool)
            sel[min(1, len(times) - 1)] = True
        nsel = (evidence.times >= chord["start"]) & (evidence.times < chord["end"])
        if not nsel.any():
            nsel = np.zeros(len(evidence.times), dtype=bool)
            nsel[min(1, len(evidence.times) - 1)] = True
        patch = np.concatenate([
            resample(cens[:, sel]),
            resample(chroma[:, sel]),
            resample(reg[0][:, nsel]),
            resample(reg[1][:, nsel]),
            resample(reg[2][:, nsel]),
        ], axis=0)
        rows.append((patch, int(chord["root_pc"]), QUALITIES.index(quality),
                     job["song"], job["texture"], int(job["melody"]), int(job["domain"])))
    return rows


def main(workers: int = 8):
    jobs = []
    for base, meta_file, domain in ((DATA, DATA / "dataset.json", 0), (EXT, EXT / "dataset.json", 1)):
        if not meta_file.exists():
            continue
        for meta in json.loads(meta_file.read_text(encoding="utf-8"))["samples"]:
            sid = meta["sample_id"]
            parts = sid.split("-")
            texture = parts[2] if len(parts) > 2 else "block"
            melody = int("mel" in parts)
            if base is EXT:
                # extension samples follow <song>-<texture>[-mel]-ext
                texture = parts[2]
                melody = int("mel" in parts[3:4])
            music = meta["audio"]
            if ("arpeggio" in sid and base is DATA
                    and (AUDIO_FIXED / f"{sid}.wav").is_file()):
                music = f"audio_fixed/{sid}.wav"
            jobs.append(dict(base=str(base), audio=music, labels=meta["labels"],
                             sample_id=sid, domain=domain,
                             song=parts[1] if len(parts) > 1 else sid,
                             texture=texture, melody=melody))
    print(f"{len(jobs)} samples, {workers} workers", flush=True)
    patches, roots, quals, songs, textures, melodies, domains = [], [], [], [], [], [], []
    t0 = time.time()
    with Pool(workers) as pool:
        for i, chunk in enumerate(pool.imap_unordered(process, jobs, chunksize=2), 1):
            for patch, root, q, song, texture, melody, domain in chunk:
                patches.append(patch.astype(np.float16))
                roots.append(root)
                quals.append(q)
                songs.append(song)
                textures.append(texture)
                melodies.append(melody)
                domains.append(domain)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)} samples, {len(roots)} chords, {time.time() - t0:.0f}s",
                      flush=True)
    x = np.stack(patches)
    print("patches", x.shape, x.dtype)
    np.savez_compressed(OUT / OUT_NAME, x=x,
                        root=np.array(roots), quality=np.array(quals), song=np.array(songs),
                        texture=np.array(textures), melody=np.array(melodies),
                        domain=np.array(domains))
    print("saved", OUT / OUT_NAME)
    print("quality counts:", {q: int((np.array(quals) == i).sum()) for i, q in enumerate(QUALITIES)})
    print("texture counts:", {t: int((np.array(textures) == t).sum()) for t in ("block", "arpeggio", "strum")})
    print("melody counts:", {m: int((np.array(melodies) == m).sum()) for m in (0, 1)})


if __name__ == "__main__":
    main(int(os.environ.get("WORKERS", "8")))

