"""Prepare 36-channel chord patches: CENS + CQT + bass chroma (from stem_BASS).

The 7th-family chords were the weakest class (0.62-0.66) because the 7th is masked in the
mix. The separated bass stem gives the root and, with the comp stem, the upper structure,
so this adds a low-band chroma block computed on stem_BASS.wav.
"""
from __future__ import annotations

import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

DATA = Path(r"D:\ChordStudio\research\boundary-dataset")
EXT = Path(r"D:\ChordStudio\research\extension-dataset")
STEMS = DATA / "stems"
HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)
QUALITIES = ["maj", "min", "7", "maj7", "min7", "9", "maj9", "min9", "sus2", "sus4"]
FRAMES = 64
SR = 22050


def resample(patch, frames=FRAMES):
    if patch.shape[1] == 0:
        return np.zeros((12, frames), dtype=np.float32)
    idx = np.linspace(0, patch.shape[1] - 1, frames)
    return np.stack([np.interp(idx, np.arange(patch.shape[1]), row) for row in patch], axis=0).astype(np.float32)


def bass_chroma(path):
    import soundfile as sf
    data, sr = sf.read(str(path), always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    if sr != SR:
        idx = np.linspace(0, len(mono) - 1, int(len(mono) * SR / sr))
        mono = np.interp(idx, np.arange(len(mono)), mono).astype(np.float32)
    hop, nfft = 512, 4096
    window = np.hanning(nfft).astype(np.float32)
    freqs = np.fft.rfftfreq(nfft, 1.0 / SR)
    band = (freqs >= 40) & (freqs <= 300)
    midi = 69 + 12 * np.log2(np.maximum(freqs, 1e-6) / 440.0)
    pc = np.round(midi).astype(int) % 12
    frames = 1 + (len(mono) - nfft) // hop
    out = np.zeros((12, frames), dtype=np.float32)
    for i in range(frames):
        seg = mono[i * hop:i * hop + nfft] * window
        mag = np.abs(np.fft.rfft(seg))[band] ** 2
        np.add.at(out[:, i], pc[band], mag)
    out /= (out.sum(axis=0, keepdims=True) + 1e-9)
    times = (np.arange(frames) * hop + nfft / 2) / SR
    return out, times


def process(job):
    import librosa
    base = Path(job["base"])
    labels = json.loads((base / job["labels"]).read_text(encoding="utf-8"))
    y, _ = librosa.load(str(base / job["audio"]), sr=SR, mono=True)
    y, _ = librosa.effects.hpss(y)
    cens = librosa.feature.chroma_cens(y=y, sr=SR, hop_length=512)
    cens = cens / (cens.sum(axis=0, keepdims=True) + 1e-9)
    cqt = np.abs(librosa.cqt(y, sr=SR, hop_length=512, n_bins=84, bins_per_octave=12))
    chroma = np.zeros((12, cqt.shape[1]), dtype=np.float32)
    for pc in range(12):
        chroma[pc] = cqt[pc::12].sum(axis=0)
    chroma = np.log1p(chroma)
    chroma = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-9)
    times = np.arange(cens.shape[1]) * 512 / SR

    bass = None
    stem_dir = STEMS / job["sample_id"]
    bass_path = stem_dir / "stem_BASS.wav"
    if bass_path.exists():
        bass, btimes = bass_chroma(bass_path)
    song = job["sample_id"].split("-")[1] if "-" in job["sample_id"] else job["sample_id"]
    rows = []
    for chord in labels["chords"]:
        quality = chord["quality"]
        if quality not in QUALITIES:
            continue
        sel = (times >= chord["start"]) & (times < chord["end"])
        p_cens = resample(cens[:, sel] if sel.any() else cens[:, :1])
        p_cqt = resample(chroma[:, sel] if sel.any() else chroma[:, :1])
        if bass is not None:
            bsel = (btimes >= chord["start"]) & (btimes < chord["end"])
            p_bass = resample(bass[:, bsel] if bsel.any() else bass[:, :1])
        else:
            p_bass = np.zeros((12, FRAMES), dtype=np.float32)
        rows.append((p_cens, p_cqt, p_bass, int(chord["root_pc"]),
                     QUALITIES.index(quality), song, job["domain"]))
    return rows


def main(workers: int = 8):
    jobs = []
    for base, meta_file, domain in ((DATA, DATA / "dataset.json", 0), (EXT, EXT / "dataset.json", 1)):
        if not meta_file.exists():
            continue
        for meta in json.loads(meta_file.read_text(encoding="utf-8"))["samples"]:
            jobs.append(dict(base=str(base), audio=meta["audio"], labels=meta["labels"],
                             sample_id=meta["sample_id"], domain=domain))
    print(f"{len(jobs)} samples, {workers} workers", flush=True)
    cens_list, cqt_list, bass_list, roots, quals, songs, domains = [], [], [], [], [], [], []
    with Pool(workers) as pool:
        for i, chunk in enumerate(pool.imap_unordered(process, jobs, chunksize=2), 1):
            for c, q, b, r, qu, s, d in chunk:
                cens_list.append(c); cqt_list.append(q); bass_list.append(b)
                roots.append(r); quals.append(qu); songs.append(s); domains.append(d)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)} samples, {len(roots)} chords", flush=True)
    common = dict(root=np.array(roots), quality=np.array(quals), song=np.array(songs),
                  domain=np.array(domains))
    x = np.concatenate([np.stack(cens_list), np.stack(cqt_list), np.stack(bass_list)], axis=1)
    np.savez_compressed(OUT / "chords_36.npz", x=x, **common)
    print(f"saved {x.shape} to chords_36.npz", flush=True)
    missing = sum(1 for b in bass_list if not b.any())
    print(f"patches without bass stem: {missing}/{len(bass_list)}")


if __name__ == "__main__":
    main(int(os.environ.get("WORKERS", "8")))
