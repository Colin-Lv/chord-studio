"""Parallel feature preparation (CENS + CQT), one worker per sample."""
from __future__ import annotations

import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

DATA = Path(r"D:\ChordStudio\research\boundary-dataset")
EXT = Path(r"D:\ChordStudio\research\extension-dataset")
HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)
QUALITIES = ["maj", "min", "7", "maj7", "min7", "9", "maj9", "min9", "sus2", "sus4"]
FRAMES = 64


def resample(patch, frames=FRAMES):
    if patch.shape[1] == 0:
        return np.zeros((12, frames), dtype=np.float32)
    idx = np.linspace(0, patch.shape[1] - 1, frames)
    return np.stack([np.interp(idx, np.arange(patch.shape[1]), row) for row in patch], axis=0).astype(np.float32)


def process(job):
    import librosa
    base = Path(job["base"])
    labels = json.loads((base / job["labels"]).read_text(encoding="utf-8"))
    y, _ = librosa.load(str(base / job["audio"]), sr=22050, mono=True)
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
    song = job["sample_id"].split("-")[1] if "-" in job["sample_id"] else job["sample_id"]
    out = []
    for chord in labels["chords"]:
        quality = chord["quality"]
        if quality not in QUALITIES:
            continue
        sel = (times >= chord["start"]) & (times < chord["end"])
        out.append((resample(cens[:, sel] if sel.any() else cens[:, :1]),
                    resample(chroma[:, sel] if sel.any() else chroma[:, :1]),
                    int(chord["root_pc"]), QUALITIES.index(quality), song, job["domain"]))
    return out


def main(workers: int = 8):
    jobs = []
    for base, meta_file, domain in ((DATA, DATA / "dataset.json", 0), (EXT, EXT / "dataset.json", 1)):
        if not meta_file.exists():
            continue
        for meta in json.loads(meta_file.read_text(encoding="utf-8"))["samples"]:
            jobs.append(dict(base=str(base), audio=meta["audio"], labels=meta["labels"],
                             sample_id=meta["sample_id"], domain=domain))
    print(f"{len(jobs)} samples, {workers} workers", flush=True)
    cens_list, cqt_list, roots, quals, songs, domains = [], [], [], [], [], []
    with Pool(workers) as pool:
        for i, chunk in enumerate(pool.imap_unordered(process, jobs, chunksize=2), 1):
            for c, q, r, qu, s, d in chunk:
                cens_list.append(c); cqt_list.append(q)
                roots.append(r); quals.append(qu); songs.append(s); domains.append(d)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)} samples, {len(roots)} chords", flush=True)
    common = dict(root=np.array(roots), quality=np.array(quals), song=np.array(songs),
                  domain=np.array(domains))
    np.savez_compressed(OUT / "chords.npz", x=np.stack(cens_list), **common)
    np.savez_compressed(OUT / "chords_cqt.npz", x=np.stack(cqt_list), **common)
    print(f"saved {len(roots)} chords (CENS + CQT)")
    print("quality counts:", {q: int((np.array(quals) == i).sum()) for i, q in enumerate(QUALITIES)})


if __name__ == "__main__":
    main(int(os.environ.get("WORKERS", "8")))
