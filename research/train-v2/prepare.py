"""Prepare frame-level chord features (CENS + CQT) for the chord CNN.

One audio load per sample produces two feature sets:
  data/chords.npz      CENS chroma (smoothed, robust)
  data/chords_cqt.npz  CQT chroma (log, no smoothing, keeps 7th/9th detail)
Each chord span is resampled to 64 frames.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path(r"D:\ChordStudio\research\boundary-dataset")
EXT = Path(r"D:\ChordStudio\research\extension-dataset")
HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)
QUALITIES = ["maj", "min", "7", "maj7", "min7", "9", "maj9", "min9", "sus2", "sus4"]
FRAMES = 64


def features(path):
    import librosa
    y, _ = librosa.load(str(path), sr=22050, mono=True)
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
    return cens, chroma, times


def resample(patch, frames=FRAMES):
    if patch.shape[1] == 0:
        return np.zeros((12, frames), dtype=np.float32)
    idx = np.linspace(0, patch.shape[1] - 1, frames)
    return np.stack([np.interp(idx, np.arange(patch.shape[1]), row) for row in patch], axis=0).astype(np.float32)


def main():
    sets = {name: [] for name in ("cens", "cqt")}
    roots, quals, songs, domains = [], [], [], []
    for base, meta_file, domain in ((DATA, DATA / "dataset.json", 0), (EXT, EXT / "dataset.json", 1)):
        if not meta_file.exists():
            continue
        for meta in json.loads(meta_file.read_text(encoding="utf-8"))["samples"]:
            labels = json.loads((base / meta["labels"]).read_text(encoding="utf-8"))
            cens, cqt, times = features(base / meta["audio"])
            song = meta["sample_id"].split("-")[1] if "-" in meta["sample_id"] else meta["sample_id"]
            for chord in labels["chords"]:
                quality = chord["quality"]
                if quality not in QUALITIES:
                    continue
                sel = (times >= chord["start"]) & (times < chord["end"])
                sets["cens"].append(resample(cens[:, sel] if sel.any() else cens[:, :1]))
                sets["cqt"].append(resample(cqt[:, sel] if sel.any() else cqt[:, :1]))
                roots.append(int(chord["root_pc"]))
                quals.append(QUALITIES.index(quality))
                songs.append(song)
                domains.append(domain)
            if len(roots) % 2000 < 40:
                print(f"  {len(roots)} chords", flush=True)
    common = dict(root=np.array(roots), quality=np.array(quals), song=np.array(songs),
                  domain=np.array(domains))
    for name, patches in sets.items():
        path = OUT / ("chords.npz" if name == "cens" else "chords_cqt.npz")
        np.savez_compressed(path, x=np.stack(patches), **common)
        print(f"saved {path}  {len(patches)} patches")
    print("quality counts:", {q: int((np.array(quals) == i).sum()) for i, q in enumerate(QUALITIES)})


if __name__ == "__main__":
    main()
