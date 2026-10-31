"""Evaluate the synthetic-trained chord CNN on 虚拟's golden spans (cross-domain)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, r'D:\ChordStudio\research\key-v3\chord_seg')
import train as T

OUT = Path(r"D:\ChordStudio\research\key-v3\chord_seg")
RUNS = Path(r"D:\ChordStudio\research\train-v2\runs")
QUALITIES = T.QUALITIES
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FRAMES = 64


def cens(path):
    import librosa
    y, _ = librosa.load(str(path), sr=22050, mono=True)
    y, _ = librosa.effects.hpss(y)
    ch = librosa.feature.chroma_cens(y=y, sr=22050, hop_length=512)
    ch = ch / (ch.sum(axis=0, keepdims=True) + 1e-9)
    return ch, np.arange(ch.shape[1]) * 512 / 22050


def patch(ch, times, a, b):
    sel = (times >= a) & (times < b)
    block = ch[:, sel] if sel.any() else ch[:, :1]
    idx = np.linspace(0, block.shape[1] - 1, FRAMES)
    out = np.stack([np.interp(idx, np.arange(block.shape[1]), row) for row in block], axis=0)
    return out.astype(np.float32)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for path in sorted(RUNS.glob("fold*.pt")):
        state = torch.load(path, map_location=device)
        model = T.ChordNet(state.get("width", 128)).to(device)
        model.load_state_dict(state["state"])
        model.eval()
        models.append(model)
    print("models:", len(models))

    comp, tcomp = cens(OUT / "stems" / "other.wav")
    import head_decode as H
    bass_ch, bass_t = H.bass_chroma(OUT / "stems" / "bass.wav")
    ref = [(float(a), float(b), c) for a, b, c in json.loads((OUT / "ref.json").read_text(encoding="utf-8"))]

    exact = family = root_ok = 0
    rows = []
    for a, b, truth in ref:
        x = torch.from_numpy(patch(comp, tcomp, a, b)[None, :, :]).to(device)
        probs_root = torch.zeros(12, device=device)
        probs_q = torch.zeros(10, device=device)
        with torch.no_grad():
            for model in models:
                pr, pq = model(x)
                probs_root += F.softmax(pr[0], dim=0)
                probs_q += F.softmax(pq[0], dim=0)
        root = int(probs_root.argmax())
        quality = int(probs_q.argmax())
        pred = f"{NAMES[root]}:{QUALITIES[quality]}"
        tr, _, tq = truth.partition(":")
        exact += pred == truth
        family += (NAMES[root] == tr) and (QUALITIES[quality][:3] == tq[:3])
        base = bass_ch[(bass_t >= a) & (bass_t < b)].mean(axis=0)
        root_ok += NAMES[int(np.argmax(base))] == tr
        rows.append((round(a, 2), truth, pred))
    n = len(ref)
    print(f"虚拟 CNN: exact {exact}/{n} = {exact / n:.3f}  family {family / n:.3f}  "
          f"bass-root {root_ok / n:.3f}")
    for row in rows[:10]:
        print("   ", row)


if __name__ == "__main__":
    main()
