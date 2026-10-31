"""Test-time augmentation on the existing 24-channel fold models.

Pitch rotation is exact for chroma, so averaging the 12 rotated predictions amplifies the
same invariance used in training. Reports single vs TTA accuracy per held-out fold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train as T

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "chords_both.npz"
RUNS = HERE / "runs"
QUALITIES = T.QUALITIES


def main():
    z = np.load(DATA)
    x = torch.from_numpy(z["x"].astype(np.float32))
    root = torch.from_numpy(z["root"].astype(np.int64))
    quality = torch.from_numpy(z["quality"].astype(np.int64))
    song = z["song"]
    songs = sorted(set(song.tolist()))
    rng = np.random.default_rng(0)
    rng.shuffle(songs)
    folds = np.array_split(np.array(songs), 5)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    single_joint, tta_joint = [], []
    for fold_index, held in enumerate(folds):
        model_path = RUNS / f"fold{fold_index}.pt"
        if not model_path.exists():
            continue
        state = torch.load(model_path, map_location=device)
        model = T.ChordNet(state.get("width", 160)).to(device)
        model.load_state_dict(state["state"])
        model.eval()
        mask = np.isin(song, held)
        xt = x[mask].to(device)
        rt = root[mask].to(device)
        qt = quality[mask].to(device)
        with torch.no_grad():
            pr, pq = model(xt)
            single = ((pr.argmax(1) == rt) & (pq.argmax(1) == qt)).float().mean().item()
            acc_root = torch.zeros(len(xt), 12, device=device)
            acc_q = torch.zeros(len(xt), 10, device=device)
            for shift in range(12):
                xs = torch.cat([torch.roll(xt[:, :12], shifts=shift, dims=1), torch.roll(xt[:, 12:], shifts=shift, dims=1)], dim=1)
                pr_s, pq_s = model(xs)
                acc_root += torch.roll(F.softmax(pr_s, dim=1), shifts=-shift, dims=1)
                acc_q += F.softmax(pq_s, dim=1)
            tta = ((acc_root.argmax(1) == rt) & (acc_q.argmax(1) == qt)).float().mean().item()
        single_joint.append(single)
        tta_joint.append(tta)
        print(f"fold {fold_index}: n {len(xt)} single {single:.4f}  TTA {tta:.4f}")
    if single_joint:
        print(f"mean: single {np.mean(single_joint):.4f}  TTA {np.mean(tta_joint):.4f}")


if __name__ == "__main__":
    main()
