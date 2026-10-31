"""Split the 60-channel model's errors by texture x quality and by chord length.

The arpeggio renderer plays one upper chord tone per beat, so a 2-beat chord never sounds
all of its tones. If that is the arpeggio gap, the dataset label is unrecoverable there.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train3 as T

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DATA = Path(r"D:\ChordStudio\research\boundary-dataset")
EXT = Path(r"D:\ChordStudio\research\extension-dataset")


def lengths_by_sample():
    """Map (sample_id) -> list of chord durations in beats, in label order."""
    out = {}
    for base in (DATA, EXT):
        meta = json.loads((base / "dataset.json").read_text(encoding="utf-8"))
        for entry in meta["samples"]:
            labels = json.loads((base / entry["labels"]).read_text(encoding="utf-8"))
            out[entry["sample_id"]] = [c["end"] - c["start"] for c in labels["chords"]]
    return out


def main():
    z = np.load(HERE / "data" / "chords_full.npz")
    x = z["x"].astype(np.float16)
    root = z["root"].astype(np.int64)
    quality = z["quality"].astype(np.int64)
    song = z["song"]
    texture = z["texture"]
    melody = z["melody"].astype(np.int64)
    songs = sorted(set(song.tolist()))
    rng = np.random.default_rng(0)
    rng.shuffle(songs)
    folds = np.array_split(np.array(songs), 5)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    oks, quals, texs = [], [], []
    for fold_index, held in enumerate(folds):
        state = torch.load(RUNS / f"full60_fold{fold_index}.pt", map_location=device,
                           weights_only=False)
        model = T.ChordNet(state["channels"], state["width"], cond=state.get("cond", 0)).to(device)
        model.load_state_dict(state["state"])
        model.eval()
        mask = np.isin(song, held)
        xt = torch.from_numpy(x[mask].copy()).to(device).float()
        rt = torch.from_numpy(root[mask]).to(device)
        qt = torch.from_numpy(quality[mask]).to(device)
        with torch.no_grad():
            acc_root = torch.zeros(len(xt), 12, device=device)
            acc_q = torch.zeros(len(xt), 10, device=device)
            for shift in range(12):
                pr_s, pq_s = model(T.rotate_blocks(xt, shift))
                acc_root += torch.roll(F.softmax(pr_s, dim=1), shifts=shift, dims=1)
                acc_q += F.softmax(pq_s, dim=1)
            r_hat, q_hat = acc_root.argmax(1), acc_q.argmax(1)
        oks.append(((r_hat == rt) & (q_hat == qt)).cpu().numpy())
        quals.append(qt.cpu().numpy())
        texs.append(texture[mask])

    ok = np.concatenate(oks)
    qual = np.concatenate(quals)
    tex = np.concatenate(texs)
    print("texture x quality (joint):")
    header = "quality".ljust(8) + "".join(t.rjust(10) for t in ("block", "arpeggio", "strum")) + "      all"
    print(header)
    for q, name in enumerate(T.QUALITIES):
        cells = []
        for t in ("block", "arpeggio", "strum"):
            sel = (qual == q) & (tex == t)
            cells.append(f"{ok[sel].mean():10.3f}" if sel.sum() else "         -")
        sel = qual == q
        print(f"{name:8s}" + "".join(cells) + f"{ok[sel].mean():9.3f}   n={int(sel.sum())}")
    for t in ("block", "arpeggio", "strum"):
        sel = tex == t
        print(f"{t:8s} all {ok[sel].mean():.4f}  n={int(sel.sum())}")


if __name__ == "__main__":
    main()
