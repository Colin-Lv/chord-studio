"""Where does the remaining error live? Split held-out accuracy by melody / texture / domain."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import train as T

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "chords_both.npz"
RUNS = HERE / "runs"


def main():
    z = np.load(DATA)
    x = torch.from_numpy(z["x"].astype(np.float32))
    root = torch.from_numpy(z["root"].astype(np.int64))
    quality = torch.from_numpy(z["quality"].astype(np.int64))
    song = z["song"]
    domain = z["domain"]
    # sample id is not stored; melody samples were named pop909-XXX-<texture>-mel
    songs = sorted(set(song.tolist()))
    rng = np.random.default_rng(0)
    rng.shuffle(songs)
    folds = np.array_split(np.array(songs), 5)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for fold_index, held in enumerate(folds):
        path = RUNS / f"fold{fold_index}.pt"
        if not path.exists():
            continue
        state = torch.load(path, map_location=device)
        model = T.ChordNet(state.get("width", 160)).to(device)
        model.load_state_dict(state["state"])
        model.eval()
        mask = np.isin(song, held)
        xt = x[mask].to(device)
        rt = root[mask].to(device)
        qt = quality[mask].to(device)
        with torch.no_grad():
            pr, pq = model(xt)
        ok = ((pr.argmax(1) == rt) & (pq.argmax(1) == qt)).cpu().numpy()
        rows.append((ok, rt.cpu().numpy(), qt.cpu().numpy(), domain[mask]))
    if not rows:
        print("no fold models")
        return
    ok = np.concatenate([r[0] for r in rows])
    dom = np.concatenate([r[3] for r in rows])
    qt = np.concatenate([r[2] for r in rows])
    print(f"overall joint {ok.mean():.4f}  (n={len(ok)})")
    for label, sel in (("base (no 9th)", dom == 0), ("extension", dom == 1)):
        print(f"  {label:14s} {ok[sel].mean():.4f}  n={int(sel.sum())}")
    print("per quality:")
    for q, name in enumerate(T.QUALITIES):
        sel = qt == q
        if sel.sum():
            print(f"  {name:5s} {ok[sel].mean():.4f}  n={int(sel.sum())}")


if __name__ == "__main__":
    main()
