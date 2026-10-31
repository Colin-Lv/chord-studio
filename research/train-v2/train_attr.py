"""Attribute-factorised chord model: third / seventh / ninth / sus heads.

A chord is (third, seventh, ninth, sus), so the 10 qualities become combinations of
attributes. Rare colours (9ths) get their own head instead of competing in a 10-way
softmax against 7000 triads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "chords.npz"
RUNS = HERE / "runs"
RUNS.mkdir(exist_ok=True)
QUALITIES = ["maj", "min", "7", "maj7", "min7", "9", "maj9", "min9", "sus2", "sus4"]
# quality -> (third 0=maj 1=min 2=none, seventh 0=none 1=b7 2=maj7, ninth 0=none 1=9, sus 0=none 1=sus2 2=sus4)
ATTR = {
    "maj": (0, 0, 0, 0), "min": (1, 0, 0, 0), "7": (0, 1, 0, 0), "maj7": (0, 2, 0, 0),
    "min7": (1, 1, 0, 0), "9": (0, 1, 1, 0), "maj9": (0, 2, 1, 0), "min9": (1, 1, 1, 0),
    "sus2": (2, 0, 0, 1), "sus4": (2, 0, 0, 2),
}


class ChordNet(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.conv1 = nn.Conv1d(12, width, 5, padding=2)
        self.conv2 = nn.Conv1d(width, width, 5, padding=2)
        self.conv3 = nn.Conv1d(width, width, 3, padding=1)
        self.bn = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(3)])
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(width * 2, 256)
        self.root = nn.Linear(256, 12)
        self.third = nn.Linear(256, 3)
        self.seventh = nn.Linear(256, 3)
        self.ninth = nn.Linear(256, 2)
        self.sus = nn.Linear(256, 3)

    def forward(self, x):
        x = F.relu(self.bn[0](self.conv1(x)))
        x = F.relu(self.bn[1](self.conv2(x)))
        x = F.relu(self.bn[2](self.conv3(x)))
        h = torch.cat([x.mean(dim=2), x.max(dim=2).values], dim=1)
        h = self.drop(F.relu(self.fc(h)))
        return self.root(h), self.third(h), self.seventh(h), self.ninth(h), self.sus(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    z = np.load(DATA)
    x = z["x"].astype(np.float32)
    root = z["root"].astype(np.int64)
    quality = z["quality"].astype(np.int64)
    song = z["song"]
    attrs = np.array([ATTR[q] for q in QUALITIES], dtype=np.int64)[quality]
    print(f"chords {len(x)} songs {len(set(song.tolist()))}")

    songs = sorted(set(song.tolist()))
    rng = np.random.default_rng(0)
    rng.shuffle(songs)
    folds = np.array_split(np.array(songs), 5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    results = []
    for fold_index, held in enumerate(folds):
        test_mask = np.isin(song, held)
        train_mask = ~test_mask
        xt = torch.from_numpy(x[train_mask]); rt = torch.from_numpy(root[train_mask])
        at = torch.from_numpy(attrs[train_mask])
        xv = torch.from_numpy(x[test_mask]).to(device); rv = torch.from_numpy(root[test_mask]).to(device)
        av = torch.from_numpy(attrs[test_mask]).to(device)
        qv = torch.from_numpy(quality[test_mask]).to(device)

        model = ChordNet(args.width).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best = 0.0
        for epoch in range(args.epochs):
            model.train()
            order = torch.randperm(len(xt))
            for start in range(0, len(order), args.batch):
                idx = order[start:start + args.batch]
                xb = xt[idx].to(device)
                rb = rt[idx].to(device)
                ab = at[idx].to(device)
                shift = int(torch.randint(0, 12, (1,)).item())
                xb = torch.roll(xb, shifts=-shift, dims=1)
                rb = (rb - shift) % 12
                opt.zero_grad()
                pr, p3, p7, p9, ps = model(xb + torch.randn_like(xb) * 0.01)
                loss = (F.cross_entropy(pr, rb) + F.cross_entropy(p3, ab[:, 0])
                        + F.cross_entropy(p7, ab[:, 1]) + F.cross_entropy(p9, ab[:, 2])
                        + F.cross_entropy(ps, ab[:, 3]))
                loss.backward()
                opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                pr, p3, p7, p9, ps = model(xv)
                root_acc = (pr.argmax(1) == rv).float().mean().item()
                # rebuild the quality index from predicted attributes
                pred_q = combine(p3.argmax(1), p7.argmax(1), p9.argmax(1), ps.argmax(1)).to(device)
                q_acc = (pred_q == qv).float().mean().item()
                joint = ((pr.argmax(1) == rv) & (pred_q == qv)).float().mean().item()
            if joint > best:
                best = joint
                torch.save(dict(state=model.state_dict(), width=args.width), RUNS / f"attr_fold{fold_index}.pt")
            if epoch % 10 == 0 or epoch == args.epochs - 1:
                print(f"  fold {fold_index} epoch {epoch:2d} root {root_acc:.3f} quality {q_acc:.3f} "
                      f"joint {joint:.3f}", flush=True)
        results.append(dict(fold=fold_index, root=root_acc, quality=q_acc, joint=best))
        print(f"fold {fold_index} done: root {root_acc:.3f} quality {q_acc:.3f} joint(best) {best:.3f}")

    mean = {k: float(np.mean([r[k] for r in results])) for k in ("root", "quality", "joint")}
    print("mean:", {k: round(v, 3) for k, v in mean.items()})
    (RUNS / "cv_attr.json").write_text(json.dumps(dict(results=results, mean=mean), indent=1), encoding="utf-8")


def combine(third, seventh, ninth, sus):
    """attribute predictions -> quality index (10 classes)."""
    out = torch.zeros_like(third)
    out = torch.where((third == 0) & (seventh == 0) & (ninth == 0), torch.tensor(0), out)   # maj
    out = torch.where((third == 1) & (seventh == 0) & (ninth == 0), torch.tensor(1), out)   # min
    out = torch.where((third == 0) & (seventh == 1) & (ninth == 0), torch.tensor(2), out)   # 7
    out = torch.where((third == 0) & (seventh == 2) & (ninth == 0), torch.tensor(3), out)   # maj7
    out = torch.where((third == 1) & (seventh == 1) & (ninth == 0), torch.tensor(4), out)   # min7
    out = torch.where((third == 0) & (seventh == 1) & (ninth == 1), torch.tensor(5), out)   # 9
    out = torch.where((third == 0) & (seventh == 2) & (ninth == 1), torch.tensor(6), out)   # maj9
    out = torch.where((third == 1) & (seventh == 1) & (ninth == 1), torch.tensor(7), out)   # min9
    out = torch.where((sus == 1) & (third == 2), torch.tensor(8), out)                      # sus2
    out = torch.where((sus == 2) & (third == 2), torch.tensor(9), out)                      # sus4
    return out


if __name__ == "__main__":
    main()
