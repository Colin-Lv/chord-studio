"""Train the 60-channel chord CNN (CENS + CQT + NNLS registers) with pitch rotation.

Channels are grouped in 12-pitch-class blocks, so pitch rotation = rolling each block by
the same semitone shift, exactly like the 12-dim runs. Optional extra channels can append a
conditioning vector (melody flag) for the label-noise experiment.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
RUNS.mkdir(exist_ok=True)
QUALITIES = ["maj", "min", "7", "maj7", "min7", "9", "maj9", "min9", "sus2", "sus4"]


class ChordNet(nn.Module):
    def __init__(self, channels=60, width=160, cond=0):
        super().__init__()
        self.channels = channels
        self.cond = cond
        self.conv1 = nn.Conv1d(channels + cond, width, 5, padding=2)
        self.conv2 = nn.Conv1d(width, width, 5, padding=2)
        self.conv3 = nn.Conv1d(width, width, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(width)
        self.bn2 = nn.BatchNorm1d(width)
        self.bn3 = nn.BatchNorm1d(width)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(width * 2, 256)
        self.root = nn.Linear(256, 12)
        self.quality = nn.Linear(256, 10)

    def forward(self, x):
        if self.cond:
            x = torch.cat([x, self.cond_vec.expand(len(x), -1, x.shape[2])], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.cat([x.mean(dim=2), x.max(dim=2).values], dim=1)
        x = self.drop(F.relu(self.fc(x)))
        return self.root(x), self.quality(x)


def rotate_blocks(x, shift, blocks=None):
    """Roll each 12-channel block by `shift` semitones (in-place safe)."""
    shifts = blocks if blocks is not None else x.shape[1] // 12
    out = x.clone()
    for b in range(shifts):
        out[:, 12 * b:12 * (b + 1)] = torch.roll(out[:, 12 * b:12 * (b + 1)], shifts=-shift, dims=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data" / "chords_full.npz"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--melody-channel", action="store_true",
                    help="append the (oracle) melody flag as a conditioning channel")
    args = ap.parse_args()

    z = np.load(args.data)
    x = z["x"].astype(np.float16)                     # N, 60, 64
    root = z["root"].astype(np.int64)
    quality = z["quality"].astype(np.int64)
    song = z["song"]
    texture = z["texture"]
    melody = z["melody"].astype(np.int64)
    domain = z["domain"].astype(np.int64)
    channels = int(x.shape[1])
    print(f"chords {len(x)}  channels {channels}  songs {len(set(song.tolist()))}")
    print("classes", {q: int((quality == i).sum()) for i, q in enumerate(QUALITIES)})

    cond = 0
    cond_arr = None
    if args.melody_channel:
        cond = 1
        cond_arr = melody.astype(np.float32).reshape(-1, 1, 1)

    songs = sorted(set(song.tolist()))
    rng = np.random.default_rng(0)
    rng.shuffle(songs)
    folds = np.array_split(np.array(songs), args.folds)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    results = []
    for fold_index, held in enumerate(folds):
        test_mask = np.isin(song, held)
        train_mask = ~test_mask
        xt = torch.from_numpy(x[train_mask].copy())
        rt = torch.from_numpy(root[train_mask])
        qt = torch.from_numpy(quality[train_mask])
        xv = torch.from_numpy(x[test_mask].copy()).to(device).float()
        rv = torch.from_numpy(root[test_mask]).to(device)
        qv = torch.from_numpy(quality[test_mask]).to(device)
        if cond:
            ct = torch.from_numpy(cond_arr[train_mask].copy())
            cv = torch.from_numpy(cond_arr[test_mask].copy()).to(device).float()

        model = ChordNet(channels, args.width, cond=cond).to(device)
        counts = np.bincount(quality[train_mask], minlength=10).astype(np.float32)
        weights = torch.from_numpy((counts.sum() / (counts + 1.0)) ** 0.5).to(device)
        weights = weights / weights.mean()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best = 0.0
        t0 = time.time()
        for epoch in range(args.epochs):
            model.train()
            order = torch.randperm(len(xt))
            total = 0.0
            for start in range(0, len(order), args.batch):
                idx = order[start:start + args.batch]
                xb = xt[idx].to(device).float()
                rb = rt[idx].to(device)
                qb = qt[idx].to(device)
                shift = int(torch.randint(0, 12, (1,)).item())
                xb = rotate_blocks(xb, shift)
                rb = (rb - shift) % 12
                noise = torch.randn_like(xb) * 0.01
                opt.zero_grad()
                if cond:
                    model.cond_vec = ct[idx].to(device).float()
                pr, pq = model(xb + noise)
                loss = F.cross_entropy(pr, rb) + F.cross_entropy(pq, qb, weight=weights)
                loss.backward()
                opt.step()
                total += float(loss) * len(idx)
            sched.step()
            model.eval()
            with torch.no_grad():
                if cond:
                    model.cond_vec = cv
                pr, pq = model(xv)
                root_acc = (pr.argmax(1) == rv).float().mean().item()
                q_acc = (pq.argmax(1) == qv).float().mean().item()
                joint = ((pr.argmax(1) == rv) & (pq.argmax(1) == qv)).float().mean().item()
            if joint > best:
                best = joint
                torch.save(dict(state=model.state_dict(), width=args.width, channels=channels,
                                cond=cond), RUNS / f"{args.tag}_fold{fold_index}.pt")
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                print(f"  fold {fold_index} epoch {epoch:2d} loss {total / len(xt):.3f} "
                      f"root {root_acc:.3f} quality {q_acc:.3f} joint {joint:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        results.append(dict(fold=fold_index, held=int(len(held)), root=root_acc,
                            quality=q_acc, joint=best))
        print(f"fold {fold_index} done: root {root_acc:.3f} quality {q_acc:.3f} joint(best) {best:.3f}")

    mean = {k: float(np.mean([r[k] for r in results])) for k in ("root", "quality", "joint")}
    print("mean:", {k: round(v, 4) for k, v in mean.items()})
    out = dict(results=results, mean=mean, channels=channels, width=args.width,
               epochs=args.epochs, cond=cond, tag=args.tag)
    (RUNS / f"cv_{args.tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
