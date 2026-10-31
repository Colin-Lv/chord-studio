"""Frame-level chord CNN: CENS patch -> (root, quality), trained on GPU with pitch rotation."""
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


class ChordNet(nn.Module):
    def __init__(self, width=96):
        super().__init__()
        self.conv1 = nn.Conv1d(36, width, 5, padding=2)
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
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.cat([x.mean(dim=2), x.max(dim=2).values], dim=1)
        x = self.drop(F.relu(self.fc(x)))
        return self.root(x), self.quality(x)


def rotate(x, root, shift):
    """Pitch rotation: roll chroma and the root label together (free 12x augmentation)."""
    x = torch.roll(x, shifts=-shift, dims=1)
    return x, (root - shift) % 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--oversample", type=int, default=1,
                    help="repeat 7th-family samples N times (classes 2,3,4)")
    args = ap.parse_args()

    z = np.load(args.data)
    x = z["x"].astype(np.float32)          # N, 12, 64
    root = z["root"].astype(np.int64)
    quality = z["quality"].astype(np.int64)
    song = z["song"]
    print(f"chords {len(x)}  songs {len(set(song.tolist()))}  classes "
          f"{dict(zip(QUALITIES, np.bincount(quality, minlength=10)))}")

    if args.oversample > 1:
        seventh = np.isin(quality, [2, 3, 4])
        extra = np.where(seventh)[0].repeat(args.oversample - 1)
        x = np.concatenate([x, x[extra]])
        root = np.concatenate([root, root[extra]])
        quality = np.concatenate([quality, quality[extra]])
        song = np.concatenate([song, song[extra]])
        print(f'oversampled 7th family x{args.oversample}: {len(x)} chords')
    songs = sorted(set(song.tolist()))
    rng = np.random.default_rng(0)
    rng.shuffle(songs)
    folds = np.array_split(np.array(songs), args.folds)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    results = []
    for fold_index, held in enumerate(folds):
        test_mask = np.isin(song, held)
        train_mask = ~test_mask
        xt = torch.from_numpy(x[train_mask])
        rt = torch.from_numpy(root[train_mask])
        qt = torch.from_numpy(quality[train_mask])
        xv = torch.from_numpy(x[test_mask]).to(device)
        rv = torch.from_numpy(root[test_mask]).to(device)
        qv = torch.from_numpy(quality[test_mask]).to(device)

        model = ChordNet(args.width).to(device)
        counts = np.bincount(quality[train_mask], minlength=10).astype(np.float32)
        weights = torch.from_numpy((counts.sum() / (counts + 1.0)) ** 0.5).to(device)
        weights = weights / weights.mean()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best = 0.0
        for epoch in range(args.epochs):
            model.train()
            order = torch.randperm(len(xt))
            total = 0.0
            for start in range(0, len(order), args.batch):
                idx = order[start:start + args.batch]
                xb = xt[idx].to(device)
                rb = rt[idx].to(device)
                qb = qt[idx].to(device)
                shift = int(torch.randint(0, 12, (1,)).item())
                xb, rb = rotate(xb, rb, shift)
                noise = torch.randn_like(xb) * 0.01
                opt.zero_grad()
                pr, pq = model(xb + noise)
                loss = F.cross_entropy(pr, rb) + F.cross_entropy(pq, qb, weight=weights)
                loss.backward()
                opt.step()
                total += float(loss) * len(idx)
            sched.step()
            model.eval()
            with torch.no_grad():
                pr, pq = model(xv)
                root_acc = (pr.argmax(1) == rv).float().mean().item()
                q_acc = (pq.argmax(1) == qv).float().mean().item()
                joint = ((pr.argmax(1) == rv) & (pq.argmax(1) == qv)).float().mean().item()
            if joint > best:
                best = joint
                torch.save(dict(state=model.state_dict(), width=args.width), RUNS / f"fold{fold_index}.pt")
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                print(f"  fold {fold_index} epoch {epoch:2d} loss {total / len(xt):.3f} "
                      f"root {root_acc:.3f} quality {q_acc:.3f} joint {joint:.3f}", flush=True)
        results.append(dict(fold=fold_index, held=len(held), root=root_acc, quality=q_acc, joint=best))
        print(f"fold {fold_index} done: root {root_acc:.3f} quality {q_acc:.3f} joint(best) {best:.3f}")

    mean = {k: float(np.mean([r[k] for r in results])) for k in ("root", "quality", "joint")}
    print("mean:", {k: round(v, 3) for k, v in mean.items()})
    (RUNS / "cv.json").write_text(json.dumps(dict(results=results, mean=mean), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
