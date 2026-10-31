"""Evaluate the 60-channel model: single vs TTA, plus every split that matters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train3 as T

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data" / "chords_full.npz"))
    ap.add_argument("--tag", default="full")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    z = np.load(args.data)
    x = z["x"].astype(np.float16)
    root = z["root"].astype(np.int64)
    quality = z["quality"].astype(np.int64)
    song = z["song"]
    texture = z["texture"]
    melody = z["melody"].astype(np.int64)
    domain = z["domain"].astype(np.int64)
    songs = sorted(set(song.tolist()))
    rng = np.random.default_rng(0)
    rng.shuffle(songs)
    folds = np.array_split(np.array(songs), args.folds)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    oks, preds_q, preds_r, roots, quals = [], [], [], [], []
    texs, mels, doms = [], [], []
    for fold_index, held in enumerate(folds):
        path = RUNS / f"{args.tag}_fold{fold_index}.pt"
        if not path.exists():
            print("missing", path)
            continue
        state = torch.load(path, map_location=device, weights_only=False)
        model = T.ChordNet(state.get("channels", x.shape[1]), state.get("width", 160),
                           cond=state.get("cond", 0)).to(device)
        model.load_state_dict(state["state"])
        model.eval()
        mask = np.isin(song, held)
        xt = torch.from_numpy(x[mask].copy()).to(device).float()
        rt = torch.from_numpy(root[mask]).to(device)
        qt = torch.from_numpy(quality[mask]).to(device)
        with torch.no_grad():
            pr, pq = model(xt)
            single = ((pr.argmax(1) == rt) & (pq.argmax(1) == qt)).float().mean().item()
            print(f"fold {fold_index}: n {len(xt)} single {single:.4f}")
            if args.tta:
                acc_root = torch.zeros(len(xt), 12, device=device)
                acc_q = torch.zeros(len(xt), 10, device=device)
                for shift in range(12):
                    xs = T.rotate_blocks(xt, shift)
                    pr_s, pq_s = model(xs)
                    acc_root += torch.roll(F.softmax(pr_s, dim=1), shifts=shift, dims=1)
                    acc_q += F.softmax(pq_s, dim=1)
                pr, pq = acc_root, acc_q
            r_hat = pr.argmax(1)
            q_hat = pq.argmax(1)
        oks.append(((r_hat == rt) & (q_hat == qt)).cpu().numpy())
        preds_r.append(r_hat.cpu().numpy())
        preds_q.append(q_hat.cpu().numpy())
        roots.append(rt.cpu().numpy())
        quals.append(qt.cpu().numpy())
        texs.append(texture[mask])
        mels.append(melody[mask])
        doms.append(domain[mask])

    ok = np.concatenate(oks)
    pred_r = np.concatenate(preds_r)
    pred_q = np.concatenate(preds_q)
    root_t = np.concatenate(roots)
    qual_t = np.concatenate(quals)
    tex = np.concatenate(texs)
    mel = np.concatenate(mels)
    dom = np.concatenate(doms)
    print(f"\noverall joint {ok.mean():.4f} (n={len(ok)})  root {(pred_r == root_t).mean():.4f} "
          f"quality {(pred_q == qual_t).mean():.4f}")
    for label, sel in (("base (no 9th)", dom == 0), ("extension", dom == 1),
                       ("melody=0", mel == 0), ("melody=1", mel == 1),
                       ("block", tex == "block"), ("arpeggio", tex == "arpeggio"),
                       ("strum", tex == "strum")):
        if sel.sum():
            print(f"  {label:14s} {ok[sel].mean():.4f}  n={int(sel.sum())}")
    print("per quality:")
    rows = []
    for q, name in enumerate(T.QUALITIES):
        sel = qual_t == q
        if sel.sum():
            acc = ok[sel].mean()
            rows.append(dict(quality=name, joint=float(acc), n=int(sel.sum()),
                             root=float((pred_r[sel] == root_t[sel]).mean()),
                             quality_acc=float((pred_q[sel] == qual_t[sel]).mean())))
            print(f"  {name:5s} {acc:.4f}  n={int(sel.sum()):5d}  "
                  f"root {(pred_r[sel] == root_t[sel]).mean():.3f} "
                  f"qual {(pred_q[sel] == qual_t[sel]).mean():.3f}")
    out = dict(overall=dict(joint=float(ok.mean()), root=float((pred_r == root_t).mean()),
                            quality=float((pred_q == qual_t).mean()), n=int(len(ok))),
               per_quality=rows, tta=bool(args.tta), tag=args.tag)
    (RUNS / f"eval_{args.tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
