"""Combine CENS and CQT patches into one 24-channel dataset."""
from pathlib import Path

import numpy as np

DATA = Path(r"D:\ChordStudio\research\train-v2\data")
a = np.load(DATA / "chords.npz")
b = np.load(DATA / "chords_cqt.npz")
assert len(a["x"]) == len(b["x"])
x = np.concatenate([a["x"], b["x"]], axis=1)      # N, 24, 64
np.savez_compressed(DATA / "chords_both.npz", x=x, root=a["root"], quality=a["quality"],
                    song=a["song"], domain=a["domain"])
print("saved", x.shape)
