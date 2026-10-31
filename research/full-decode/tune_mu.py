"""Trade-off check for the full decoder's tone term: does a higher mu keep more chord tones?

Reports exact / family / boundary F1 / tone recall on 虚拟's golden for mu in 0.5..0.9.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Import the research decoder before decode_lib: decode_lib puts the app's web directory on
# sys.path, which also contains a full_decode.py (the shipped decoder).
import full_decode as F  # noqa: E402
import decode_lib as D  # noqa: E402
from tone_recovery import pcs  # noqa: E402
assert hasattr(F, "_prepare"), "wrong full_decode module imported"


def tone_metrics(rows, ref):
    recall = 0.0
    n = 0
    for a, b, label in ref:
        mid = (a + b) / 2
        pred = next((lab for x, y, lab in rows if x <= mid < y), "N")
        g, p = pcs(label), pcs(pred)
        if not g:
            continue
        n += 1
        recall += len(g & p) / len(g)
    return recall / n


def main():
    art = F._prepare(D.load_run(D.latest_xuni_run()))
    model = F.load_model()
    ref = D.load_lab(D.GOLDEN)
    print(f"{'mu':>4} {'segments':>9} {'exact':>7} {'family':>7} {'F1':>6} {'toneRecall':>11} {'extShare':>9}")
    for mu in (0.5, 0.6, 0.7, 0.8, 1.0):
        rows = F.decode_semimarkov(art, model, mu=mu)
        m = D.score(rows, ref)
        ext = sum(1 for _, _, label in rows if pcs(label) and len(pcs(label)) > 3) / len(rows)
        print(f"{mu:4.1f} {m['segments']:9d} {m['exact']:7.3f} {m['family']:7.3f} "
              f"{m['f1']:6.3f} {tone_metrics(rows, ref):11.3f} {ext:9.3f}")


if __name__ == "__main__":
    main()
