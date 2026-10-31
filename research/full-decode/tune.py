"""Sweep the full decoder's two free knobs (boundary penalty, minimum span)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import boundary_lib as B  # noqa: E402
import decode_lib as D  # noqa: E402
import full_decode as F  # noqa: E402


def main():
    run = D.latest_xuni_run()
    art = D.load_run(run)
    model = F.load_model()
    ref = D.load_lab(D.GOLDEN)
    beat = float(np.median(np.diff(art["beats"])))
    bars = D.phase_corrected_bars(art["beats"], art["beat_pos"], art["chroma"])
    cand = B.bar_candidates(bars)
    times = np.array([t for t, _ in cand], dtype=float)
    X = B.candidate_features(art["act"], art["note_times"], cand, beat)
    p = model.predict_proba(X)[:, 1]
    logit = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))

    truth = D.load_lab(D.GOLDEN)
    rows = []
    for mult in (1.5, 2.0, 2.5, 3.0, 4.0):
        for lam in (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0):
            chosen = F.boundary_dp(times, logit, lam, mult * beat - 1e-6)
            bounds = [float(times[0])] + chosen + [float(min(times[-1], art["ends"][-1]))]
            bounds = sorted(set(round(b, 4) for b in bounds))
            if "bass_cells" not in art:
                art["bass_cells"] = D.cell_bass(art["act"], art["note_times"],
                                                art["starts"], art["ends"])
            segs = []
            for a, b in zip(bounds, bounds[1:]):
                if b - a <= 1e-6:
                    continue
                cells = [i for i, (x, y) in enumerate(zip(art["starts"], art["ends"]))
                         if x < b and y > a]
                if not cells:
                    continue
                segs.append((a, b, D.pooled_label(art["prob"], art["bass_cells"], cells, 1.5)))
            segs = D.merge_rows(segs)
            m = D.score(segs, truth)
            rows.append(dict(mult=mult, lam=lam, **m))
            print(f"minspan {mult:4.1f} beats  lam {lam:+5.1f}: segments {m['segments']:3d} "
                  f"exact {m['exact']:.3f} family {m['family']:.3f} P {m['precision']:.2f} "
                  f"R {m['recall']:.2f} F1 {m['f1']:.2f}")
    (HERE / "sweep.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
