"""Aggregate the MUSDB benchmark across the songs that have demucs / ft / roformer scores."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"D:\ChordStudio\datasets")


def main():
    rows = []
    for index in range(15):
        song = ROOT / f"musdb_song{index}"
        if not (song / "scores.json").is_file() or not (song / "scores_ft.json").is_file():
            continue
        base = json.loads((song / "scores.json").read_text(encoding="utf-8"))
        ft = json.loads((song / "scores_ft.json").read_text(encoding="utf-8"))
        rows.append((base["song"], base["demucs"], base["roformer"], ft))

    header = f"{'song':44s} {'d.v':>6s} {'ft.v':>6s} {'rf.v':>6s} | {'d.i':>6s} {'ft.i':>6s} {'rf.i':>6s}"
    print(header)
    totals = dict.fromkeys(("dv", "fv", "rv", "di", "fi", "ri"), 0.0)
    for name, demucs, roformer, ft in rows:
        values = {
            "dv": demucs["stems"]["vocals"]["si_sdr_db"],
            "di": demucs["stems"]["instrumental"]["si_sdr_db"],
            "fv": ft["stems"]["vocals"]["si_sdr_db"],
            "fi": ft["stems"]["instrumental"]["si_sdr_db"],
            "rv": roformer["stems"]["vocals"]["si_sdr_db"],
            "ri": roformer["stems"]["instrumental"]["si_sdr_db"],
        }
        for key, value in values.items():
            totals[key] += value
        print(f"{name[:44]:44s} {values['dv']:6.2f} {values['fv']:6.2f} {values['rv']:6.2f} | "
              f"{values['di']:6.2f} {values['fi']:6.2f} {values['ri']:6.2f}")
    count = max(1, len(rows))
    print(f"{'MEAN':44s} {totals['dv'] / count:6.2f} {totals['fv'] / count:6.2f} "
          f"{totals['rv'] / count:6.2f} | {totals['di'] / count:6.2f} "
          f"{totals['fi'] / count:6.2f} {totals['ri'] / count:6.2f}   (n={len(rows)})")


if __name__ == "__main__":
    main()

