"""Which of the six stems actually matches MUSDB's 'other'? Score each one in turn."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf

EVAL = Path(r"D:\ChordStudio\research\separation\eval\stem_eval.py")
song = Path(r"D:\ChordStudio\datasets\musdb_song0")
src = song / "sw"

for cand in ("other", "guitar", "piano"):
    path = next(src.glob(f"*({cand})*.wav"))
    est = song / f"est_sw_{cand}"
    est.mkdir(exist_ok=True)
    data, rate = sf.read(str(path), always_2d=True, dtype="float32")
    sf.write(str(est / "vocals.wav"), data, rate, subtype="FLOAT")
    sf.write(str(est / "instrumental.wav"), data, rate, subtype="FLOAT")
    out = subprocess.run([sys.executable, str(EVAL), "--truth", str(song / "truth_two"),
                          "--stems", str(est)], capture_output=True, text=True)
    result = json.loads(out.stdout)
    value = result["stems"]["instrumental"]["si_sdr_db"]
    print(f"{cand:7s} vs truth instrumental: {value:6.2f} dB")
