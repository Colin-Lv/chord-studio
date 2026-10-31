"""Score htdemucs_ft on the MUSDB songs we already unpacked, in the same two-stem framing."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

EVAL = Path(r"D:\ChordStudio\research\separation\eval\stem_eval.py")
GPU_PY = r"D:\ChordStudio\runtime\gpu\Scripts\python.exe"
SONGS = [Path(p) for p in sys.argv[1:]]


def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr[-1500:])
    return result


def main():
    for song in SONGS:
        mixture = song / "mixture.wav"
        out = song / "demucs_ft" / "htdemucs_ft" / "mixture"
        if not out.is_dir():
            run([GPU_PY, "-m", "demucs.separate", "-n", "htdemucs_ft", "--device", "cuda",
                 "-o", str(song / "demucs_ft"), str(mixture)])
        est = song / "est_ft"
        est.mkdir(exist_ok=True)
        vocals, rate = sf.read(str(out / "vocals.wav"), always_2d=True, dtype="float32")
        bed = sum(sf.read(str(out / f"{name}.wav"), always_2d=True, dtype="float32")[0]
                  for name in ("drums", "bass", "other"))
        length = min(len(vocals), len(bed))
        sf.write(str(est / "vocals.wav"), vocals[:length], rate, subtype="FLOAT")
        sf.write(str(est / "instrumental.wav"), bed[:length], rate, subtype="FLOAT")
        result = run([sys.executable, str(EVAL), "--truth", str(song / "truth_two"),
                      "--stems", str(est)])
        data = json.loads(result.stdout)
        print(f"{song.name}: ft vocals {data['stems']['vocals']['si_sdr_db']:.2f}  "
              f"instr {data['stems']['instrumental']['si_sdr_db']:.2f}  "
              f"mean {data['mean_si_sdr_db']:.2f}")
        (song / "scores_ft.json").write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                             encoding="utf-8")


if __name__ == "__main__":
    main()
