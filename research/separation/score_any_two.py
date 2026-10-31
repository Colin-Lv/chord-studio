"""Score an arbitrary two-stem model output against MUSDB ground truth."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf

EVAL = Path(r"D:\ChordStudio\research\separation\eval\stem_eval.py")


def main():
    song = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    tag = sys.argv[3]
    truth = song / "truth_two"
    est = song / f"est_{tag}"
    est.mkdir(exist_ok=True)
    vocals = next(outdir.glob("*vocals*.wav"), None)
    bed = next((p for p in outdir.glob("*.wav") if p != vocals), None)
    if not vocals or not bed:
        raise SystemExit("missing outputs")
    for source, target in ((vocals, "vocals.wav"), (bed, "instrumental.wav")):
        data, rate = sf.read(str(source), always_2d=True, dtype="float32")
        sf.write(str(est / target), data, rate, subtype="FLOAT")
    result = subprocess.run([sys.executable, str(EVAL), "--truth", str(truth), "--stems", str(est)],
                            capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr[-1500:])
    data = json.loads(result.stdout)
    print(f"{tag}: vocals {data['stems']['vocals']['si_sdr_db']:.2f}  "
          f"instr {data['stems']['instrumental']['si_sdr_db']:.2f}  "
          f"mean {data['mean_si_sdr_db']:.2f}")


if __name__ == "__main__":
    main()
