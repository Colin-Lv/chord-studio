"""Run the two-stem benchmark (demucs vs BS-RoFormer) on one MUSDB song."""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

EVAL = Path(r"D:\ChordStudio\research\separation\eval\stem_eval.py")
ROFORMER_DIR = Path(r"D:\ChordStudio\research\separation\bsroformer")
GPU_PY = r"D:\ChordStudio\runtime\gpu\Scripts\python.exe"


def extract(parquet: Path, out: Path) -> str:
    table = pq.read_table(parquet)
    name = table.column("name")[0].as_py()
    out.mkdir(parents=True, exist_ok=True)
    for field in ("mixture", "vocals", "drums", "bass", "other"):
        blob = table.column(field).combine_chunks()[0].as_py()
        data, rate = sf.read(io.BytesIO(blob["bytes"]), always_2d=True, dtype="float32")
        sf.write(str(out / f"{field}.wav"), data, rate, subtype="FLOAT")
    return name


def run(cmd, **kwargs):
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode:
        print(result.stdout[-1500:], result.stderr[-1500:])
        raise SystemExit(f"failed: {' '.join(str(c) for c in cmd[:3])}")
    return result


def score(truth: Path, estimate: Path) -> dict:
    result = run([sys.executable, str(EVAL), "--truth", str(truth), "--stems", str(estimate)])
    return json.loads(result.stdout)


def main():
    parquet = Path(sys.argv[1])
    out = Path(sys.argv[2])
    name = extract(parquet, out)
    print("song:", name)

    # ground truth in two-stem form
    truth = out / "truth_two"
    truth.mkdir(exist_ok=True)
    vocals, rate = sf.read(str(out / "vocals.wav"), always_2d=True, dtype="float32")
    bed = sum(sf.read(str(out / f"{field}.wav"), always_2d=True, dtype="float32")[0]
              for field in ("drums", "bass", "other"))
    sf.write(str(truth / "vocals.wav"), vocals, rate, subtype="FLOAT")
    sf.write(str(truth / "instrumental.wav"), bed, rate, subtype="FLOAT")

    # demucs
    run([GPU_PY, "-m", "demucs.separate", "-n", "htdemucs", "--device", "cuda",
         "-o", str(out / "demucs"), str(out / "mixture.wav")])
    demucs_dir = out / "demucs" / "htdemucs" / "mixture"
    est = out / "est_demucs"
    est.mkdir(exist_ok=True)
    dv, _ = sf.read(str(demucs_dir / "vocals.wav"), always_2d=True, dtype="float32")
    db = sum(sf.read(str(demucs_dir / f"{f}.wav"), always_2d=True, dtype="float32")[0]
             for f in ("drums", "bass", "other"))
    length = min(len(dv), len(db))
    sf.write(str(est / "vocals.wav"), dv[:length], rate, subtype="FLOAT")
    sf.write(str(est / "instrumental.wav"), db[:length], rate, subtype="FLOAT")
    demucs = score(truth, est)

    # BS-RoFormer
    run([GPU_PY, str(ROFORMER_DIR / "run_roformer.py"), "--input", str(out / "mixture.wav"),
         "--outdir", str(out / "roformer")])
    est2 = out / "est_roformer"
    est2.mkdir(exist_ok=True)
    for pattern, target in (("*Vocals*.wav", "vocals.wav"), ("*Instrumental*.wav", "instrumental.wav")):
        path = next((out / "roformer").glob(pattern))
        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        sf.write(str(est2 / target), data, sr, subtype="FLOAT")
    roformer = score(truth, est2)

    print(f"{'model':12s} {'vocals':>8s} {'instr':>8s} {'mean':>8s}")
    for label, data in (("demucs", demucs), ("bs-roformer", roformer)):
        print(f"{label:12s} {data['stems']['vocals']['si_sdr_db']:8.2f} "
              f"{data['stems']['instrumental']['si_sdr_db']:8.2f} {data['mean_si_sdr_db']:8.2f}")
    (out / "scores.json").write_text(json.dumps({"song": name, "demucs": demucs,
                                                 "roformer": roformer}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")


if __name__ == "__main__":
    main()
