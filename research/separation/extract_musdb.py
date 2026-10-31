"""Unpack one MUSDB18 song (mixture + four stems) from the HuggingFace parquet mirror.

The mirror (danjacobellis/musdb) stores one song per parquet shard with stereo 44.1 kHz
audio; this script writes standard wav files so the separation models and the eval harness
can use them as ground truth.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

PARQUET = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else PARQUET.with_suffix("")

FIELDS = ("mixture", "vocals", "drums", "bass", "other")


def main():
    table = pq.read_table(PARQUET)
    name = table.column("name")[0].as_py()
    print("song:", name)
    OUT.mkdir(parents=True, exist_ok=True)
    for field in FIELDS:
        blob = table.column(field).combine_chunks()[0].as_py()
        data, rate = sf.read(io.BytesIO(blob["bytes"]), always_2d=True, dtype="float32")
        target = OUT / f"{field}.wav"
        sf.write(str(target), data, rate, subtype="FLOAT")
        print(f"  {field:8s} {len(data) / rate:7.1f}s  {rate} Hz  {data.shape[1]} ch  "
              f"{20 * __import__('math').log10(float((data ** 2).mean()) ** 0.5 + 1e-12):6.1f} dBFS")


if __name__ == "__main__":
    main()
