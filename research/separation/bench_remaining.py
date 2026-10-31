"""Benchmark every remaining MUSDB shard: download, unpack, run both models, score."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import requests

DATA = Path(r"D:\ChordStudio\datasets")
BASE = "https://huggingface.co/datasets/danjacobellis/musdb/resolve/main/data/"
SHARDS = {
    0: "test-00000-of-00014-f8abe09f7e2a03ec.parquet",
    6: "test-00006-of-00014-20e66c9b9da16876.parquet",
    7: "test-00007-of-00014-92b9cc4f509acad5.parquet",
    8: "test-00008-of-00014-5f40afb7ca8444f6.parquet",
    9: "test-00009-of-00014-6d06c82d6cb34236.parquet",
    10: "test-00010-of-00014-80caa3ed90717553.parquet",
}
PY = sys.executable
BENCH = Path(r"D:\ChordStudio\research\separation\musdb_bench.py")
SCORE_FT = Path(r"D:\ChordStudio\research\separation\score_ft_musdb.py")


def download(index: int, name: str) -> Path:
    target = DATA / f"musdb_danj_test{index}.parquet"
    if target.is_file() and target.stat().st_size > 100_000_000:
        return target
    started = time.time()
    with requests.get(BASE + name, stream=True, timeout=600) as response:
        response.raise_for_status()
        with target.open("wb") as stream:
            for chunk in response.iter_content(1 << 20):
                stream.write(chunk)
    print(f"  downloaded {target.name} in {time.time() - started:.0f}s", flush=True)
    return target


def main():
    for index, name in SHARDS.items():
        song_dir = DATA / f"musdb_song{index}"
        if (song_dir / "scores_ft.json").is_file():
            print(f"song{index}: already done", flush=True)
            continue
        print(f"== song{index}", flush=True)
        parquet = download(index, name)
        result = subprocess.run([PY, str(BENCH), str(parquet), str(song_dir)],
                                capture_output=True, text=True)
        print(result.stdout.strip()[-600:] or result.stderr.strip()[-600:], flush=True)
        if not (song_dir / "scores.json").is_file():
            print("  bench failed", flush=True)
            continue
        result = subprocess.run([PY, str(SCORE_FT), str(song_dir)],
                                capture_output=True, text=True)
        print(result.stdout.strip()[-300:] or result.stderr.strip()[-300:], flush=True)


if __name__ == "__main__":
    main()
