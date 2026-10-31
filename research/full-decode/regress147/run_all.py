"""Run all 147 regression cases with a small process pool (resumable)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache-root", default=r"D:\ChordStudio\data\audio-cache-147")
    args = ap.parse_args()
    manifest = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    cases = sorted(manifest)
    todo = [c for c in cases if not (HERE / "cases" / c / "result.json").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(cases)} cases, {len(todo)} to run, {args.workers} workers", flush=True)
    t0 = time.time()
    done = [0]
    slots = list(range(args.workers))
    free = list(slots)
    import threading
    guard = threading.Lock()

    def one(case):
        with guard:
            slot = free.pop(0)
        env = dict(os.environ, REGRESS_CACHE=str(Path(args.cache_root) / f"w{slot}"))
        try:
            proc = subprocess.run([sys.executable, "-u", str(HERE / "run_case.py"), "--case", case],
                                  capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", env=env)
        finally:
            with guard:
                free.append(slot)
        done[0] += 1
        ok = proc.returncode == 0
        line = (proc.stdout or "").strip().splitlines()[-1] if proc.stdout else ""
        if not ok:
            line = "FAILED: " + (proc.stderr or "").strip().splitlines()[-1]
        print(f"[{done[0]}/{len(todo)}] {case} {line} ({time.time() - t0:.0f}s)", flush=True)
        return ok

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, todo))
    print(f"finished {sum(results)}/{len(results)} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

