"""Run a separation with a chosen model and report timing + summary."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

SESSION = Path(r"D:\ChordStudio\data\active-session.json")
PROJECT = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "htdemucs_ft"


def main():
    session = json.loads(SESSION.read_text(encoding="utf-8"))
    base = f"http://127.0.0.1:{session['port']}"
    headers = {"x-chord-session": session["token"], "X-Harmony-Client": "studio",
               "Origin": base, "Referer": base + "/"}
    started = time.time()
    response = requests.post(f"{base}/api/projects/{PROJECT}/separate", headers=headers,
                             json={"force": False, "model": MODEL}, timeout=60)
    print("submit:", response.status_code)
    response.raise_for_status()
    job_id = response.json()["id"]
    last = None
    for _ in range(400):
        time.sleep(5)
        row = requests.get(f"{base}/api/jobs/{job_id}", headers=headers, timeout=30).json()
        label = row.get("stage_label")
        if label != last:
            last = label
            print(f"  [{time.time() - started:6.0f}s] {row['status']:9s} {label}")
        if row["status"] in {"completed", "failed", "interrupted"}:
            if row["status"] != "completed":
                print("FAILED:", row.get("error"))
                return 1
            break
    manifest = requests.get(f"{base}/api/projects/{PROJECT}/stems?model={MODEL}",
                            headers=headers, timeout=30).json()
    print(f"  model {manifest['model']}  bpm {manifest['bpm']}")
    for track in manifest["tracks"]:
        print(f"    {track['id']:7s} rms {track['rms_db']:6.1f}  active {track['active_ratio']:.2f}")
    print(f"total {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
