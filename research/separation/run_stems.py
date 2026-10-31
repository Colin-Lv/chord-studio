"""Drive a separation job through the desktop API and inspect the manifest."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

SESSION = Path(r"D:\ChordStudio\data\active-session.json")
PROJECT = sys.argv[1] if len(sys.argv) > 1 else "aa5a41797bb343e590abcf4d8b27ae54"
FORCE = "--force" in sys.argv


def main():
    session = json.loads(SESSION.read_text(encoding="utf-8"))
    base = f"http://127.0.0.1:{session['port']}"
    headers = {"x-chord-session": session["token"], "X-Harmony-Client": "studio",
               "Origin": base, "Referer": base + "/"}
    started = time.time()
    response = requests.post(f"{base}/api/projects/{PROJECT}/separate", headers=headers,
                             json={"force": FORCE}, timeout=60)
    print("submit:", response.status_code, response.text[:200])
    response.raise_for_status()
    job = response.json()["id"]
    last = None
    for _ in range(300):
        time.sleep(3)
        row = requests.get(f"{base}/api/jobs/{job}", headers=headers, timeout=30).json()
        label = row.get("stage_label") or row.get("stage")
        if label != last:
            last = label
            print(f"  [{time.time() - started:6.1f}s] {row['status']:9s} {label}")
        if row["status"] in {"completed", "failed", "interrupted"}:
            if row["status"] != "completed":
                print("FAILED:", row.get("error"))
                return 1
            break
    manifest = requests.get(f"{base}/api/projects/{PROJECT}/stems", headers=headers, timeout=30)
    print("manifest:", manifest.status_code)
    manifest.raise_for_status()
    data = manifest.json()
    print(f"  duration {data['duration']}s  bpm {data['bpm']} ({data['bpm_source']})  "
          f"bars {len(data['bars'])}  model {data['model']}")
    for track in data["tracks"]:
        steps = len(track.get("steps") or [])
        head = requests.head(f"{base}{data['urls'][track['id']]}", headers=headers, timeout=30)
        print(f"  {track['id']:7s} {track['name']:3s} rms {track['rms_db']:6.1f} dB  "
              f"active {track['active_ratio']:.2f}  steps {steps}  mp3 {head.status_code} "
              f"{head.headers.get('content-length', '?')} bytes")
    print(f"total wall {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
