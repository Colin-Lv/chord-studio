"""Submit a drum-parts job through the app and report the resulting manifest."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

PROJECT = sys.argv[1] if len(sys.argv) > 1 else "aa5a41797bb343e590abcf4d8b27ae54"
FORCE = "force" in sys.argv
value = json.loads(Path(r"D:\ChordStudio\data\active-session.json").read_text(encoding="utf-8"))
base = f"http://127.0.0.1:{value['port']}"
headers = {"x-chord-session": value["token"], "X-Harmony-Client": "studio",
           "Origin": base, "Referer": base + "/"}

started = time.time()
response = requests.post(f"{base}/api/projects/{PROJECT}/drums-parts", headers=headers,
                         json={"force": FORCE}, timeout=60)
print("submit", response.status_code)
job = response.json()
for _ in range(200):
    time.sleep(3)
    row = requests.get(f"{base}/api/jobs/{job['id']}", headers=headers, timeout=30).json()
    if row["status"] in {"completed", "failed", "interrupted"}:
        print(row["status"], row.get("error"), f"{time.time() - started:.0f}s")
        break
manifest = requests.get(f"{base}/api/projects/{PROJECT}/stems/drums-parts", headers=headers,
                        timeout=30)
print("manifest", manifest.status_code)
if manifest.status_code == 200:
    data = manifest.json()
    for track in data["tracks"]:
        print(f"  {track['id']:7s} rms {track['rms_db']:6.1f}  active {track['active_ratio']:.2f}  "
              f"level {track.get('level_factor')}")
