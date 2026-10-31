"""Force one separation with an explicit model (used after code changes to refresh metrics)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

PROJECT = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "htdemucs"
value = json.loads(Path(r"D:\ChordStudio\data\active-session.json").read_text(encoding="utf-8"))
base = f"http://127.0.0.1:{value['port']}"
headers = {"x-chord-session": value["token"], "X-Harmony-Client": "studio",
           "Origin": base, "Referer": base + "/"}

started = time.time()
job = requests.post(f"{base}/api/projects/{PROJECT}/separate", headers=headers,
                    json={"force": True, "model": MODEL}, timeout=60).json()
print("job", job["id"], "force", job.get("force"))
for _ in range(200):
    time.sleep(3)
    row = requests.get(f"{base}/api/jobs/{job['id']}", headers=headers, timeout=30).json()
    if row["status"] in {"completed", "failed", "interrupted"}:
        print(row["status"], row.get("error"), f"{time.time() - started:.0f}s")
        break
manifest = requests.get(f"{base}/api/projects/{PROJECT}/stems?model={MODEL}",
                        headers=headers, timeout=30).json()
print("model", manifest["model"])
print("reconstruction", manifest.get("reconstruction"))
