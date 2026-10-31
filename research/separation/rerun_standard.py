"""Force a standard re-separation through the app (GPU path) and report timing."""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

PROJECT = "aa5a41797bb343e590abcf4d8b27ae54"
value = json.loads(Path(r"D:\ChordStudio\data\active-session.json").read_text(encoding="utf-8"))
base = f"http://127.0.0.1:{value['port']}"
headers = {"x-chord-session": value["token"], "X-Harmony-Client": "studio",
           "Origin": base, "Referer": base + "/"}

started = time.time()
job = requests.post(f"{base}/api/projects/{PROJECT}/separate", headers=headers,
                    json={"force": True, "model": "htdemucs"}, timeout=60).json()
print("job", job["id"], job.get("model"))
for _ in range(120):
    time.sleep(3)
    row = requests.get(f"{base}/api/jobs/{job['id']}", headers=headers, timeout=30).json()
    if row["status"] in {"completed", "failed", "interrupted"}:
        print(row["status"], row.get("error"), f"{time.time() - started:.0f}s")
        break
manifest = requests.get(f"{base}/api/projects/{PROJECT}/stems?model=htdemucs",
                        headers=headers, timeout=30).json()
print("model", manifest["model"], "options",
      [(item["id"], item["ready"]) for item in manifest["model_options"]])
