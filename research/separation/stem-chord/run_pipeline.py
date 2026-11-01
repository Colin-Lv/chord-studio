"""End-to-end check of the one-button pipeline on a project that already has stems + analysis."""
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

body = {"model": "bs_roformer_sw", "vocal_parts": True, "guitar_parts": True,
        "drums_parts": True, "harmony": True, "harmony_default": True, "force_stems": False}
started = time.time()
response = requests.post(f"{base}/api/projects/{PROJECT}/pipeline", headers=headers, json=body,
                         timeout=60)
print("submit", response.status_code, response.text[:160])
response.raise_for_status()
job = response.json()
last = None
for _ in range(300):
    time.sleep(4)
    row = requests.get(f"{base}/api/jobs/{job['id']}", headers=headers, timeout=30).json()
    label = f"{row.get('profile')} | {row.get('stage_label')}"
    if label != last:
        last = label
        print(f"  [{time.time() - started:5.0f}s] {row['status']:9s} {label}")
    if row["status"] in {"completed", "failed", "interrupted"}:
        print("final:", row["status"], row.get("error"))
        break

fusion = requests.get(f"{base}/api/projects/{PROJECT}/stem-fusion", headers=headers, timeout=30)
print("stem-fusion:", fusion.status_code)
if fusion.status_code == 200:
    data = fusion.json()
    print("  segments:", len(data["source"]["rows"]), "meta:", data["meta"].get("lambda_"),
          data["meta"].get("harmonic_stems"))
result = requests.get(f"{base}/api/projects/{PROJECT}/result", headers=headers, timeout=30).json()
print("default_source:", result.get("default_source"),
      "sources:", [source["id"] for source in result["sources"]])
print(f"total {time.time() - started:.0f}s")
