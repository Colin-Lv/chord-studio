"""Does separation change the chords? Compare the model's timeline on mix vs harmonic stems."""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests
import soundfile as sf

SESSION = Path(r"D:\ChordStudio\data\active-session.json")
STEP = 0.25
WORK = Path(r"D:\ChordStudio\research\separation\downstream")


def api():
    value = json.loads(SESSION.read_text(encoding="utf-8"))
    base = f"http://127.0.0.1:{value['port']}"
    headers = {"x-chord-session": value["token"], "X-Harmony-Client": "studio",
               "Origin": base, "Referer": base + "/"}
    return base, headers


def sum_wavs(paths, target: Path):
    data, rate = None, None
    for path in paths:
        audio, rate = sf.read(str(path), always_2d=True, dtype="float32")
        data = audio if data is None else data + audio
    sf.write(str(target), data, rate, subtype="PCM_16")
    return target


def analyze(path: Path, label: str) -> dict:
    base, headers = api()
    with path.open("rb") as stream:
        project = requests.post(f"{base}/api/upload", headers=headers,
                                files={"file": (path.name, stream, "audio/wav")},
                                timeout=300).json()
    job = requests.post(f"{base}/api/projects/{project['id']}/analyze", headers=headers,
                        json={"profile": "retained10", "key_mode": "continuous"},
                        timeout=60).json()
    started = time.time()
    status = None
    for _ in range(240):
        time.sleep(3)
        row = requests.get(f"{base}/api/jobs/{job['id']}", headers=headers, timeout=30).json()
        if row["status"] in {"completed", "failed", "interrupted"}:
            status = row["status"]
            print(f"  {label}: {status} in {time.time() - started:.0f}s")
            if status != "completed":
                print("   ", row.get("error"))
                return {}
            break
    if status != "completed":
        return {}
    return requests.get(f"{base}/api/projects/{project['id']}/result",
                        headers=headers, timeout=60).json()


def timeline(result: dict, duration: float):
    rows = []
    for source in result.get("sources", []):
        if source["id"] == result.get("default_source"):
            rows = source["rows"]
            break
    grid = {}
    t = 0.0
    while t < duration:
        label = "N"
        for row in rows:
            if row["start"] <= t < row["end"]:
                label = row["label"]
                break
        grid[round(t, 2)] = label
        t += STEP
    return grid


def compare(a, b):
    keys = sorted(set(a) & set(b))
    exact = sum(1 for k in keys if a[k] == b[k]) / max(1, len(keys))
    root = sum(1 for k in keys if a[k].split(":")[0] == b[k].split(":")[0]) / max(1, len(keys))
    return exact, root, len(keys)


def run_song(name, mix: Path, stems: Path, parts):
    if not mix.is_file():
        print(f"{name}: mix missing")
        return
    print(f"== {name}")
    harmonic = WORK / f"{name}_harmonic.wav"
    sum_wavs([stems / part for part in parts], harmonic)
    duration = sf.info(str(mix)).duration
    mix_result = analyze(mix, "mix")
    harmonic_result = analyze(harmonic, "harmonic")
    if not mix_result or not harmonic_result:
        return
    a, b = timeline(mix_result, duration), timeline(harmonic_result, duration)
    exact, root, n = compare(a, b)
    print(f"  frames {n}  exact {exact * 100:.1f}%  root-only {root * 100:.1f}%")


def main():
    WORK.mkdir(exist_ok=True)
    songs = [
        ("虚拟", Path(r"D:\ChordStudio\research\separation\virtual.wav"),
         Path(r"D:\ChordStudio\data\stems\d4867eec9bec73fb4dbf0575cff26e3e6550cdba9553c8f886b3a4adab6ef72f\htdemucs_ft\wav"),
         ["bass.wav", "other.wav"]),
        ("BKS-Bulldozer", Path(r"D:\ChordStudio\datasets\musdb_song3\mixture.wav"),
         Path(r"D:\ChordStudio\datasets\musdb_song3"), ["bass.wav", "other.wav"]),
    ]
    for name, mix, stems, parts in songs:
        run_song(name, mix, stems, parts)


if __name__ == "__main__":
    main()
