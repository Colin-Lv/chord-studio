"""Prototype: fuse the frozen chord model's mix posterior with a harmonic-stem posterior.

  1. H = bass + other + guitar + piano from the separated stems
  2. run the frozen audio frontend on H (inference/audio_predict.py)
  3. fuse log-probabilities with the mix posterior, optionally adding a bass-root prior
  4. decode and score against the golden timeline (frame + segment level)
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
QUALITIES = ("maj", "min", "7", "maj7", "min7", "9", "maj9", "min9", "sus2", "sus4")
SHA = "d4867eec9bec73fb4dbf0575cff26e3e6550cdba9553c8f886b3a4adab6ef72f"
STEMS = Path(r"D:\ChordStudio\data\stems") / SHA / "bs_roformer_sw" / "wav"
PY = r"D:\ChordStudio\Dialogue\.tmp\structure_env\Scripts\python.exe"
FRONTEND = Path(r"D:\ChordStudio\Dialogue\inference\audio_predict.py")
RUNTIME = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\inference\runtime.local.json")
CACHE = Path(r"D:\ChordStudio\data\audio-cache")
GOLDEN = Path(r"D:\ChordStudio\Dialogue\research\golden\reference_golden.lab")
MIX_RUN = Path(r"D:\ChordStudio\data\projects\aa5a41797bb343e590abcf4d8b27ae54\runs\75ad217baca0484c83fa23f7f0f75442\retained10")
WORK = Path(r"D:\ChordStudio\research\separation\stem-chord")
DURATION = 238.123


def build_harmonic() -> Path:
    target = WORK / "harmonic.wav"
    if target.is_file():
        return target
    data = rate = None
    for name in ("bass", "other", "guitar", "piano"):
        audio, rate = sf.read(str(STEMS / f"{name}.wav"), always_2d=True, dtype="float32")
        data = audio if data is None else data + audio
    peak = float(np.abs(data).max())
    if peak > 0.999:
        data *= 0.999 / peak
    sf.write(str(target), data, rate, subtype="PCM_16")
    return target


def run_frontend(audio: Path) -> Path:
    out = WORK / "front_harmonic"
    if (out / "posterior.npz").is_file():
        return out
    command = [PY, str(FRONTEND), "--audio", str(audio), "--output-dir", str(out),
               "--cache-dir", str(CACHE), "--runtime", str(RUNTIME), "--max-seconds", "600"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        print(result.stdout[-1200:], result.stderr[-1200:])
        raise SystemExit("frontend failed")
    return out


def load_posterior(folder: Path):
    with np.load(folder / "posterior.npz") as z:
        return z["chord"], z["N"]


def decode_labels(chord, nprob, threshold=0.5):
    labels = []
    for row, pn in zip(chord, nprob):
        root, quality = divmod(int(row.argmax()), 10)
        labels.append("N" if pn >= threshold else f"{NAMES[root]}:{QUALITIES[quality]}")
    return labels


def read_golden():
    events = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3:
            events.append((float(parts[0]), float(parts[1]), parts[2]))
    return events


def frame_accuracy(labels, times, events):
    total = hit = root_hit = 0
    for t, label in zip(times, labels):
        truth = next((lab for start, end, lab in events if start <= t < end), None)
        if truth is None:
            continue
        total += 1
        hit += label == truth
        root_hit += label.split(":")[0] == truth.split(":")[0]
    return hit / max(1, total), root_hit / max(1, total), total


def segment_accuracy(labels, times, events):
    exact = root = weighted = 0.0
    total = weight = 0.0
    for start, end, truth in events:
        inner = [lab for t, lab in zip(times, labels) if start <= t < end]
        if not inner:
            continue
        values, counts = np.unique(inner, return_counts=True)
        guess = values[counts.argmax()]
        duration = end - start
        total += 1
        weight += duration
        exact += guess == truth
        root += guess.split(":")[0] == truth.split(":")[0]
        weighted += (guess == truth) * duration
    return exact / max(1, total), root / max(1, total), weighted / max(1, weight)


def bass_chroma(count: int):
    cache = WORK / "bass_chroma.npy"
    if cache.is_file():
        return np.load(cache)
    import librosa
    y, sr = librosa.load(str(STEMS / "bass.wav"), sr=22050, mono=True)
    hop = max(1, int(sr * 0.25))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    chroma = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-9)
    chroma = chroma.T
    if len(chroma) < count:
        chroma = np.pad(chroma, ((0, count - len(chroma)), (0, 0)), mode="edge")
    chroma = chroma[:count].astype(np.float32)
    np.save(cache, chroma)
    return chroma


def fuse(p_mix, p_h, lam, beta, chroma):
    score = (1 - lam) * np.log(np.clip(p_mix, 1e-9, 1)) + lam * np.log(np.clip(p_h, 1e-9, 1))
    if beta:
        roots = np.arange(120) // 10
        score = score + beta * np.log(np.clip(chroma[:, roots], 1e-6, 1))
    out = np.exp(score - score.max(axis=1, keepdims=True))
    return out / out.sum(axis=1, keepdims=True)


def report(name, labels, times, events):
    fe, fr, total = frame_accuracy(labels, times, events)
    se, sr_, sw = segment_accuracy(labels, times, events)
    print(f"{name:>16s}: frame {fe * 100:5.2f}% / root {fr * 100:5.2f}% | "
          f"segment {se * 100:5.2f}% / root {sr_ * 100:5.2f}% / dur {sw * 100:5.2f}%  ({total} frames)")
    return fe, se


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)

    folder = run_frontend(build_harmonic())
    p_h, n_h = load_posterior(folder)
    p_mix, n_mix = load_posterior(MIX_RUN)
    count = min(len(p_mix), len(p_h))
    p_mix, p_h, n_mix, n_h = p_mix[:count], p_h[:count], n_mix[:count], n_h[:count]
    events = read_golden()
    times = np.array([(i + 0.5) * DURATION / count for i in range(count)])
    chroma = bass_chroma(count)

    print(f"cells {count}  golden events {len(events)}  coverage {events[0][0]:.1f}-{events[-1][1]:.1f}s")
    report("mix", decode_labels(p_mix, n_mix), times, events)
    report("harmonic", decode_labels(p_h, n_h), times, events)

    combos = ([(0.3, 0.0), (0.5, 0.0), (0.7, 0.0), (1.0, 0.0)] +
              [(l, b) for l in (0.3, 0.5) for b in (0.2, 0.5, 1.0)] +
              [(0.0, b) for b in (0.2, 0.5, 1.0)]) if args.sweep else [(0.5, 0.0)]
    best = None
    for lam, beta in combos:
        fused = fuse(p_mix, p_h, lam, beta, chroma)
        nprob = np.maximum(n_mix, n_h)
        fe, se = report(f"lambda={lam:.1f} beta={beta:.1f}", decode_labels(fused, nprob), times, events)
        if best is None or se > best["segment"]:
            best = {"lambda": lam, "beta": beta, "frame": fe, "segment": se}
    print("best by segment:", best)
    (WORK / "fusion-result.json").write_text(json.dumps(
        {"cells": count, "best": best}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
