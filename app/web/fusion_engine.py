"""Fuse the frozen chord model's mix posterior with a separation-derived harmonic posterior.

The frozen model is audio-in, so the only ways to use separation are: feed it a different
signal, fuse its posterior with a second run, or add a bass-root prior. This module does all
three in one place and is shared by the fusion job and the full pipeline.

Legacy experiment only. Earlier raw-cell gains did not survive comparison with
the released full decoder. Its posterior index alignment is also not a reliable
cross-audio clock. New requests use stem_harmony.py, with exact interval pooling.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .storage import ROOT, atomic_json, now, read_json

NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
QUALITIES = ("maj", "min", "7", "maj7", "min7", "9", "maj9", "min9", "sus2", "sus4")
HARMONIC_STEMS = ("bass", "other", "guitar", "piano", "strings")
DEFAULT_LAMBDA = 0.65
DEFAULT_BETA = 0.0
N_THRESHOLD = 0.5


def _data_root() -> Path:
    from app.runtime_paths import data_root
    return data_root()


def harmonic_wav(manifest: dict, stems_wav_dir: Path, target: Path) -> Path:
    """Sum the non-percussive, non-vocal stems into one harmonic mix."""
    import soundfile as sf

    ids = [track["id"] for track in manifest["tracks"] if track["id"] in HARMONIC_STEMS]
    if not ids:
        raise ValueError("这一版分轨没有可用于和声的轨道")
    total = None
    rate = None
    for stem_id in ids:
        from .stems_worker import resolve_stem
        path = resolve_stem(stems_wav_dir.parent, stem_id, manifest)
        if not path.is_file():
            continue
        audio, rate = sf.read(str(path), always_2d=True, dtype="float32")
        total = audio if total is None else total + audio
    if total is None or rate is None:
        raise ValueError("找不到分轨的 WAV 文件")
    # Match the loudness of the full mixture (sum of every stem) so the two posterior runs
    # see comparable levels; without this the fusion partly measures a gain difference.
    import math

    mixture_rms = math.sqrt(sum(10 ** (track["rms_db"] / 10)
                                for track in manifest["tracks"] if "rms_db" in track))
    harmonic_rms = float(np.sqrt((total.astype(np.float64) ** 2).mean()))
    if harmonic_rms > 0 and mixture_rms > 0:
        gain = mixture_rms / harmonic_rms
        gain = max(0.25, min(4.0, gain))
        total *= gain
    peak = float(np.abs(total).max())
    if peak > 0.999:
        total *= 0.999 / peak
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), total, rate, subtype="PCM_16")
    return target


def run_frontend(audio: Path, outdir: Path, label: str = "fusion") -> Path:
    """Run the frozen audio frontend on an arbitrary wav (same entry the analysis uses)."""
    if (outdir / "posterior.npz").is_file():
        return outdir
    if outdir.exists():
        raise RuntimeError(f"{outdir} exists without a posterior")
    runtime = read_json(ROOT / "desktop_runtime.json")
    command = [sys.executable, "-I", "-B", str(ROOT / "inference_windowless.py"),
               "--audio", str(audio), "--output-dir", str(outdir),
               "--cache-dir", str(_data_root() / "audio-cache"), "--max-seconds", "600",
               "--memory-budget-mib", str(runtime.get("inference_memory_budget_mib", 1500))]
    outdir.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (outdir.parent / f"{label}.stdout").open("wb") as stdout, \
            (outdir.parent / f"{label}.stderr").open("wb") as stderr:
        child = subprocess.run(command, cwd=str(ROOT), stdout=stdout, stderr=stderr,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                               timeout=3600)
    if child.returncode:
        tail = ""
        path = outdir.parent / f"{label}.stderr"
        if path.is_file():
            tail = path.read_text(encoding="utf-8", errors="replace")[-1200:]
        raise RuntimeError("和声轨推理失败：" + tail)
    if not (outdir / "posterior.npz").is_file():
        raise RuntimeError("和声轨推理没有产出 posterior")
    return outdir


def load_posterior(folder: Path):
    with np.load(folder / "posterior.npz") as archive:
        return archive["chord"], archive["N"]


def fuse(mix_chord: np.ndarray, mix_n: np.ndarray, harm_chord: np.ndarray, harm_n: np.ndarray,
         lam: float = DEFAULT_LAMBDA, beta: float = DEFAULT_BETA, chroma: np.ndarray | None = None):
    count = min(len(mix_chord), len(harm_chord))
    a = np.log(np.clip(mix_chord[:count], 1e-9, 1))
    b = np.log(np.clip(harm_chord[:count], 1e-9, 1))
    score = (1 - lam) * a + lam * b
    if beta and chroma is not None:
        roots = np.arange(120) // 10
        score = score + beta * np.log(np.clip(chroma[:count][:, roots], 1e-6, 1))
    out = np.exp(score - score.max(axis=1, keepdims=True))
    out /= out.sum(axis=1, keepdims=True)
    nprob = np.maximum(mix_n[:count], harm_n[:count])
    return out, nprob


def labels_from(probability: np.ndarray, nprob: np.ndarray, threshold: float = N_THRESHOLD):
    labels = []
    for row, pn in zip(probability, nprob):
        root, quality = divmod(int(row.argmax()), 10)
        labels.append("N" if pn >= threshold else f"{NAMES[root]}:{QUALITIES[quality]}")
    return labels


def cells_for(count: int, duration: float):
    step = duration / count
    return [(i * step, (i + 1) * step) for i in range(count)]


def segments_from(labels, cells):
    from .source_rows import canonical_rows
    if len(labels)!=len(cells):raise ValueError('Labels and intervals have different lengths')
    checked=canonical_rows([dict(start=a,end=b,label=label,candidates=[]) for label,(a,b) in zip(labels,cells)])
    rows = []
    for row in checked:
        label,start,end=row['label'],row['start'],row['end']
        if rows and rows[-1]["label"] == label and rows[-1]["end"] == start:
            rows[-1]["end"] = end
        else:
            rows.append(dict(start=start, end=end, label=label))
    return canonical_rows([dict(row,candidates=[]) for row in rows])


def full_decode_segments(run_dir: Path, probability) -> list[dict]:
    """Run the app's full decoder on an arbitrary posterior (bar grid + notes from the mix run)."""
    from .full_decode import decode as decode_full

    prediction = read_json(run_dir / "retained10" / "prediction.json")
    cache = Path(prediction["cache_directory"])
    cells = read_json(cache / "cells.json")
    deep = __import__("numpy").load(cache / "deep.npz")
    notes = __import__("numpy").load(cache / "notes.npz")
    beats_rows = read_json(run_dir / "beats.json")
    count = min(len(probability), len(cells))
    rows = decode_full(
        posterior=probability[:count],
        starts=__import__("numpy").array([cell["start"] for cell in cells[:count]], dtype=float),
        ends=__import__("numpy").array([cell["end"] for cell in cells[:count]], dtype=float),
        chroma=deep["chroma"],
        act=notes["activations"], note_times=notes["times"],
        beats=__import__("numpy").array([float(row["time"]) for row in beats_rows], dtype=float),
        beat_pos=__import__("numpy").array([int(row.get("beat", 0)) for row in beats_rows], dtype=int))
    return segments_from([str(r[2]) for r in rows],[(float(r[0]),float(r[1])) for r in rows])


def apply_to_analysis(run_dir: Path, segments, meta: dict, source_id: str = "stem_fusion",
                      label: str = "分轨融合（实验）", make_default: bool = True) -> dict:
    """Add the fused timeline to analysis.json as a real source (lab file + sha), like the others."""
    import hashlib

    path = run_dir / "analysis.json"
    result = read_json(path)
    from .source_rows import serialize_rows
    from .storage import atomic_text
    segments,content=serialize_rows(segments,result.get('duration'))
    lab = run_dir / f"{source_id}.lab"
    atomic_text(lab,content)
    raw = lab.read_bytes()
    sources = [source for source in result["sources"] if source["id"] != source_id]
    sources.append(dict(id=source_id, name=label, path=lab.name,
                        sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), rows=segments))
    result["sources"] = sources
    result.setdefault("stem_fusion", {}).update(meta)
    if make_default:
        result["default_source"] = source_id
    atomic_json(path, result)
    return result


def duration_of(run_dir: Path) -> float:
    result = read_json(run_dir / "analysis.json")
    return float(result["duration"])


def fuse_project(store, project: dict, model: str, job_path: Path | None = None,
                 lam: float = DEFAULT_LAMBDA, beta: float = DEFAULT_BETA,
                 make_default: bool = True) -> dict:
    """Build the harmonic mix, run the frozen frontend on it, fuse, and store the source."""
    from .stems_worker import stems_dir
    from .storage import signature

    project_dir = store.project_dir(project["id"])
    audio = project_dir / project["audio"]
    sha = signature(audio)["sha256"]
    base = stems_dir(sha, model)
    manifest_path = base / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("请先完成这一版分轨")
    manifest = read_json(manifest_path)
    run_id = project.get("current_run")
    if not run_id:
        raise ValueError("请先完成一次和声分析，融合需要混音的后验分布")
    run_dir = store.run_dir(project["id"], run_id)
    mix_dir = run_dir / "retained10"
    if not (mix_dir / "posterior.npz").is_file():
        raise ValueError("这次分析没有保留后验分布")

    work = base / "fusion"
    work.mkdir(parents=True, exist_ok=True)
    harmonic = harmonic_wav(manifest, base / "wav", work / "harmonic.wav")
    started = time.perf_counter()
    folder = run_frontend(harmonic, work / "retained10")
    seconds = time.perf_counter() - started

    mix_chord, mix_n = load_posterior(mix_dir)
    harm_chord, harm_n = load_posterior(folder)
    probability, nprob = fuse(mix_chord, mix_n, harm_chord, harm_n, lam, beta)
    duration = duration_of(run_dir)
    cells = cells_for(len(probability), duration)
    try:
        segments = full_decode_segments(run_dir, probability)
        decoder = "full"
    except Exception:                                  # noqa: BLE001
        segments = segments_from(labels_from(probability, nprob), cells)
        decoder = "cell"
    meta = dict(created_at=now(), model=model, lambda_=lam, beta=beta, decoder=decoder,
                harmonic_stems=[track["id"] for track in manifest["tracks"]
                                if track["id"] in HARMONIC_STEMS],
                cells=len(probability), segments=len(segments), seconds=round(seconds, 1),
                note="融合了混音与和声轨两次冻结推理的后验；不是逐轨真值校准")
    apply_to_analysis(run_dir, segments, meta, make_default=make_default)
    return meta
