"""Expand a separated drum stem into six parts (kick/snare/toms/hihat/ride/crash).

Runs MDX23C-DrumSep on the already separated drums stem, then annotates each part with the
same bar grid so the multitrack view can show them as extra rows.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from .storage import ROOT, atomic_json, now, read_json, signature
from app.runtime_paths import models_root
from .stems_worker import (_annotate, _load, _runtime, _to_mp3, beat_grid, data_root,
                           stems_dir, stems_for, stem_wav)

PARTS = (
    dict(id="kick", name="底鼓", short="KICK", instrument="Kick", color="#e6c07b", match="kick"),
    dict(id="snare", name="军鼓", short="SNR", instrument="Snare", color="#f0a58a", match="snare"),
    dict(id="toms", name="通鼓", short="TOM", instrument="Toms", color="#c9a0e6", match="toms"),
    dict(id="hihat", name="踩镲", short="HH", instrument="Hi-hat", color="#8fd3c7", match="hh"),
    dict(id="ride", name="叮叮镲", short="RIDE", instrument="Ride", color="#7fb3e6", match="ride"),
    dict(id="crash", name="强音镲", short="CRASH", instrument="Crash", color="#b6e7ce", match="crash"),
)
PARTS_DIR = "drums_parts"


def parts_dir(sha: str, model: str) -> Path:
    return stems_dir(sha, model) / PARTS_DIR


def parts_manifest(sha: str, model: str) -> Path:
    return parts_dir(sha, model) / "drums_parts.json"


def _publish(job_path, job, **update):
    job.update(**update)
    atomic_json(job_path, job)


def execute(store, job_id):
    job_path = store.job_path(job_id)
    job = read_json(job_path)
    project = store.project(job["project_id"])
    folder = store.project_dir(project["id"])
    audio = folder / project["audio"]
    sha = signature(audio)["sha256"]
    model = job.get("model") or "htdemucs"
    base_manifest_path = stems_dir(sha, model) / "manifest.json"
    if not base_manifest_path.is_file():
        raise ValueError("请先完成这首歌的分轨，再细分鼓组")
    base = read_json(base_manifest_path)
    drums = stem_wav(sha, model, "drums")
    if not drums.is_file():
        raise ValueError("这一版分轨没有鼓轨")

    target = parts_dir(sha, model)
    manifest_path = parts_manifest(sha, model)
    if manifest_path.is_file() and not job.get("force") and read_json(manifest_path).get("parent_sha256") == signature(drums)["sha256"]:
        manifest = read_json(manifest_path)
        manifest["cached"] = True
        project["drum_parts"] = dict(sha256=sha, model=model, manifest=str(manifest_path),
                                     updated_at=now())
        store.save_project(project)
        _publish(job_path, job, stage_label="复用已有鼓组细分")
        return manifest

    target.mkdir(parents=True, exist_ok=True)
    outdir = target / "raw"
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)

    def separate():
        python, device = _runtime()
        job["device"] = device
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env.setdefault("TORCH_HOME", str(Path.home() / ".cache" / "torch"))
        result = subprocess.run([python, "-B", str(ROOT / "app/web/drumsep_runner.py"),
                                 "--input", str(drums), "--outdir", str(outdir),
                                 "--model-dir", str(models_root() / 'drumsep')],
                                cwd=str(ROOT), env=env, capture_output=True, text=True,
                                timeout=3600,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode:
            raise RuntimeError("鼓组细分失败：" + (result.stderr or result.stdout or "")[-1500:])

    started = time.perf_counter()
    _publish(job_path, job, stage="separate", stage_label="正在细分鼓组（6 件）")
    separate()

    produced = {}
    for path in outdir.glob("*.wav"):
        name = path.name.lower()
        for part in PARTS:
            if f"({part['match']})" in name or f"_{part['match']}.wav" in name:
                produced[part["id"]] = path
    missing = [part["id"] for part in PARTS if part["id"] not in produced]
    if missing:
        raise RuntimeError("鼓组细分缺少：" + ", ".join(missing))

    _publish(job_path, job, stage="encode", stage_label="转码鼓组细分")
    wav_dir = target / "wav"
    wav_dir.mkdir(exist_ok=True)
    for part_id, path in produced.items():
        _to_mp3(path, target / f"{part_id}.mp3")
        path.replace(wav_dir / f"{part_id}.wav")
    shutil.rmtree(outdir, ignore_errors=True)

    _publish(job_path, job, stage="annotate", stage_label="标注鼓组")
    data, sr = _load(wav_dir / "kick.wav")
    length = len(data) / sr
    bars = [float(bar["start"]) for bar in base.get("bars", [])]
    if len(bars) < 2:
        grid = beat_grid(audio, length, None)
        bars = [float(b) for b in grid["bars"]]
    tracks = []
    for part in PARTS:
        entry = dict(part)
        entry.pop("match", None)
        entry["file"] = f"{part['id']}.mp3"
        entry["wav"] = f"wav/{part['id']}.wav"
        _annotate(wav_dir / f"{part['id']}.wav", entry, bars, length)
        tracks.append(entry)
    loudest = max(track["rms_db"] for track in tracks)
    for track in tracks:
        # The grid normalises each row by itself, so a -70 dB cymbal would look as dense as
        # the kick. Carry an explicit level factor so the UI can draw quiet parts faintly.
        track["level_factor"] = round(max(0.05, min(1.0, 10 ** ((track["rms_db"] - loudest) / 20))), 3)
    manifest = dict(
        schema="chord_studio_drum_parts_v1", parent_sha256=signature(drums)["sha256"],
        project_id=project["id"], audio_sha256=sha, model=model,
        created_at=now(), cached=False,
        model_info=dict(name="MDX23C-DrumSep-aufr33-jarredou", device=job.get("device"),
                        seconds=round(time.perf_counter() - started, 1)),
        duration=base.get("duration"), bpm=base.get("bpm"), bpm_source=base.get("bpm_source"),
        bars=base.get("bars"), tracks=tracks,
    )
    atomic_json(manifest_path, manifest)
    project["drum_parts"] = dict(sha256=sha, model=model, manifest=str(manifest_path),
                                 updated_at=now())
    store.save_project(project)
    _publish(job_path, job, stage_label="鼓组细分完成")
    return manifest
