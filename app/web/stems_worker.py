"""Separate a project's audio into stems and annotate them for the multitrack grid.

Runs demucs (htdemucs) in the model environment, converts each stem to a browser-friendly
mp3, measures per-stem level / activity / register and per-16th energy for the grid, reuses
the project's beat grid when it exists, and caches everything by audio sha256.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .storage import AUDIO_EXTENSIONS, ROOT, atomic_json, now, read_json, signature

STEMS = (
    dict(id="vocals", name="人声", short="VOX", instrument="主唱 / 和声", color="#b6e7ce"),
    dict(id="drums", name="鼓", short="DRM", instrument="鼓组与打击乐", color="#e6c07b"),
    dict(id="bass", name="贝斯", short="BAS", instrument="贝斯", color="#7fb3e6"),
    dict(id="other", name="其他", short="OTH", instrument="钢琴 / 吉他 / 铺底", color="#c9a0e6"),
)
MODEL_NAME = "htdemucs"
STEPS_PER_BAR = 16
ACTIVE_DBFS = -45.0


def data_root() -> Path:
    value = os.environ.get("CHORD_DESKTOP_DATA")
    return Path(value) if value else Path(r"D:\ChordStudio\data")


def stems_dir(sha: str) -> Path:
    return data_root() / "stems" / sha


def _ffmpeg() -> str:
    return str(read_json(ROOT / "inference/runtime.local.json")["ffmpeg"]["path"])


def _model_python() -> str:
    value = read_json(ROOT / "desktop_runtime.json").get("model_python")
    return str(value) if value else sys.executable


def _run(cmd, timeout=1800):
    return subprocess.run([str(part) for part in cmd], capture_output=True, timeout=timeout,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


def _decode_to_wav(audio: Path, target: Path):
    result = _run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", audio,
                   "-ac", "2", "-ar", "44100", "-c:a", "pcm_f32le", target])
    if result.returncode:
        raise RuntimeError("音频解码失败：" + result.stderr.decode("utf-8", "replace")[-1200:])


def _to_mp3(wav: Path, target: Path):
    result = _run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", wav,
                   "-codec:a", "libmp3lame", "-b:a", "192k", target])
    if result.returncode:
        raise RuntimeError("mp3 编码失败：" + result.stderr.decode("utf-8", "replace")[-1200:])


def _load(path: Path):
    import soundfile as sf
    data, rate = sf.read(str(path), always_2d=True, dtype="float32")
    return data, rate


def _rms_db(x: np.ndarray) -> float:
    return float(20 * math.log10(float(np.sqrt((x.astype(np.float64) ** 2).mean())) + 1e-12))


def _peak_db(x: np.ndarray) -> float:
    return float(20 * math.log10(float(np.abs(x).max()) + 1e-12))


def _active_mask(x: np.ndarray, sr: int) -> np.ndarray:
    hop = max(1, int(sr * 0.05))
    frames = len(x) // hop
    if frames < 1:
        return np.zeros(0, dtype=bool)
    trimmed = x[:frames * hop].reshape(frames, hop)
    rms = np.sqrt((trimmed.astype(np.float64) ** 2).mean(axis=1))
    return 20 * np.log10(rms + 1e-12) > ACTIVE_DBFS


def _centroid_hz(x: np.ndarray, sr: int) -> float:
    n = 2048
    if len(x) < n:
        return 0.0
    frames = 1 + (len(x) - n) // (n // 2)
    window = np.hanning(n).astype(np.float32)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    total, weight = 0.0, 0.0
    for i in range(0, frames, max(1, frames // 400)):
        seg = x[i * (n // 2):i * (n // 2) + n] * window
        mag = np.abs(np.fft.rfft(seg))
        total += float((mag * freqs).sum())
        weight += float(mag.sum())
    return total / weight if weight else 0.0


def _bass_pc(x: np.ndarray, sr: int):
    n = 4096
    if len(x) < n:
        return None
    window = np.hanning(n).astype(np.float32)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    band = (freqs >= 40) & (freqs <= 300)
    if not band.any():
        return None
    midi = 69 + 12 * np.log2(np.maximum(freqs, 1e-6) / 440.0)
    pc = np.round(midi).astype(int) % 12
    energy = np.zeros(12)
    frames = 1 + (len(x) - n) // (n // 2)
    for i in range(0, frames, max(1, frames // 300)):
        seg = x[i * (n // 2):i * (n // 2) + n] * window
        mag = np.abs(np.fft.rfft(seg))[band] ** 2
        np.add.at(energy, pc[band], mag)
    return int(energy.argmax()) if energy.sum() > 0 else None


def _register(centroid: float) -> str:
    if centroid <= 0:
        return "未知"
    if centroid < 400:
        return "低频"
    if centroid < 1600:
        return "中频"
    return "高频"


def beat_grid(audio: Path, duration: float, run_dir: Path | None):
    """Beat/bpm grid: the project's beat tracker when available, else librosa's estimate."""
    if run_dir is not None and (run_dir / "beats.json").is_file():
        rows = read_json(run_dir / "beats.json")
        times = [float(row["time"]) for row in rows]
        if len(times) > 4:
            period = float(np.median(np.diff(times)))
            bars = [t for t, row in zip(times, rows) if int(row.get("beat", 0)) == 1]
            if len(bars) < 2:
                bars = times[0::4]
            return dict(bpm=60.0 / period, bpm_source="beats.json", beats=times, bars=bars)
    try:
        import librosa
        mono, sr = librosa.load(str(audio), sr=22050, mono=True)
        tempo, beat_times = librosa.beat.beat_track(y=mono, sr=sr, units="time")
        tempo = float(np.atleast_1d(tempo)[0])
        beats = [float(t) for t in np.atleast_1d(beat_times)]
        bars = beats[0::4] if len(beats) >= 4 else ([0.0] if beats else [])
        return dict(bpm=tempo, bpm_source="librosa estimate", beats=beats, bars=bars)
    except Exception:                                    # noqa: BLE001
        period = 0.5
        bars = [i * 4 * period for i in range(max(1, int(duration / (4 * period))))]
        return dict(bpm=120.0, bpm_source="fallback 120", beats=[], bars=bars)


def _steps_for(x: np.ndarray, sr: int, bars: list[float], duration: float):
    steps = []
    for index, start in enumerate(bars):
        end = bars[index + 1] if index + 1 < len(bars) else duration
        span = max(1e-3, end - start)
        for k in range(STEPS_PER_BAR):
            a = start + span * k / STEPS_PER_BAR
            b = start + span * (k + 1) / STEPS_PER_BAR
            segment = x[int(a * sr):int(b * sr)]
            steps.append(float(np.sqrt((segment.astype(np.float64) ** 2).mean()))
                         if len(segment) else 0.0)
    values = np.asarray(steps, dtype=np.float32)
    if not len(values):
        return []
    floor = float(np.percentile(values, 95)) or float(values.max()) or 1.0
    return [round(float(min(1.0, v / floor)), 3) for v in values]


def _annotate(path: Path, entry: dict, bars: list[float], duration: float):
    data, sr = _load(path)
    mono = data.mean(axis=1)
    mask = _active_mask(mono, sr)
    active_ratio = float(mask.mean()) if len(mask) else 0.0
    first = last = None
    if mask.any():
        first = float(np.argmax(mask)) / 20.0
        last = float(len(mask) - np.argmax(mask[::-1])) / 20.0
    centroid = _centroid_hz(mono, sr)
    entry.update(
        sample_rate=int(sr), channels=int(data.shape[1]),
        rms_db=round(_rms_db(mono), 1), peak_db=round(_peak_db(mono), 1),
        active_ratio=round(active_ratio, 3),
        first_active=None if first is None else round(first, 2),
        last_active=None if last is None else round(min(last, duration), 2),
        centroid_hz=round(centroid, 1), register=_register(centroid),
        steps=_steps_for(mono, sr, bars, duration),
        silent=active_ratio < 0.01,
    )
    if entry["id"] == "bass":
        entry["dominant_pitch_class"] = _bass_pc(mono, sr)


def _download(project: dict, folder: Path, store) -> dict:
    target = folder / "source"
    target.mkdir(exist_ok=True)
    python = os.environ.get("HARMONY_DOWNLOAD_PYTHON", sys.executable)
    result = _run([python, "-B", "-m", "yt_dlp", "--ignore-config", "--no-playlist",
                   "--no-update", "--no-cache-dir", "--socket-timeout", "20", "--retries", "2",
                   "--max-filesize", "512M", "--write-info-json", "--no-progress",
                   "-f", "bestaudio", "-o", str(target / "audio.%(ext)s"), "--",
                   project["source_url"]], timeout=1800)
    if result.returncode:
        raise RuntimeError("下载音频失败：" + result.stderr.decode("utf-8", "replace")[-1200:])
    files = [p for p in target.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS]
    if len(files) != 1:
        raise ValueError("下载未产生有效音频，请改为上传本地文件。")
    metadata = read_json(target / "audio.info.json")
    project.update(audio=str(files[0].relative_to(folder)), audio_identity=signature(files[0]),
                   original_filename=files[0].name,
                   title=str(metadata.get("title") or project["title"]))
    store.save_project(project)
    return project


def _latest_beats_run(folder: Path) -> Path | None:
    candidates = sorted((folder / "runs").glob("*/beats.json"),
                        key=lambda path: path.stat().st_mtime)
    return candidates[-1].parent if candidates else None


def execute(store, job_id):
    job_path = store.job_path(job_id)
    job = read_json(job_path)
    if not job.get("stages"):
        job["stages"] = []
    project = store.project(job["project_id"])
    folder = store.project_dir(project["id"])
    started = time.perf_counter()

    def publish(**update):
        job.update(**update)
        atomic_json(job_path, job)

    def stage(name, label, action):
        record = dict(id=name, label=label, started_at=now())
        publish(stage=name, stage_label=label)
        begin = time.perf_counter()
        try:
            value = action()
        except Exception as exc:                          # noqa: BLE001
            record.update(status="failed", error=str(exc),
                          seconds=round(time.perf_counter() - begin, 3))
            job["stages"] = job.get("stages", []) + [record]
            job["error"] = str(exc)[-2000:]
            publish()
            raise
        record.update(status="completed", seconds=round(time.perf_counter() - begin, 3))
        job["stages"] = job.get("stages", []) + [record]
        publish()
        return value

    if not project.get("audio"):
        if project.get("source_url"):
            project = stage("download", "正在下载 B 站音频",
                            lambda: _download(project, folder, store))
        else:
            raise ValueError("项目没有音频，请先上传或粘贴链接。")
    audio = folder / project["audio"]
    if not audio.is_file():
        raise ValueError("音频文件不存在，请重新导入。")
    sha = signature(audio)["sha256"]
    job["audio_sha256"] = sha
    publish()

    target = stems_dir(sha)
    manifest_path = target / "manifest.json"
    if manifest_path.is_file() and not job.get("force"):
        manifest = read_json(manifest_path)
        manifest["cached"] = True
        project["stems"] = dict(sha256=sha, manifest=str(manifest_path), updated_at=now(),
                                model=manifest.get("model", {}).get("name", MODEL_NAME),
                                bpm=manifest.get("bpm"))
        store.save_project(project)
        publish(stage_label="复用已有分轨")
        return manifest

    target.mkdir(parents=True, exist_ok=True)
    work = target / ".work"
    if work.exists():
        for path in sorted(work.rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
        work.rmdir()
    work.mkdir(parents=True)

    def separate():
        source_wav = work / "input.wav"
        _decode_to_wav(audio, source_wav)
        command = [_model_python(), "-m", "demucs.separate", "-n", MODEL_NAME, "-o", str(work),
                   "--filename", "{stem}.{ext}", "--device", "cpu", "-j", "1", str(source_wav)]
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env.setdefault("TORCH_HOME", str(Path.home() / ".cache" / "torch"))
        env["OMP_NUM_THREADS"] = env.get("STEMS_THREADS", "8")
        process = subprocess.Popen([str(part) for part in command], cwd=str(ROOT), env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW
                                   if os.name == "nt" else 0)
        # demucs draws a carriage-return progress bar, so read chunks and split on CR/LF.
        last = -10
        buffer = ""
        while True:
            chunk = process.stdout.read(512) if process.stdout else b""
            if not chunk:
                break
            buffer += chunk.decode("utf-8", "replace")
            parts = re.split(r"[\r\n]+", buffer)
            buffer = parts.pop() if parts else ""
            for part in parts:
                match = re.search(r"(\d{1,3})%\|", part)
                if not match:
                    continue
                percent = int(match.group(1))
                if percent >= last + 5:
                    last = percent
                    publish(stage_label=f"正在分离音轨（{percent}%）")
        if process.wait():
            raise RuntimeError("demucs 分离失败，请查看任务日志。")

    stage("separate", "正在分离音轨（htdemucs，4 轨）", separate)

    def collect():
        wanted = {stem["id"] for stem in STEMS}
        produced = {path.stem: path for path in work.rglob("*.wav") if path.stem in wanted}
        missing = sorted(wanted - set(produced))
        if missing:
            raise RuntimeError("分离结果缺少轨道：" + ", ".join(missing))
        return produced

    produced = stage("collect", "收集分离结果", collect)

    def encode():
        wav_dir = target / "wav"
        wav_dir.mkdir(exist_ok=True)
        for stem_id, path in produced.items():
            _to_mp3(path, target / f"{stem_id}.mp3")
            path.replace(wav_dir / f"{stem_id}.wav")
        for leftover in ("input.wav", "input.mp3"):
            (work / leftover).unlink(missing_ok=True)
            (target / leftover).unlink(missing_ok=True)
            (wav_dir / leftover).unlink(missing_ok=True)

    stage("encode", "转码为可播放音频", encode)

    def annotate():
        data, sr = _load(target / "wav" / "vocals.wav")
        length = len(data) / sr
        run_dir = None
        if project.get("current_run"):
            candidate = store.run_dir(project["id"], project["current_run"])
            if (candidate / "beats.json").is_file():
                run_dir = candidate
        run_dir = run_dir or _latest_beats_run(folder)
        grid = beat_grid(audio, length, run_dir)
        bars = [float(b) for b in grid["bars"] if 0 <= b < length]
        if len(bars) < 2:
            period = 60.0 / max(40.0, float(grid["bpm"]))
            bars = [i * 4 * period for i in range(max(1, int(length / (4 * period))))]
        tracks = []
        for stem in STEMS:
            entry = dict(stem)
            entry["file"] = f"{stem['id']}.mp3"
            entry["wav"] = f"wav/{stem['id']}.wav"
            _annotate(target / "wav" / f"{stem['id']}.wav", entry, bars, length)
            tracks.append(entry)
        bar_rows = []
        for index, start in enumerate(bars):
            end = bars[index + 1] if index + 1 < len(bars) else length
            bar_rows.append(dict(index=index + 1, start=round(start, 3), end=round(end, 3)))
        return dict(duration=round(length, 3), sample_rate=sr, bpm=round(float(grid["bpm"]), 2),
                    bpm_source=grid["bpm_source"], bars=bar_rows,
                    beats=[round(b, 3) for b in grid["beats"][:4000]], tracks=tracks)

    analysis = stage("annotate", "标注音轨信息", annotate)

    manifest = dict(
        schema="chord_studio_stems_v1",
        project_id=project["id"], audio_sha256=sha,
        created_at=now(), cached=False,
        model=dict(name=MODEL_NAME, package="demucs 4.1.0", device="cpu",
                   seconds=round(time.perf_counter() - started, 1)),
        source_audio=project["audio"],
        **analysis,
    )
    atomic_json(manifest_path, manifest)
    for path in sorted(work.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    work.rmdir()

    project["stems"] = dict(sha256=sha, manifest=str(manifest_path), updated_at=now(),
                            model=MODEL_NAME, bpm=manifest["bpm"])
    store.save_project(project)
    publish(stage_label="分轨完成")
    return manifest
