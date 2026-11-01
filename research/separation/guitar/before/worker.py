"""Run existing recognition CLIs and produce data, never per-song HTML."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .storage import (ROOT, AUDIO_EXTENSIONS, PROFILES, V1_MODEL, V2_MODEL,
                      Store, atomic_json, now, read_json, signature)
from .source_rows import lab_rows

sys.path.insert(0, str(ROOT / "app"))


def review_candidates(rows, sources):
    """Unscored, source-attributed proposals; never manufacture confidence."""
    from chord_semantics_v2 import parse_chord
    for row in rows:
        candidates = {}
        for source in sources:
            for event in source["rows"]:
                if event["start"] >= row["end"]:
                    break
                overlap = min(row["end"], event["end"]) - max(row["start"], event["start"])
                if overlap <= 0:
                    continue
                label = event["label"]
                semantic = parse_chord(label)
                identity = semantic.identity if not semantic.parse_error else ("unparsed", label)
                candidate = candidates.setdefault(identity, dict(label=label, score=None, sources={}))
                candidate["sources"][source["id"]] = candidate["sources"].get(source["id"], 0) + overlap
        row["candidates"] = sorted(candidates.values(), key=lambda item: (-len(item["sources"]), item["label"]))


def build_bars(beats, duration, chords):
    starts = [row["time"] for row in beats if row["beat"] == 1 and 0 <= row["time"] < duration]
    end_has_downbeat = any(row["beat"] == 1 and abs(row["time"] - duration) <= 1e-7 for row in beats)
    bars = []
    for start, end in zip(starts, starts[1:] + [duration]):
        spans = [dict(start=max(start, row["start"]), end=min(end, row["end"]), chord=row["label"])
                 for row in chords if row["end"] > start and row["start"] < end]
        bars.append(dict(bar=len(bars) + 1, start=start, end=end, spans=spans,
                         grid_origin="automatic_unverified", start_boundary_source="predicted_downbeat",
                         end_boundary_source="predicted_downbeat" if end < duration or end_has_downbeat else "audio_end_unconfirmed"))
    return bars


def execute(store, job_id):
    if read_json(store.job_path(job_id))["profile"] == "retained10":
        from .retained_worker import execute as retained_execute
        return retained_execute(store, job_id)
    job_path = store.job_path(job_id)
    job = read_json(job_path)
    project = store.project(job["project_id"])
    folder = store.project_dir(project["id"])
    output = store.run_dir(project["id"], job_id)
    output.mkdir(parents=True, exist_ok=False)
    start_time = time.perf_counter()
    warnings = []

    def publish():
        # After a server restart, an old child must not resurrect its job.
        if read_json(job_path)["status"] == "interrupted":
            raise RuntimeError("此任务已被服务重启中断，请使用新任务的结果。")
        atomic_json(job_path, job)

    def stage(name, title, action, optional=False):
        job.update(stage=name, stage_label=title)
        publish()
        started = time.perf_counter()
        record = dict(id=name, label=title, started_at=now())
        try:
            value = action(record)
            record["status"] = "completed"
            return value
        except Exception as exc:
            record.update(status="failed", error=str(exc))
            if not optional:
                raise
            warnings.append(title + "未完成，可继续复核和弦。详情见任务日志。")
            traceback.print_exc()
            return None
        finally:
            record["seconds"] = time.perf_counter() - started
            job["stages"].append(record)
            publish()

    def command(record, *arguments, python=sys.executable):
        args = [python, "-B", *map(str, arguments)]
        record.setdefault("commands", []).append(args)
        print("Stage: " + job["stage"], flush=True)
        subprocess.run(args, cwd=ROOT, check=True, timeout=1800)

    if job["profile"] != "madmom":
        model = V2_MODEL if job["profile"] == "v2" else V1_MODEL
        # Both profiles expose the frozen v1 comparison; validate before acoustics.
        def validate(record):
            command(record, "-c", "import sys;sys.path.insert(0,'app');from apply_multisource_candidates import load_frozen_artifact;load_frozen_artifact(sys.argv[1])", V1_MODEL)
            if model == V2_MODEL:
                command(record, "-c", "import sys;sys.path.insert(0,'app');from multisource_model_v2 import load_artifact;load_artifact(sys.argv[1])", model)
        stage("preflight", "检查模型与运行环境", validate)

    if job["download"]:
        def download(record):
            target = output / "source"
            target.mkdir(exist_ok=True)
            command(record, "-m", "yt_dlp", "--ignore-config", "--no-playlist", "--no-update", "--no-cache-dir",
                    "--socket-timeout", "20", "--retries", "2", "--fragment-retries", "2", "--max-filesize", "512M",
                    "--write-info-json", "--no-progress", "-f", "bestaudio", "-o", target / "audio.%(ext)s", "--", project["source_url"],
                    python=os.environ.get("HARMONY_DOWNLOAD_PYTHON", sys.executable))
            files = [p for p in target.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS]
            if len(files) != 1 or files[0].stat().st_size == 0:
                raise ValueError("下载没有产生唯一的有效音频文件；请查看下载日志或改为上传本地音频。")
            metadata = read_json(target / "audio.info.json")
            project.update(audio=str(files[0].relative_to(folder)), original_filename=files[0].name,
                           title=str(metadata.get("title") or project["title"]), audio_identity=signature(files[0]))
            publish()
            if store.project(project["id"]).get("active_job") != job_id:
                raise RuntimeError("下载任务已失效，请查看当前任务。")
            store.save_project(project)
        stage("download", "正在从 B 站获取音频", download)

    audio = folder / project["audio"]
    from chord_recognizer import decode_audio
    from audio_waveform import peak_envelope
    import numpy as np

    def decode(record):
        samples = decode_audio(audio, sample_rate=22050, duration=1801)
        if len(samples) / 22050 > 1800:
            raise ValueError("当前工作台支持 30 分钟以内的单曲，请裁剪后再上传。")
        if not np.isfinite(samples).all() or len(samples) / 22050 < 1:
            raise ValueError("音频应至少有 1 秒，且不能包含无效采样值。")
        return samples
    y = stage("decode", "解码音频与生成波形", decode)
    duration = len(y) / 22050
    waveform = peak_envelope(y, 22050)
    manifest = output / "audio_only.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "audio"])
        writer.writeheader()
        writer.writerow(dict(id="audio", audio=str(audio)))
    raw = output / "raw_madmom.lab"
    stage("madmom", "madmom 正在识别和弦", lambda r: command(r, ROOT / "app/madmom_bridge.py", audio, "-o", raw))
    paths = [("raw_madmom", "madmom", raw)]
    models = {}
    selected = "raw_madmom"
    if job["profile"] != "madmom":
        chordino = output / "chordino/labs/audio.lab"
        large = output / "largevocab/labs/audio.lab"
        stage("chordino", "Chordino 正在识别和弦", lambda r: command(r, ROOT / "app/generate_audio_only_chordino_labs.py", manifest, "--output", output / "chordino", "--expected-count", "1"))
        stage("largevocab", "大词表模型正在识别", lambda r: command(r, ROOT / "app/generate_audio_only_largevocab_labs_v2.py", "--audio", audio, "--output-dir", output / "largevocab", "--export-posteriors", "--threads", "2"))
        cache = output / "nnls_cache"
        v1 = output / "v1.lab"
        stage("v1", "融合多来源候选 · 冻结 v1", lambda r: command(r, ROOT / "app/apply_multisource_candidates.py", "--audio", audio, "--raw", raw, "--chordino", chordino, "--largevocab", large, "--model", V1_MODEL, "--output", v1, "--cache", cache, "--preserve-root"))
        paths = [("v1", "冻结 v1 · B66", v1), ("raw_madmom", "madmom", raw), ("chordino", "Chordino", chordino), ("largevocab", "大词表", large)]
        models["v1"] = dict(weights=signature(V1_MODEL), contract=signature(V1_MODEL.with_suffix(".contract.json")))
        selected = "v1"
        if job["profile"] == "v2":
            notes = list(cache.glob("*.npz"))
            if len(notes) != 1:
                raise ValueError("Expected one whole-audio NNLS cache")
            v2 = output / "v2.lab"
            stage("v2", "完整语义候选 · 实验 v2", lambda r: command(r, ROOT / "app/apply_multisource_candidates_v2.py", "--audio", audio, "--raw", raw, "--chordino", chordino, "--largevocab", large, "--notes", notes[0], "--model", V2_MODEL, "--output", v2))
            paths.insert(0, ("v2", "实验 v2 · B52", v2))
            models["v2"] = dict(weights=signature(V2_MODEL), contract=signature(V2_MODEL.with_suffix(".contract.json")))
            selected = "v2"
    key_path = output / "key.json"
    stage("key", "识别全曲主调", lambda r: command(r, ROOT / "app/key_bridge.py", audio, "-o", key_path), optional=True)
    beat_path = output / "beats.json"
    stage("beats", "定位拍点与小节", lambda r: command(r, ROOT / "app/downbeat_bridge.py", audio, "-o", beat_path), optional=True)
    beats = read_json(beat_path) if beat_path.is_file() else []
    key_analysis = read_json(key_path) if key_path.is_file() else None
    key_fusion = None
    if job.get("key_mode") == "fused":
        def fuse_key(record):
            from key_fusion import read_raw_lab
            from key_disagreement import analyze, load_model
            neutral_base = output / "key_neutral"
            extra = ["--downbeats", beat_path] if beat_path.is_file() else []
            command(record, ROOT / "app/chord_studio.py", audio, "-o", neutral_base,
                    "--analysis-model", "template", "--mode", "seventh", "--feature-mode", "fast",
                    "--all-key-candidates", "--no-html", "--no-structure", *extra)
            neutral_path = output / "key_neutral_analysis.json"
            result = analyze(key_analysis, read_json(neutral_path), read_raw_lab(raw), model=load_model())
            result["neutral_artifact"] = str(neutral_path.relative_to(output))
            # Web chord sources are key-independent; no final chord rerun is needed.
            result["studio_passes"] = 1
            atomic_json(output / "key_fusion.json", result)
            return result
        key_fusion = stage("key_fusion", "独立和声证据与调性冲突检查 · 实验", fuse_key)
    sources = [dict(id=name, name=title, path=str(path.relative_to(output)), **signature(path), rows=lab_rows(path)) for name, title, path in paths]
    acoustic = [s for s in sources if s["id"] in {"raw_madmom", "chordino", "largevocab"}]
    for source in sources:
        review_candidates(source["rows"], acoustic)
    chords = next(s["rows"] for s in sources if s["id"] == selected)
    bars = build_bars(beats, duration, chords)

    def structure_analysis(record):
        from song_structure import analyze_song_structure
        structure = analyze_song_structure(dict(bars=bars, duration=duration, beat_source="madmom",
                                               key=key_fusion["key"] if key_fusion else (key_analysis or {}).get("key")), y=y, sr=22050)
        source = next(source for source in sources if source["id"] == selected)
        structure["harmony_source"] = dict(source_id=source["id"], source_sha256=source["sha256"],
                                           run_id=job_id, audio_sha256=project["audio_identity"]["sha256"])
        return structure
    structure = stage("structure", "分析段落相似性与和声循环", structure_analysis, optional=True)
    if signature(audio) != project["audio_identity"]:
        raise ValueError("分析期间音频文件发生变化，请重新上传")
    result = dict(schema="harmony_web_analysis_v1", project_id=project["id"], run_id=job_id, title=project["title"],
                  profile=job["profile"], profile_name=PROFILES[job["profile"]], duration=duration,
                  audio=dict(url=f"/api/projects/{project['id']}/audio", **project["audio_identity"]),
                  key_analysis=key_analysis,
                  beats=beats, bars=bars, waveform=waveform, structure=structure,
                  sources=sources, default_source=selected, warnings=warnings,
                  candidate_score_kind="unscored_source_proposals_not_model_ranking",
                  reference_available=False, model_identity=models,
                  provenance=dict(created_at=now(), worker=signature(__file__), fitting=False, reference_inputs=False,
                                  elapsed_seconds=time.perf_counter() - start_time, stages=job["stages"],
                                  original_intervals_preserved=True))
    if key_fusion is not None:
        from key_fusion import attach_key_metadata
        attach_key_metadata(result, key_fusion, key_analysis)
    atomic_json(output / "analysis.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    store = Store(args.store)
    try:
        job = read_json(store.job_path(args.job))
        if job.get("profile") == "stems":
            from .stems_worker import execute as execute_stems
            execute_stems(store, args.job)
        elif job.get("profile") == "drums_parts":
            from .drums_worker import execute as execute_drums
            execute_drums(store, args.job)
        elif job.get("profile") == "vocal_parts":
            from .vocal_parts_worker import execute as execute_vocals
            execute_vocals(store, args.job)
        else:
            execute(store, args.job)
    except Exception as exc:
        job = read_json(store.job_path(args.job))
        job["error"] = f"{type(exc).__name__}: {exc}"
        atomic_json(store.job_path(args.job), job)
        raise


if __name__ == "__main__":
    main()
