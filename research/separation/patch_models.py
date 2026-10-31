"""Support two demucs checkpoints per song (standard + fine-tuned) with separate caches.

Layout: <data>/stems/<sha>/<model>/{manifest.json, *.mp3, wav/*.wav}
The existing htdemucs cache is migrated in place once.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

WORKER = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\stems_worker.py")
text = WORKER.read_text(encoding="utf-8")

OLD_HEAD = '''MODEL_NAME = "htdemucs"
STEPS_PER_BAR = 16
ACTIVE_DBFS = -45.0


def data_root() -> Path:
    value = os.environ.get("CHORD_DESKTOP_DATA")
    return Path(value) if value else Path(r"D:\\ChordStudio\\data")


def stems_dir(sha: str) -> Path:
    return data_root() / "stems" / sha
'''
NEW_HEAD = '''MODELS = {
    "htdemucs": dict(id="htdemucs", label="标准", note="4 轨，约 2–3 分钟"),
    "htdemucs_ft": dict(id="htdemucs_ft", label="精细", note="4 个微调模型集成，约 2 倍时间，质量更好"),
}
MODEL_NAME = "htdemucs"
STEPS_PER_BAR = 16
ACTIVE_DBFS = -45.0


def data_root() -> Path:
    value = os.environ.get("CHORD_DESKTOP_DATA")
    return Path(value) if value else Path(r"D:\\ChordStudio\\data")


def stems_dir(sha: str, model: str = MODEL_NAME) -> Path:
    return data_root() / "stems" / sha / model


def available_models(sha: str) -> list[dict]:
    found = []
    for model in MODELS.values():
        ready = (stems_dir(sha, model["id"]) / "manifest.json").is_file()
        found.append(dict(model, ready=ready))
    return found
'''
assert text.count(OLD_HEAD) == 1, text.count(OLD_HEAD)
text = text.replace(OLD_HEAD, NEW_HEAD)
WORKER.write_text(text, encoding="utf-8")
print("stems_worker: model table + per-model dirs")

# --- execute(): take the model from the job, cache per model
text = WORKER.read_text(encoding="utf-8")
OLD_EXEC = '''    audio = folder / project["audio"]
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
'''
NEW_EXEC = '''    audio = folder / project["audio"]
    if not audio.is_file():
        raise ValueError("音频文件不存在，请重新导入。")
    sha = signature(audio)["sha256"]
    model = job.get("model") or MODEL_NAME
    if model not in MODELS:
        raise ValueError("未知的分轨模型")
    job["audio_sha256"] = sha
    publish()

    target = stems_dir(sha, model)
    manifest_path = target / "manifest.json"
    if manifest_path.is_file() and not job.get("force"):
        manifest = read_json(manifest_path)
        manifest["cached"] = True
        stems = dict(project.get("stems") or {})
        stems[model] = dict(sha256=sha, manifest=str(manifest_path), updated_at=now(),
                            model=model, bpm=manifest.get("bpm"))
        project["stems"] = stems
        store.save_project(project)
        publish(stage_label="复用已有分轨")
        return manifest
'''
assert text.count(OLD_EXEC) == 1, text.count(OLD_EXEC)
text = text.replace(OLD_EXEC, NEW_EXEC)

# separate(): use the selected checkpoint
OLD_CMD = '''        command = [_model_python(), "-m", "demucs.separate", "-n", MODEL_NAME, "-o", str(work),'''
NEW_CMD = '''        command = [_model_python(), "-m", "demucs.separate", "-n", model, "-o", str(work),'''
assert text.count(OLD_CMD) == 1, text.count(OLD_CMD)
text = text.replace(OLD_CMD, NEW_CMD)

OLD_MANIFEST = '''        model=dict(name=MODEL_NAME, package="demucs 4.1.0", device="cpu",
                   seconds=round(time.perf_counter() - started, 1)),'''
NEW_MANIFEST = '''        model=dict(name=model, label=MODELS[model]["label"], package="demucs 4.1.0",
                   device="cpu", seconds=round(time.perf_counter() - started, 1)),'''
assert text.count(OLD_MANIFEST) == 1, text.count(OLD_MANIFEST)
text = text.replace(OLD_MANIFEST, NEW_MANIFEST)

OLD_SAVE = '''    project["stems"] = dict(sha256=sha, manifest=str(manifest_path), updated_at=now(),
                            model=MODEL_NAME, bpm=manifest["bpm"])
    store.save_project(project)'''
NEW_SAVE = '''    stems = dict(project.get("stems") or {})
    stems[model] = dict(sha256=sha, manifest=str(manifest_path), updated_at=now(),
                        model=model, bpm=manifest.get("bpm"))
    project["stems"] = stems
    store.save_project(project)'''
assert text.count(OLD_SAVE) == 1, text.count(OLD_SAVE)
text = text.replace(OLD_SAVE, NEW_SAVE)
WORKER.write_text(text, encoding="utf-8")
print("stems_worker: model-aware execute")

# --- migrate existing caches: <data>/stems/<sha>/* -> <data>/stems/<sha>/htdemucs/*
root = Path(r"D:\ChordStudio\data\stems")
moved = 0
for sha_dir in root.iterdir():
    if not sha_dir.is_dir():
        continue
    manifest = sha_dir / "manifest.json"
    if not manifest.is_file():
        continue
    target = sha_dir / "htdemucs"
    target.mkdir(exist_ok=True)
    for entry in list(sha_dir.iterdir()):
        if entry.name in {"manifest.json", "htdemucs"} or entry.name.endswith(".mp3") or entry.name == "wav":
            entry.rename(target / entry.name)
            moved += 1
print(f"migrated {moved} entries into htdemucs/ subdirs")
