"""Run-scoped human decisions, independent of immutable automatic analysis."""
import os
import math
import re
import subprocess
import sys
from pathlib import Path
from fastapi import HTTPException

from .storage import ROOT, atomic_json, now, read_json, signature

CONTRACT = ROOT / "benchmarks/reports/harmonic_key_v2_20261006/engine_contract.json"
REGISTRY = ROOT / "benchmarks/key_review_cache_registry.json"
ADAPTER = ROOT / "app/key_review_adapter.py"


def checked_path(base, value):
    path = (base / value).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("Input path escapes project")
    return path


def validate_contract():
    contract = read_json(CONTRACT)
    implementation = contract["implementation"]
    for name, expected in implementation.items():
        path = ROOT / "app" / name
        actual = signature(path)
        digest = expected.get("sha256") if isinstance(expected, dict) else expected
        if actual["sha256"] != digest:
            raise ValueError("Frozen implementation identity changed: " + name)


def inputs(store, project_id):
    project, result = store.project(project_id), store.result(project_id)
    folder = store.run_dir(project_id, result["run_id"])
    audio = checked_path(store.project_dir(project_id), project["audio"])
    raw_source = next(row for row in result["sources"] if row["id"] == "raw_madmom")
    raw = checked_path(folder, raw_source["path"])
    a, r = signature(audio), signature(raw)
    if a["sha256"] != project["audio_identity"]["sha256"] or a["sha256"] != result["audio"]["sha256"] or r["sha256"] != raw_source["sha256"]:
        raise ValueError("音频或原始和弦结果身份不一致")
    provenance = read_json(Path(str(raw) + ".provenance.json"))
    if provenance["audio"]["sha256"] != a["sha256"] or provenance["lab"]["sha256"] != r["sha256"]:
        raise ValueError("原始和弦来源与音频不一致")
    if provenance.get("offset", 0) != 0 or abs(provenance["duration"] - result["duration"]) > 1 / 22050 + 1e-6:
        raise ValueError("原始音频时间支持与显示结果不一致")
    validate_contract()
    # Registry contains only input/prediction identities, never references or song names.
    for entry in read_json(REGISTRY).get("entries", []) if REGISTRY.exists() else []:
        if entry["audio"]["sha256"] == a["sha256"] and entry["raw"]["sha256"] == r["sha256"]:
            neutral = ROOT / entry["neutral"]["path"]
            if signature(neutral)["sha256"] != entry["neutral"]["sha256"]:
                raise ValueError("冻结中性输入已变更")
            if abs(read_json(neutral)["duration"] - result["duration"]) > 1 / 22050 + 1e-6:
                raise ValueError("冻结中性输入时间支持不一致")
            return result, audio, raw, neutral
    neutral = folder / "key_review_neutral.json"
    # Only measured time support is consumed by this frozen engine. No key,
    # corrected chords, reference, title or artist are sent into prediction.
    projection = dict(duration=result["duration"], bars=[{k: row[k] for k in ("start", "end")} for row in result["bars"]],
                      schema="measured_key_independent_grid_v1", audio=a, raw=r)
    if not neutral.exists() or read_json(neutral) != projection:
        atomic_json(neutral, projection)
    return result, audio, raw, neutral


def evidence(store, project_id, generate=False):
    if store.result(project_id).get("profile") == "retained10":
        return dict(available=False, reason="retained10 不输出主调；可使用独立 CNN 估计或人工调整。")
    result = store.result(project_id)
    path = store.run_dir(project_id, result["run_id"]) / "key_review_evidence.json"
    try:
        result, audio, raw, neutral = inputs(store, project_id)
        expected = {k: signature(p) for k, p in dict(audio=audio, raw=raw, neutral=neutral, contract=CONTRACT).items()}
        if path.exists():
            cached = read_json(path)
            if cached["inputs"] == expected and cached["adapter"] == signature(ADAPTER):
                return dict(available=True, **cached)
        if not generate:
            return dict(available=False, reason="未生成或缓存身份已失效")
        for entry in read_json(REGISTRY).get("entries", []) if REGISTRY.exists() else []:
            if all(entry[k]["sha256"] == expected[k]["sha256"] for k in ("audio", "raw", "neutral")) and entry.get("contract_sha256") == expected["contract"]["sha256"] and entry.get("prediction"):
                cached_prediction = ROOT / entry["prediction"]["path"]
                if signature(cached_prediction)["sha256"] != entry["prediction"]["sha256"]:
                    raise ValueError("冻结预测缓存已变更")
                cached = dict(schema="harmonic_key_review_evidence_v1", inputs=expected, adapter=signature(ADAPTER),
                              prediction=read_json(cached_prediction), frozen_cache=entry["prediction"],
                              score_kind="uncalibrated", automatic_override_enabled=False,
                              new_acoustic_inference=False, new_training=False)
                from app.key_review_adapter import contributions
                cached['weighted_contributions'] = contributions(cached['prediction'])
                atomic_json(path, cached)
                return dict(available=True, **cached)
        python = os.environ.get("HARMONY_MODEL_PYTHON")
        config = ROOT / ".tmp/harmony_web_model_python.txt"
        python = python or (config.read_text(encoding="utf-8-sig").strip() if config.exists() else getattr(sys, "_base_executable", sys.executable))
        argv = [python, "-B", "-m", "app.key_review_adapter"]
        for k, p in dict(audio=audio, raw=raw, neutral=neutral, contract=CONTRACT, output=path).items():
            argv += ["--" + k, str(p)]
        completed = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=90,
                                   env={**os.environ, "PYTHONUTF8": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
        atomic_json(path.with_suffix(".execution.json"), dict(argv=argv, returncode=completed.returncode,
                    stdout=completed.stdout, stderr=completed.stderr, new_acoustic_inference=False, timestamp=now()))
        if completed.returncode:
            raise ValueError("冻结证据生成失败，详见 key_review_evidence.execution.json")
        return dict(available=True, **read_json(path))
    except (OSError, ValueError, KeyError, StopIteration, subprocess.SubprocessError) as exc:
        return dict(available=False, reason=str(exc) or "缺少原始来源")


def validate_record(record, project_id, run_id, audio_sha, duration):
    """Read older run/audio-bound records without assigning missing identities."""
    required = dict(schema="global_key_human_review_v1", run_id=run_id, audio_sha256=audio_sha)
    if (not isinstance(record, dict) or any(record.get(k) != v for k, v in required.items())
            or ("project_id" in record and record["project_id"] != project_id)
            or ("duration" in record and (type(record["duration"]) not in {int, float} or record["duration"] != duration))
            or type(record.get("revision")) is not int or record["revision"] < 0
            or type(record.get("enabled")) is not bool or not isinstance(record.get("history"), list)
            or any(not isinstance(event, dict) for event in record["history"])):
        raise HTTPException(409, "主调审阅记录的项目、运行、音频、时长或版本无法验证；原记录已保留")
    manual = record.get("manual")
    if manual is not None and (not isinstance(manual, dict) or not isinstance(manual.get("key"), str)
            or manual["key"] and not re.fullmatch(r"(?:C|C#|Db|D|D#|Eb|E|F|F#|Gb|G|G#|Ab|A|A#|Bb|B)m?", manual["key"])):
        raise HTTPException(409, "已保存的人工主调格式无效；原记录已保留")
    return record


def state(store, project_id):
    from .editor import file_identity
    project, result = store.project(project_id), store.result(project_id)
    if (project.get("id") != project_id or result.get("project_id") != project_id
            or result.get("run_id") != project.get("current_run")):
        raise HTTPException(409, "主调审阅的项目与分析身份不一致；原记录已保留")
    audio_sha = result.get("audio", result.get("audio_identity", {})).get("sha256")
    duration = result.get("duration")
    if (not audio_sha or project.get("audio_identity", {}).get("sha256") != audio_sha
            or type(duration) not in {int, float} or not math.isfinite(duration) or duration <= 0):
        raise HTTPException(409, "主调审阅的音频或时长无法验证；原记录已保留")
    try:
        actual = file_identity(checked_path(store.project_dir(project_id), project["audio"]))
    except (OSError, KeyError, ValueError) as exc:
        raise HTTPException(409, "主调审阅音频无法验证；原记录已保留") from exc
    if actual != audio_sha:
        raise HTTPException(409, "音频文件已改变，请重新分析后再审阅；原记录已保留")
    path = store.run_dir(project_id, result["run_id"]) / "key_review.json"
    if path.exists():
        try:
            record = read_json(path)
        except (OSError, ValueError) as exc:
            raise HTTPException(409, "已保存的主调审阅无法解析；原记录已保留") from exc
        validate_record(record, project_id, result["run_id"], audio_sha, duration)
    else:
        record = dict(schema="global_key_human_review_v1", revision=0, enabled=False, history=[], manual=None)
    if not path.exists():
        legacy = []
        for correction in (store.run_dir(project_id, result["run_id"]) / "corrections").glob("*/corrections.json"):
            from .corrections import context_and_record
            row = context_and_record(store, project_id, correction.parent.name)[-1]
            if row.get("key") is not None:
                legacy.append(row)
        if legacy:
            latest = max(legacy, key=lambda row: row.get("saved_at", ""))
            if latest.get('key_origin') == 'manual':
                record["manual"] = dict(key=latest["key"], source="legacy_source_correction", at=latest.get("saved_at"), label_status="assisted_unverified")
    automatic = dict(source="fused" if result.get("key_fusion") else "madmom", key=(result.get("key_fusion") or result.get("key_analysis") or {}).get("key"),
                     evidence=result.get("key_fusion") or result.get("key_analysis"))
    proposal = evidence(store, project_id) if record["enabled"] else dict(available=False, reason="辅助审阅开关未启用")
    manual = record.get("manual")
    current = manual["key"] if manual else automatic["key"]
    return dict(**{k:v for k,v in record.items() if k not in ("project_id", "run_id", "audio_sha256", "duration")},
                project_id=project_id, run_id=result["run_id"], audio_sha256=audio_sha, duration=duration,
                automatic=automatic, suggestion=proposal, effective_key=current,
                review_status="human_confirmed_unverified" if manual else "unconfirmed",
                automatic_override_enabled=False, derived_key_analysis_stale=bool(manual and current != automatic["key"]))


def update(store, project_id, body):
    current = state(store, project_id)
    if any(body[k] != current[k] for k in ("run_id", "audio_sha256", "revision")):
        raise ValueError("CONFLICT: 运行、音频或主调审阅版本已变更，请重新打开")
    record = {k: current[k] for k in ("schema", "revision", "enabled", "history", "manual")}
    record.update(project_id=project_id, run_id=current["run_id"], audio_sha256=current["audio_sha256"], duration=current["duration"])
    action = body["action"]
    event = dict(action=action, at=now(), revision=current["revision"] + 1,
                 reviewer="local_user", prediction_visible=True, label_status="assisted_unverified")
    if action in ("enable", "disable"):
        record["enabled"] = action == "enable"
        if record["enabled"]:
            proposal = evidence(store, project_id, generate=True)
            event["evidence_available"] = proposal["available"]
    elif action == "restore":
        record["manual"] = None
    elif action in ("adopt", "custom"):
        key = body.get("key", "")
        if action == "adopt":
            proposal = current["suggestion"]
            if not proposal["available"] or not proposal["prediction"].get("key"):
                raise ValueError("建议不可用，不能采用")
            key = proposal["prediction"]["key"]
            event["contract"] = proposal["inputs"]["contract"]
            event["source"] = "harmonic_key_v2"
        else:
            event["source"] = "user_custom"
        if key and not re.fullmatch(r"(?:C|C#|Db|D|D#|Eb|E|F|F#|Gb|G|G#|Ab|A|A#|Bb|B)m?", key):
            raise ValueError("无效的主调")
        event["key"] = key
        record["manual"] = event.copy()
    else:
        raise ValueError("无效审阅操作")
    record["revision"] += 1
    record["history"].append(event)
    atomic_json(store.run_dir(project_id, current["run_id"]) / "key_review.json", record)
    return state(store, project_id)
