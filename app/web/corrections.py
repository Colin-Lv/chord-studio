"""Validate saved source corrections before displaying, exporting or migrating."""
import math
import re

from fastapi import HTTPException

from app.chord_semantics_v2 import parse_chord
from .source_rows import lab_rows
from .storage import read_json

SCHEMA = "harmony_web_corrections_v1"


def validate_record(record, project_id, result, source, duration, audio_sha):
    required = dict(run_id=result["run_id"], source_id=source["id"], source_sha256=source["sha256"])
    optional = dict(schema=SCHEMA, project_id=project_id, audio_sha256=audio_sha)
    if (not isinstance(record, dict)
            or any(record.get(k) != value for k, value in required.items())
            or any(k in record and record[k] != value for k, value in optional.items())
            or type(record.get("revision")) is not int or record["revision"] < 0
            or "duration" in record and (type(record["duration"]) not in {int, float}
                or not math.isfinite(record["duration"]) or record["duration"] != duration)):
        raise HTTPException(409, "已保存和弦修正的项目、运行、来源、音频或版本不一致，请保留草稿")
    edits = record.get("edits")
    if not isinstance(edits, list) or len(edits) > 20000:
        raise HTTPException(409, "已保存和弦修正格式无效，请保留原文件")
    seen = set()
    for edit in edits:
        if (not isinstance(edit, dict) or type(edit.get("index")) is not int
                or not 0 <= edit["index"] < len(source["rows"]) or edit["index"] in seen
                or not isinstance(edit.get("label"), str) or not 1 <= len(edit["label"]) <= 160
                or any(c.isspace() for c in edit["label"])):
            raise HTTPException(409, "已保存和弦修正含无效或重复区间，请保留原文件")
        semantic = parse_chord(edit["label"])
        if semantic.parse_error or semantic.state == "missing":
            raise HTTPException(409, "已保存和弦标签无法解析，请保留原文件")
        seen.add(edit["index"])
    if (record.get("display", "roman") not in {"roman", "name"}
            or record.get("key_origin", "automatic") not in {"automatic", "manual", "unknown"}
            or not isinstance(record.get("key", ""), str)
            or record.get("key") and not re.fullmatch(r"(?:C|C#|Db|D|D#|Eb|E|F|F#|Gb|G|G#|Ab|A|A#|Bb|B)m?", record["key"])):
        raise HTTPException(409, "已保存和弦修正的主调或显示设置无效，请保留原文件")
    return record


def read_validated(store, project_id, result, source, duration, audio_sha):
    path = store.run_dir(project_id, result["run_id"])/"corrections"/source["id"]/"corrections.json"
    if not path.exists():
        return dict(schema=SCHEMA, project_id=project_id, run_id=result["run_id"], source_id=source["id"],
                    source_sha256=source["sha256"], audio_sha256=audio_sha, duration=duration, revision=0, edits=[])
    try:
        record = read_json(path)
    except (OSError, ValueError) as exc:
        raise HTTPException(409, "已保存和弦修正无法读取，请保留原文件") from exc
    return validate_record(record, project_id, result, source, duration, audio_sha)


def context_and_record(store, project_id, source_id):
    from .editor import context
    project, result, source, duration, audio_sha = context(store, project_id, source_id)
    if "audio" in result and result["audio"].get("sha256") != audio_sha:
        raise HTTPException(409, "分析音频与当前文件身份不一致，请保留草稿")
    record = read_validated(store, project_id, result, source, duration, audio_sha)
    # Export exact timestamp spellings from the verified LAB, never mutable JSON metadata.
    source = {**source, "rows": lab_rows(store.run_dir(project_id, result["run_id"])/source["path"])}
    return project, result, source, duration, audio_sha, record
