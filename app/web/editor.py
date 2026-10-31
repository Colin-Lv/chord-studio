"""Independent, versioned human workbench; predictions and references stay immutable."""
from __future__ import annotations

import math
import re
from copy import deepcopy

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.chord_semantics_v2 import parse_chord
from .storage import atomic_json, now, read_json, signature
from .source_rows import lab_rows

_identity_cache = {}


def file_identity(path):
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in _identity_cache:
        if len(_identity_cache) > 1024:
            _identity_cache.clear()
        _identity_cache[key] = signature(path)["sha256"]
    return _identity_cache[key]


def verified_audio_hash(store, project):
    expected = project.get("audio_identity", {}).get("sha256")
    if not expected or not project.get("audio"):
        raise ValueError("音频身份缺失，不能保存人工修订")
    try:
        actual = file_identity(store.project_dir(project["id"]) / project["audio"])
    except FileNotFoundError as exc:
        raise HTTPException(409, "当前音频文件已移动，请保留草稿") from exc
    if expected != actual:
        raise HTTPException(409, "音频内容已变化，请保留草稿并重新分析")
    return actual


class EditorRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    run_id: str
    source_id: str
    source_sha256: str
    audio_sha256: str
    revision: int = Field(ge=0, strict=True)
    bars: list[dict] = Field(max_length=10000)
    sections: list[dict] = Field(default_factory=list, max_length=1000)
    events: list[dict] = Field(max_length=20000)


def token(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value):
        raise ValueError("无效的工作区对象 ID")
    return value


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限秒数")
    return float(value)


def context(store, project_id, source_id):
    token(source_id)
    project = store.project(project_id)
    result = store.result(project_id)
    if (project.get("id") != project_id or project["current_run"] != result["run_id"]
            or result.get("project_id") != project_id):
        raise HTTPException(409, "项目与分析身份不一致，请保留草稿后重新打开")
    source = next((s for s in result["sources"] if s["id"] == source_id), None)
    if source is None:
        raise HTTPException(404, "结果中没有该来源")
    audio_hash = verified_audio_hash(store, project)
    # The current upload's hash cannot establish which recording produced this
    # analysis. Accept either historical field name, but never infer a missing
    # binding or ignore a contradictory second declaration.
    declared_audio = [result[key] for key in ("audio", "audio_identity") if key in result]
    if (not declared_audio or any(not isinstance(value, dict)
            or value.get("sha256") != audio_hash for value in declared_audio)):
        raise HTTPException(409, "分析音频与当前文件身份不一致，请保留草稿并重新分析")
    source_file = store.run_dir(project_id, result["run_id"]) / source["path"]
    try:
        actual_source = file_identity(source_file)
    except FileNotFoundError as exc:
        raise HTTPException(409, "模型来源文件已移动，请保留草稿") from exc
    if actual_source != source["sha256"]:
        raise HTTPException(409, "模型来源内容已变化，请保留草稿并重新分析")
    try:
        verified_rows = lab_rows(source_file)
    except (OSError, ValueError) as exc:
        raise HTTPException(409, "模型来源时间轴无法验证，请保留草稿") from exc
    rows = source.get("rows")
    if (not isinstance(rows, list) or len(rows) != len(verified_rows)
            or any(not isinstance(row, dict)
                   or any(type(row.get(key)) not in {int, float} or row[key] != original[key]
                          for key in ("start", "end"))
                   or row.get("label") != original["label"]
                   for row, original in zip(rows, verified_rows))):
        raise HTTPException(409, "分析区间与模型来源文件不一致，请保留草稿并重新分析")
    duration = finite(result["duration"] if "duration" in result else verified_rows[-1]["end"], "音频长度")
    if duration <= 0:
        raise ValueError("尚无可编辑的音频时间轴")
    return project, result, source, duration, audio_hash


def location(store, project_id, result, source_id):
    return store.run_dir(project_id, result["run_id"]) / "editor" / (token(source_id) + ".json")


def initial_document(store, project_id, source_id):
    project, result, source, duration, audio_hash = context(store, project_id, source_id)
    raw_bars = result.get("bars", [])
    starts = sorted(set([0.0] + [float(b["start"]) for b in raw_bars if 0 < float(b["start"]) < duration] + [duration]))
    bars = []
    # A leading region can be an intro, silence or free rhythm. Its missing
    # downbeat does not establish a musical pickup.
    leading_region = bool(raw_bars and float(raw_bars[0]["start"]) > 1e-6)
    unconfirmed_ending = any(bar.get("end_boundary_source") == "audio_end_unconfirmed"
                             and abs(float(bar["end"]) - duration) <= 1e-7 for bar in raw_bars)
    for i, (start, end) in enumerate(zip(starts, starts[1:])):
        bars.append(dict(id=f"bar-{i+1:04d}", number=i if leading_region else i+1,
                         kind="unconfirmed" if not raw_bars or leading_region and i == 0 or unconfirmed_ending and end == duration else "measure",
                         grid_origin="automatic_unverified", start=start, end=end,
                         beats=[b["time"] for b in result.get("beats", []) if start <= b["time"] < end], meter=None))
    # Legacy corrections are migrated by value once, and retain unverified provenance.
    from .corrections import read_validated
    legacy = read_validated(store, project_id, result, source, duration, audio_hash)
    edits = {e["index"]: e["label"] for e in legacy["edits"]}
    events = [dict(id=f"event-{i:05d}", start=r["start"], end=r["end"], label=edits.get(i, r["label"]),
                   origin="manual" if i in edits else "model", source_index=i) for i, r in enumerate(source["rows"])]
    return dict(schema="harmony_bar_editor_v1", project_id=project_id, run_id=result["run_id"], source_id=source_id,
                source_sha256=source["sha256"], audio_sha256=audio_hash, revision=0, duration=duration,
                bars=bars, sections=[], events=events, saved_at=None, human_edited_unverified=True,
                warnings=["拍号未知时不推断为 4/4；弱起与边界需听辨核对", "各识别来源保留独立人工修订"])


def read_document(store, project_id, source_id):
    _, result, source, duration, audio_hash = context(store, project_id, source_id)
    path = location(store, project_id, result, source_id)
    if not path.exists():
        return initial_document(store, project_id, source_id)
    doc = read_json(path)
    expected = dict(schema="harmony_bar_editor_v1", project_id=project_id, run_id=result["run_id"],
                    source_id=source_id, source_sha256=source["sha256"], audio_sha256=audio_hash)
    if any(doc.get(key) != value for key, value in expected.items()):
        raise HTTPException(409, "已保存修订的项目、分析、来源或音频身份不一致，请保留草稿")
    if (type(doc.get("duration")) not in {int, float} or doc["duration"] != duration
            or type(doc.get("revision")) is not int or doc["revision"] < 0):
        raise HTTPException(409, "已保存修订的时间范围或版本无效，请保留草稿")
    return doc


def validate_document(doc, source, duration):
    bars = deepcopy(doc["bars"])
    seen = set()
    previous = 0.0
    if not bars:
        raise ValueError("小节网格不能为空")
    for bar in bars:
        ident = token(bar.get("id"))
        if ident in seen:
            raise ValueError("小节 ID 重复")
        seen.add(ident)
        start, end = finite(bar.get("start"), "小节开始"), finite(bar.get("end"), "小节结束")
        if abs(start - previous) > 1e-6 or end <= start or end > duration + 1e-6:
            raise ValueError("小节必须连续覆盖音频，不能重叠或留下间隙")
        previous = end
        if type(bar.get("number")) is not int or bar["number"] < 0 or bar.get("kind") not in {"pickup", "measure", "unconfirmed"}:
            raise ValueError("无效的小节编号或弱起标记")
        meter = bar.get("meter")
        if meter is not None and (not isinstance(meter, dict) or type(meter.get("numerator")) is not int
                or not 1 <= meter["numerator"] <= 32 or type(meter.get("denominator")) is not int
                or meter.get("denominator") not in {1, 2, 4, 8, 16, 32}):
            raise ValueError("无效拍号；未知拍号请留空")
        beats = [finite(t, "拍点") for t in bar.get("beats", [])]
        if len(beats) > 128 or beats != sorted(set(beats)) or any(not start <= t < end for t in beats):
            raise ValueError("拍点应递增且位于当前小节内")
    if abs(previous - duration) > 1e-6:
        raise ValueError("小节末端必须与音频长度一致")
    sections = deepcopy(doc.get("sections", []))
    order = {b["id"]: i for i, b in enumerate(bars)}
    ids = set()
    for section in sections:
        ident = token(section.get("id"))
        if ident in ids or not isinstance(section.get("name"), str) or not section["name"].strip() or len(section["name"]) > 80:
            raise ValueError("段落应有唯一 ID 与简短名称")
        ids.add(ident)
        if section.get("start_bar_id") not in order or section.get("end_bar_id") not in order or order[section["start_bar_id"]] > order[section["end_bar_id"]]:
            raise ValueError("段落范围引用不存在或倒序的小节")
    events = deepcopy(doc["events"])
    ids, previous = set(), 0.0
    for event in events:
        ident = token(event.get("id"))
        start, end = finite(event.get("start"), "和弦开始"), finite(event.get("end"), "和弦结束")
        if ident in ids or start < previous - 1e-6 or end <= start or start < 0 or end > duration + 1e-6:
            raise ValueError("和弦区间重复、重叠、倒序或越界")
        ids.add(ident)
        previous = end
        label = event.get("label")
        if not isinstance(label, str) or not 1 <= len(label) <= 160 or any(c.isspace() for c in label):
            raise ValueError("和弦标签无效")
        semantic = parse_chord(label)
        if semantic.parse_error or semantic.state == "missing":
            raise ValueError("和弦标签无法解析：" + label)
        if event.get("origin") not in {"model", "manual", "score"}:
            raise ValueError("无效的和弦来源")
        if event["origin"] == "model":
            index = event.get("source_index")
            if type(index) is not int or not 0 <= index < len(source["rows"]):
                raise ValueError("模型来源序号无效")
            original = source["rows"][index]
            if label != original["label"] or start < original["start"] - 1e-6 or end > original["end"] + 1e-6:
                raise ValueError("人工变化不能标为原始模型预测")
        if event["origin"] == "score" and not event.get("reference_event_id"):
            raise ValueError("曲谱采用必须保留候选身份")
    return bars, sections, events


def register_editor(application, store, save_lock):
    router = APIRouter()

    @router.get("/api/projects/{project_id}/editor/{source_id}")
    def get_editor(project_id: str, source_id: str):
        with save_lock:
            return read_document(store, project_id, source_id)

    @router.post("/api/projects/{project_id}/editor")
    def save_editor(project_id: str, body: EditorRequest):
        with save_lock:
            _, result, source, duration, audio_hash = context(store, project_id, body.source_id)
            if body.run_id != result["run_id"] or body.source_sha256 != source["sha256"] or body.audio_sha256 != audio_hash:
                raise HTTPException(409, "分析、音频或来源已变化，草稿不会覆盖新结果")
            old = read_document(store, project_id, body.source_id)
            if body.revision != old["revision"]:
                raise HTTPException(409, "另一窗口已保存修改，请导出草稿后重新载入")
            bars, sections, events = validate_document(body.model_dump(), source, duration)
            old_events = {e["id"]: e for e in old["events"]}
            for event in events:
                if event["origin"] == "score":
                    from .score_api import verified_label, transpose_label
                    # A later reference review must not rewrite or invalidate an already
                    # committed adoption snapshot. New/changed adoptions are checked afresh.
                    if old_events.get(event["id"]) == event:
                        continue
                    if transpose_label(event["label"], 0) != verified_label(store, project_id, event):
                        raise ValueError("曲谱采用标签与已审核实音候选不一致")
                    from .score_api import get_reference
                    event["reference_revision"] = get_reference(store, project_id, event["reference_id"])["revision"]
            # Do not allow arbitrary client fields to alter authoritative identity/provenance.
            record = {k: old[k] for k in ("schema", "project_id", "run_id", "source_id", "source_sha256", "audio_sha256", "duration", "warnings")}
            record.update(bars=bars, sections=sections, events=events, revision=old["revision"]+1,
                          saved_at=now(), human_edited_unverified=True)
            atomic_json(location(store, project_id, result, body.source_id), record)
            return record

    application.include_router(router)
