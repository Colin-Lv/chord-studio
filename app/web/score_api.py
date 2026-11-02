"""Local score references with independent review and identity guards."""
from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from app.chord_semantics_v2 import parse_chord
from .editor import token, verified_audio_hash
from .storage import atomic_json, atomic_text, now, read_json, signature

MAX_SCORE = 20 * 1024 * 1024
EXTENSIONS = {".xml", ".musicxml", ".mxl", ".mid", ".midi", ".gp", ".gpif", ".json", ".pdf", ".png", ".jpg", ".jpeg", ".webp"}


class ScoreReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=0, strict=True)
    events: list[dict] = Field(max_length=20000)
    metadata: dict = Field(default_factory=dict)


def transpose_label(label, semitones):
    parsed = parse_chord(label)
    if parsed.parse_error or parsed.state != "chord" or not parsed.harte:
        if parsed.state in {"N", "X"}:
            return parsed.state
        raise ValueError("曲谱和弦无法规范化")
    root = re.match(r"[A-G][b#]*", parsed.harte).group()
    names = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
    return names[(parsed.root_pc + semitones) % 12] + parsed.harte[len(root):]


def folder(store, project_id, run_id):
    return store.run_dir(project_id, run_id) / "score_references"


def get_reference(store, project_id, reference_id):
    token(reference_id)
    project = store.project(project_id)
    result = store.result(project_id)
    doc = read_json(folder(store, project_id, result["run_id"]) / reference_id / "reference.json")
    if (doc.get("project_id") != project_id or doc.get("run_id") != project["current_run"]
            or doc.get("audio_sha256") != verified_audio_hash(store, project)):
        raise HTTPException(409, "曲谱参考与当前音频或分析不匹配")
    return doc


def verified_label(store, project_id, event):
    doc = get_reference(store, project_id, token(event.get("reference_id")))
    candidate = next((c for c in doc["events"] if c["id"] == event.get("reference_event_id")), None)
    shift = doc.get("metadata", {}).get("transposition_semitones")
    if (candidate is None or candidate.get("review_status") != "verified" or candidate.get("kind") == "note_evidence"
            or not candidate.get("normalized_harte") or type(shift) is not int or event.get("alignment_confirmed") is not True):
        raise ValueError("曲谱采用前须核对和弦、实音转调与音频区间")
    return transpose_label(candidate["normalized_harte"], shift)


def register_scores(application, store, save_lock):
    router = APIRouter()

    @router.get('/api/score-capabilities')
    def score_capabilities():
        from .printed_score_ocr import available
        return dict(printed_ocr=available())

    @router.get("/api/projects/{project_id}/score-references")
    def list_references(project_id: str):
        result = store.result(project_id)
        paths = folder(store, project_id, result["run_id"]).glob("*/reference.json")
        return dict(items=[get_reference(store, project_id, p.parent.name) for p in sorted(paths)])

    @router.post("/api/projects/{project_id}/score-references", status_code=201)
    async def import_score(project_id: str, file: UploadFile = File(...)):
        filename = (file.filename or "score").replace("\\", "/").rsplit("/", 1)[-1]
        extension = Path(filename).suffix.lower()
        if extension not in EXTENSIONS:
            raise HTTPException(400, "请选择图片、PDF、MusicXML/MXL、MIDI 或审核 JSON")
        data = await file.read(MAX_SCORE + 1)
        await file.close()
        if not data or len(data) > MAX_SCORE:
            raise HTTPException(400, "曲谱应非空且不超过 20 MB")
        # Capture identity before parsing; don't hold the save lock during OCR.
        project = store.project(project_id)
        result = store.result(project_id)
        audio_hash = verified_audio_hash(store, project)
        if extension in {".pdf", ".png", ".jpg", ".jpeg", ".webp"}:
            from .printed_score_ocr import parse_printed_score
            parsed = await run_in_threadpool(parse_printed_score, data, filename)
        else:
            from .score_reference import parse_score
            parsed = await run_in_threadpool(parse_score, data, filename)
        with save_lock:
            current = store.project(project_id)
            if current["current_run"] != result["run_id"] or verified_audio_hash(store, current) != audio_hash:
                raise HTTPException(409, "解析期间音频或分析已变更，请重新导入")
            reference_id = uuid4().hex
            target = folder(store, project_id, result["run_id"]) / reference_id
            target.mkdir(parents=True)
            asset = target / ("source" + extension)
            asset.write_bytes(data)
            record = {**parsed, "id": reference_id, "revision": 0, "filename": filename, **signature(asset),
                      "asset": asset.name, "project_id": project_id, "run_id": result["run_id"],
                      "audio_sha256": audio_hash, "created_at": now(), "human_edited_unverified": True}
            for i, event in enumerate(record.get("events", [])):
                event.setdefault("id", f"score-event-{i:05d}")
                event["review_status"] = "unreviewed"
            metadata = record.get("metadata", {})
            if isinstance(metadata, list):
                # MIDI tempo/key/time-signature messages are evidence, not review settings.
                record["midi_metadata"] = metadata
                record["metadata"] = {}
            elif not isinstance(metadata, dict):
                raise ValueError("曲谱元数据格式无效")
            else:
                record["metadata"] = metadata
            # Even a score-supplied shift must be explicitly reviewed before adoption.
            record["metadata"]["transposition_semitones"] = None
            atomic_json(target / "reference.json", record)
            return record

    @router.get("/api/projects/{project_id}/score-references/{reference_id}/asset")
    def asset(project_id: str, reference_id: str):
        doc = get_reference(store, project_id, reference_id)
        return FileResponse(folder(store, project_id, doc["run_id"]) / reference_id / doc["asset"],
                            content_disposition_type="inline", headers={"Content-Security-Policy": "sandbox"})

    @router.post("/api/projects/{project_id}/score-references/{reference_id}/review")
    def review(project_id: str, reference_id: str, body: ScoreReview):
        with save_lock:
            old = get_reference(store, project_id, reference_id)
            if body.revision != old["revision"]:
                raise HTTPException(409, "曲谱审核已在其他窗口修改，请保留当前输入")
            metadata = body.metadata
            shift = metadata.get("transposition_semitones")
            capo = metadata.get("capo")
            if shift is not None and (type(shift) is not int or not -24 <= shift <= 24):
                raise ValueError("实音转调应为 -24 到 24 的整数半音，未知留空")
            if capo is not None and (type(capo) is not int or not 0 <= capo <= 24):
                raise ValueError("变调夹品位无效")
            for key in ("fingering_key", "sounding_key", "arrangement_version"):
                if metadata.get(key) is not None and (not isinstance(metadata[key], str) or len(metadata[key]) > 200):
                    raise ValueError("曲谱调性和编配说明过长或无效")
            known = {e["id"]: e for e in old["events"]}
            seen, events = set(), []
            editable = {"raw_chord", "normalized_harte", "review_status", "measure_id", "offset_quarter", "measure_quarters", "alternatives", "review_note"}
            for change in body.events:
                ident = change.get("id")
                if ident not in known or ident in seen:
                    raise ValueError("审核不能伪造或重复曲谱候选身份")
                seen.add(ident)
                event = {**known[ident], **{k: v for k, v in change.items() if k in editable}}
                if event.get("review_status") not in {"unreviewed", "verified", "rejected"}:
                    raise ValueError("无效审核状态")
                label = event.get("normalized_harte")
                if label:
                    if not isinstance(label, str) or len(label) > 160:
                        raise ValueError("和弦标签无效")
                    event["normalized_harte"] = transpose_label(label, 0)
                if event.get("kind") == "note_evidence" and (label or event["review_status"] == "verified"):
                    raise ValueError("MIDI 音符不能冒充已确认的和弦符号")
                if event["review_status"] == "verified" and not label:
                    raise ValueError("已审核候选必须有可解析和弦")
                event["sounding_harte"] = transpose_label(event["normalized_harte"], shift) if shift is not None and label else None
                events.append(event)
            if seen != set(known):
                raise ValueError("审核需保留全部候选；忽略项请标为 rejected")
            record = {**old, "events": events, "metadata": {**old.get("metadata", {}), **metadata},
                      "revision": old["revision"]+1, "reviewed_at": now(), "human_edited_unverified": True}
            atomic_json(folder(store, project_id, old["run_id"]) / reference_id / "reference.json", record)
            return record

    application.include_router(router)
