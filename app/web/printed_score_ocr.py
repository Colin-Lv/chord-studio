"""Offline printed chord-text candidates, never OMR or audio ground truth.

The web interpreter may lack OCR dependencies. A bounded local worker uses an
existing Python/model installation; this module never downloads or uploads data.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading

from app.chord_semantics_v2 import parse_chord

MAX_BYTES = 12 * 1024 * 1024
MAX_BLOCKS = 3000
TIMEOUT_SECONDS = 75
_OCR_LOCK = threading.BoundedSemaphore(1)
_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".pdf"}
_ROOT = Path(__file__).resolve().parents[2]
_METADATA = re.compile(r"capo|变调夹|調夾|指法调|指法調|实音调|實音調|原调|原調|选调|選調|演奏调|演奏調|\b(?:key|tuning)\b", re.I)
_TOKEN = re.compile(r"[A-G](?:[#b♯♭]{0,2})(?::?[A-Za-z0-9+#b♯♭øΔ()*,\-]*)?(?:/(?:[A-G][#b♯♭]{0,2}|[#b]*[1-9][0-9]*))?")


def _filename(value):
    name = str(value).replace("\\", "/").rsplit("/", 1)[-1]
    if not name or len(name) > 240:
        raise ValueError("Invalid printed score filename")
    return name


def _run_worker(data, filename):
    executable = Path(os.environ.get("HARMONY_SCORE_OCR_PYTHON", str(Path.home() / "miniconda3" / "python.exe")))
    if not executable.is_file():
        raise ValueError("Local OCR Python unavailable; set HARMONY_SCORE_OCR_PYTHON to an existing installation")
    worker = _ROOT / "helpers" / "printed_score_ocr_worker.py"
    scratch = _ROOT / ".tmp" / "score_ocr"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="import-", dir=scratch) as directory:
        path = Path(directory) / ("input" + Path(filename).suffix.lower())
        output = Path(directory) / "ocr.json"
        path.write_bytes(data)
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            completed = subprocess.run([str(executable), "-B", str(worker), str(path), str(output)],
                                       capture_output=True, timeout=TIMEOUT_SECONDS, env=environment,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Local printed-score OCR exceeded 75 seconds; use fewer/smaller pages") from exc
        if completed.returncode:
            # Worker emits concise controlled errors, not the private OCR text.
            message = completed.stderr.decode("utf-8", "replace").strip()[-500:]
            raise ValueError("Local printed-score OCR failed: " + message)
        if not output.is_file() or output.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("Local OCR produced missing or oversized output")
        try:
            return json.loads(output.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ValueError("Local OCR returned invalid JSON") from exc


def candidates_from_ocr(payload, filename, source_sha256):
    """Validate worker output; token boxes refer to their enclosing OCR block."""
    if not isinstance(payload, dict) or not isinstance(payload.get("blocks"), list):
        raise ValueError("Invalid local OCR block output")
    if len(payload["blocks"]) > MAX_BLOCKS:
        raise ValueError("Printed score has too many OCR blocks")
    blocks, events, metadata = [], [], []
    for index, item in enumerate(payload["blocks"]):
        if not isinstance(item, dict):
            raise ValueError("Invalid OCR block")
        text = item.get("text")
        box = item.get("bbox")
        confidence = item.get("confidence")
        page = item.get("page")
        if not isinstance(text, str) or len(text) > 2000:
            raise ValueError("Invalid OCR text")
        if type(page) is not int or not 1 <= page <= 3:
            raise ValueError("Invalid OCR page")
        if not isinstance(box, list) or len(box) != 4 or any(type(x) not in {int, float} or not math.isfinite(x) or not 0 <= x <= 1 for x in box):
            raise ValueError("Invalid normalized OCR bounding box")
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("Invalid OCR bounding box ordering")
        if type(confidence) not in {int, float} or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid OCR confidence")
        block = dict(id=f"ocr-{index+1}", text=text, page=page, bbox=box,
                     bbox_format="ltrb", bbox_units="normalized", confidence=confidence)
        blocks.append(block)
        if _METADATA.search(text):
            metadata.append(dict(kind="pitch_context_text", raw_text=text, page=page, bbox=box,
                                 bbox_format="ltrb", bbox_units="normalized",
                                 confidence=confidence, review_status="unreviewed"))
            continue
        # Whole-block all-token matching avoids harvesting a letter from lyrics.
        # Do not guess ambiguous OCR substitutions (e.g. 0 -> D or 8 -> B).
        tokens = text.strip().split()
        if not tokens or len(tokens) > 32 or any(not _TOKEN.fullmatch(token) for token in tokens):
            continue
        parsed = [parse_chord(token) for token in tokens]
        if any(item.state != "chord" for item in parsed):
            continue
        for ordinal, (token, sem) in enumerate(zip(tokens, parsed)):
            events.append(dict(id=f"printed-{index+1}-{ordinal+1}", kind="printed_chord_candidate",
                               source_file=filename, page=page, bbox=box, bbox_scope="ocr_block",
                               bbox_format="ltrb", bbox_units="normalized",
                               ocr_block_id=block["id"], token_index=ordinal, raw_chord=token,
                               normalized_harte=sem.harte, confidence=confidence,
                               confidence_kind="ocr_text_score_not_chord_accuracy", review_status="unreviewed",
                               measure_id=None, offset_quarter=None, duration_quarter=None,
                               alternatives=[], concert_pitch_harte=None))
    return dict(schema="score_reference_candidates_v1", format="printed_ocr", source_file=filename,
                source_sha256=source_sha256, events=events, measures=[], ocr_blocks=blocks,
                pages=payload.get("pages", []), metadata={"pitch_context_candidates": metadata,
                "capo": None, "fingering_key": None, "concert_key": None,
                "pitch_context_review_status": "unreviewed"},
                provenance=payload.get("provenance", {}),
                warnings=list(payload.get("warnings", [])) + [
                    "仅识别印刷和弦文字；未识别六线谱音符、指法图或小节结构。",
                    "候选列表保持OCR返回顺序，不代表乐曲演奏顺序；需按原图位置审核和标定小节。",
                    "OCR分数不是和弦准确率；歌词、水印与单字母可能混淆，所有候选均待人工核对。",
                    "变调夹、指法调、实音调及编配版本必须核对；未自动移调或对齐音频。"])


def parse_printed_score(data: bytes, filename: str) -> dict:
    filename = _filename(filename)
    if Path(filename).suffix.lower() not in _FORMATS:
        raise ValueError("Printed score must be PNG, JPEG, WebP or PDF")
    if not isinstance(data, bytes) or not data or len(data) > MAX_BYTES:
        raise ValueError("Printed score must be nonempty and at most 12 MiB")
    if not _OCR_LOCK.acquire(blocking=False):
        raise ValueError("Local printed-score OCR is busy; retry after the current import")
    try:
        payload = _run_worker(data, filename)
        return candidates_from_ocr(payload, filename, hashlib.sha256(data).hexdigest())
    finally:
        _OCR_LOCK.release()
