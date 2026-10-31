"""Small, atomic on-disk project store shared with the inference worker."""
from __future__ import annotations

import os
import hashlib
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
# The desktop shell launches this backend through a canonicalized \\?\ path. Children
# inherit ROOT as their working directory, and verbatim paths break pkg_resources /
# madmom (os.listdir('\\?\...\.') -> WinError 123), which silently killed the key and
# beat stages. Keep one plain path and every subprocess gets a normal cwd.
if str(ROOT).startswith("\\\\?\\"):
    ROOT = Path(str(ROOT)[4:])
DEFAULT_STORE = ROOT / "projects/web"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif", ".wma", ".webm"}
MAX_UPLOAD = 512 * 1024 * 1024
LEGACY_ROOT = Path(os.environ.get("HARMONY_LEGACY_ROOT", r"D:\ChordStudio\Dialogue"))
PROFILES = {
    "retained10": "当前冻结模型",
    "v1": "多来源和声 · 冻结 v1",
    "v2": "多来源和声 · 实验 v2",
    "madmom": "madmom · 单来源",
}
V1_MODEL = LEGACY_ROOT / "benchmarks/reports/session_20260926_accuracy2/multisource_candidate_reliability/selected_train66_head.joblib"
V2_MODEL = LEGACY_ROOT / "benchmarks/reports/model_v2_20261003/semantic_e1/semantic_e1_B52_v2.joblib"


def now():
    return datetime.now(timezone.utc).isoformat()


def identifier(value):
    if not re.fullmatch(r"[0-9a-f]{32}", str(value)):
        raise ValueError("无效的项目或任务 ID")
    return str(value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, payload):
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def signature(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(sha256=digest.hexdigest(), bytes=path.stat().st_size)


def bilibili_url(value):
    """Accept a single public video, retaining only its explicit part number."""
    text = value.strip()
    if re.fullmatch(r"BV[0-9A-Za-z]{10}", text):
        text = "https://www.bilibili.com/video/" + text
    parts = urlsplit(text)
    if (parts.scheme not in {"https", "http"} or parts.hostname not in {"www.bilibili.com", "bilibili.com", "m.bilibili.com"}
            or parts.username or parts.password or parts.port is not None):
        raise ValueError("请输入 bilibili.com/video/ 下的视频链接或 BV 号")
    match = re.fullmatch(r"/video/(BV[0-9A-Za-z]{10}|av[0-9]+)/?", parts.path)
    if not match:
        raise ValueError("仅支持单个 B 站视频链接，不支持合集、直播或用户主页")
    number = parse_qs(parts.query).get("p", ["1"])[0]
    if not number.isdigit() or not 1 <= int(number) <= 10000:
        raise ValueError("无效的视频分 P 编号")
    return "https://www.bilibili.com/video/" + match[1] + "/" + ("?p=" + str(int(number)) if int(number) != 1 else "")


class Store:
    def __init__(self, root=DEFAULT_STORE):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "jobs").mkdir(exist_ok=True)

    def project_dir(self, project_id):
        return self.root / identifier(project_id)

    def project(self, project_id):
        return read_json(self.project_dir(project_id) / "project.json")

    def save_project(self, project):
        atomic_json(self.project_dir(project["id"]) / "project.json", project)

    def create(self, title, **fields):
        project = dict(id=uuid4().hex, title=title, created_at=now(), status="ready", current_run=None, active_job=None, **fields)
        self.save_project(project)
        return project

    def job_path(self, job_id):
        return self.root / "jobs" / (identifier(job_id) + ".json")

    def run_dir(self, project_id, run_id):
        return self.project_dir(project_id) / "runs" / identifier(run_id)

    def result(self, project_id):
        project = self.project(project_id)
        if not project["current_run"]:
            raise FileNotFoundError("本项目尚未完成分析")
        return read_json(self.run_dir(project_id, project["current_run"]) / "analysis.json")
