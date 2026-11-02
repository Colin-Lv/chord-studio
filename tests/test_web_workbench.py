"""HTTP/storage regressions; these fixtures are not acoustic accuracy tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.web.server import create_app
from app.web.storage import Store, atomic_json, bilibili_url, signature
from app.web.worker import build_bars, lab_rows, review_candidates

HEADERS = {"X-Harmony-Client": "studio"}


class FakeJobs:
    def __init__(self, store):
        self.store = store

    def submit(self, project_id, profile, download=False, key_mode="madmom"):
        return dict(id=uuid4().hex, project_id=project_id, profile=profile, download=download, key_mode=key_mode, status="queued")

    def shutdown(self):
        pass


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path, manager_factory=FakeJobs)
    with TestClient(app) as client:
        yield client


def upload(client, filename="song.wav", data=b"RIFFaudio bytes"):
    response = client.post("/api/upload", headers=HEADERS, files={"file": (filename, data, "audio/wav")})
    assert response.status_code == 201, response.text
    return response.json()


def test_experimental_key_mode_is_explicit_and_validated(client):
    project=upload(client)
    url=f"/api/projects/{project['id']}/analyze"
    assert client.post(url,headers=HEADERS,json={}).json()["key_mode"]=="madmom"
    assert client.post(url,headers=HEADERS,json={"key_mode":"fused"}).json()["key_mode"]=="fused"
    assert client.post(url,headers=HEADERS,json={"key_mode":"invalid"}).status_code==422


def fixture_result(client):
    project = upload(client)
    store = client.app.state.store
    run_id = uuid4().hex
    directory = store.run_dir(project["id"], run_id)
    directory.mkdir(parents=True)
    raw = "0.0000\t1.00\tC:maj\n1.00\t2.500\tG:7\n"
    path = directory / "raw.lab"
    path.write_bytes(raw.encode("utf-8"))
    source = dict(id="raw_madmom", name="madmom", path="raw.lab", **signature(path), rows=lab_rows(path))
    result = dict(schema="harmony_web_analysis_v1", project_id=project["id"], run_id=run_id,
                  audio=project['audio_identity'], sources=[source])
    atomic_json(directory / "analysis.json", result)
    project["current_run"] = run_id
    store.save_project(project)
    return project, result, raw


def correction(result, **overrides):
    return dict(run_id=result["run_id"], source_id="raw_madmom", source_sha256=result["sources"][0]["sha256"],
                revision=0, edits=[dict(index=0, label="C9/E")], key="C", key_origin="manual", display="roman", **overrides)


def test_upload_separate_ids_safe_names_and_playback_range(client):
    first = upload(client, "../../same.wav", b"0123456789")
    second = upload(client, "same.wav", b"different")
    assert first["id"] != second["id"]
    assert first["original_filename"] == "same.wav"
    assert first["audio_identity"]["sha256"] == hashlib.sha256(b"0123456789").hexdigest()
    response = client.get(f"/api/projects/{first['id']}/audio", headers={"Range": "bytes=2-5"})
    assert response.status_code == 206
    assert response.content == b"2345"
    assert len(client.get("/api/projects").json()) == 2


@pytest.mark.parametrize("filename,data", [("x.exe", b"123"), ("x.wav", b"")])
def test_bad_upload_leaves_no_project(client, filename, data):
    response = client.post("/api/upload", headers=HEADERS, files={"file": (filename, data)})
    assert response.status_code == 400
    assert client.get("/api/projects").json() == []


def test_mutations_require_local_client_and_matching_origin(client):
    assert client.post("/api/upload", files={"file": ("x.wav", b"bytes")}).status_code == 403
    assert client.post("/api/bilibili", headers={**HEADERS, "Origin": "https://evil.example"}, json={"url": "BV15P411W7Af"}).status_code == 403
    assert client.get("/api/config", headers={"Host": "evil.example"}).status_code == 400


@pytest.mark.parametrize("url", ["https://127.0.0.1/video/BV15P411W7Af", "https://www.bilibili.com.evil.test/video/BV15P411W7Af", "https://u:p@www.bilibili.com/video/BV15P411W7Af", "file:///etc/passwd", "https://www.bilibili.com:90/video/BV15P411W7Af", "https://www.bilibili.com/video/BV15P411W7Af?p=-1", "https://www.bilibili.com/medialist/play/1"])
def test_bilibili_rejects_nonvideo_or_foreign_url(url):
    with pytest.raises(ValueError):
        bilibili_url(url)


def test_bilibili_normalization_and_enqueue(client):
    value = "https://www.bilibili.com/video/BV15P411W7Af/?p=2&spm_id_from=ignored"
    assert bilibili_url(value) == "https://www.bilibili.com/video/BV15P411W7Af/?p=2"
    response = client.post("/api/bilibili", headers=HEADERS, json={"url": value, "profile": "v2"})
    assert response.status_code == 202
    assert response.json()["download"] is True
    project = client.get("/api/projects").json()[0]
    assert project["source_url"] == bilibili_url(value)


def test_corrections_preserve_raw_and_timestamps_and_survive_new_app(client):
    project, result, raw = fixture_result(client)
    prefix = f"/api/projects/{project['id']}"
    response = client.post(prefix + "/corrections", headers=HEADERS, json=correction(result))
    assert response.status_code == 200, response.text
    assert response.json()["revision"] == 1
    assert client.get(prefix + "/export/raw_madmom/original.lab").text == raw
    assert client.get(prefix + "/export/raw_madmom/corrected.lab").text == raw.replace("C:maj", "C9/E")
    app2 = create_app(client.app.state.store.root, manager_factory=FakeJobs)
    with TestClient(app2) as other:
        assert other.get(prefix + "/corrections/raw_madmom").json()["edits"] == [dict(index=0, label="C9/E")]
        assert other.get(prefix + "/result").json() == result


def test_corrections_reject_conflict_stale_run_and_boundary_changes(client):
    project, result, _ = fixture_result(client)
    url = f"/api/projects/{project['id']}/corrections"
    body = correction(result)
    assert client.post(url, headers=HEADERS, json=body).status_code == 200
    assert client.post(url, headers=HEADERS, json=body).status_code == 409
    body["revision"] = 1
    body["run_id"] = uuid4().hex
    assert client.post(url, headers=HEADERS, json=body).status_code == 409
    body = correction(result)
    body["edits"][0]["start"] = 0.5
    assert client.post(url, headers=HEADERS, json=body).status_code == 422


@pytest.mark.parametrize("edits", [[dict(index=999, label="C")], [dict(index=0, label="garbage")], [dict(index=0, label="C\nG")], [dict(index=0, label="C"),dict(index=0,label="G")]])
def test_invalid_corrections_are_not_committed(client, edits):
    project, result, _ = fixture_result(client)
    body = correction(result)
    body["edits"] = edits
    response = client.post(f"/api/projects/{project['id']}/corrections", headers=HEADERS, json=body)
    assert response.status_code == 400
    assert client.get(f"/api/projects/{project['id']}/corrections/raw_madmom").json()["revision"] == 0


def test_unknown_n_and_x_are_preserved(client):
    project, result, _ = fixture_result(client)
    body = correction(result)
    body["edits"] = [dict(index=0, label="X"), dict(index=1, label="N")]
    assert client.post(f"/api/projects/{project['id']}/corrections", headers=HEADERS, json=body).status_code == 200


def test_source_proposals_preserve_identity_without_duplicate_support():
    rows = [dict(start=0, end=2)]
    sources = [dict(id="chordino", rows=[dict(start=0,end=1,label="C:9"),dict(start=1,end=2,label="C9")]),
               dict(id="largevocab", rows=[dict(start=0,end=2,label="C:7")])]
    review_candidates(rows, sources)
    assert len(rows[0]["candidates"]) == 2
    assert all(candidate["score"] is None for candidate in rows[0]["candidates"])
    nine = next(c for c in rows[0]["candidates"] if "chordino" in c["sources"])
    assert nine["sources"] == {"chordino": 2}


def test_bars_follow_real_downbeats_without_leading_grid_invention():
    beats = [dict(time=.2,beat=3),dict(time=1,beat=1),dict(time=2,beat=2),dict(time=3,beat=1)]
    bars = build_bars(beats, 4, [dict(start=0,end=2.5,label="C"),dict(start=2.5,end=4,label="G")])
    assert [(bar["start"],bar["end"]) for bar in bars] == [(1,3),(3,4)]
    assert bars[0]["spans"][0]["start"] == 1


def test_job_restart_marks_orphan_interrupted(tmp_path):
    from app.web.jobs import Jobs
    store = Store(tmp_path)
    project = store.create("interrupted")
    job_id = uuid4().hex
    project["active_job"] = job_id
    store.save_project(project)
    atomic_json(store.job_path(job_id), dict(id=job_id, project_id=project["id"], status="running"))
    jobs = Jobs(store)
    try:
        assert json.loads(store.job_path(job_id).read_text(encoding="utf-8"))["status"] == "interrupted"
        assert store.project(project["id"])["active_job"] is None
    finally:
        jobs.shutdown()


def test_static_page_and_reused_notation_are_served(client):
    assert "主调" in client.get("/").text
    script = client.get("/assets/notation.js").text
    assert "function parseChord(label)" in script
    assert "function parseEditableChord(label)" in script
    assert "__ANALYSIS_JSON__" not in script


def test_queue_deduplicates_and_publishes_failure(tmp_path, monkeypatch,fake_job_process):
    import subprocess
    import threading
    import time
    from app.web.jobs import Jobs
    entered, release = threading.Event(), threading.Event()

    def failed_command(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        raise subprocess.CalledProcessError(1, args[0])

    fake_job_process(failed_command)
    store = Store(tmp_path)
    project = store.create("worker failure")
    jobs = Jobs(store)
    try:
        first = jobs.submit(project["id"], "v1")
        assert entered.wait(5)
        second = jobs.submit(project["id"], "v2")
        assert first["id"] == second["id"]
        release.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            saved = json.loads(store.job_path(first["id"]).read_text(encoding="utf-8"))
            if saved["status"] == "failed":
                break
            time.sleep(.01)
        assert saved["status"] == "failed"
        assert "CalledProcessError" in saved["error"]
    finally:
        release.set()
        jobs.shutdown()
    assert store.project(project["id"])["active_job"] is None
    assert store.project(project["id"])["current_run"] is None


def test_one_service_per_project_store(tmp_path):
    from app.web.jobs import Jobs
    store = Store(tmp_path)
    first = Jobs(store)
    try:
        with pytest.raises(RuntimeError, match="已有工作台"):
            Jobs(store)
    finally:
        first.shutdown()
    reopened = Jobs(store)
    reopened.shutdown()


def test_interrupted_worker_cannot_restart_acoustics(tmp_path, monkeypatch):
    from app.web.worker import execute
    import chord_recognizer
    store = Store(tmp_path)
    project = store.create("interrupted", audio="audio.wav")
    job_id = uuid4().hex
    atomic_json(store.job_path(job_id), dict(id=job_id, project_id=project["id"], profile="madmom", download=False, stages=[], status="interrupted"))
    monkeypatch.setattr(chord_recognizer, "decode_audio", lambda *a, **kw: pytest.fail("Interrupted worker decoded audio"))
    with pytest.raises(RuntimeError, match="重启中断"):
        execute(store, job_id)
    assert json.loads(store.job_path(job_id).read_text(encoding="utf-8"))["status"] == "interrupted"


def test_download_worker_keeps_media_inside_its_own_run(tmp_path, monkeypatch):
    """Mock CLI outputs verify wiring, not music predictions or network access."""
    import numpy as np
    import chord_recognizer
    from app.web.worker import execute
    store = Store(tmp_path)
    project = store.create("download", source_url="https://www.bilibili.com/video/BV15P411W7Af/")
    job_id = uuid4().hex
    project["active_job"] = job_id
    store.save_project(project)
    atomic_json(store.job_path(job_id), dict(id=job_id, project_id=project["id"], profile="madmom", download=True, stages=[], status="running"))
    monkeypatch.setenv("HARMONY_DOWNLOAD_PYTHON", "download-python")

    def fake_command(command, **kwargs):
        if "yt_dlp" in command:
            assert command[0] == "download-python"
            target = Path(command[command.index("-o") + 1])
            assert target.parent == store.run_dir(project["id"], job_id) / "source"
            (target.parent / "audio.m4a").write_bytes(b"mock media")
            atomic_json(target.parent / "audio.info.json", dict(title="mock download"))
        else:
            target = Path(command[command.index("-o") + 1])
            if command[2].endswith("madmom_bridge.py"):
                target.write_text("0 1 C:maj\n", encoding="utf-8")
            elif command[2].endswith("key_bridge.py"):
                atomic_json(target, dict(key="C"))
            elif command[2].endswith("downbeat_bridge.py"):
                atomic_json(target, [])
            else:
                pytest.fail(str(command))

    monkeypatch.setattr("app.web.worker.subprocess.run", fake_command)
    monkeypatch.setattr(chord_recognizer, "decode_audio", lambda *a, **kw: np.ones(22050, dtype=np.float32))
    execute(store, job_id)
    saved = store.project(project["id"])
    assert saved["audio"].replace("\\", "/") == f"runs/{job_id}/source/audio.m4a"
    result = json.loads((store.run_dir(project["id"], job_id) / "analysis.json").read_text(encoding="utf-8"))
    assert result["audio"]["sha256"] == hashlib.sha256(b"mock media").hexdigest()
    assert result["sources"][0]["rows"][0]["label"] == "C:maj"
