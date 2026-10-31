"""Fix the missing /api/jobs route and drop the temp input from the stems cache."""
from __future__ import annotations

from pathlib import Path

SERVER = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\server.py")
text = SERVER.read_text(encoding="utf-8")
OLD = """    @application.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
"""
NEW = """    @application.get("/api/jobs/{job_id}")
    def job(job_id: str):
        return read_json(store.job_path(job_id))

    @application.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
"""
assert text.count(OLD) == 1
SERVER.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("server.py: jobs route restored")

WORKER = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\stems_worker.py")
text = WORKER.read_text(encoding="utf-8")

OLD_COLLECT = """    def collect():
        produced = {path.stem: path for path in work.rglob("*.wav")}
        missing = [stem["id"] for stem in STEMS if stem["id"] not in produced]
        if missing:
            raise RuntimeError("分离结果缺少轨道：" + ", ".join(missing))
        return produced
"""
NEW_COLLECT = """    def collect():
        wanted = {stem["id"] for stem in STEMS}
        produced = {path.stem: path for path in work.rglob("*.wav") if path.stem in wanted}
        missing = sorted(wanted - set(produced))
        if missing:
            raise RuntimeError("分离结果缺少轨道：" + ", ".join(missing))
        return produced
"""
assert text.count(OLD_COLLECT) == 1
text = text.replace(OLD_COLLECT, NEW_COLLECT)

OLD_ENC = """        for stem_id, path in produced.items():
            _to_mp3(path, target / f"{stem_id}.mp3")
            path.replace(wav_dir / f"{stem_id}.wav")
"""
NEW_ENC = """        for stem_id, path in produced.items():
            _to_mp3(path, target / f"{stem_id}.mp3")
            path.replace(wav_dir / f"{stem_id}.wav")
        for leftover in ("input.wav", "input.mp3"):
            (work / leftover).unlink(missing_ok=True)
            (target / leftover).unlink(missing_ok=True)
            (wav_dir / leftover).unlink(missing_ok=True)
"""
assert text.count(OLD_ENC) == 1
text = text.replace(OLD_ENC, NEW_ENC)
WORKER.write_text(text, encoding="utf-8")
print("stems_worker.py: collect/cleanup fixed")
