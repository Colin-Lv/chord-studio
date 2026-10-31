"""Add the BS-RoFormer two-stem model to the stems worker."""
from __future__ import annotations

from pathlib import Path

WORKER = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\stems_worker.py")
text = WORKER.read_text(encoding="utf-8")

OLD_STEMS = '''STEMS_6 = STEMS + (
    dict(id="guitar", name="吉他", short="GTR", instrument="吉他", color="#d98bd9"),
    dict(id="piano", name="钢琴", short="KEY", instrument="钢琴", color="#9fe08a"),
)'''
NEW_STEMS = '''STEMS_6 = STEMS + (
    dict(id="guitar", name="吉他", short="GTR", instrument="吉他", color="#d98bd9"),
    dict(id="piano", name="钢琴", short="KEY", instrument="钢琴", color="#9fe08a"),
)
STEMS_2 = (
    dict(id="vocals", name="人声", short="VOX", instrument="主唱 / 和声", color="#b6e7ce"),
    dict(id="instrumental", name="伴奏", short="INST", instrument="除人声以外的全部", color="#e6a06f"),
)'''
assert text.count(OLD_STEMS) == 1
text = text.replace(OLD_STEMS, NEW_STEMS)

OLD_MODELS = '''    "htdemucs_6s": dict(id="htdemucs_6s", label="六轨", note="额外分出吉他/钢琴，整体质量弱于四轨模型", stems=6),
}'''
NEW_MODELS = '''    "htdemucs_6s": dict(id="htdemucs_6s", label="六轨", note="额外分出吉他/钢琴，整体质量弱于四轨模型", stems=6),
    "bs_roformer": dict(id="bs_roformer", label="人声优先", note="BS-RoFormer 两轨，GPU 约 3 分钟，人声更强", stems=2),
}'''
assert text.count(OLD_MODELS) == 1
text = text.replace(OLD_MODELS, NEW_MODELS)

OLD_FOR = '''def stems_for(model: str):
    return STEMS_6 if MODELS.get(model, {}).get("stems") == 6 else STEMS'''
NEW_FOR = '''def stems_for(model: str):
    count = MODELS.get(model, {}).get("stems", 4)
    if count == 6:
        return STEMS_6
    if count == 2:
        return STEMS_2
    return STEMS'''
assert text.count(OLD_FOR) == 1
text = text.replace(OLD_FOR, NEW_FOR)

OLD_SEP = '''    def separate():
        source_wav = work / "input.wav"
        _decode_to_wav(audio, source_wav)
        python, device = _runtime()
        job["device"] = device
        publish()
        command = [python, "-m", "demucs.separate", "-n", model, "-o", str(work),
                   "--filename", "{stem}.{ext}", "--device", device, "-j", "1", str(source_wav)]
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env.setdefault("TORCH_HOME", str(Path.home() / ".cache" / "torch"))
        env["OMP_NUM_THREADS"] = env.get("STEMS_THREADS", "8")
'''
NEW_SEP = '''    def separate():
        source_wav = work / "input.wav"
        _decode_to_wav(audio, source_wav)
        python, device = _runtime()
        job["device"] = device
        publish()
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env.setdefault("TORCH_HOME", str(Path.home() / ".cache" / "torch"))
        env["OMP_NUM_THREADS"] = env.get("STEMS_THREADS", "8")
        if model == "bs_roformer":
            outdir = work / "roformer"
            result = subprocess.run([python, "-B", str(ROOT / "app/web/bsroformer_runner.py"),
                                     "--input", str(source_wav), "--outdir", str(outdir)],
                                    cwd=str(ROOT), env=env, capture_output=True, text=True,
                                    timeout=3600,
                                    creationflags=subprocess.CREATE_NO_WINDOW
                                    if os.name == "nt" else 0)
            if result.returncode:
                raise RuntimeError("BS-RoFormer 分离失败：" + (result.stderr or result.stdout or "")[-1500:])
            for produced in outdir.glob("*.wav"):
                name = produced.name.lower()
                if "vocal" in name:
                    shutil.copyfile(produced, work / "vocals.wav")
                elif "instrument" in name or "inst" in name:
                    shutil.copyfile(produced, work / "instrumental.wav")
            publish(stage_label="正在标注音轨信息")
            return
        command = [python, "-m", "demucs.separate", "-n", model, "-o", str(work),
                   "--filename", "{stem}.{ext}", "--device", device, "-j", "1", str(source_wav)]
'''
assert text.count(OLD_SEP) == 1
text = text.replace(OLD_SEP, NEW_SEP)

text = text.replace("import math\nimport os\n", "import math\nimport os\nimport shutil\n", 1)

WORKER.write_text(text, encoding="utf-8")
print("stems_worker: bs_roformer model wired")
