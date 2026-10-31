"""Run separation on the CUDA runtime when it is present, with a CPU fallback.

The backend itself runs on CPU-only torch; the GPU environment lives at runtime/gpu and is
probed once per job. If the CUDA run fails (OOM, driver hiccup) the job retries on CPU.
"""
from __future__ import annotations

from pathlib import Path

WORKER = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\stems_worker.py")
text = WORKER.read_text(encoding="utf-8")

OLD_MODELS = '''MODELS = {
    "htdemucs": dict(id="htdemucs", label="标准", note="4 轨，约 2–3 分钟"),
    "htdemucs_ft": dict(id="htdemucs_ft", label="精细", note="4 个微调模型集成，约 2 倍时间，质量更好"),
}'''
NEW_MODELS = '''MODELS = {
    "htdemucs": dict(id="htdemucs", label="标准", note="4 轨，GPU 约 20 秒，CPU 约 2.5 分钟"),
    "htdemucs_ft": dict(id="htdemucs_ft", label="精细", note="4 个微调模型集成，GPU 约 45 秒，质量更好"),
}
GPU_PYTHON = Path(r"D:\\ChordStudio\\runtime\\gpu\\Scripts\\python.exe")'''
assert text.count(OLD_MODELS) == 1
text = text.replace(OLD_MODELS, NEW_MODELS)

OLD_PY = '''def _model_python() -> str:
    value = read_json(ROOT / "desktop_runtime.json").get("model_python")
    return str(value) if value else sys.executable
'''
NEW_PY = '''def _model_python() -> str:
    value = read_json(ROOT / "desktop_runtime.json").get("model_python")
    return str(value) if value else sys.executable


def _runtime() -> tuple[str, str]:
    """(interpreter, device): the CUDA runtime when importable, else the CPU model python."""
    candidates = []
    configured = read_json(ROOT / "desktop_runtime.json").get("stems_python")
    if configured:
        candidates.append(Path(configured))
    candidates.append(GPU_PYTHON)
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            probe = subprocess.run([str(candidate), "-c",
                                    "import torch;print(1 if torch.cuda.is_available() else 0)"],
                                   capture_output=True, timeout=180)
        except Exception:                              # noqa: BLE001
            continue
        if probe.stdout.strip() == b"1":
            return str(candidate), "cuda"
    return _model_python(), "cpu"
'''
assert text.count(OLD_PY) == 1
text = text.replace(OLD_PY, NEW_PY)

OLD_SEP = '''    def separate():
        source_wav = work / "input.wav"
        _decode_to_wav(audio, source_wav)
        command = [_model_python(), "-m", "demucs.separate", "-n", model, "-o", str(work),
                   "--filename", "{stem}.{ext}", "--device", "cpu", "-j", "1", str(source_wav)]
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
        command = [python, "-m", "demucs.separate", "-n", model, "-o", str(work),
                   "--filename", "{stem}.{ext}", "--device", device, "-j", "1", str(source_wav)]
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env.setdefault("TORCH_HOME", str(Path.home() / ".cache" / "torch"))
        env["OMP_NUM_THREADS"] = env.get("STEMS_THREADS", "8")
'''
assert text.count(OLD_SEP) == 1
text = text.replace(OLD_SEP, NEW_SEP)

# retry on CPU if a CUDA run dies
OLD_WAIT = '''        if process.wait():
            raise RuntimeError("demucs 分离失败，请查看任务日志。")
'''
NEW_WAIT = '''        code = process.wait()
        if code and device == "cuda":
            publish(stage_label="GPU 分离失败，改用 CPU 重试")
            command = [_model_python(), "-m", "demucs.separate", "-n", model, "-o", str(work),
                       "--filename", "{stem}.{ext}", "--device", "cpu", "-j", "1", str(source_wav)]
            job["device"] = "cpu"
            publish()
            backup = subprocess.run([str(part) for part in command], cwd=str(ROOT), env=env,
                                    capture_output=True,
                                    creationflags=subprocess.CREATE_NO_WINDOW
                                    if os.name == "nt" else 0)
            if backup.returncode:
                raise RuntimeError("demucs 分离失败，请查看任务日志。")
            return
        if code:
            raise RuntimeError("demucs 分离失败，请查看任务日志。")
'''
assert text.count(OLD_WAIT) == 1
text = text.replace(OLD_WAIT, NEW_WAIT)

OLD_MAN = '''        model=dict(name=model, label=MODELS[model]["label"], package="demucs 4.1.0",
                   device="cpu", seconds=round(time.perf_counter() - started, 1)),'''
NEW_MAN = '''        model=dict(name=model, label=MODELS[model]["label"], package="demucs 4.1.0",
                   device=job.get("device", "cpu"), seconds=round(time.perf_counter() - started, 1)),'''
assert text.count(OLD_MAN) == 1
text = text.replace(OLD_MAN, NEW_MAN)

WORKER.write_text(text, encoding="utf-8")
print("stems_worker: CUDA runtime with CPU fallback")
