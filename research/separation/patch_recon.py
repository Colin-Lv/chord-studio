"""Add a reconstruction check to the manifest: how much of the mix the stems explain."""
from __future__ import annotations

from pathlib import Path

WORKER = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\stems_worker.py")
text = WORKER.read_text(encoding="utf-8")

OLD_ENC = '''        for leftover in ("input.wav", "input.mp3"):
            (work / leftover).unlink(missing_ok=True)
            (target / leftover).unlink(missing_ok=True)
            (wav_dir / leftover).unlink(missing_ok=True)
'''
NEW_ENC = '''        (target / "input.mp3").unlink(missing_ok=True)
        (wav_dir / "input.wav").unlink(missing_ok=True)
'''
assert text.count(OLD_ENC) == 1
text = text.replace(OLD_ENC, NEW_ENC)

OLD_ANN = '''        return dict(duration=round(length, 3), sample_rate=sr, bpm=round(float(grid["bpm"]), 2),
                    bpm_source=grid["bpm_source"], bars=bar_rows,
                    beats=[round(b, 3) for b in grid["beats"][:4000]], tracks=tracks)
'''
NEW_ANN = '''        reconstruction = _reconstruction(work / "input.wav", target / "wav", tracks)
        return dict(duration=round(length, 3), sample_rate=sr, bpm=round(float(grid["bpm"]), 2),
                    bpm_source=grid["bpm_source"], bars=bar_rows,
                    beats=[round(b, 3) for b in grid["beats"][:4000]], tracks=tracks,
                    reconstruction=reconstruction)
'''
assert text.count(OLD_ANN) == 1
text = text.replace(OLD_ANN, NEW_ANN)

OLD_TAIL = '''    atomic_json(manifest_path, manifest)
    for path in sorted(work.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    work.rmdir()
'''
NEW_TAIL = '''    atomic_json(manifest_path, manifest)
    for path in sorted(work.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    work.rmdir()
'''
assert text.count(OLD_TAIL) == 1

OLD_HELPER = '''def _download(project: dict, folder: Path, store) -> dict:'''
NEW_HELPER = '''def _reconstruction(mix_path: Path, wav_dir: Path, tracks) -> dict:
    """Residual = mix - sum(stems). Low residual means the stems explain the mix."""
    if not mix_path.is_file():
        return {}
    mix, rate = _load(mix_path)
    total = None
    for track in tracks:
        data, _ = _load(wav_dir / f"{track['id']}.wav")
        total = data if total is None else total + data
    if total is None:
        return {}
    length = min(len(mix), len(total))
    residual = mix[:length] - total[:length]
    mix_db = _rms_db(mix[:length].mean(axis=1))
    residual_db = _rms_db(residual.mean(axis=1))
    return dict(residual_db=round(residual_db, 1), mix_db=round(mix_db, 1),
                relative_db=round(residual_db - mix_db, 1),
                note="残差为混音减四轨之和；低不代表归属正确，只代表没有丢内容")


def _download(project: dict, folder: Path, store) -> dict:'''
assert text.count(OLD_HELPER) == 1
text = text.replace(OLD_HELPER, NEW_HELPER)

WORKER.write_text(text, encoding="utf-8")
print("stems_worker: reconstruction metric")
