"""Match the two-stem output levels to the mix and skip the tautological residual.

audio-separator rescales stems when a sample exceeds full scale (seen on the accompaniment,
+1.09). For a vocals/instrumental split the accompaniment is defined as mix - vocals, so we
fit that gain back; and a reconstruction check is meaningless when the split is complementary.
"""
from __future__ import annotations

from pathlib import Path

WORKER = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\stems_worker.py")
text = WORKER.read_text(encoding="utf-8")

OLD_COPY = '''            for produced in outdir.glob("*.wav"):
                name = produced.name.lower()
                if "vocal" in name:
                    shutil.copyfile(produced, work / "vocals.wav")
                elif "instrument" in name or "inst" in name:
                    shutil.copyfile(produced, work / "instrumental.wav")
            publish(stage_label="正在标注音轨信息")
            return
'''
NEW_COPY = '''            for produced in outdir.glob("*.wav"):
                name = produced.name.lower()
                if "vocal" in name:
                    shutil.copyfile(produced, work / "vocals.wav")
                elif "instrument" in name or "inst" in name:
                    shutil.copyfile(produced, work / "instrumental.wav")
            _balance_complementary(work / "vocals.wav", work / "instrumental.wav", source_wav)
            publish(stage_label="正在标注音轨信息")
            return
'''
assert text.count(OLD_COPY) == 1
text = text.replace(OLD_COPY, NEW_COPY)

OLD_HELPER = '''def _reconstruction(mix_path: Path, wav_dir: Path, tracks) -> dict:'''
NEW_HELPER = '''def _balance_complementary(vocals_path: Path, bed_path: Path, mix_path: Path):
    """Rescale the accompaniment so vocals + accompaniment matches the mix."""
    import soundfile as sf
    vocals, rate = sf.read(str(vocals_path), always_2d=True, dtype="float32")
    bed, _ = sf.read(str(bed_path), always_2d=True, dtype="float32")
    mix, _ = sf.read(str(mix_path), always_2d=True, dtype="float32")
    length = min(len(vocals), len(bed), len(mix))
    remain = (mix[:length] - vocals[:length]).mean(axis=1)
    bed_mono = bed[:length].mean(axis=1)
    denominator = float((bed_mono.astype("float64") ** 2).sum())
    if denominator <= 0:
        return
    gain = float((bed_mono.astype("float64") * remain.astype("float64")).sum() / denominator)
    gain = max(0.5, min(2.0, gain))
    if abs(gain - 1.0) > 0.005:
        sf.write(str(bed_path), bed * gain, rate, subtype="FLOAT")


def _reconstruction(mix_path: Path, wav_dir: Path, tracks) -> dict:'''
assert text.count(OLD_HELPER) == 1
text = text.replace(OLD_HELPER, NEW_HELPER)

OLD_CALL = '''        reconstruction = _reconstruction(work / "input.wav", target / "wav", tracks)'''
NEW_CALL = '''        reconstruction = ({} if len(tracks) == 2
                          else _reconstruction(work / "input.wav", target / "wav", tracks))'''
assert text.count(OLD_CALL) == 1
text = text.replace(OLD_CALL, NEW_CALL)

WORKER.write_text(text, encoding="utf-8")
print("stems_worker: two-stem gain balance")
