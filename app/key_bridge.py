#!/usr/bin/env python
"""Optional madmom CNN key estimation in the existing isolated environment."""
from __future__ import annotations

import os
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from chord_recognizer import decode_audio
from downbeat_bridge import DEFAULT_MADMOM_PYTHON

KEY_HELPER = r'''
import hashlib, json, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import madmom
from madmom.features.key import CNNKeyRecognitionProcessor, KEY_LABELS
from madmom.models import KEY_CNN
prediction = np.asarray(CNNKeyRecognitionProcessor()(np.load(sys.argv[1]))).reshape(-1)
if len(prediction) != 24 or not np.isfinite(prediction).all():
    raise ValueError('Invalid key network output')
rank = np.argsort(prediction)[::-1]
print(json.dumps({'label': KEY_LABELS[int(rank[0])],
    'candidates': [{'label': KEY_LABELS[int(i)], 'score': float(prediction[i])} for i in rank],
    'model_files': [{'name': __import__('pathlib').Path(p).name, 'sha256': hashlib.sha256(open(p, 'rb').read()).hexdigest()} for p in KEY_CNN],
    'runtime_version': madmom.__version__}))
'''


def run_madmom_key(audio_path: Path, madmom_python: Path = DEFAULT_MADMOM_PYTHON,
                   offset: float = 0., duration: float | None = None, timeout: int = 180) -> dict:
    from chord_studio import normalize_key_label
    if not madmom_python.is_file():
        raise FileNotFoundError(f"madmom Python not found: {madmom_python}")
    if not np.isfinite(offset) or offset < 0 or duration is not None and (not np.isfinite(duration) or duration <= 0):
        raise ValueError("Invalid key-analysis audio window")
    audio_path = audio_path.resolve()
    y = decode_audio(audio_path, sample_rate=44100, offset=offset, duration=duration)
    if not len(y) or np.max(np.abs(y)) < 1e-6:
        raise ValueError("Silent or empty audio has no supported global key")
    folder = Path(__file__).resolve().parents[1] / ".tmp/key_bridge"
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npy", dir=folder, delete=False) as handle:
        path = Path(handle.name)
        np.save(handle, y)
    try:
        completed = subprocess.run([str(madmom_python), "-c", KEY_HELPER, str(path)],
                                   text=True, encoding="utf-8", capture_output=True, timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or "madmom key model failed")
        result = json.loads(completed.stdout)
    finally:
        path.unlink(missing_ok=True)
    result["key"] = normalize_key_label(result.pop("label"))
    for item in result["candidates"]:
        item["key"] = normalize_key_label(item.pop("label"))
    result.update(source="madmom_cnn_key", score_kind="uncalibrated_classifier_score",
                  audio_offset=offset, duration=len(y) / 44100,
                  limitation="Global major/minor key; local modulation and pretraining overlap unresolved")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--madmom-python", type=Path, default=DEFAULT_MADMOM_PYTHON)
    parser.add_argument("--offset", type=float, default=0.)
    parser.add_argument("--duration", type=float)
    args = parser.parse_args()
    result = run_madmom_key(args.audio, args.madmom_python, args.offset, args.duration)
    content = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
