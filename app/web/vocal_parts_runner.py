"""Karaoke model on an existing vocal stem; preserve PCM gain and timing."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import time

import numpy as np
import soundfile as sf

MODEL = 'mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt'
CONFIG = 'mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956_config.yaml'


def split_complement(parent, lead):
    if parent.shape != lead.shape or parent.ndim != 2:
        raise ValueError('Model changed audio length or channel layout')
    if not np.isfinite(parent).all() or not np.isfinite(lead).all():
        raise ValueError('Non-finite audio')
    # A subtraction residual, not an independently verified backing-vocal estimate.
    return parent - lead


def main(source, output, model_dir):
    import torch
    from audio_separator.separator import Separator
    output.mkdir(parents=True, exist_ok=False)
    parent, sr = sf.read(source, dtype='float32', always_2d=True)
    if sr != 44100 or parent.shape[1] != 2 or not len(parent):
        raise ValueError('Expected nonempty stereo 44100 Hz vocal stem')
    start = time.perf_counter()
    sep = Separator(log_level=logging.ERROR, model_file_dir=str(model_dir),
                    output_dir=str(output), use_autocast=True,
                    normalization_threshold=1.0, amplification_threshold=0.0)
    sep.load_model(model_filename=MODEL)
    engine = sep.model_instance
    # Bypass per-output peak normalisation and integer WAV export. Feed normalized
    # PCM only when necessary, then restore its exact common input gain.
    scale = max(1.0, float(np.abs(parent).max()))
    mix = np.ascontiguousarray(parent.T / scale)
    with torch.inference_mode():
        estimates = engine.demix(mix, override_model_segment_size=engine._use_model_segment_override(len(parent)/sr))
    lead = np.asarray(estimates['Vocals'] if isinstance(estimates, dict) else estimates,
                      dtype=np.float32).T * scale
    backing = split_complement(parent, lead)
    sf.write(output/'lead_vocals.wav', lead, sr, subtype='FLOAT')
    sf.write(output/'backing_vocals.wav', backing, sr, subtype='FLOAT')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    meta = dict(model=MODEL, input_sha256=sha(source), weights_sha256=sha(model_dir/MODEL),
                config_sha256=sha(model_dir/CONFIG), runner_sha256=sha(__file__),
                audio_separator=importlib.metadata.version('audio-separator'), torch=torch.__version__,
                device=str(sep.torch_device), seconds=time.perf_counter()-start,
                samples=len(parent), sample_rate=sr, channels=2,
                peak_vram_mb=torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else 0,
                residual_kind='parent_minus_lead; includes backing, reverberation, leakage and error',
                sum_max_abs_error=float(np.max(np.abs(parent-(lead+backing)))))
    (output/'inference.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta),flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--outdir',type=Path,required=True)
    ap.add_argument('--model-dir',type=Path,default=Path('D:/ChordStudio/data/models/audio_separator'))
    args=ap.parse_args();main(args.input,args.outdir,args.model_dir)
