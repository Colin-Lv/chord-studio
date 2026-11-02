"""Frozen Transkun V2 inference, isolated from chord/separation environments."""
from pathlib import Path
import argparse
import hashlib
import json
import time


def transcribe(source, target, device='auto'):
    import numpy as np
    import soundfile as sf
    import soxr
    import torch
    import moduleconf
    import transkun
    from transkun.Data import writeMidi

    torch.set_num_threads(2)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    package = Path(transkun.__file__).parent
    weights = package / 'pretrained/2.0.pt'
    config = package / 'pretrained/2.0.conf'
    conf = moduleconf.parseFromFile(str(config))['Model']
    # This is the checkpoint shipped in the pinned upstream wheel, not an upload.
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    model = conf.module.TransKun(conf=conf.config)
    model.load_state_dict(checkpoint.get('best_state_dict', checkpoint.get('state_dict')), strict=True)
    model.to(device).eval()
    audio, sr = sf.read(str(source), dtype='float32', always_2d=True)
    duration = len(audio) / sr
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('音频为空或包含无效采样')
    if sr != model.fs:
        audio = soxr.resample(audio, sr, model.fs)
    started = time.perf_counter()
    with torch.inference_mode():
        events = model.transcribe(torch.from_numpy(audio).to(device),
                                  stepInSecond=8, segmentSizeInSecond=16,
                                  discardSecondHalf=False)
    # Retain notes/velocities/pedals. Only clip model context beyond the input.
    kept, clipped = [], 0
    for event in events:
        start, end = max(0., float(event.start)), min(duration, float(event.end))
        if end <= start:
            clipped += 1
            continue
        event.start, event.end = start, end
        kept.append(event)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    writeMidi(kept, resolution=6720).write(str(target))
    record = dict(model='Transkun V2 No Pedal Extension', package_version='2.0.1',
                  upstream='https://github.com/Yujia-Yan/Transkun',
                  weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                  config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                  device=device, seconds=time.perf_counter()-started, duration=duration,
                  notes=sum(e.pitch > 0 for e in kept), pedal_intervals=sum(e.pitch < 0 for e in kept),
                  discarded_outside_audio=clipped, segment_seconds=16, hop_seconds=8,
                  offset_semantics='key release; pedal stored separately as CC')
    target.with_suffix('.inference.json').write_text(json.dumps(record, indent=2), encoding='utf8')
    print(json.dumps(record), flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    transcribe(args.input, args.output, args.device)
