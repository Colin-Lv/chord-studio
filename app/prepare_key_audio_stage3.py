"""Prepare the historical mono float32 PCM input for the stage-three replay.

Run as a script from the repository root; existing outputs are never replaced.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import chord_recognizer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    ffmpeg = chord_recognizer.find_ffmpeg()
    rows = []
    for sid in ('191347', '853834'):
        audio = Path('D:/ChordStudio/_archive/2026-10-30/benchmarks/external_key_data/audio') / (sid + '.mp3')
        provenance = json.loads((Path('benchmarks/reports/harmonic_key_v2_20261006/acoustics') / sid / 'raw.lab.provenance.json').read_text(encoding='utf-8'))
        audio_sha = sha(audio)
        if audio_sha != provenance['audio']['sha256']:
            raise ValueError('Audio source mismatch: ' + sid)
        pcm = chord_recognizer.decode_audio(audio, sample_rate=44100).astype(np.float32)
        target = out / (sid + '.npy')
        np.save(target, pcm)
        rows.append(dict(id=sid, audio_sha256=audio_sha, pcm_file_sha256=sha(target),
                         samples=len(pcm), duration=len(pcm)/44100))
    manifest = dict(rows=rows, ffmpeg=str(ffmpeg), ffmpeg_sha256=sha(ffmpeg),
                    decoder_sha256=sha(chord_recognizer.__file__))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    main(parser.parse_args().out)
