"""Run the local ISMIR2019 five-network ensemble without reference inputs.

The original full-song CQT, CNN/biLSTM and HMM are retained. In particular,
there is no chunking of the instance-normalized CNN or oracle beat input.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from chord_recognizer import decode_audio, find_ffmpeg
from harmonic_note_features import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = ROOT / "tools/models/large-vocabulary-chord-recognition-20260809081849/ISMIR2019-Large-Vocabulary-Chord-Recognition-master"
SAMPLE_RATE = 22050
HOP_LENGTH = 512


def signature(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def write_lab_atomic(path, rows):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        for start, end, label in rows:
            if not np.isfinite([start, end]).all() or start < 0 or end <= start:
                raise ValueError("Recognizer emitted an invalid interval")
            stream.write(f"{float(start)!r}\t{float(end)!r}\t{label}\n")
    temporary.replace(path)


class LargeVocabularyRecognizer:
    def __init__(self, repository=DEFAULT_REPOSITORY, threads=2):
        import librosa
        import torch
        self.repository = Path(repository).resolve()
        sys.path.insert(0, str(self.repository))
        previous_directory = Path.cwd()
        try:
            # The vendor network imports its published training-ID list using
            # a relative path. No annotation is read or used for inference.
            os.chdir(self.repository)
            from chordnet_ismir_naive import ChordNet
            from extractors.xhmm_ismir import XHMMDecoder
        finally:
            os.chdir(previous_directory)
        torch.set_num_threads(threads)
        self.torch, self.librosa = torch, librosa
        paths = [self.repository / "cache_data" /
                 f"joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s{i}.best.sdict" for i in range(5)]
        self.networks = []
        for path in paths:
            # These are trusted, locally installed original PyTorch checkpoints.
            state = torch.load(path, map_location="cpu", weights_only=False)
            network = ChordNet(None)
            network.use_gpu = False
            network.load_state_dict(state["net"], strict=True)
            network.cpu().eval()
            self.networks.append(network)
            del state
        template = self.repository / "data/submission_chord_list.txt"
        self.decoder = XHMMDecoder(template_file=str(template))
        self.identity = dict(
            schema="ismir2019_full_song_five_model_audio_only_v1",
            models=[signature(path) for path in paths], template=signature(template),
            implementation=signature(__file__),
            vendor_implementations=[signature(self.repository / path) for path in
                ("chordnet_ismir_naive.py", "complex_chord.py", "extractors/xhmm_ismir.py", "mir/nn/train.py")],
            audio_decoder=signature(find_ffmpeg()),
            audio_decoder_implementation=signature(Path(__file__).with_name("chord_recognizer.py")),
            versions=dict(python=sys.version, numpy=np.__version__, librosa=librosa.__version__, torch=torch.__version__),
            configuration=dict(sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH, bins_per_octave=36,
                n_bins=288, fmin_note="F#0", tuning=None, cqt="librosa.hybrid_cqt",
                full_song_instance_normalization=True, device="cpu", threads=threads,
                ensemble="arithmetic mean of 5 original probability outputs",
                hmm_transition_penalty=30., use_bass=True, use_seventh=True, use_extended=True,
                use_beats=False, use_downbeats=False, layer_decode=False,
                time_axis="native frame centers, interval end may exceed audio by <= 1 hop"),
            reference_inputs=False, fitting=False,
        )

    def predict(self, audio, duration=None):
        started = time.perf_counter()
        samples = decode_audio(Path(audio), SAMPLE_RATE, duration=duration)
        cqt = np.abs(self.librosa.hybrid_cqt(samples, sr=SAMPLE_RATE,
            bins_per_octave=36, fmin=self.librosa.note_to_hz("F#0"), n_bins=288,
            tuning=None, hop_length=HOP_LENGTH)).T.astype(np.float32)
        if not len(cqt) or not np.isfinite(cqt).all():
            raise ValueError("CQT must contain finite full-song frames")
        tensor = self.torch.from_numpy(cqt)
        with self.torch.inference_mode():
            probabilities = [network.inference(tensor) for network in self.networks]
        averaged = [np.mean([prediction[i] for prediction in probabilities], axis=0)
                    for i in range(6)]
        if any(not np.isfinite(p).all() for p in averaged):
            raise ValueError("Nonfinite model output")
        entry = SimpleNamespace(prop=SimpleNamespace(sr=SAMPLE_RATE, hop_length=HOP_LENGTH))
        rows = self.decoder.decode_to_chordlab(entry, averaged, False,
                                               use_beats=False, use_downbeats=False)
        metadata = dict(decoded_samples=len(samples), audio_seconds=len(samples) / SAMPLE_RATE,
                        cqt_frames=len(cqt), intervals=len(rows),
                        elapsed_seconds=time.perf_counter() - started,
                        requested_duration_seconds=duration)
        return rows, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="CSV with exactly id,audio columns")
    source.add_argument("--audio", type=Path)
    parser.add_argument("--id", default="audio")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--duration", type=float, help="Single-audio smoke extraction only; never for a manifest")
    args = parser.parse_args()
    if args.threads < 1 or (args.duration is not None and (args.manifest or not np.isfinite(args.duration) or args.duration <= 0)):
        parser.error("threads must be positive; duration is allowed only for a single audio smoke run")
    if args.manifest:
        with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if set(reader.fieldnames or []) != {"id", "audio"}:
                parser.error("Audio-only manifest must have exactly id,audio columns")
            rows = list(reader)
    else:
        rows = [dict(id=args.id, audio=str(args.audio.resolve()))]
    if not rows or (args.expected_count is not None and len(rows) != args.expected_count):
        parser.error("Unexpected audio count")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", sid) or sid in {".", ".."} for sid in ids):
        parser.error("Song IDs must be unique safe filenames")
    if any(not Path(row["audio"]).is_absolute() or not Path(row["audio"]).is_file() for row in rows):
        parser.error("All audio paths must name existing absolute files")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lab_dir = args.output_dir / "labs"
    lab_dir.mkdir(exist_ok=True)
    model = LargeVocabularyRecognizer(args.repository, args.threads)
    report = dict(status="running", expected_count=len(rows), recognizer=model.identity,
                  manifest=signature(args.manifest) if args.manifest else None, songs=[], failures=[])
    write_json(args.output_dir / "report.json", report)
    for row in rows:
        sid, audio = row["id"], Path(row["audio"])
        output = lab_dir / (sid + ".lab")
        sidecar = output.with_suffix(".provenance.json")
        source_identity = signature(audio)
        try:
            if output.exists() or sidecar.exists():
                if not output.is_file() or not sidecar.is_file():
                    raise ValueError("Incomplete output/sidecar pair from an interrupted run; preserved for inspection. Use a new output directory for this item")
                previous = json.loads(sidecar.read_text(encoding="utf-8"))
                if (previous["source_audio"] != source_identity or previous["recognizer"] != model.identity
                        or previous["requested_duration_seconds"] != args.duration
                        or previous["output"] != signature(output)):
                    raise ValueError("Existing output provenance differs; refusing overwrite or stale reuse")
                metadata = previous
                status = "reused_verified_output"
            else:
                prediction, metadata = model.predict(audio, args.duration)
                if signature(audio) != source_identity:
                    raise RuntimeError("Audio content changed during recognition; refusing to publish predictions")
                write_lab_atomic(output, prediction)
                metadata.update(source_audio=source_identity, recognizer=model.identity, output=signature(output))
                write_json(sidecar, metadata)
                status = "complete"
            report["songs"].append(dict(id=sid, status=status, output=str(output.resolve()),
                                        audio_seconds=metadata["audio_seconds"], intervals=metadata["intervals"],
                                        elapsed_seconds=metadata["elapsed_seconds"]))
            print(json.dumps(report["songs"][-1], ensure_ascii=False), flush=True)
        except Exception as exc:
            failure = dict(id=sid, error=f"{type(exc).__name__}: {exc}")
            report["failures"].append(failure)
            print(json.dumps(failure, ensure_ascii=False), flush=True)
        gc.collect()
        write_json(args.output_dir / "report.json", report)
    report["status"] = "complete" if not report["failures"] else "incomplete"
    write_json(args.output_dir / "report.json", report)
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
