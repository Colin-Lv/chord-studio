"""Opt-in six-head posterior export around the unchanged v1 audio-only bridge.

Example: python app/generate_audio_only_largevocab_labs_v2.py --audio song.wav
  --output-dir results --export-posteriors
Only trusted locally installed original checkpoints are loaded. This entry point
never reads reference labels. Existing v1 output directories are not v2 caches.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from generate_audio_only_largevocab_labs import (
    DEFAULT_REPOSITORY, LargeVocabularyRecognizer, signature, write_lab_atomic,
)
from largevocab_posteriors_v2 import (
    HEAD_NAMES, HOP_LENGTH, SAMPLE_RATE, SCHEMA,
    load_posteriors, make_evidence, save_posteriors, validate_probabilities,
)

BRIDGE_SCHEMA = "ismir2019_audio_only_bridge_v2"


class PosteriorDecoderTap:
    """Capture without replacing, normalizing, casting, or editing HMM inputs."""
    def __init__(self, decoder):
        self.decoder = decoder
        self.probabilities = None
        self.calls = 0

    def decode_to_chordlab(self, entry, probabilities, *args, **kwargs):
        self.calls += 1
        if self.calls != 1:
            raise ValueError("Expected one complete-song HMM call")
        if entry.prop.sr != SAMPLE_RATE or entry.prop.hop_length != HOP_LENGTH:
            raise ValueError("Unexpected original recognizer sample clock")
        if len(probabilities) != len(HEAD_NAMES):
            raise ValueError("Expected six original probability heads")
        heads = dict(zip(HEAD_NAMES, probabilities))
        validate_probabilities(heads)
        self.probabilities = {name: values.copy() for name, values in heads.items()}
        return self.decoder.decode_to_chordlab(entry, probabilities, *args, **kwargs)


class LargeVocabularyRecognizerV2(LargeVocabularyRecognizer):
    # Inherited predict remains exactly the original v1 acoustic/HMM path.
    def posterior_identity(self, source_identity, duration=None):
        return dict(schema=SCHEMA, source_audio=source_identity,
            recognizer=self.identity, requested_duration_seconds=duration, audio_offset=0.,
            bridge_implementation=signature(__file__),
            posterior_implementation=signature(Path(__file__).with_name("largevocab_posteriors_v2.py")))

    def predict_with_posteriors(self, audio, duration=None, *, identity):
        if getattr(self, "_capturing_posteriors", False):
            raise RuntimeError("Concurrent or recursive posterior extraction is unsupported")
        if Path(audio).is_file() and Path(audio).stat().st_size == 0:
            raise ValueError("Cannot export empty audio")
        original_decoder = self.decoder
        tap = PosteriorDecoderTap(original_decoder)
        self._capturing_posteriors = True
        self.decoder = tap
        try:
            rows, statistics = super().predict(audio, duration)
            if tap.calls != 1 or tap.probabilities is None:
                raise ValueError("Original recognizer did not emit complete-song probabilities")
            evidence = make_evidence(tap.probabilities, identity,
                                     decoded_samples=statistics["decoded_samples"])
            if statistics["cqt_frames"] != evidence.metadata["frame_count"]:
                raise ValueError("Reported original frame count differs from posterior evidence")
            return rows, statistics, evidence
        finally:
            self.decoder = original_decoder
            self._capturing_posteriors = False


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run_item(model, sid, audio, output_dir, *, duration=None, export_posteriors=False):
    """An item is complete only when LAB, optional NPZ and final sidecar agree."""
    audio, output_dir = Path(audio), Path(output_dir)
    output = output_dir / "labs" / f"{sid}.lab"
    sidecar = output.with_suffix(".provenance.json")
    posterior_path = output_dir / "posteriors" / f"{sid}.npz"
    source_identity = signature(audio)
    identity = model.posterior_identity(source_identity, duration)
    expected = [output, sidecar] + ([posterior_path] if export_posteriors else [])
    # An unexpected posterior also prevents silently reusing an export-on run
    # as a different extraction contract.
    if any(path.exists() for path in [output, sidecar, posterior_path]):
        if not all(path.is_file() for path in expected):
            raise ValueError("Incomplete v2 cache; preserve it for inspection and use a new output directory")
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
        if (previous.get("schema") != BRIDGE_SCHEMA or previous.get("identity") != identity
                or previous.get("export_posteriors") is not export_posteriors
                or previous.get("output") != signature(output)):
            raise ValueError("Stale or v1 cache identity; refusing overwrite or reuse")
        if export_posteriors:
            if previous.get("posteriors") != signature(posterior_path):
                raise ValueError("Posterior cache SHA differs from its completed sidecar")
            load_posteriors(posterior_path, expected_identity=identity)
        elif posterior_path.exists():
            raise ValueError("Unexpected posterior cache for export-disabled invocation")
        if signature(audio) != source_identity:
            raise RuntimeError("Audio content changed during cache validation")
        return dict(id=sid, status="reused_verified_output", output=str(output.resolve()),
            posteriors=str(posterior_path.resolve()) if export_posteriors else None,
            audio_seconds=previous["audio_seconds"], intervals=previous["intervals"])
    if export_posteriors:
        rows, statistics, evidence = model.predict_with_posteriors(audio, duration, identity=identity)
    else:
        rows, statistics = model.predict(audio, duration)
        evidence = None
    if statistics.get("decoded_samples", 0) <= 0 or not rows:
        raise ValueError("Cannot publish empty audio or an empty recognition")
    if signature(audio) != source_identity:
        raise RuntimeError("Audio content changed during recognition; refusing publication")
    output.parent.mkdir(parents=True, exist_ok=True)
    if export_posteriors:
        posterior_path.parent.mkdir(parents=True, exist_ok=True)
    # Stage every artifact first. Publish the sidecar last as completion marker.
    # A process interruption leaves an explicit incomplete bundle, never a
    # reusable completed prediction. The outer CLI owns the directory lock.
    with tempfile.TemporaryDirectory(prefix=".v2-stage-", dir=output_dir) as temporary:
        stage = Path(temporary)
        staged_lab = stage / "output.lab"
        write_lab_atomic(staged_lab, rows)
        if export_posteriors:
            save_posteriors(stage / "posteriors.npz", evidence)
        staged_lab.replace(output)
        if export_posteriors:
            (stage / "posteriors.npz").replace(posterior_path)
        metadata = dict(statistics, schema=BRIDGE_SCHEMA, identity=identity,
            source_audio=source_identity, recognizer=model.identity,
            output=signature(output), export_posteriors=export_posteriors,
            posteriors=signature(posterior_path) if export_posteriors else None,
            requested_duration_seconds=duration, reference_inputs=False,
            completion="LAB and optional NPZ published; this sidecar is written last")
        write_json_atomic(sidecar, metadata)
    return dict(id=sid, status="complete", output=str(output.resolve()),
        posteriors=str(posterior_path.resolve()) if export_posteriors else None,
        audio_seconds=statistics["audio_seconds"], intervals=statistics["intervals"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="CSV with exactly id,audio columns")
    source.add_argument("--audio", type=Path)
    parser.add_argument("--id", default="audio")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--duration", type=float, help="Single-audio smoke extraction only")
    parser.add_argument("--export-posteriors", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1 or (args.duration is not None and (args.manifest or not np.isfinite(args.duration) or args.duration <= 0)):
        parser.error("Positive threads required; finite positive duration is allowed only with single audio")
    if args.manifest:
        with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if len(reader.fieldnames or []) != 2 or set(reader.fieldnames) != {"id", "audio"}:
                parser.error("Audio-only manifest must have exactly id,audio columns")
            rows = list(reader)
            if any(set(row) != {"id", "audio"} or None in row.values() for row in rows):
                parser.error("Malformed audio-only manifest row")
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
    lock = args.output_dir / ".v2-extraction.lock"
    try:
        stream = lock.open("x", encoding="utf-8")
    except FileExistsError:
        parser.error("Output directory is locked; inspect any interrupted process before reusing it")
    report = dict(schema=BRIDGE_SCHEMA, status="running", expected_count=len(rows),
        export_posteriors=args.export_posteriors,
        manifest=signature(args.manifest) if args.manifest else None, songs=[], failures=[])
    try:
        with stream:
            stream.write(str(os.getpid()))
        write_json_atomic(args.output_dir / "report.json", report)
        try:
            model = LargeVocabularyRecognizerV2(args.repository, args.threads)
        except Exception as exc:
            report["failures"].append(dict(stage="model_initialization", error=f"{type(exc).__name__}: {exc}"))
            report["status"] = "incomplete"
            write_json_atomic(args.output_dir / "report.json", report)
            return 1
        report["recognizer"] = model.identity
        for row in rows:
            try:
                result = run_item(model, row["id"], row["audio"], args.output_dir,
                    duration=args.duration, export_posteriors=args.export_posteriors)
                report["songs"].append(result)
            except Exception as exc:
                result = dict(id=row["id"], error=f"{type(exc).__name__}: {exc}")
                report["failures"].append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            gc.collect()
            write_json_atomic(args.output_dir / "report.json", report)
        report["completed_count"] = len(report["songs"])
        report["status"] = "complete" if not report["failures"] and len(report["songs"]) == len(rows) else "incomplete"
        write_json_atomic(args.output_dir / "report.json", report)
        return 0 if report["status"] == "complete" else 1
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
