"""Official NNLS Chroma note evidence, independent of reference annotations.

This bridge runs Matthias Mauch/QMUL's installed Vamp NNLS Chroma plugin.
It does not implement or fit another note dictionary. Frame timestamps are
read as integer PCM sample positions; source audio is decoded by our ffmpeg
path first, avoiding a second MP3 decoder's delay convention.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import NamedTuple

import numpy as np
import soundfile as sf

from chord_recognizer import decode_audio, find_ffmpeg

SAMPLE_RATE = 44100
BLOCK_SIZE = 16384
HOP_LENGTH = 2048
MIDI_NOTES = np.arange(21, 105, dtype=np.int16)
FEATURE_SCHEMA = "official_nnls_root_relative_current_interval_v1"
FEATURE_DIMENSION = 97
SUPPORT_THRESHOLD = 0.10
FEATURE_NAMES = tuple(
    f"{group}_relative_pc_{pc}" for group in
    ("mean", "q25", "q75", "support", "low_mean", "mid_mean", "high_mean", "longest_support")
    for pc in range(12)
) + ("base_is_minor",)
DEFAULT_SONIC = Path(__file__).resolve().parents[1] / "tools/sonic-annotator-1.7-win64/sonic-annotator-win64/sonic-annotator.exe"
TRANSFORM_ID = "vamp:nnls-chroma:nnls-chroma:semitonespectrum"
TRANSFORM = '''@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix vamp: <http://purl.org/ontology/vamp/> .
@prefix : <#> .
:transform a vamp:Transform ;
 vamp:plugin <http://vamp-plugins.org/rdf/plugins/nnls-chroma#nnls-chroma> ;
 vamp:step_size "2048"^^xsd:int ;
 vamp:block_size "16384"^^xsd:int ;
 vamp:sample_rate "44100"^^xsd:float ;
 vamp:plugin_version "5" ;
 vamp:parameter_binding [ vamp:parameter [ vamp:identifier "chromanormalize" ] ; vamp:value "0"^^xsd:float ] ;
 vamp:parameter_binding [ vamp:parameter [ vamp:identifier "rollon" ] ; vamp:value "0"^^xsd:float ] ;
 vamp:parameter_binding [ vamp:parameter [ vamp:identifier "s" ] ; vamp:value "0.7"^^xsd:float ] ;
 vamp:parameter_binding [ vamp:parameter [ vamp:identifier "tuningmode" ] ; vamp:value "0"^^xsd:float ] ;
 vamp:parameter_binding [ vamp:parameter [ vamp:identifier "useNNLS" ] ; vamp:value "1"^^xsd:float ] ;
 vamp:parameter_binding [ vamp:parameter [ vamp:identifier "whitening" ] ; vamp:value "1"^^xsd:float ] ;
 vamp:output <http://vamp-plugins.org/rdf/plugins/nnls-chroma#nnls-chroma_output_semitonespectrum> .
'''


class NoteEvidence(NamedTuple):
    times: np.ndarray
    activations: np.ndarray
    metadata: dict


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plugin_path(plugin_path=None):
    if plugin_path is not None:
        candidates = [Path(plugin_path)]
    else:
        candidates = [Path(path) / "nnls-chroma.dll" for path in
                      os.environ.get("VAMP_PATH", "").split(os.pathsep) if path]
        candidates += [Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Vamp Plugins/nnls-chroma.dll"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Official NNLS Chroma DLL unavailable; provide plugin_path")


def extraction_identity(audio_path, sonic_annotator=DEFAULT_SONIC, plugin_path=None, offset=0.0, duration=None):
    source = Path(audio_path).resolve()
    sonic = Path(sonic_annotator).resolve()
    plugin = _plugin_path(plugin_path)
    ffmpeg = Path(find_ffmpeg()).resolve()
    if not np.isfinite(offset) or offset < 0 or (duration is not None and (not np.isfinite(duration) or duration <= 0)):
        raise ValueError("Audio offset/duration must describe a finite positive window")
    identity = dict(cache_schema="nnls_note_evidence_v1", source_audio=str(source),
                    source_sha256=sha256_file(source), offset_seconds=float(offset),
                    requested_duration_seconds=None if duration is None else float(duration),
                    sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE, hop_length=HOP_LENGTH,
                    right_padding_samples=BLOCK_SIZE,
                    sonic_path=str(sonic), sonic_sha256=sha256_file(sonic),
                    plugin_path=str(plugin), plugin_sha256=sha256_file(plugin),
                    decoder_path=str(ffmpeg), decoder_sha256=sha256_file(ffmpeg),
                    decoder_implementation_sha256=sha256_file(Path(__file__).with_name("chord_recognizer.py")),
                    bridge_implementation_sha256=sha256_file(__file__),
                    transform_id=TRANSFORM_ID, transform=TRANSFORM,
                    transform_sha256=hashlib.sha256(TRANSFORM.encode()).hexdigest())
    return identity


def _validate_evidence(evidence):
    times, activations, metadata = evidence
    if (times.ndim != 1 or activations.shape != (84, len(times))
            or not np.all(np.isfinite(times)) or not np.all(np.isfinite(activations))
            or np.any(activations < 0) or np.any(times < 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("NNLS evidence requires finite increasing times and nonnegative (84,T) activations")
    if len(times) > 1 and not np.allclose(np.diff(times), HOP_LENGTH / SAMPLE_RATE, rtol=0, atol=1e-10):
        raise ValueError("Unexpected NNLS frame clock")
    duration = float(metadata["duration_seconds"])
    if not np.isfinite(duration) or duration < 0 or (len(times) and times[-1] >= duration + HOP_LENGTH / SAMPLE_RATE):
        raise ValueError("NNLS frames exceed decoded audio support")
    expected_positions = np.arange(0, int(metadata["decoded_samples"]), HOP_LENGTH, dtype=np.int64)
    if times.shape != expected_positions.shape or not np.allclose(times, expected_positions / SAMPLE_RATE, rtol=0, atol=1e-10):
        raise ValueError("NNLS frame clock must cover the entire decoded audio from sample zero")


def _read_csv(text):
    samples, frames = [], []
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        if len(row) != 85:
            raise ValueError(f"Expected timestamp and 84 NNLS values, got {len(row)} columns")
        position = float(row[0])
        if not np.isfinite(position) or position != round(position):
            raise ValueError("NNLS sample timestamp must be an integer")
        samples.append(int(position))
        frames.append([float(value) for value in row[1:]])
    activations = np.asarray(frames, dtype=np.float32).reshape(-1, 84).T
    return np.asarray(samples, dtype=np.int64), activations


def extract_note_evidence(audio_path, cache_dir=None, *, sonic_annotator=DEFAULT_SONIC,
                          plugin_path=None, offset=0.0, duration=None, timeout=600,
                          original_source=None, original_offset=0.0):
    """Extract once per audio, returning times, MIDI21..104 activations, metadata.

    Times are relative to the decoded window (offset is recorded separately).
    Cached identity covers audio bytes, binaries, source implementation and the
    complete fixed transform. Invalid caches raise instead of silently fitting.
    """
    identity = extraction_identity(audio_path, sonic_annotator, plugin_path, offset, duration)
    if original_source is not None:
        if not np.isfinite(original_offset) or original_offset < 0:
            raise ValueError("Original audio offset must be finite and nonnegative")
        identity["original_source"] = str(Path(original_source).resolve())
        identity["original_source_sha256"] = sha256_file(original_source)
        identity["original_offset_seconds"] = float(original_offset)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = None
    if cache_dir is not None:
        directory = Path(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        cache = directory / (key + ".npz")
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as saved:
                evidence = NoteEvidence(saved["times"], saved["activations"], json.loads(str(saved["metadata"])))
            if evidence.metadata.get("identity") != identity:
                raise ValueError("NNLS cache identity mismatch")
            _validate_evidence(evidence)
            return evidence
    started = time.perf_counter()
    y = decode_audio(Path(audio_path), sample_rate=SAMPLE_RATE, offset=offset, duration=duration)
    decoded_duration = len(y) / SAMPLE_RATE
    with tempfile.TemporaryDirectory(prefix="nnls_notes_") as temporary:
        directory = Path(temporary)
        wav = directory / "input.wav"
        transform = directory / "notes.ttl"
        # The host's buffering adapter otherwise omits trailing frame centres.
        # ShiftData already supplies left context; append one complete block
        # to flush the right side, then retain only original audio timestamps.
        sf.write(wav, np.pad(y, (0, BLOCK_SIZE)), SAMPLE_RATE, subtype="FLOAT")
        transform.write_text(TRANSFORM, encoding="utf-8")
        env = dict(os.environ, VAMP_PATH=str(Path(identity["plugin_path"]).parent),
                   OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
        if len(y):
            result = subprocess.run([identity["sonic_path"], "-q", "-t", str(transform),
                "-w", "csv", "--csv-stdout", "--csv-omit-filename", "--csv-sample-timing",
                "--csv-digits", "9", str(wav)], capture_output=True, text=True,
                encoding="utf-8", errors="replace", env=env, timeout=timeout)
            if result.returncode:
                raise RuntimeError(f"NNLS Chroma exited {result.returncode}: {result.stderr[-4000:]}")
            sample_positions, activations = _read_csv(result.stdout)
            keep = (sample_positions >= 0) & (sample_positions < len(y))
            sample_positions, activations = sample_positions[keep], activations[:, keep]
        else:
            sample_positions, activations = np.empty(0, dtype=np.int64), np.empty((84, 0), dtype=np.float32)
    metadata = dict(identity=identity, duration_seconds=decoded_duration,
                    decoded_samples=len(y), sample_rate=SAMPLE_RATE, midi_min=21, midi_max=104,
                    frame_timestamp_kind="sonic_annotator_pcm_sample_centres",
                    frame_hop_samples=HOP_LENGTH, analysis_window_samples=BLOCK_SIZE,
                    analysis_window_seconds=BLOCK_SIZE / SAMPLE_RATE,
                    right_padding_samples=BLOCK_SIZE, padding_frames_retained=False,
                    first_frame_sample=int(sample_positions[0]) if len(sample_positions) else None,
                    elapsed_seconds=time.perf_counter() - started,
                    normalization="raw official NNLS nonnegative note activation",
                    reference_inputs=False, global_tuning="estimated from the decoded audio window")
    evidence = NoteEvidence(sample_positions.astype(np.float64) / SAMPLE_RATE, activations, metadata)
    _validate_evidence(evidence)
    if sha256_file(audio_path) != identity["source_sha256"]:
        raise RuntimeError("Audio changed during NNLS extraction")
    if cache is not None:
        # Unique temporary name allows separate workers to finish safely.
        with tempfile.NamedTemporaryFile(dir=cache.parent, suffix=".npz", delete=False) as pending:
            pending_path = Path(pending.name)
        try:
            np.savez_compressed(pending_path, times=evidence.times, activations=evidence.activations,
                                metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
            pending_path.replace(cache)
        finally:
            pending_path.unlink(missing_ok=True)
    return evidence


def _longest_support(support):
    longest = np.zeros(12, dtype=np.float64)
    current = np.zeros(12, dtype=np.float64)
    for frame in support.T:
        current = np.where(frame, current + 1, 0)
        longest = np.maximum(longest, current)
    return longest / max(1, support.shape[1])


def features_for_intervals(evidence, rows):
    """97 current-interval features, rotated to each upstream chord's root.

    Groups: mean/q25/q75/support; low(A0..B2), mid(C3..B4), high(C5..G#7)
    means; longest support; base minor flag. Spectral mass is L1-normalized
    jointly across all notes in each frame, so register strengths stay honest.
    All quantiles include silence. N/unsupported qualities return zero rows.
    Sub-hop intervals use only the closest frame centre to their midpoint.
    """
    from chord_extension_model import chord_attributes

    _validate_evidence(evidence)
    times, notes, metadata = evidence
    output = np.zeros((len(rows), FEATURE_DIMENSION), dtype=np.float32)
    normalized = notes / np.maximum(notes.sum(axis=0, keepdims=True), 1e-12)
    registers = []
    for mask in (np.ones(84, dtype=bool), MIDI_NOTES < 48,
                 (MIDI_NOTES >= 48) & (MIDI_NOTES < 72), MIDI_NOTES >= 72):
        chroma = np.zeros((12, len(times)), dtype=np.float32)
        np.add.at(chroma, MIDI_NOTES[mask] % 12, normalized[mask])
        registers.append(chroma)
    for index, row in enumerate(rows):
        start, end = float(row.start), float(row.end)
        if not np.isfinite(start + end) or start < 0 or end <= start:
            raise ValueError("Intervals require finite nonnegative start and positive duration")
        if end > metadata["duration_seconds"] + 1 / SAMPLE_RATE:
            raise ValueError("Interval exceeds decoded audio duration; do not silently extrapolate")
        attributes = chord_attributes(row.chord)
        if attributes is None:
            continue
        output[index, -1] = attributes[1]
        if not len(times):
            continue
        left, right = np.searchsorted(times, [start, end], side="left")
        if left == right:
            left = int(np.argmin(np.abs(times - (start + end) / 2)))
            right = left + 1
        order = (attributes[0] + np.arange(12)) % 12
        current = registers[0][order, left:right]
        support = current >= SUPPORT_THRESHOLD
        output[index, :12] = current.mean(axis=1)
        output[index, 12:36] = np.quantile(current, [.25, .75], axis=1).reshape(-1)
        output[index, 36:48] = support.mean(axis=1)
        for register in range(3):
            output[index, 48 + 12 * register:60 + 12 * register] = registers[register + 1][order, left:right].mean(axis=1)
        output[index, 84:96] = _longest_support(support)
    return output
