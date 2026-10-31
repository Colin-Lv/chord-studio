#!/usr/bin/env python
"""Run the pinned SheetSage2 model in an isolated environment.

This is an optional experimental recognizer. A shorter --context-seconds changes
the upstream 300-second acoustic context and must not be reported as the official
benchmark setting. Original weights, code and tokenizer vocabulary stay intact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_REPOSITORY = "m-a-p/SheetSage2"
MODEL_REVISION = "4f89269db831bdc1880124164a00d4f9385cd129"
PARENT_REPOSITORY = "m-a-p/MERT-v2-FullSong"
PARENT_REVISION = "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"
DEFAULT_PYTHON = REPO_ROOT / ".tmp" / "sheetsage2_env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
DEFAULT_MODEL = REPO_ROOT / "tools" / "sheetsage2"
SAMPLE_RATE = 24000
DEFAULT_PROMPTS = ["timestamp", "downbeat_meter", "key", "chord_full"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_metadata(model_dir: Path) -> dict:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    metadata_path = model_dir / ".cache/huggingface/download/config.json.metadata"
    revision = metadata_path.read_text(encoding="utf-8").splitlines()[0] if metadata_path.exists() else None
    if revision is not None and revision != MODEL_REVISION:
        raise ValueError(f"Expected reviewed model revision {MODEL_REVISION}, found {revision}.")
    if config.get("base_model_revision") != PARENT_REVISION:
        raise ValueError("The model's parent revision differs from the reviewed checkpoint.")
    return {"repository": MODEL_REPOSITORY, "revision": revision or MODEL_REVISION,
            "parent_repository": config["base_model_name_or_path"],
            "parent_revision": config["base_model_revision"], "parent_sha256": config["base_model_sha256"],
            "tokenizer_fingerprint": config["tokenizer_fingerprint"],
            "original_context_seconds": float(config["input_audio_length"]),
            "code_sha256": {path.name: sha256(path) for path in sorted(model_dir.glob("*.py"))},
            "license": "CC-BY-NC-4.0"}


def configure_context(model, seconds: float) -> dict:
    """Change padding/window length after loading; never rebuild the tokenizer."""
    original = float(model.config.input_audio_length)
    if not math.isfinite(seconds) or seconds <= 0 or seconds > original:
        raise ValueError(f"context-seconds must be in (0, {original}].")
    fingerprint = model.tokenizer.vocab_fingerprint
    vocabulary = model.tokenizer.n_tokens
    model.config.input_audio_length = float(seconds)
    model.hparams.input_audio_length = float(seconds)
    if model.tokenizer.vocab_fingerprint != fingerprint or model.tokenizer.n_tokens != vocabulary:
        raise RuntimeError("Short-context inference must preserve the pretrained vocabulary.")
    return {"context_seconds": seconds, "original_context_seconds": original,
            "experimental_short_context": seconds != original,
            "tokenizer_fingerprint": fingerprint, "vocabulary_size": vocabulary}


def package_versions() -> dict:
    names = ("torch", "torchaudio", "transformers", "huggingface-hub", "safetensors", "numpy", "scipy", "mir_eval", "pretty_midi", "mido")
    return {name: importlib.metadata.version(name) for name in names}


def import_pinned_model_class(model_dir: Path):
    """Import the reviewed snapshot as a package without modifying upstream code.

    Transformers 4.45's dynamic copier misses newer transitive relative modules
    (notably chord_spelling_sheetsage2.py). Native package imports resolve the
    complete pinned snapshot; the model still verifies its pinned parent hashes.
    """
    package = "chord_studio_sheetsage2_pinned"
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(package, model_dir / "__init__.py",
                                                      submodule_search_locations=[str(model_dir.resolve())])
        if spec is None or spec.loader is None:
            raise ImportError("Cannot import the local SheetSage2 package")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    return importlib.import_module(package + ".modeling_sheetsage2").SheetSage2Model


def shift_lab(text: str, offset: float, point_events: bool = False) -> str:
    """Make original-recording timestamps explicit without altering raw exports."""
    lines = []
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        fields[0] = f"{float(fields[0]) + offset:.6f}"
        if not point_events:
            fields[1] = f"{float(fields[1]) + offset:.6f}"
        lines.append("\t".join(fields))
    return "\n".join(lines) + ("\n" if lines else "")


def audit_windows(windows: list[dict], duration: float, max_tokens: int) -> dict:
    """Verify accepted audio support and detect forced token-limit endings."""
    cursor = 0.0
    limited = []
    for index, window in enumerate(windows):
        start, end = float(window["accept_start"]), float(window["accept_end"])
        if not all(math.isfinite(value) for value in (start, end)) or abs(start - cursor) > 1e-5 or end <= start or end > duration + 1e-5:
            raise ValueError(f"Non-contiguous accepted window support at window {index}")
        cursor = end
        # Upstream appends EOS when the generation limit is reached. The
        # exporter may then extend the final chord to the song end; reject that
        # as a complete transcription even when its LAB appears fully covered.
        if int(window["tokens"]) >= max_tokens:
            limited.append(index)
    complete = bool(windows) and abs(cursor - duration) < 1e-5
    return {"accepted_support_complete": complete, "accepted_seconds": cursor,
            "token_limit_windows": limited, "complete": complete and not limited,
            "token_budget_scope": "per_window_including_overlap_prefix"}


def worker(args) -> int:
    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = model_metadata(args.model_dir)
    provenance["packages"] = package_versions()
    provenance["python"] = sys.version
    provenance["device"] = "cpu"
    provenance["dtype"] = "float32"
    provenance["threads"] = args.threads
    model_class = import_pinned_model_class(args.model_dir)
    if args.check:
        # Importing the class checks dependencies without constructing 632M weights.
        provenance["model_class"] = model_class.__name__
        provenance["status"] = "metadata_and_imports_ok; weights_not_loaded"
        (output / "preflight.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(provenance, ensure_ascii=False, indent=2), flush=True)
        return 0
    started = time.monotonic()
    provenance.update({"status": "loading_weights", "source_audio": str(args.audio.resolve()),
                       "original_offset_seconds": args.offset, "requested_context_seconds": args.context_seconds,
                       "requested_duration_seconds": args.duration, "prompts": args.prompts})
    (output / "run.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SheetSage2: loading pinned local adapter and verified cached parent", flush=True)
    model = model_class.from_pretrained(str(args.model_dir), local_files_only=True,
                                        torch_dtype=torch.float32, attn_implementation="sdpa").eval().to("cpu")
    provenance["model_load_seconds"] = time.monotonic() - started
    provenance.update(configure_context(model, args.context_seconds))
    model.max_output_seq_len = min(int(model.max_output_seq_len), args.max_tokens)
    provenance["max_output_tokens"] = model.max_output_seq_len
    samples = np.load(args.worker_audio, allow_pickle=False)
    actual_duration = len(samples) / SAMPLE_RATE
    provenance.update({"clip_duration_seconds": actual_duration, "original_offset_seconds": args.offset,
                       "source_audio": str(args.audio.resolve()), "prompts": args.prompts,
                       "decode_protocol": "ffmpeg_from_recording_start_then_sample_crop_24khz_v2",
                       "audio_array_sha256": sha256(args.worker_audio), "preset": "default", "overlap_seconds": 0, "lookahead_seconds": 0})
    (output / "run.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

    def progress(value):
        print("SheetSage2: " + json.dumps(value, ensure_ascii=False), flush=True)

    with torch.inference_mode():
        result = model.transcribe(samples, sampling_rate=SAMPLE_RATE, output_dir=output / "raw",
                                  prompts=args.prompts, dtype="fp32", preset="default",
                                  overlap_seconds=0, lookahead_seconds=0, progress=progress)
    labs = result.get("labs", {})
    for name in ("chord", "key", "structure", "beat", "downbeat"):
        text = labs.get(name, "")
        (output / f"{name}.lab").write_text(text, encoding="utf-8")
        (output / f"{name}_absolute.lab").write_text(shift_lab(text, args.offset, name in {"beat", "downbeat"}), encoding="utf-8")
    beat_rows = [line.split() for line in labs.get("beat", "").splitlines() if line.strip()]
    (output / "beats.json").write_text(json.dumps({"source": "sheetsage2", "beats": [
        {"time": float(row[0]), "beat": int(row[1]), "meter_numerator": int(row[2]), "meter_denominator": int(row[3])}
        for row in beat_rows], "original_offset_seconds": args.offset}, indent=2), encoding="utf-8")
    window_audit = audit_windows(result.get("windows", []), actual_duration, model.max_output_seq_len)
    provenance.update({"status": "ok" if window_audit["complete"] else "incomplete", "elapsed_seconds": time.monotonic() - started,
                       "num_events": result.get("num_events"), "windows": result.get("windows"),
                       "window_audit": window_audit,
                       "warnings": result.get("warnings", []), "abc_error": result.get("abc_error"),
                       "chord_segments": len(labs.get("chord", "").splitlines())})
    (output / "run.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: provenance[key] for key in ("status", "elapsed_seconds", "chord_segments", "experimental_short_context")}), flush=True)
    return 0 if window_audit["complete"] else 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, nargs="?", default=Path("unused_for_check.wav"))
    parser.add_argument("-o", "--output", type=Path, required=True, help="A new experiment directory.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--offset", type=float, default=0)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--full-song", action="store_true", help="Decode through the end of the recording, in context-sized windows.")
    parser.add_argument("--context-seconds", type=float, default=300,
                        help="Original is 300. Explicit 30/60 are experimental shorter acoustic contexts.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout-active-seconds", type=float, default=900,
                        help="Stop this worker after this much sampled active wall time (large host-pause gaps excluded).")
    parser.add_argument("--timeout-wall-seconds", type=float, default=1800)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--check", action="store_true", help="Check pinned config/imports without loading model weights.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-audio", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def run_worker_monitored(command, environment, output, active_timeout=900, wall_timeout=1800):
    """Monitor only this worker's process tree; never manage unrelated jobs."""
    try:
        import psutil
    except ImportError:
        return subprocess.run(command, env=environment, check=False, timeout=wall_timeout).returncode
    report = {"interval_seconds": 1, "samples": 0, "peak_tree_rss_bytes": 0,
              "peak_process_rss_bytes": 0, "peak_process_private_bytes": 0,
              "max_sampling_gap_seconds": 0., "sampled_active_seconds": 0., "max_observed_process_cpu_seconds": 0.,
              "minimum_system_available_bytes": psutil.virtual_memory().available}
    started = time.monotonic()
    previous_sample = started
    process = subprocess.Popen(command, env=environment)
    tracked = psutil.Process(process.pid)
    while process.poll() is None:
        now = time.monotonic()
        gap = now - previous_sample
        previous_sample = now
        report["max_sampling_gap_seconds"] = max(report["max_sampling_gap_seconds"], gap)
        report["sampled_active_seconds"] += min(gap, 5.)
        try:
            processes = [tracked, *tracked.children(recursive=True)]
            total = 0
            for item in processes:
                try:
                    info = item.memory_info()
                    cpu = item.cpu_times()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                total += info.rss
                report["max_observed_process_cpu_seconds"] = max(report["max_observed_process_cpu_seconds"], cpu.user + cpu.system)
                report["peak_process_rss_bytes"] = max(report["peak_process_rss_bytes"], getattr(info, "peak_wset", info.rss))
                report["peak_process_private_bytes"] = max(report["peak_process_private_bytes"], getattr(info, "peak_pagefile", getattr(info, "private", info.vms)))
            report["peak_tree_rss_bytes"] = max(report["peak_tree_rss_bytes"], total)
            report["minimum_system_available_bytes"] = min(report["minimum_system_available_bytes"], psutil.virtual_memory().available)
            report["samples"] += 1
        except psutil.NoSuchProcess:
            pass
        if now - started >= wall_timeout or report["sampled_active_seconds"] >= active_timeout:
            report["termination_reason"] = "wall_timeout" if now - started >= wall_timeout else "active_timeout"
            for owned in reversed([tracked, *tracked.children(recursive=True)]):
                try:
                    owned.terminate()
                except psutil.NoSuchProcess:
                    pass
            process.wait(timeout=10)
            break
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    report.update(exit_code=process.returncode, wall_seconds=time.monotonic() - started,
                  note="Sampled worker tree only. CPU value is the largest observed process user+system CPU time. Active wall clips monitor gaps to 5s; long host pauses remain in wall time.")
    if "termination_reason" in report:
        path = output / "run.json"
        run = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        run.update(status="interrupted", termination_reason=report["termination_reason"])
        path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    (output / "resource_monitor.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return process.returncode


def main(argv=None) -> int:
    args = parse_args(argv)
    if not 1 <= args.threads <= 4:
        raise ValueError("threads must be between 1 and 4 for this CPU bridge")
    if args.worker:
        try:
            return worker(args)
        except Exception as exc:
            if not args.check:
                report_path = args.output.resolve() / "run.json"
                provenance = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
                provenance.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
    if not args.python.is_file():
        raise FileNotFoundError(f"SheetSage2 environment missing: {args.python}; run app/setup_sheetsage2.py")
    model_metadata(args.model_dir)
    if not args.check and (not math.isfinite(args.offset) or args.offset < 0 or not math.isfinite(args.duration)
                           or args.duration <= 0 or not 0 < args.context_seconds <= 300):
        raise ValueError("Require finite offset>=0, duration>0 and 0<context-seconds<=300.")
    if args.max_tokens < 32:
        raise ValueError("max-tokens must be at least 32")
    args.output.mkdir(parents=True, exist_ok=True)
    command = [str(args.python.resolve()), str(Path(__file__).resolve()), str(args.audio.resolve()),
               "--worker", "-o", str(args.output.resolve()), "--model-dir", str(args.model_dir.resolve()),
               "--context-seconds", str(args.context_seconds), "--offset", str(args.offset),
               "--duration", str(args.duration),
               "--threads", str(args.threads), "--max-tokens", str(args.max_tokens), "--prompts", *args.prompts]
    audio_array = None
    if args.check:
        command.append("--check")
    else:
        import numpy as np
        import soundfile as sf
        from chord_recognizer import decode_audio
        if not args.audio.is_file():
            raise FileNotFoundError(args.audio)
        # Input-side FFmpeg -ss can prepend MP3 decoder warm-up samples. Decode
        # from source zero so a clip equals the same sample range of a full run.
        source_samples = decode_audio(args.audio, sample_rate=SAMPLE_RATE, offset=0,
                                      duration=None if args.full_song else args.offset + args.duration)
        start_sample = round(args.offset * SAMPLE_RATE)
        samples = source_samples[start_sample:]
        if not args.full_song:
            samples = samples[:round(args.duration * SAMPLE_RATE)]
        if not len(samples):
            raise ValueError("Requested offset is beyond the recording")
        audio_array = args.output.resolve() / "input_24k.npy"
        np.save(audio_array, samples.astype(np.float32))
        sf.write(args.output / "audio.wav", samples, SAMPLE_RATE)
        command += ["--worker-audio", str(audio_array)]
        if args.full_song:
            command += ["--duration", str(len(samples) / SAMPLE_RATE), "--full-song"]
    environment = dict(os.environ)
    environment.update({"OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads),
                        "OPENBLAS_NUM_THREADS": str(args.threads), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    return_code = (subprocess.run(command, env=environment, check=False).returncode if args.check else
                   run_worker_monitored(command, environment, args.output.resolve(),
                                        args.timeout_active_seconds, args.timeout_wall_seconds))
    if audio_array is not None:
        audio_array.unlink(missing_ok=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
