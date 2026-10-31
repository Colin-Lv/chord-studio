"""Fresh Chordino inference from a strict id/audio-only CSV, with resumable provenance.

Each source is decoded in full to 44100 Hz mono PCM32 by FFmpeg. Exactly one
Sonic Annotator process runs at a time. No annotation, key, beat or score input
is accepted. Existing successful files are verified and reused, never rerun.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import wave


TRANSFORM = "vamp:nnls-chroma:chordino:simplechord"
DEFAULT_SONIC = Path("tools/sonic-annotator-1.7-win64/sonic-annotator-win64/sonic-annotator.exe")
DEFAULT_PLUGIN = Path("C:/Program Files/Vamp Plugins/nnls-chroma.dll")


def identity(path: Path) -> dict:
    path = path.resolve()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while hashing: {path}")
    return dict(path=str(path), bytes=after.st_size, sha256=digest.hexdigest())


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(value, encoding="utf-8", newline="\n")
    pending.replace(path)


def write_json(path: Path, payload: dict) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def load_audio_manifest(path: Path) -> list[dict]:
    """Reject extra fields so even reference-path columns cannot enter this runner."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or sorted(reader.fieldnames) != ["audio", "id"]:
            raise ValueError("Audio-only manifest must contain exactly id,audio columns")
        rows = []
        for row in reader:
            identifier = row["id"].strip()
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier):
                raise ValueError(f"Unsafe recording ID: {identifier!r}")
            audio = (path.parent / row["audio"]).resolve()
            if not audio.is_file():
                raise FileNotFoundError(audio)
            rows.append(dict(id=identifier, audio=str(audio)))
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Manifest must contain nonempty unique IDs")
    if len({os.path.normcase(row["audio"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate audio paths")
    return rows


def intervals_from_points(text: str, duration: float) -> tuple[str, dict]:
    """Preserve source change times; close the last state at decoded duration.

    Leading missing coverage remains missing; it is never filled with N.
    A point exactly at EOF is a boundary, not a positive-duration prediction.
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Decoded duration must be finite and positive")
    points = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"Malformed Chordino point at line {line_number}")
        start = float(fields[0])
        label = fields[1].strip()
        if len(label) >= 2 and label[0] == label[-1] and label[0] in "\"'":
            label = label[1:-1]
        if not label or any(char.isspace() for char in label):
            raise ValueError(f"Blank/unsupported Chordino label at line {line_number}")
        if not math.isfinite(start) or start < 0 or start > duration + 1e-6:
            raise ValueError(f"Point outside decoded audio at line {line_number}")
        start = min(start, duration)
        if points and start <= points[-1][0]:
            raise ValueError("Chordino points must be strictly increasing")
        points.append((start, label))
    if not points:
        raise ValueError("Chordino returned no points")
    intervals = []
    for index, (start, label) in enumerate(points):
        end = points[index + 1][0] if index + 1 < len(points) else duration
        if end <= start:
            continue
        if intervals and intervals[-1][2] == label:
            intervals[-1] = (intervals[-1][0], end, label)
        else:
            intervals.append((start, end, label))
    if not intervals:
        raise ValueError("Chordino returned no positive-duration intervals")
    lab = "".join(f"{start:.9f} {end:.9f} {label}\n" for start, end, label in intervals)
    covered = sum(end - start for start, end, _ in intervals)
    return lab, dict(point_count=len(points), interval_count=len(intervals),
                     audio_duration=duration, first_prediction=intervals[0][0],
                     last_prediction_end=intervals[-1][1], covered_seconds=covered,
                     uncovered_leading_seconds=intervals[0][0], internal_gap_seconds=0.,
                     coverage_fraction=covered / duration,
                     full_coverage=intervals[0][0] <= 1e-6,
                     final_interval_end_policy="decoded_audio_duration",
                     source_change_times_preserved=True, synthetic_leading_N=False)


def checked_run(command: list[str], environment: dict) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env=environment)
    if result.returncode:
        raise RuntimeError(f"Process exit {result.returncode}: {result.stderr[-6000:]}")
    return result


def verify_existing(lab: Path, sidecar: Path, audio_identity: dict,
                    configuration_sha256: str) -> dict | None:
    if not lab.exists() and not sidecar.exists():
        return None
    if not lab.is_file() or not sidecar.is_file():
        raise FileExistsError(f"Incomplete existing output; refused overwrite: {lab}")
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    if (data.get("status") != "complete"
            or data.get("source_audio", {}).get("sha256") != audio_identity["sha256"]
            or data.get("configuration_sha256") != configuration_sha256
            or data.get("output_lab", {}).get("sha256") != identity(lab)["sha256"]):
        raise FileExistsError(f"Existing output identity/configuration mismatch: {lab}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--sonic-annotator", type=Path, default=DEFAULT_SONIC)
    parser.add_argument("--plugin", type=Path, default=DEFAULT_PLUGIN)
    parser.add_argument("--ffmpeg", type=Path)
    args = parser.parse_args()
    rows = load_audio_manifest(args.manifest.resolve())
    if len(rows) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} recordings, found {len(rows)}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".run.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    os.close(descriptor)
    started = time.perf_counter()
    try:
        sonic = args.sonic_annotator.resolve()
        plugin = args.plugin.resolve()
        ffmpeg = args.ffmpeg
        if ffmpeg is None:
            found = shutil.which("ffmpeg")
            if not found:
                import imageio_ffmpeg
                found = imageio_ffmpeg.get_ffmpeg_exe()
            ffmpeg = Path(found)
        ffmpeg = ffmpeg.resolve()
        environment = dict(os.environ, VAMP_PATH=str(plugin.parent),
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        default_transform = checked_run([str(sonic), "-s", TRANSFORM], environment).stdout
        sonic_version = checked_run([str(sonic), "--version"], environment)
        ffmpeg_version = checked_run([str(ffmpeg), "-version"], environment).stdout.splitlines()[0]
        configuration = dict(transform=TRANSFORM, parameters="unchanged plugin defaults",
                             default_transform=default_transform, sonic=identity(sonic),
                             plugin=identity(plugin), ffmpeg=identity(ffmpeg),
                             sonic_version=(sonic_version.stdout or sonic_version.stderr).strip(),
                             ffmpeg_version=ffmpeg_version, code=identity(Path(__file__)),
                             sample_rate=44100, channels=1, codec="pcm_s32le", offset=0.,
                             duration_limit=None, concurrent_sonic_processes=1, threads=1,
                             vamp_path=str(plugin.parent))
        configuration_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
        atomic_text(output / "default_transform.n3", default_transform)
        report = dict(schema="fresh_audio_only_chordino_batch_v1", status="running",
                      created_at_utc=datetime.now(timezone.utc).isoformat(),
                      source_manifest=identity(args.manifest), manifest_fields_used=["id", "audio"],
                      references_opened=False, scoring_performed=False,
                      configuration=configuration, configuration_sha256=configuration_hash,
                      requested_count=len(rows), songs=[])
        write_json(output / "report.json", report)
        for index, row in enumerate(rows, 1):
            entry = dict(id=row["id"], audio=row["audio"], status="running")
            report["songs"].append(entry)
            write_json(output / "report.json", report)
            item_start = time.perf_counter()
            try:
                source_identity = identity(Path(row["audio"]))
                lab = output / "labs" / (row["id"] + ".lab")
                sidecar = lab.with_suffix(".provenance.json")
                previous = verify_existing(lab, sidecar, source_identity, configuration_hash)
                if previous is not None:
                    entry.update(previous, reused_verified_success=True)
                    print(f"[{index}/{len(rows)}] reused {row['id']}", flush=True)
                    continue
                print(f"[{index}/{len(rows)}] generating {row['id']}", flush=True)
                with tempfile.TemporaryDirectory(prefix=".chordino_audio_", dir=output) as temporary:
                    decoded = Path(temporary) / "complete_44100_mono.wav"
                    decode_command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin",
                                      "-threads", "1", "-filter_threads", "1", "-i", row["audio"],
                                      "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "44100",
                                      "-c:a", "pcm_s32le", "-threads", "1", str(decoded)]
                    checked_run(decode_command, environment)
                    with wave.open(str(decoded), "rb") as handle:
                        if (handle.getnchannels(), handle.getframerate(), handle.getsampwidth()) != (1, 44100, 4):
                            raise ValueError("Decoded WAV must be mono 44100 Hz PCM32")
                        samples = handle.getnframes()
                    if not samples:
                        raise ValueError("Empty decoded source")
                    decoded_identity = identity(decoded)
                    command = [str(sonic), "-q", "-d", TRANSFORM,
                               "-w", "lab", "--lab-stdout", str(decoded)]
                    result = checked_run(command, environment)
                    lab_text, coverage = intervals_from_points(result.stdout, samples / 44100)
                    if identity(Path(row["audio"]))["sha256"] != source_identity["sha256"]:
                        raise RuntimeError("Source audio changed during inference")
                    raw_points = output / "evidence" / (row["id"] + ".points.lab")
                    atomic_text(raw_points, result.stdout)
                    atomic_text(output / "evidence" / (row["id"] + ".stderr.txt"), result.stderr)
                    atomic_text(lab, lab_text)
                    entry.update(status="complete", source_audio=source_identity,
                                 configuration_sha256=configuration_hash,
                                 decoded_audio=decoded_identity, decoded_sample_count=samples,
                                 output_lab=identity(lab), original_point_output=identity(raw_points),
                                 coverage=coverage, audio_offset=0., references_opened=False,
                                 scoring_performed=False, reused_verified_success=False,
                                 elapsed_seconds=time.perf_counter() - item_start,
                                 temporary_decoded_audio_retained=False)
                    write_json(sidecar, entry)
            except Exception as exc:
                entry.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                             elapsed_seconds=time.perf_counter() - item_start)
                print(f"FAILED {row['id']}: {entry['error']}", flush=True)
            finally:
                write_json(output / "report.json", report)
        complete = [row for row in report["songs"] if row["status"] == "complete"]
        failed = [row for row in report["songs"] if row["status"] == "failed"]
        report.update(status="complete" if not failed else "completed_with_failures",
                      completed_count=len(complete), failed_count=len(failed),
                      full_coverage_count=sum(row["coverage"]["full_coverage"] for row in complete),
                      reused_count=sum(row["reused_verified_success"] for row in complete),
                      failures=[dict(id=row["id"], error=row["error"]) for row in failed],
                      elapsed_seconds=time.perf_counter() - started,
                      completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "report.json", report)
        print(f"Complete {len(complete)}/{len(rows)}; failures {len(failed)}; no scoring.", flush=True)
        return int(bool(failed))
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
