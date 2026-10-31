"""Optional All-In-One neural structure expert, isolated from chord inference.

Use the optional environment installed by setup_structure_env.ps1. Inference
uses actual four-stem separation and the released Harmonix checkpoints. The
default one-fold CPU mode trades ensemble accuracy for runtime. Neural roles
are model suggestions; manual sections and existing chord annotations survive.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from song_structure import apply_manual_sections, detect_harmonic_loops, grid_quality, valid_bars


HARMONIX_LABELS = ["start", "end", "intro", "outro", "break", "bridge", "inst", "solo", "verse", "chorus"]
ROLE_NAMES = {"start": "开头", "end": "结尾", "intro": "前奏", "outro": "尾奏", "break": "间歇", "bridge": "桥段", "inst": "器乐段", "solo": "独奏", "verse": "主歌", "chorus": "副歌"}


def shift_expert_timebase(external: dict[str, Any], offset_seconds: float, duration: float) -> dict[str, Any]:
    """Translate a independently verified decoder offset, never a fitted beat offset.

    Cached activations keep their original frame index. The separate activation
    offset lets role/boundary evidence be read correctly after translating times.
    """
    if not np.isfinite([offset_seconds, duration]).all() or duration <= 0:
        raise ValueError("Invalid expert timebase translation")
    result = copy.deepcopy(external)
    if len(result.get("beats", [])) != len(result.get("beat_positions", [])):
        raise ValueError("Expert beat arrays must be equally sized")
    pairs = [(float(time) + offset_seconds, position) for time, position in zip(result.get("beats", []), result.get("beat_positions", [])) if 0 <= float(time) + offset_seconds <= duration]
    result["beats"] = [time for time, _ in pairs]
    result["beat_positions"] = [position for _, position in pairs]
    result["downbeats"] = [float(time) + offset_seconds for time in result.get("downbeats", []) if 0 <= float(time) + offset_seconds <= duration]
    result["segments"] = [{**segment, "start": max(0.0, float(segment["start"]) + offset_seconds), "end": min(duration, float(segment["end"]) + offset_seconds)} for segment in result.get("segments", []) if min(duration, float(segment["end"]) + offset_seconds) > max(0.0, float(segment["start"]) + offset_seconds)]
    result["activation_time_offset_seconds"] = float(result.get("activation_time_offset_seconds", 0)) + offset_seconds
    result["timebase_transform"] = {"kind": "explicit_decoder_time_offset", "verification": "caller_supplies_independent_alignment_evidence", "offset_seconds": offset_seconds, "duration": duration, "original_cached_result_unchanged": True}
    return result


def compare_beat_grids(bars: list[dict[str, Any]], external: dict[str, Any], tolerance: float = 0.07) -> dict[str, Any]:
    """Explain disagreement between two hypotheses without treating either as truth."""
    existing = sorted({float(beat["start"]) for bar in bars for beat in bar.get("beats", [])})
    proposed = np.asarray(external.get("beats", []), dtype=float)
    if not existing or not len(proposed):
        return {"available": False}
    if not np.isfinite(proposed).all() or np.any(np.diff(proposed) <= 0):
        raise ValueError("Expert beat times must be finite and strictly increasing")

    def nearest_distances(values, anchors):
        anchors = np.asarray(sorted(anchors), dtype=float)
        values = np.asarray(values, dtype=float)
        if not len(anchors) or not len(values):
            return np.asarray([])
        indexes = np.searchsorted(anchors, values)
        return np.minimum(np.abs(values - anchors[np.clip(indexes, 0, len(anchors) - 1)]), np.abs(values - anchors[np.clip(indexes - 1, 0, len(anchors) - 1)]))

    proposed_errors = nearest_distances(proposed, existing)
    existing_errors = nearest_distances(existing, proposed)
    # Retime output includes uncovered regions and partial bars. Their start
    # is not an observed downbeat; compare only markers explicitly numbered 1.
    existing_downbeats = [float(beat["start"]) for bar in bars for beat in bar.get("beats", []) if int(beat.get("beat", 0)) == 1]
    proposed_downbeats = external.get("downbeats", [])
    downbeat_errors = nearest_distances(proposed_downbeats, existing_downbeats)
    reverse_downbeat_errors = nearest_distances(existing_downbeats, proposed_downbeats)
    downbeat_agreement = float(0.5 * (np.mean(downbeat_errors <= tolerance) + np.mean(reverse_downbeat_errors <= tolerance))) if len(downbeat_errors) and len(reverse_downbeat_errors) else None
    agreement = float(0.5 * (np.mean(proposed_errors <= tolerance) + np.mean(existing_errors <= tolerance)))
    return {"available": True, "reference_is_ground_truth": False, "tolerance_seconds": tolerance, "beat_agreement": round(agreement, 4), "median_nearest_beat_difference_ms": round(float(np.median(proposed_errors)) * 1000, 2), "downbeat_agreement": round(downbeat_agreement, 4) if downbeat_agreement is not None else None, "existing_beat_count": len(existing), "proposed_beat_count": len(proposed), "pulse_density_ratio": round(len(proposed) / len(existing), 4), "requires_review": agreement < 0.8 or downbeat_agreement is not None and downbeat_agreement < 0.7}


def import_expert_structure(payload: dict[str, Any], external: dict[str, Any], activations: dict[str, np.ndarray] | None = None, model: str = "harmonix-fold0", role_threshold: float = 0.55) -> dict[str, Any]:
    """Map expert time segments to the existing grid without changing any chords."""
    bars = valid_bars(payload)
    if not bars:
        raise ValueError("An existing bar grid is required to attach structure suggestions")
    duration = float(payload.get("duration", bars[-1]["end"]))
    starts = np.asarray([bar["start"] for bar in bars])
    ends = np.asarray([bar["end"] for bar in bars])
    fps = float(external.get("activation_fps", 100))
    activation_offset = float(external.get("activation_time_offset_seconds", 0))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid expert activation frame rate")
    if not np.isfinite(activation_offset):
        raise ValueError("Invalid expert activation time offset")
    labels = None if activations is None else activations.get("label")
    boundary_activations = None if activations is None else activations.get("segment")
    if labels is not None and (labels.ndim != 2 or labels.shape[0] != len(HARMONIX_LABELS) or not np.isfinite(labels).all()):
        raise ValueError("Expected ten finite Harmonix label activation rows")
    sections, boundaries = [], []
    previous_end = 0.0
    for raw in external.get("segments", []):
        start, end = float(raw["start"]), float(raw["end"])
        if not np.isfinite([start, end]).all() or start < 0 or end <= start or start < previous_end - 0.02:
            raise ValueError("Invalid or overlapping expert segments")
        previous_end = end
        start, end = max(0.0, start), min(duration, end)
        if end <= start:
            continue
        predicted_role = str(raw["label"])
        probabilities = {}
        if labels is not None:
            lo, hi = max(0, int((start - activation_offset) * fps)), min(labels.shape[1], int((end - activation_offset) * fps))
            if hi > lo:
                probabilities = {name: float(score) for name, score in zip(HARMONIX_LABELS, np.mean(labels[:, lo:hi], axis=1))}
        role_confidence = probabilities.get(predicted_role)
        accepted_role = predicted_role if role_confidence is not None and role_confidence >= role_threshold else "unknown"
        left = int(np.clip(np.searchsorted(ends, start, side="right"), 0, len(bars) - 1))
        right = int(np.clip(np.searchsorted(starts, end, side="left") - 1, 0, len(bars) - 1))
        boundary_strength = None
        if boundary_activations is not None and start > 0:
            lo, hi = max(0, int((start - activation_offset - 0.25) * fps)), min(len(boundary_activations), int((start - activation_offset + 0.25) * fps) + 1)
            if hi > lo:
                boundary_strength = float(np.max(boundary_activations[lo:hi]))
        index = len(sections)
        label = f"S{index + 1}"
        # Semantic-role equality is not evidence of identical musical material.
        # Keep groups separate until an acoustic repeat detector confirms them.
        section = {
            "id": f"section-{index + 1:03d}", "label": label, "group": label, "repeat_group": label,
            "name": f"{ROLE_NAMES.get(predicted_role, predicted_role)}建议" if accepted_role != "unknown" else f"段落 {label}",
            "role": {"inst": "instrumental", "start": "unknown", "end": "unknown"}.get(accepted_role, accepted_role),
            "role_candidate": predicted_role, "role_confidence": role_confidence,
            "role_candidates": [{"role": name, "score": round(score, 5)} for name, score in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:3]],
            "start": start, "end": end, "start_bar": int(bars[left]["bar"]), "end_bar": int(bars[right]["bar"]),
            "confidence": boundary_strength if boundary_strength is not None else 0.0,
            "source": "allin1", "evidence": {"model": model, "boundary_activation": boundary_strength, "snapped_to_existing_bars": False, "uncalibrated": True},
        }
        sections.append(section)
        if start > 0:
            boundaries.append({"time": start, "bar": int(bars[left]["bar"]), "confidence": section["confidence"], "source": "allin1", "evidence": {"boundary_activation": boundary_strength}})
    if not sections:
        raise ValueError("No usable expert sections overlap the analysis window")
    result = {
        "version": "1.0", "method": "allin1_harmonix", "model": model,
        "grid_quality": grid_quality(payload, bars), "sections": sections, "boundaries": boundaries,
        "loops": detect_harmonic_loops(bars), "manual_overrides": [],
        "expert_grid": {"bpm": external.get("bpm"), "beats": external.get("beats", []), "beat_positions": external.get("beat_positions", []), "downbeats": external.get("downbeats", []), "applied": False, "comparison": compare_beat_grids(bars, external)},
        "warnings": ["Section roles are suggestions from a model trained on Harmonix; confidence is not calibrated on this music.", "Existing chords and beat grid are preserved. Neural beat/downbeat estimates are supplied separately.", "Sections sharing a functional role are not automatically considered acoustic repeats."],
        "provenance": {"paper": "https://arxiv.org/abs/2307.16425", "upstream": "https://github.com/mir-aidj/all-in-one", "runtime": "https://github.com/openmirlab/all-in-one-infer", "external_path": str(external.get("path", ""))},
    }
    if external.get("timebase_transform"):
        result["provenance"]["timebase_transform"] = copy.deepcopy(external["timebase_transform"])
        result["provenance"]["activation_time_offset_seconds"] = activation_offset
    existing = payload.get("structure", {})
    if existing:
        result["previous_structure"] = copy.deepcopy({key: value for key, value in existing.items() if key != "previous_structure"})
        manual = existing.get("manual_overrides", [])
        if not manual and existing.get("sections") and all(item.get("source") == "manual" for item in existing["sections"]):
            manual = existing["sections"]
        if manual:
            result = apply_manual_sections(result, manual, payload)
    return result


def _convert_window(source: Path, destination: Path, offset: float, duration: float) -> None:
    from chord_recognizer import find_ffmpeg

    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [find_ffmpeg(), "-v", "error", "-y", "-i", str(source)]
    if offset:
        command += ["-ss", str(offset)]
    command += ["-t", str(duration), "-ar", "44100", "-ac", "2", str(destination)]
    subprocess.run(command, check=True)


def run_expert(payload: dict[str, Any], cache: Path, model: str = "harmonix-fold0", audio: Path | None = None, stems: Path | None = None, threads: int = 4) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    try:
        import allin1_infer
        import torch
        from allin1_infer.stems_input import StemsInput
    except ImportError as exc:
        raise RuntimeError("Install the optional environment with setup_structure_env.ps1 and run its Scripts/python.exe") from exc
    torch.set_num_threads(max(1, threads))
    source = audio or Path(payload["source"])
    offset, duration = float(payload.get("audio_offset", 0)), float(payload["duration"])
    if duration <= 0 or offset < 0 or not np.isfinite([offset, duration]).all():
        raise ValueError("Invalid analysis offset/duration")
    source_paths = [stems / (name + ".wav") for name in ("bass", "drums", "other", "vocals")] if stems else [source]
    signature = [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in source_paths]
    cache_identity = [signature, offset, duration]
    # Nonzero clips previously used input seeking, which can omit MP3 decoder
    # warmup. Invalidate only those entries; full-file inputs are unchanged.
    if offset:
        cache_identity.append("output_seek_v2")
    identifier = "track_" + hashlib.sha256(json.dumps(cache_identity).encode()).hexdigest()[:16]
    cache = cache.resolve()
    result_dir = cache / "results" / model
    kwargs = {"model": model, "device": "cpu", "out_dir": result_dir, "demix_dir": cache / "demix", "spec_dir": cache / "spec", "include_activations": True, "keep_byproducts": True, "multiprocess": False, "overwrite": True}
    began = time.monotonic()
    result_file = result_dir / (identifier + ".json")
    cache_hit = result_file.is_file() and result_file.with_suffix(".activ.npz").is_file()
    if cache_hit:
        external = json.loads(result_file.read_text(encoding="utf-8"))
    else:
        if stems:
            prepared = cache / "inputs" / identifier
            for name in ("bass", "drums", "other", "vocals"):
                _convert_window(stems / (name + ".wav"), prepared / (name + ".wav"), offset, duration)
            input_stems = StemsInput(**{name: prepared / (name + ".wav") for name in ("bass", "drums", "other", "vocals")}, identifier=identifier)
            allin1_infer.analyze(stems_input=input_stems, **kwargs)
        else:
            wav = cache / "inputs" / (identifier + ".wav")
            _convert_window(source, wav, offset, duration)
            allin1_infer.analyze(wav, **kwargs)
        external = json.loads(result_file.read_text(encoding="utf-8"))
    with np.load(result_file.with_suffix(".activ.npz")) as data:
        activations = {name: data[name] for name in data.files}
    provenance = {"runtime_version": importlib.metadata.version("all-in-one-infer"), "model": model, "elapsed_seconds": round(time.monotonic() - began, 3), "cache_reused": cache_hit, "result": str(result_file), "audio_offset": offset, "duration": duration, "source_signature": signature, "crop_policy": "decode_from_start_then_output_seek", "activation_fps": float(external.get("activation_fps", 100)), "activation_shapes": {name: list(value.shape) for name, value in activations.items()}}
    return external, activations, provenance


def write_expert_beats(path: Path, external: dict[str, Any]) -> None:
    beats, positions = external.get("beats", []), external.get("beat_positions", [])
    if len(beats) != len(positions) or not beats:
        raise ValueError("Expert beats and beat positions must be nonempty and equally sized")
    rows, previous = [], -1.0
    for time_value, position in zip(beats, positions):
        timestamp, position = float(time_value), int(position)
        if not np.isfinite(timestamp) or timestamp <= previous or timestamp < 0 or position < 1:
            raise ValueError("Invalid expert beat grid")
        rows.append(f"{timestamp:.6f}\t{position}")
        previous = timestamp
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--stems", type=Path, help="Folder with bass/drums/other/vocals.wav matching the full source")
    parser.add_argument("--model", default="harmonix-fold0", choices=["harmonix-all"] + [f"harmonix-fold{i}" for i in range(8)])
    parser.add_argument("--cache", type=Path, default=Path(".tmp/structure_expert"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--import-result", type=Path, help="Reuse an All-In-One JSON for this exact analysis window")
    parser.add_argument("--activations", type=Path, help="Optional matching activation NPZ when importing")
    parser.add_argument("--import-time-offset", type=float, help="Independently verified decoder offset in seconds; only for --import-result, never fitted to reference beats")
    parser.add_argument("--beat-grid", type=Path, help="Export proposed beats without changing the existing analysis")
    parser.add_argument("--html", action="store_true")
    args = parser.parse_args()
    if args.import_time_offset is not None and not args.import_result:
        parser.error("--import-time-offset requires --import-result")
    payload = json.loads(args.analysis.read_text(encoding="utf-8-sig"))
    provenance = {}
    if args.import_result:
        external = json.loads(args.import_result.read_text(encoding="utf-8-sig"))
        activation_path = args.activations or args.import_result.with_suffix(".activ.npz")
        activations = None
        if activation_path.exists():
            with np.load(activation_path) as data:
                activations = {name: data[name] for name in data.files}
        if args.import_time_offset is not None:
            external = shift_expert_timebase(external, args.import_time_offset, float(payload["duration"]))
    else:
        external, activations, provenance = run_expert(payload, args.cache, args.model, args.audio, args.stems, args.threads)
    payload["structure"] = import_expert_structure(payload, external, activations, args.model)
    payload["structure"]["provenance"].update(provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.beat_grid:
        write_expert_beats(args.beat_grid, external)
    if args.html:
        from chord_studio import write_html
        write_html(args.output.with_suffix(".html"), payload)
    print(json.dumps({"output": str(args.output), "sections": len(payload["structure"]["sections"]), "model": args.model}, ensure_ascii=False))


if __name__ == "__main__":
    main()
