"""Conservative, bar-synchronous music structure and harmonic-loop analysis.

The result is evidence for reviewing an existing chord analysis: it never changes
the beat grid or chord labels. Acoustic repetition groups (A, B, ...) are not
semantic verse/chorus predictions. All confidence values are heuristic evidence
strengths, not calibrated probabilities.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "1.2"
FEATURE_WEIGHTS = {"chroma": 0.30, "timbre": 0.45, "rhythm": 0.15, "energy": 0.10}


@dataclass(frozen=True)
class StructureConfig:
    min_section_bars: int = 3
    novelty_threshold: float = 0.04
    repeat_threshold: float = 0.79
    max_bars: int = 2000
    novelty_widths: tuple[int, ...] = (4,)

    def __post_init__(self) -> None:
        if self.min_section_bars < 1 or self.max_bars < 1:
            raise ValueError("Section distance and bar limit must be positive")
        if not self.novelty_widths or any(width < 1 for width in self.novelty_widths):
            raise ValueError("Novelty windows must be positive bar counts")
        if not 0 <= self.novelty_threshold <= 1 or not 0 <= self.repeat_threshold <= 1:
            raise ValueError("Structure thresholds must be finite values between zero and one")


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-9)


def _robust_scale(values: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    center = np.median(values, axis=0, keepdims=True)
    scale = np.percentile(values, 90, axis=0) - np.percentile(values, 10, axis=0)
    return np.clip((values - center) / np.maximum(scale, floor), -3.0, 3.0)


def valid_bars(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the supplied grid instead of silently inventing replacement bars."""
    # Feature rows may already be paired with these entries. Reordering only
    # the grid silently attributes cached acoustic evidence to different times.
    bars = list(payload.get("bars", []))
    previous_start = -float("inf")
    previous_end = -float("inf")
    seen = set()
    for bar in bars:
        start, end = float(bar["start"]), float(bar["end"])
        number = int(bar["bar"])
        if not np.isfinite([start, end]).all() or end <= start or start < 0:
            raise ValueError(f"Invalid time interval for bar {number}")
        if start < previous_start:
            raise ValueError("Bars must be in chronological order and match feature-row order")
        if start < previous_end - 0.02:
            raise ValueError("Overlapping bars cannot be used for structure analysis")
        if number in seen:
            raise ValueError("Duplicate bar numbers cannot be used for structure analysis")
        previous_end = end
        previous_start = start
        seen.add(number)
    return bars


def grid_quality(payload: dict[str, Any], bars: list[dict[str, Any]]) -> dict[str, Any]:
    durations = np.array([float(bar["end"]) - float(bar["start"]) for bar in bars])
    median = float(np.median(durations)) if len(durations) else 0.0
    inner = durations[1:-1] if len(durations) > 3 else durations
    relative_deviation = float(np.median(np.abs(inner - median)) / median) if median else 1.0
    # Regularity alone does not prove correct bar phase or meter.
    meter_confidence = payload.get("meter_confidence")
    evidence = float(np.clip(float(meter_confidence), 0, 1)) if meter_confidence is not None else 0.5
    gaps = sum(float(right["start"]) - float(left["end"]) > 0.02 for left, right in zip(bars, bars[1:]))
    return {
        "bar_count": len(bars), "median_bar_seconds": round(median, 6),
        "relative_duration_deviation": round(relative_deviation, 4),
        "meter_confidence": meter_confidence, "source": payload.get("beat_source", payload.get("beat_grid", "unknown")),
        "confidence": round(evidence * max(0.4, 1.0 - relative_deviation) * (0.8 if gaps else 1), 4),
        "gap_count": gaps, "phase_verified": False,
        "leading_ungridded_seconds": round(float(bars[0]["start"]), 6) if bars else None,
        "unconfirmed_ending_intervals": sum(bar.get("end_boundary_source") == "audio_end_unconfirmed" for bar in bars),
    }


def extract_bar_features(y: np.ndarray, sr: int, bars: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Pool audio features on real bar times, preserving variable tempo and meter."""
    import librosa

    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1 or sr <= 0 or not len(y) or not np.isfinite(y).all():
        raise ValueError("Structure analysis requires finite, nonempty mono audio and a positive sample rate")
    n_fft, hop = 2048, 512
    power = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    chroma = librosa.feature.chroma_stft(S=power, sr=sr, n_fft=n_fft, hop_length=hop)
    mel = librosa.feature.melspectrogram(S=power, sr=sr, n_fft=n_fft, n_mels=40)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=14)[1:]
    onset = np.maximum(0, np.diff(log_mel, axis=1, prepend=log_mel[:, :1])).mean(axis=0)
    rms = librosa.feature.rms(S=np.sqrt(power), frame_length=n_fft, hop_length=hop)[0]
    frame_times = np.arange(power.shape[1]) * hop / sr
    features: dict[str, list[np.ndarray]] = {name: [] for name in FEATURE_WEIGHTS}
    for bar in bars:
        start, end = float(bar["start"]), float(bar["end"])
        lo, hi = np.searchsorted(frame_times, [start, end])
        lo = min(int(lo), len(frame_times) - 1)
        hi = min(len(frame_times), max(lo + 1, int(hi)))
        features["chroma"].append(np.mean(chroma[:, lo:hi], axis=1))
        features["timbre"].append(np.mean(mfcc[:, lo:hi], axis=1))
        edges = np.linspace(start, end, 9)
        pulse = []
        for left, right in zip(edges[:-1], edges[1:]):
            a, b = np.searchsorted(frame_times, [left, right])
            pulse.append(float(np.mean(onset[a:b])) if b > a and a < len(onset) else 0.0)
        features["rhythm"].append(np.asarray(pulse))
        features["energy"].append(np.asarray([20 * np.log10(max(float(np.mean(rms[lo:hi])), 1e-8))]))
    return {name: np.asarray(values, dtype=float) for name, values in features.items()}


def feature_similarities(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    similarities = {}
    for name, values in features.items():
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f"Invalid {name} feature matrix")
        if name in {"chroma", "rhythm"}:
            normalized = _unit_rows(values)
            similarity = np.clip(normalized @ normalized.T, 0.0, 1.0)
            silent = np.linalg.norm(values, axis=1) < 1e-8
            similarity[np.ix_(silent, silent)] = 1.0
        else:
            # Fixed dB scale keeps negligible volume jitter from becoming a boundary.
            normalized = values / 10.0 if name == "energy" else _robust_scale(values, floor=8.0)
            distance = np.maximum(0, np.mean((normalized[:, None, :] - normalized[None, :, :]) ** 2, axis=2))
            similarity = np.exp(-2.0 * distance)
        similarities[name] = similarity
    return similarities


def novelty_curve(similarity: np.ndarray, widths: tuple[int, ...] = (2, 4, 8)) -> np.ndarray:
    """Multi-scale checkerboard novelty; no endpoint zero-padding artifacts."""
    n = len(similarity)
    novelty = np.zeros(n + 1, dtype=float)
    for boundary in range(1, n):
        scores = []
        for width in widths:
            radius = min(width, boundary, n - boundary)
            if radius < min(2, width):
                continue
            left = similarity[boundary - radius:boundary, boundary - radius:boundary]
            right = similarity[boundary:boundary + radius, boundary:boundary + radius]
            cross = similarity[boundary - radius:boundary, boundary:boundary + radius]
            scores.append(max(0.0, float(0.5 * (left.mean() + right.mean()) - cross.mean())))
        novelty[boundary] = float(np.mean(scores)) if scores else 0.0
    return novelty


def select_boundaries(curve: np.ndarray, config: StructureConfig) -> list[int]:
    """Keep strong local changes without imposing a fixed eight-bar song form."""
    n = len(curve) - 1
    if n < 2:
        return [0, n] if n else [0]
    inner = curve[1:n]
    median = float(np.median(inner))
    mad = float(np.median(np.abs(inner - median)))
    threshold = max(config.novelty_threshold, median + 0.7 * mad)
    candidates = [i for i in range(2, n - 1) if curve[i] >= threshold and curve[i] >= curve[i - 1] and curve[i] > curve[i + 1]]
    selected: list[int] = []
    for index in sorted(candidates, key=lambda value: (-curve[value], value)):
        if all(abs(index - existing) >= config.min_section_bars for existing in selected):
            selected.append(index)
    return [0, *sorted(selected), n]


def _resample_rows(values: np.ndarray, count: int = 16) -> np.ndarray:
    source = (np.arange(len(values)) + 0.5) / max(1, len(values))
    target = (np.arange(count) + 0.5) / count
    return np.column_stack([np.interp(target, source, values[:, col]) for col in range(values.shape[1])])


def _sequence_similarity(features: dict[str, np.ndarray], left: tuple[int, int], right: tuple[int, int]) -> tuple[float, dict[str, float]]:
    scores = {}
    for name, values in features.items():
        a = _resample_rows(values[left[0]:left[1]])
        b = _resample_rows(values[right[0]:right[1]])
        if name in {"chroma", "rhythm"}:
            score = float(np.mean(np.sum(_unit_rows(a) * _unit_rows(b), axis=1)))
        else:
            scale = np.maximum(np.percentile(values, 90, axis=0) - np.percentile(values, 10, axis=0), 8.0)
            if name == "energy":
                scale = np.full(values.shape[1], 10.0)
            score = float(np.exp(-2 * np.mean(((a - b) / scale) ** 2)))
        scores[name] = max(0.0, min(1.0, score))
    weight_sum = sum(FEATURE_WEIGHTS.get(name, 0) for name in scores)
    total = sum(FEATURE_WEIGHTS.get(name, 0) * score for name, score in scores.items()) / max(weight_sum, 1e-8)
    ratio = min(left[1] - left[0], right[1] - right[0]) / max(left[1] - left[0], right[1] - right[0])
    return total * (0.7 + 0.3 * ratio), scores


def _group_label(index: int) -> str:
    label = ""
    while index >= 0:
        label = chr(65 + index % 26) + label
        index = index // 26 - 1
    return label


LOOP_NUMERIC_TOLERANCE = 1e-9


def _bar_pattern(bar: dict[str, Any]) -> list[dict] | None:
    """A complete observed harmonic timeline, never a majority-label summary."""
    # A file boundary can cut a measure anywhere. Keeping all harmony in that
    # interval does not establish a complete musical bar or repeat cycle.
    if bar.get("end_boundary_source") == "audio_end_unconfirmed" or bar.get("kind") in {"pickup", "unconfirmed"}:
        return None
    try:
        from .chord_semantics_v2 import parse_chord
    except ImportError:
        from chord_semantics_v2 import parse_chord
    start, end = float(bar["start"]), float(bar["end"])
    if not np.isfinite([start, end]).all() or end <= start:
        return None
    spans = bar.get("spans", bar.get("beats"))
    if spans is None:
        spans = [{"start": start, "end": end, "chord": bar.get("chord")}]
    if not isinstance(spans, list) or not spans:
        return None
    result = []; cursor = start
    for span in spans:
        if not isinstance(span, dict):
            return None
        raw_start, raw_end = span.get("start"), span.get("end")
        if (type(raw_start) not in {int, float} or type(raw_end) not in {int, float}
                or not np.isfinite([raw_start, raw_end]).all() or raw_end <= raw_start):
            return None
        left, right = max(start, raw_start), min(end, raw_end)
        if right <= left or abs(left - cursor) > LOOP_NUMERIC_TOLERANCE:
            return None
        label = span.get("chord")
        if not isinstance(label, str):
            return None
        semantic = parse_chord(label)
        if semantic.state != "chord" or semantic.parse_error:
            return None
        left_fraction, right_fraction = (left-start)/(end-start), (right-start)/(end-start)
        if result and result[-1]["identity"] == semantic.identity:
            result[-1]["end_fraction"] = right_fraction
        else:
            result.append(dict(start_fraction=left_fraction, end_fraction=right_fraction,
                               label=label, identity=semantic.identity))
        cursor = right
    return result if abs(cursor-end) <= LOOP_NUMERIC_TOLERANCE else None


def _same_bar_pattern(left: list[dict] | None, right: list[dict] | None) -> bool:
    # Fixed bins make equivalence transitive; pairwise isclose could allow
    # successive repetitions to drift progressively away from the first one.
    return bool(left and right and len(left) == len(right) and all(
        a["identity"] == b["identity"] and all(round(a[k], 9) == round(b[k], 9)
        for k in ("start_fraction", "end_fraction")) for a, b in zip(left, right)))


def detect_harmonic_loops(bars: list[dict[str, Any]], max_period: int = 16) -> list[dict[str, Any]]:
    """Repeat complete relative-time progressions; gaps/unknowns cannot prove one.

    Variable bar duration is permitted; relative times match at nine decimal
    places. This is annotation agreement, not acoustic truth.
    """
    if type(max_period) is not int or max_period < 1:
        raise ValueError("Loop period must be a positive integer")
    patterns = [_bar_pattern(bar) for bar in bars]
    candidates = []
    n = len(bars)
    continuous_end = [n] * n
    for i in range(n-2, -1, -1):
        continuous_end[i] = (continuous_end[i+1] if abs(float(bars[i]["end"])-float(bars[i+1]["start"]))
                             <= LOOP_NUMERIC_TOLERANCE else i+1)
    for period in range(1, min(max_period, n // 2) + 1):
        # Count equal forward pairs once, avoiding a quadratic rescan of long
        # repeated songs for every possible starting bar.
        matched_run = [0] * (n+1)
        for i in range(n-period-1, -1, -1):
            if _same_bar_pattern(patterns[i], patterns[i+period]):
                matched_run[i] = matched_run[i+1] + 1
        for start in range(n - 2 * period + 1):
            if matched_run[start] < period:
                continue
            pattern = patterns[start:start + period]
            if any(item is None for item in pattern) or len({span["identity"] for item in pattern for span in item}) < 2:
                continue
            if any(period % smaller == 0 and all(_same_bar_pattern(pattern[i], pattern[i % smaller])
                    for i in range(period)) for smaller in range(1, period)):
                continue
            repetitions = min(1 + matched_run[start] // period, (continuous_end[start]-start) // period)
            if repetitions < 2:
                continue
            end = start + repetitions * period
            candidates.append((start, end, period, repetitions, pattern))
    kept = []
    for item in sorted(candidates, key=lambda value: (-(value[1] - value[0]), value[2], value[0])):
        start, end, period, repetitions, pattern = item
        if any(max(0, min(end, old[1]) - max(start, old[0])) >= 0.5 * (end - start) for old in kept):
            continue
        kept.append(item)
    result = []
    for index, (start, end, period, repetitions, pattern) in enumerate(sorted(kept)):
        result.append({
            "id": f"loop-{index + 1:03d}", "kind": "harmonic_loop", "start": float(bars[start]["start"]),
            "end": float(bars[end - 1]["end"]), "start_bar": int(bars[start]["bar"]), "end_bar": int(bars[end - 1]["bar"]),
            "period_bars": period, "repetitions": repetitions,
            "chords": [" → ".join(span["label"] for span in item) for item in pattern],
            "bar_patterns": [[{key: value for key, value in span.items() if key != "identity"}
                              for span in item] for item in pattern],
            "match_basis": "full_bar_relative_harmonic_timeline", "relative_time_decimal_places": 9,
            "confidence": round(min(0.95, 0.55 + 0.1 * repetitions), 4), "source": "chord_labels",
            "evidence": "Complete within-bar harmonic and relative-time agreement; not acoustic confirmation. Depends on chord labels and unverified bar alignment.",
        })
    return result


def analyze_feature_structure(payload: dict[str, Any], features: dict[str, np.ndarray], config: StructureConfig | None = None) -> dict[str, Any]:
    """Analyze supplied bar features; useful for repeatable tests and cached evaluation."""
    config = config or StructureConfig()
    bars = valid_bars(payload)
    n = len(bars)
    if n > config.max_bars:
        raise ValueError(f"Structure analysis supports at most {config.max_bars} bars per window")
    if any(len(values) != n for values in features.values()):
        raise ValueError("Feature rows must match the supplied bar grid")
    quality = grid_quality(payload, bars)
    warnings = ["Automatic labels A/B indicate acoustic similarity, not verse/chorus roles.", "Confidence is an uncalibrated evidence score."]
    if quality["confidence"] < 0.6:
        warnings.append("Bar alignment is uncertain; review the beat grid before interpreting section lengths.")
    if quality["gap_count"]:
        warnings.append("The supplied bar grid has gaps; structure coverage follows the available bars.")
    if quality["leading_ungridded_seconds"]:
        warnings.append("The opening region before the first detected downbeat is outside this automatic structure grid; it is not a confirmed pickup.")
    if quality["unconfirmed_ending_intervals"]:
        warnings.append("The final interval ends at the audio-file boundary, not a confirmed downbeat; it cannot complete a harmonic loop.")
    if n == 0:
        return {"version": VERSION, "method": "bar_multifeature_novelty", "grid_quality": quality, "sections": [], "loops": [], "boundaries": [], "warnings": warnings, "manual_overrides": []}
    similarities = feature_similarities(features)
    curves = {name: novelty_curve(matrix, config.novelty_widths) for name, matrix in similarities.items()}
    weight_sum = sum(FEATURE_WEIGHTS.get(name, 0) for name in curves)
    if not weight_sum:
        raise ValueError("At least one supported audio feature is required")
    curve = sum(FEATURE_WEIGHTS.get(name, 0) * values for name, values in curves.items()) / weight_sum
    positions = select_boundaries(curve, config)
    intervals = list(zip(positions[:-1], positions[1:]))
    groups: list[list[int]] = []
    assignments = []
    for current, interval in enumerate(intervals):
        best_group, best_score, evidence = -1, -1.0, {}
        for group_index, members in enumerate(groups):
            comparisons = [_sequence_similarity(features, interval, intervals[member]) for member in members]
            score = min(item[0] for item in comparisons)  # complete linkage avoids A~B~C drift
            if score > best_score:
                best_group, best_score, evidence = group_index, score, comparisons[0][1]
        if best_score < config.repeat_threshold or ("timbre" in evidence and evidence["timbre"] < 0.63):
            groups.append([current])
            assignments.append((len(groups) - 1, 0.0))
        else:
            groups[best_group].append(current)
            assignments.append((best_group, best_score))
    sections = []
    for index, ((start, end), (group, repeat_score)) in enumerate(zip(intervals, assignments)):
        label = _group_label(group)
        boundary_support = float(max(curve[start], curve[end]))
        confidence = min(0.95, 0.35 + 0.6 * min(1.0, boundary_support / 0.25))
        sections.append({
            "id": f"section-{index + 1:03d}", "label": label, "name": f"段落 {label}", "group": label, "repeat_group": label,
            "role": "unknown", "role_confidence": 0.0, "start": float(bars[start]["start"]), "end": float(bars[end - 1]["end"]),
            "start_bar": int(bars[start]["bar"]), "end_bar": int(bars[end - 1]["bar"]), "confidence": round(confidence, 4),
            "source": "automatic", "evidence": {"boundary_strength": round(boundary_support, 4), "repeat_similarity": round(repeat_score, 4), "group_occurrences": len(groups[group])},
        })
    boundaries = [{"time": float(bars[index]["start"]), "bar": int(bars[index]["bar"]), "confidence": round(min(0.95, curve[index] / 0.3), 4), "evidence": {name: round(float(values[index]), 4) for name, values in curves.items()}} for index in positions[1:-1]]
    return {
        "version": VERSION, "method": "bar_multifeature_novelty", "config": vars(config), "grid_quality": quality,
        "sections": sections, "loops": detect_harmonic_loops(bars), "boundaries": boundaries,
        "novelty": [{"time": float(bar["start"]), "bar": int(bar["bar"]), "value": round(float(curve[index]), 5)} for index, bar in enumerate(bars)],
        "warnings": warnings, "manual_overrides": [],
    }


def apply_manual_sections(structure: dict[str, Any], sections: list[dict[str, Any]], payload: dict[str, Any]) -> dict[str, Any]:
    """Preserve validated manual times, or expand explicitly supplied bar ranges.

    Neural boundaries may lie inside a bar, so two adjacent sections can share
    a bar number. A complete list of start/end intervals takes precedence over
    the bar indexes and is validated on the time axis. Bar-only input retains
    the strict whole-bar coverage contract.
    """
    bars = valid_bars(payload)
    by_number = {int(bar["bar"]): (index, bar) for index, bar in enumerate(bars)}
    output = copy.deepcopy(structure)
    validated = []
    next_index = 0
    timed = bool(sections) and all("start" in section and "end" in section for section in sections)
    if not timed and any("start" in section or "end" in section for section in sections):
        raise ValueError("Manual sections must either all provide start/end times or all use bar ranges")
    starts = np.asarray([bar["start"] for bar in bars])
    ends = np.asarray([bar["end"] for bar in bars])
    previous_end = None
    for index, section in enumerate(sections):
        if timed:
            start_time, end_time = float(section["start"]), float(section["end"])
            if not np.isfinite([start_time, end_time]).all() or start_time < 0 or end_time <= start_time:
                raise ValueError("Manual section has an invalid time interval")
            if previous_end is not None and abs(start_time - previous_end) > 0.02:
                raise ValueError("Manual sections must be contiguous without time overlaps or gaps")
            if not bars or (index == 0 and start_time > starts[0] + 0.02):
                raise ValueError("Manual sections must cover every supplied bar")
            if end_time > float(payload.get("duration", ends[-1])) + 0.1:
                raise ValueError("Manual section exceeds the analysis window")
            start = int(np.clip(np.searchsorted(ends, start_time, side="right"), 0, len(bars) - 1))
            end = int(np.clip(np.searchsorted(starts, end_time, side="left") - 1, 0, len(bars) - 1))
            start_bar, end_bar = int(bars[start]["bar"]), int(bars[end]["bar"])
            previous_end = end_time
        else:
            start_bar, end_bar = int(section["start_bar"]), int(section["end_bar"])
            if start_bar not in by_number or end_bar not in by_number:
                raise ValueError("Manual section refers to an unknown bar")
            start, left = by_number[start_bar]
            end, right = by_number[end_bar]
            if start != next_index or end < start:
                raise ValueError("Manual sections must cover all bars in order without overlaps or gaps")
            start_time, end_time = float(left["start"]), float(right["end"])
        item = copy.deepcopy(section)
        label = str(item.get("label", _group_label(index)))
        item.update({"id": str(item.get("id", f"section-{index + 1:03d}")), "label": label, "name": str(item.get("name", f"段落 {label}")), "group": str(item.get("group", label)), "repeat_group": str(item.get("group", label)), "role": str(item.get("role", "unknown")), "start": start_time, "end": end_time, "start_bar": start_bar, "end_bar": end_bar, "confidence": 1.0, "source": "manual"})
        validated.append(item)
        next_index = end + 1
    if (timed and (previous_end is None or previous_end < ends[-1] - 0.02)) or (not timed and next_index != len(bars)):
        raise ValueError("Manual sections must cover every supplied bar")
    output["automatic_sections"] = output.get("automatic_sections", output.get("sections", []))
    output["sections"] = validated
    output["boundaries"] = [{"time": section["start"], "bar": section["start_bar"], "confidence": 1.0, "source": "manual", "evidence": {}} for section in validated[1:]]
    output["manual_overrides"] = copy.deepcopy(validated)
    return output


def analyze_song_structure(payload: dict[str, Any], y: np.ndarray | None = None, sr: int = 22050, config: StructureConfig | None = None) -> dict[str, Any]:
    """Return structure metadata; times are relative to the analyzed audio window."""
    bars = valid_bars(payload)
    if not bars:
        return analyze_feature_structure(payload, {})
    config = config or StructureConfig()
    if len(bars) > config.max_bars:
        raise ValueError(f"Structure analysis supports at most {config.max_bars} bars per window")
    if y is None:
        from chord_recognizer import decode_audio
        source = payload.get("source")
        if not source or not Path(source).is_file():
            raise FileNotFoundError("Provide audio or a valid analysis source path for structure analysis")
        y = decode_audio(Path(source), sample_rate=sr, offset=float(payload.get("audio_offset", 0)), duration=float(payload.get("duration", bars[-1]["end"])))
    audio_duration = len(y) / sr
    if float(bars[-1]["end"]) > audio_duration + 0.1:
        raise ValueError("The bar grid extends beyond the supplied audio window")
    result = analyze_feature_structure(payload, extract_bar_features(y, sr, bars), config)
    previous = payload.get("structure", {})
    manual = previous.get("manual_overrides", []) if isinstance(previous, dict) else []
    if manual:
        result = apply_manual_sections(result, manual, payload)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--manual-sections", type=Path, help="JSON list of contiguous sections using start_bar/end_bar")
    parser.add_argument("--html", action="store_true", help="Render the updated analysis using Chord Studio")
    args = parser.parse_args()
    payload = json.loads(args.analysis.read_text(encoding="utf-8-sig"))
    y = None
    if args.audio:
        from chord_recognizer import decode_audio
        y = decode_audio(args.audio, offset=float(payload.get("audio_offset", 0)), duration=float(payload["duration"]))
    payload["structure"] = analyze_song_structure(payload, y)
    if args.manual_sections:
        manual = json.loads(args.manual_sections.read_text(encoding="utf-8-sig"))
        payload["structure"] = apply_manual_sections(payload["structure"], manual, payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.html:
        from chord_studio import write_html
        write_html(args.output.with_suffix(".html"), payload)
    print(json.dumps({"output": str(args.output), "sections": len(payload["structure"]["sections"]), "loops": len(payload["structure"]["loops"])}))


if __name__ == "__main__":
    main()
