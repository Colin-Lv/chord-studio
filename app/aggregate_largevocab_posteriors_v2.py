"""Aggregate cached six-head evidence on explicit LAB boundaries, without inference.

Raw LAB boundaries are the default. Large-vocabulary and union anchors are
separate experimental timing factors, never new acoustic evidence sources.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_audio_only_largevocab_labs import signature
from generate_audio_only_largevocab_labs_v2 import write_json_atomic
from largevocab_posteriors_v2 import aggregate_posteriors, load_posteriors

SCHEMA = "ismir2019_six_head_boundary_aggregation_v2"


def read_anchor_lab(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=2)
        if len(fields) != 3:
            raise ValueError(f"Malformed LAB row {line_number}: require start end label")
        start, end = map(float, fields[:2])
        if not np.isfinite([start, end]).all() or start < 0 or end <= start:
            raise ValueError(f"Invalid LAB boundary at row {line_number}")
        if rows and start < rows[-1][1] - 1e-9:
            raise ValueError(f"Overlapping or unordered LAB intervals at row {line_number}")
        rows.append((start, end, fields[2]))
    if not rows:
        raise ValueError("An empty LAB provides no aggregation boundaries")
    return rows


def boundary_anchors(raw_rows, *, mode="raw", largevocab_rows=None):
    if mode == "raw":
        return [[row[0], row[1]] for row in raw_rows]
    if mode not in {"largevocab", "union"} or not largevocab_rows:
        raise ValueError("Experimental largevocab/union anchors require a second LAB")
    if mode == "largevocab":
        return [[row[0], row[1]] for row in largevocab_rows]
    edges = sorted({float(value) for row in list(raw_rows) + list(largevocab_rows) for value in row[:2]})
    return [[start, end] for start, end in zip(edges, edges[1:]) if end > start]


def verify_identity_files(identity):
    """Require the audio, all five weights and frontend files still match.

This checks the artifact's recorded source identity. It does not assert that
an arbitrary anchor LAB was computed from that source audio.
"""
    if not isinstance(identity, dict) or not isinstance(identity.get("source_audio"), dict):
        raise ValueError("Posterior source audio identity is missing")
    recognizer = identity.get("recognizer", {})
    if len(recognizer.get("models", [])) != 5:
        raise ValueError("Posterior identity must identify all five original model weights")
    required = [identity["source_audio"], *recognizer["models"],
        recognizer.get("template"), recognizer.get("implementation"),
        recognizer.get("audio_decoder"), recognizer.get("audio_decoder_implementation"),
        identity.get("bridge_implementation"), identity.get("posterior_implementation")]
    required += recognizer.get("vendor_implementations", [])
    if not recognizer.get("vendor_implementations"):
        raise ValueError("Posterior vendor frontend identity is missing")
    verified = []
    for expected in required:
        if not isinstance(expected, dict) or not expected.get("path") or not expected.get("sha256"):
            raise ValueError("Incomplete posterior source/model/frontend file identity")
        if signature(expected["path"]) != expected:
            raise ValueError(f"Stale posterior source/model/frontend identity: {expected['path']}")
        verified.append(expected)
    return verified


def run_aggregation(posteriors, raw_lab, output, *, anchors="raw", largevocab_lab=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing aggregation: {output}")
    posterior_identity = signature(posteriors)
    evidence = load_posteriors(posteriors)
    verified = verify_identity_files(evidence.metadata["identity"])
    raw_identity = signature(raw_lab)
    raw_rows = read_anchor_lab(raw_lab)
    if anchors != "raw" and largevocab_lab is None:
        raise ValueError("--largevocab-lab is required for experimental anchors")
    large_identity = signature(largevocab_lab) if largevocab_lab else None
    large_rows = read_anchor_lab(largevocab_lab) if largevocab_lab else None
    anchor_source = dict(mode=anchors, raw_lab=raw_identity, largevocab_lab=large_identity,
        labels_used_as_acoustic_features=False,
        source_audio_identity_verified_for_anchor=False,
        limitation="LAB SHA identifies boundary bytes; no audio identity is inferred from filenames",
        experimental=anchors != "raw")
    intervals = boundary_anchors(raw_rows, mode=anchors, largevocab_rows=large_rows)
    aggregated = aggregate_posteriors(evidence, intervals, anchor_source=anchor_source)
    if signature(raw_lab) != raw_identity or (largevocab_lab and signature(largevocab_lab) != large_identity):
        raise RuntimeError("Anchor LAB changed while aggregating")
    if signature(posteriors) != posterior_identity:
        raise RuntimeError("Posterior artifact changed while aggregating")
    probabilities = {name: [[float(value) if np.isfinite(value) else None for value in row] for row in values]
                     for name, values in aggregated.probabilities.items()}
    result = dict(schema=SCHEMA, status="complete", posterior_artifact=posterior_identity,
        source_identity=evidence.metadata["identity"], anchor_source=anchor_source,
        source_files_verified=verified, implementation=signature(__file__),
        audio_seconds=evidence.metadata["audio_seconds"],
        requested_duration_seconds=evidence.metadata["identity"].get("requested_duration_seconds"),
        intervals=aggregated.intervals.tolist(), coverage=aggregated.coverage.tolist(),
        probabilities=probabilities, metadata=aggregated.metadata,
        interval_count=len(intervals), zero_coverage_intervals=int(np.count_nonzero(aggregated.coverage == 0)),
        partial_coverage_intervals=int(np.count_nonzero((aggregated.coverage > 0) & (aggregated.coverage < 1-1e-9))),
        reference_inputs=False, fitting=False, acoustic_inference=False,
        interpretation="Boundary aggregation diagnostic only; no accuracy or training gain is measured")
    write_json_atomic(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posteriors", type=Path, required=True)
    parser.add_argument("--raw-lab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchors", choices=("raw", "largevocab", "union"), default="raw")
    parser.add_argument("--largevocab-lab", type=Path)
    args = parser.parse_args(argv)
    if args.anchors != "raw" and args.largevocab_lab is None:
        parser.error("--largevocab-lab is required for experimental anchors")
    result = run_aggregation(args.posteriors, args.raw_lab, args.output,
                             anchors=args.anchors, largevocab_lab=args.largevocab_lab)
    print(json.dumps({key: result[key] for key in ("status", "interval_count", "zero_coverage_intervals", "partial_coverage_intervals")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
