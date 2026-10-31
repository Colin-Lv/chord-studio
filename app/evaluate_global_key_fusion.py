"""Offline four-way key comparison; reference labels are used only after inference.

CSV columns: id, cnn, neutral, raw, reference [, audio]. Paths resolve relative
to the manifest. Inputs must be saved CNN output, key-independent LAB, and an
unconditioned studio pass (--all-key-candidates). No audio inference/training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

from chord_studio import parse_chord_name
from key_fusion import POLICY, analyze_key_evidence, key_identity, read_raw_lab
from dataclasses import asdict


def signature(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path.resolve()), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def confusion(reference, prediction):
    if prediction is None:
        return "missing"
    reference, prediction = key_identity(reference), key_identity(prediction)
    if reference == prediction:
        return "exact"
    a, am = parse_chord_name(reference)
    b, bm = parse_chord_name(prediction)
    if am != bm and ((am == "" and (b-a)%12 == 9) or (am == "m" and (b-a)%12 == 3)):
        return "relative_major_minor"
    if am == bm and (b-a)%12 in {5,7}:
        return "dominant_fourth"
    return "other"


def evaluate(manifest, output, dataset_kind="regression"):
    started = time.perf_counter()
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig")))
    songs, failures, ids = [], [], set()
    for index, row in enumerate(rows):
        identifier = row["id"]
        if identifier in ids:
            raise ValueError("Duplicate evaluation ID: " + identifier)
        ids.add(identifier)
        try:
            paths = {k: (manifest.parent / row[k]).resolve() for k in ("cnn", "neutral", "raw")}
            cnn, neutral = [json.loads(paths[k].read_text(encoding="utf-8")) for k in ("cnn", "neutral")]
            fusion = analyze_key_evidence(cnn, neutral, read_raw_lab(paths["raw"]))
            # Deliberately read reference only AFTER producing the entire prediction.
            reference = key_identity(row["reference"])
            evidence = fusion["evidence"]
            predictions = {"cnn":evidence["madmom_cnn"]["key"], "functional_joint":evidence["functional_joint"]["key"],
                           "raw_harmonic":evidence["raw_chord_functional"]["key"], "fused":fusion["key"]}
            path = output / f"{index:03d}_evidence.json"
            path.write_text(json.dumps(fusion, ensure_ascii=False, indent=2), encoding="utf-8")
            inputs = {k:signature(p) for k,p in paths.items()}
            if row.get("audio"):
                inputs["audio"] = signature(manifest.parent / row["audio"])
            songs.append(dict(id=identifier, reference=reference, predictions=predictions,
                              confusions={k:confusion(reference,v) for k,v in predictions.items()},
                              status=fusion["status"], needs_review=fusion["needs_review"], conflict=fusion["conflict"],
                              gate_checks=fusion["gate_checks"], inputs=inputs, evidence_file=path.name,
                              cnn_models=cnn.get("model_files", []),
                              pitched_coverage=evidence["raw_chord_functional"]["pitched_coverage"],
                              key_candidate_coverage={k:evidence[k].get("complete_24",False) for k in ("madmom_cnn","functional_joint","raw_chord_functional")}))
        except Exception as exc:
            failures.append(dict(id=identifier, error=f"{type(exc).__name__}: {exc}"))
    summary = {}
    for method in ("cnn", "functional_joint", "raw_harmonic", "fused"):
        counts = {kind:sum(s["confusions"][method] == kind for s in songs) for kind in ("exact","relative_major_minor","dominant_fourth","other","missing")}
        summary[method] = dict(**counts, evaluated=len(songs), total_requested=len(rows),
                               accuracy=counts["exact"]/len(songs) if songs else None,
                               resolved_wrong=sum(s["confusions"][method]!="exact" and s["status"]=="resolved" for s in songs) if method=="fused" else None)
    result = dict(schema="global_key_fusion_evaluation_v1", dataset_kind=dataset_kind,
                  independent_validation_completed=dataset_kind=="held-out" and not failures and bool(songs),
                  split_is_caller_attested=True, upstream_pretraining_overlap="unknown",
                  manifest=signature(manifest), policy=asdict(POLICY),
                  code=[signature(Path(__file__).parent / p) for p in ("key_fusion.py","detect_key_timeline.py","chord_studio.py","evaluate_global_key_fusion.py")],
                  summary=summary, songs=songs, failures=failures,
                  unresolved=sum(s["status"]=="ambiguous" for s in songs),
                  unresolved_conflicts=sum(s["status"]=="ambiguous" and s["conflict"]["detected"] for s in songs),
                  regressions=[s["id"] for s in songs if s["confusions"]["cnn"]=="exact" and s["confusions"]["fused"]!="exact"],
                  improvements=[s["id"] for s in songs if s["confusions"]["cnn"]!="exact" and s["confusions"]["fused"]=="exact"],
                  elapsed_seconds=time.perf_counter()-started, acoustic_inference=False, training=False,
                  confidence_note="Resolved means the engineering gate passed; NOT calibrated confidence.",
                  timing_note="Each method predicts the same full-song key; raw labels clipped to neutral audio duration; no reference timing input.")
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-kind", choices=["regression","development","held-out"], default="regression")
    args = parser.parse_args()
    result = evaluate(args.manifest, args.output, args.dataset_kind)
    print(json.dumps({k:result[k] for k in ("summary","failures","unresolved","regressions")}, indent=2))
    return int(bool(result["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
