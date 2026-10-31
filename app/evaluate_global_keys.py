"""Evaluate dedicated global key inference without reference beats or chords."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import mir_eval

from chord_studio import normalize_key_label
from downbeat_bridge import DEFAULT_MADMOM_PYTHON
from key_bridge import run_madmom_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--madmom-python", type=Path, default=DEFAULT_MADMOM_PYTHON)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig")))
    if args.limit:
        rows = rows[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(protocol="audio_only_global_major_minor_key", upstream_pretraining_overlap="unknown",
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(), songs=[])
    for row in rows:
        audio = (args.manifest.parent / row["audio"]).resolve()
        began = time.perf_counter()
        prediction = run_madmom_key(audio, args.madmom_python)
        # Only after inference, consult the reference for scoring.
        reference = normalize_key_label(row["key"])
        def mirex(label):
            return label[:-1] + " minor" if label.endswith("m") else label + " major"
        score = mir_eval.key.weighted_score(mirex(reference), mirex(prediction["key"]))
        record = dict(id=row["id"], source=str(audio), reference=reference, prediction=prediction["key"],
                      exact=score == 1., weighted_score=score, elapsed_seconds=time.perf_counter()-began)
        report["songs"].append(record)
        report["summary"] = {"songs": len(report["songs"]),
            "exact_accuracy": sum(r["exact"] for r in report["songs"]) / len(report["songs"]),
            "mean_weighted_score": sum(r["weighted_score"] for r in report["songs"]) / len(report["songs"])}
        (args.output / (row["id"] + "_key.json")).write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
