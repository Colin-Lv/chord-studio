"""Compare complete, paired song evaluations without selecting songs by outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = (
    "root_accuracy", "triad_accuracy", "tetrad_accuracy",
    "tetrad_inv_accuracy", "prediction_coverage",
)


def read_complete(path: Path) -> tuple[dict, dict]:
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    tracks = report["tracks"]
    by_id = {track["id"]: track for track in tracks}
    if len(by_id) != len(tracks) or len(tracks) < 2:
        raise ValueError("Need at least two songs, with unique IDs")
    if any(track.get("status") != "complete" for track in tracks):
        raise ValueError("Both reports must contain only complete songs")
    if report.get("remaining_count", 0) or report.get("failure_count", 0):
        raise ValueError("Cannot silently drop incomplete/failed songs")
    return report, by_id


def compare(baseline_path: Path, final_path: Path, resamples: int, seed: int) -> dict:
    if resamples < 1000:
        raise ValueError("Use at least 1000 paired resamples")
    baseline_report, baseline = read_complete(baseline_path)
    final_report, final = read_complete(final_path)
    for field in ("scoring_parameters", "annotation_alignment"):
        if not baseline_report.get(field) or baseline_report[field] != final_report.get(field):
            raise ValueError(f"Scoring protocol differs or is missing: {field}")
    manifest_hash = baseline_report.get("manifest_signature", {}).get("sha256")
    if not manifest_hash or manifest_hash != final_report.get("manifest_signature", {}).get("sha256"):
        raise ValueError("Manifest identity differs or is missing")
    if set(baseline) != set(final):
        raise ValueError("Song IDs must match exactly; no intersect-and-drop comparison")
    ids = list(final)
    for track_id in ids:
        if Path(baseline[track_id]["audio"]).resolve() != Path(final[track_id]["audio"]).resolve():
            raise ValueError(f"Audio path differs for {track_id}")
        for metric in ("evaluated_seconds", "non_no_chord_seconds"):
            a = baseline[track_id]["chord_metrics"][metric]
            b = final[track_id]["chord_metrics"][metric]
            if not np.isfinite(a) or not np.isfinite(b) or abs(a - b) > 1e-6:
                raise ValueError(f"Reference scoring support differs: {track_id}, {metric}")
    indices = np.random.default_rng(seed).integers(0, len(ids), size=(resamples, len(ids)))
    rows = []
    for metric in METRICS:
        old = np.array([baseline[i]["chord_metrics"][metric] for i in ids], dtype=float)
        new = np.array([final[i]["chord_metrics"][metric] for i in ids], dtype=float)
        if not np.all(np.isfinite(old)) or not np.all(np.isfinite(new)):
            raise ValueError(f"Nonfinite metric: {metric}")
        differences = new - old
        interval = np.quantile(differences[indices].mean(axis=1), [.025, .975])
        per_song = [dict(id=i, baseline=float(a), final=float(b), difference=float(d))
                    for i, a, b, d in zip(ids, old, new, differences)]
        rows.append(dict(
            metric=metric, baseline=float(old.mean()), final=float(new.mean()),
            difference=float(differences.mean()), improved_songs=int((differences > 1e-8).sum()),
            worse_songs=int((differences < -1e-8).sum()), unchanged_songs=int((abs(differences) <= 1e-8).sum()),
            paired_song_bootstrap_95_interval=interval.tolist(),
            per_song_sorted_by_difference=sorted(per_song, key=lambda item: item["difference"]),
        ))
    return dict(
        baseline=str(baseline_path), final=str(final_path), songs=len(ids), rows=rows,
        input_sha256={"baseline": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                      "final": hashlib.sha256(final_path.read_bytes()).hexdigest()},
        interval_method=f"{resamples} paired song resamples, seed {seed}; conditional on this corpus; "
                        "does not establish across-artist or unseen-pretraining generalization",
        final_stage=final_report.get("protocol"),
        reference_identity_limit="Legacy reports lack per-song reference hashes; matching manifest, scoring protocol and scoring support do not prove historical annotation contents were identical.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("final", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    result = compare(args.baseline, args.final, args.resamples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"songs": result["songs"], "rows": [
        {k: v for k, v in row.items() if k != "per_song_sorted_by_difference"} for row in result["rows"]
    ]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
