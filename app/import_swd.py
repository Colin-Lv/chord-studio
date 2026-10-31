"""Import a pre-frozen SWD recording list without changing audio or harmony.

References are expert score analyses transferred through manual bar anchors and
DTW, not independently annotated frame-by-frame listening labels. No model runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from evaluate_chords import harte_chord


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def safe_member(root: Path, member: str) -> Path:
    root = root.resolve()
    path = (root / member).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Archive member escapes the dataset directory")
    return path


def convert_reference(path: Path, column: str = "shorthand") -> tuple[str, dict]:
    """Only normalize label syntax; retain interval boundaries and unknowns."""
    if column != "shorthand":
        raise ValueError("The frozen protocol requires original shorthand harmony")
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline=""), delimiter=";"))
    if not rows or not {"start", "end", column} <= set(rows[0]):
        raise ValueError("SWD reference must contain start/end/shorthand columns")
    output, gaps, previous, unknown = [], 0.0, None, 0
    for index, row in enumerate(rows):
        start, end = float(row["start"]), float(row["end"])
        if not math.isfinite(start + end) or start < 0 or end <= start:
            raise ValueError(f"Invalid interval in reference row {index + 1}")
        if previous is not None:
            if start < previous - 1e-6:
                raise ValueError(f"Overlapping reference row {index + 1}")
            gaps += max(0.0, start - previous)
        original = row[column].strip()
        if not original:
            raise ValueError(f"Empty harmony label in reference row {index + 1}")
        label = harte_chord(original)
        if label == "X" and original != "X":
            raise ValueError(f"Unsupported harmony syntax in reference row {index + 1}")
        unknown += label == "X"
        # Decimal timestamp text stays unchanged; no alignment or trimming.
        output.append(f"{row['start'].strip()} {row['end'].strip()} {label}\n")
        previous = end
    return "".join(output), {"intervals": len(rows), "explicit_unknown_intervals": unknown,
                             "internal_gap_seconds": gaps, "timing_changed": False}


def prepare(dataset: Path, protocol_path: Path, output: Path, cohort: str = "all") -> Path:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema") != "swd_external_validation_reservation_v1":
        raise ValueError("A frozen SWD validation reservation is required")
    wanted = {"primary": "primary_public_domain_historical",
              "secondary": "secondary_modern_noncommercial"}.get(cohort)
    selected = [item for item in protocol["items"] if wanted is None or item["cohort"] == wanted]
    if not selected or any(item["role"] != "external_validation_only" for item in selected):
        raise ValueError("Only pre-reserved external validation recordings are eligible")
    manifest_path = output / f"manifest.{cohort}.csv"
    if manifest_path.exists():
        raise ValueError("Refusing to replace an existing frozen import manifest")
    prepared, manifest, records = [], [], []
    for item in selected:
        audio = safe_member(dataset, item["audio_archive_path"])
        reference = safe_member(dataset, item["reference_archive_path"])
        if not audio.is_file() or not reference.is_file():
            raise ValueError(f"Missing source files for {item['id']}")
        lab = output / "references" / f"{item['id']}.lab"
        if lab.exists():
            raise ValueError("Refusing to replace a previously imported reference")
        text, validation = convert_reference(reference, item["reference_column"])
        prepared.append((lab, text))
        manifest.append(dict(id=item["id"], dataset="swd_v2.1", split="external_test",
                             audio=str(audio.resolve()), chord_reference=str(lab.resolve()),
                             composition_group=item["composition_group"], performance=item["performance"],
                             cohort=item["cohort"], license=item["license"],
                             reference_kind="score_derived_expert_harmony_manual_bars_DTW",
                             notes="Reserved validation only; score-derived expert harmony with manual bars/DTW; not frame-by-frame listening gold; no reference key/grid supplied"))
        records.append(dict(id=item["id"], audio=identity(audio), original_reference=identity(reference),
                            imported_reference=str(lab.resolve()), validation=validation))
    output.mkdir(parents=True, exist_ok=True)
    for path, text in prepared:
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
    for record, (path, _) in zip(records, prepared):
        record["imported_reference"] = identity(path)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(manifest[0]))
    writer.writeheader()
    writer.writerows(manifest)
    manifest_path.write_text(buffer.getvalue(), encoding="utf-8")
    report = dict(schema="swd_format_only_import_v1", protocol=identity(protocol_path),
                  manifest=identity(manifest_path), cohort=cohort, recordings=len(selected),
                  models_run=False, predictions_read=False, audio_modified=False,
                  reference_labels_used_as_model_inputs=False, items=records)
    (output / f"import_report.{cohort}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--cohort", choices=["primary", "secondary", "all"], default="all")
    args = parser.parse_args()
    print(prepare(args.dataset, args.protocol, args.output, args.cohort))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
