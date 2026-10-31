"""Freeze grouped fit/development/calibration/sealed-test inputs before tuning.

The splitter uses supplied provenance, never inferred song names or references.
Known regression works/artists/recordings remain a separate connected component.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

SPLIT_SCHEMA = "work_artist_recording_grouped_splits_v2"
CONFIG_SCHEMA = "masked_candidate_selection_configuration_v2"
PARTITIONS = ("fit", "development", "calibration", "sealed_test", "known_regression")
HEADS = ("root", "triad", "tetrad", "bass", "ext9", "ext11", "ext13")
INPUT_NAMES = ("audio", "raw_madmom", "chordino", "largevocab", "notes", "reference")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path))


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def validate_grouped_rows(rows, require_complete=True):
    required = ("id", "work_id", "artist_id", "recording_id", "domain", "split", "grouping_evidence")
    seen, assignments = set(), {}
    counts = defaultdict(int)
    for row in rows:
        if any(not row.get(key) for key in required):
            raise ValueError("Explicit work/artist/recording/domain/grouping evidence and split are required")
        if row["id"] in seen:
            raise ValueError("Duplicate recording ID")
        seen.add(row["id"])
        if row["split"] not in PARTITIONS:
            raise ValueError("Unknown partition")
        counts[row["split"]] += 1
        keys = [(field, str(row[field])) for field in ("work_id", "artist_id", "recording_id")]
        if row.get("audio_sha256"):
            keys.append(("audio_sha256", row["audio_sha256"]))
        for key in keys:
            if key in assignments and assignments[key] != row["split"]:
                raise ValueError(f"Cross-partition grouping leakage: {key}")
            assignments[key] = row["split"]
        if row.get("known_regression") and row["split"] != "known_regression":
            raise ValueError("Known regression inputs cannot become new sealed/development data")
    if require_complete and any(counts[part] == 0 for part in PARTITIONS[:-1]):
        raise ValueError("Fit, development, calibration and sealed_test must each be nonempty")
    if require_complete and any(len({r["domain"] for r in rows if r["split"] == split}) < 2
                                for split in ("fit", "development")):
        raise ValueError("Cross-style fit and development each require at least two explicitly documented domains")
    return dict(counts)


def grouped_assignment(rows, seed=20261003):
    """Deterministic component assignment, for review before freezing.

    Entire transitive components share one split. This helper does not promise
    stratification: validate_grouped_rows must accept the reviewed result.
    """
    rows = [dict(row) for row in rows]
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, row in enumerate(rows):
        for field in ("work_id", "artist_id", "recording_id", "audio_sha256"):
            if not row.get(field):
                if field != "audio_sha256":
                    raise ValueError("Cannot invent missing grouping metadata")
                continue
            key = field, str(row[field])
            if key in owners:
                parent[find(i)] = find(owners[key])
            owners[key] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    eligible = []
    for indexes in groups.values():
        if any(rows[i].get("known_regression") for i in indexes):
            for i in indexes:
                rows[i]["split"] = "known_regression"
        else:
            key = "|".join(sorted(rows[i]["id"] for i in indexes))
            eligible.append((hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), indexes))
    if len(eligible) < 4:
        raise ValueError("Need at least four independent non-regression connected groups")
    eligible.sort()
    n = len(eligible)
    development = max(1, round(.2*n))
    calibration = max(1, round(.1*n))
    test = max(1, round(.1*n))
    fit = n-development-calibration-test
    if fit < 1:
        raise ValueError("Too few independent fit groups")
    assignments = ["fit"]*fit+["development"]*development+["calibration"]*calibration+["sealed_test"]*test
    for (_, indexes), split in zip(eligible, assignments):
        for i in indexes:
            rows[i]["split"] = split
    validate_grouped_rows(rows, require_complete=False)
    return rows


def load_split_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("schema") != SPLIT_SCHEMA or protocol.get("status") != "frozen_before_configuration":
        raise ValueError("A frozen grouped v2 split protocol is required")
    validate_grouped_rows(protocol["rows"])
    for row in protocol["rows"]:
        identities = row.get("input_identities", {})
        if set(identities) != set(INPUT_NAMES):
            raise ValueError("Every frozen row needs all six input identities")
        if row.get("audio_sha256") != identities["audio"]["sha256"]:
            raise ValueError("Audio grouping hash and frozen input identity disagree")
    return protocol


def freeze_input_rows(rows, base):
    """Hash sealed references as opaque bytes; never parse their labels."""
    result=[]
    for original in rows:
        row=dict(original)
        identities={}
        for name in INPUT_NAMES:
            if not row.get(name):
                raise ValueError(f"Missing {name} input for {row.get('id')}")
            path=(Path(base)/row[name]).resolve()
            identities[name]=signature(path)
            row[name]=str(path)
        if row.get("audio_sha256") not in (None, identities["audio"]["sha256"]):
            raise ValueError("Supplied audio grouping hash is stale")
        row.update(input_identities=identities,audio_sha256=identities["audio"]["sha256"])
        result.append(row)
    return result


def validate_configuration(config, protocol_path):
    import math
    if config.get("schema") != CONFIG_SCHEMA or config.get("split_protocol") != signature(protocol_path):
        raise ValueError("Selection configuration must bind the previously frozen split protocol")
    weights = config.get("weights", {})
    if set(weights) != set(HEADS) or any(not math.isfinite(v) or v < 0 for v in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Specify finite nonnegative weights for all seven supervised fields")
    if not math.isfinite(config.get("margin", float("nan"))) or not 0 <= config["margin"] <= 1:
        raise ValueError("Margin must be finite in [0,1]")
    if not config.get("reference_coverage_policy"):
        raise ValueError("Explicit reference coverage policy required")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-splits")
    freeze.add_argument("--manifest", type=Path, required=True, help="JSON rows with explicit grouped split metadata")
    freeze.add_argument("--output", type=Path, required=True)
    config = commands.add_parser("freeze-config")
    config.add_argument("--protocol", type=Path, required=True)
    config.add_argument("--configuration", type=Path, required=True, help="JSON weights, margin and coverage policy")
    config.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze-splits":
        rows = freeze_input_rows(json.loads(args.manifest.read_text(encoding="utf-8")),args.manifest.parent)
        counts = validate_grouped_rows(rows)
        write_new_json(args.output, dict(schema=SPLIT_SCHEMA, status="frozen_before_configuration",
            created_utc=datetime.now(timezone.utc).isoformat(), manifest=signature(args.manifest),
            counts=counts, rows=rows, sealed_reference_labels_read=False))
    else:
        load_split_protocol(args.protocol)
        config = json.loads(args.configuration.read_text(encoding="utf-8"))
        config.update(schema=CONFIG_SCHEMA, split_protocol=signature(args.protocol),
                      created_utc=datetime.now(timezone.utc).isoformat())
        validate_configuration(config, args.protocol)
        write_new_json(args.output, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
