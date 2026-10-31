"""Read extension metadata as data; verify only the explicitly supplied files."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

EXTENSION_SCHEMA = "fixed_root_seventh_application_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_identity(value):
    if not isinstance(value, dict):
        return None
    return {key: item for key, item in value.items()
            if key in {"path", "sha256", "bytes"} and isinstance(item, (str, int))}


def read_extension_provenance(audio: Path, lab: Path) -> dict | None:
    """Return verified derived-LAB provenance, or an explicit read/hash failure.

    Missing sidecars and other valid schemas are not classified as extensions.
    Model/upstream paths are copied as declared text and never opened or run.
    This record does not grant eligibility for the raw madmom cache.
    """
    audio, lab = Path(audio), Path(lab)
    sidecar = lab.with_name(lab.name + ".provenance.json")
    if not sidecar.exists():
        return None
    result = dict(sidecar={"path": str(sidecar.resolve())}, schema=None,
                  source_content_verified=False, verification_scope="current_audio_and_output_lab_sha256_only",
                  score_kind="uncalibrated_extension_evidence",
                  score_warning="Extension scores are uncalibrated and are not correctness probabilities.",
                  declared_model_and_upstream_identities_verified=False)
    try:
        raw = sidecar.read_bytes()
        result["sidecar"].update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        metadata = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(metadata, dict):
            raise ValueError("Sidecar JSON must be an object")
    except (OSError, UnicodeError, ValueError) as error:
        result["verification"] = f"Cannot read extension provenance: {type(error).__name__}"
        return result
    if metadata.get("schema") != EXTENSION_SCHEMA:
        return None
    result.update(schema=EXTENSION_SCHEMA,
                  model=_declared_identity(metadata.get("model")),
                  upstream_lab=_declared_identity(metadata.get("upstream_lab")),
                  audio=_declared_identity(metadata.get("audio")),
                  output_lab=_declared_identity(metadata.get("output_lab")),
                  feature_schema=metadata.get("feature_schema") if isinstance(metadata.get("feature_schema"), str) else None)
    for key in ("threshold", "change_count"):
        value = metadata.get(key)
        try:
            finite = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        except OverflowError:
            finite = False
        result[key] = value if finite else None
    failures = []
    for key, path, description in (("audio", audio, "audio"), ("output_lab", lab, "output LAB")):
        declared = result[key] or {}
        expected = declared.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            failures.append(f"Missing or invalid {description} SHA256")
            continue
        try:
            actual = _sha256(path)
        except OSError as error:
            failures.append(f"Cannot read current {description}: {type(error).__name__}")
            continue
        result.setdefault("verified_current_files", {})[key] = {"path": str(path.resolve()), "sha256": actual}
        if actual != expected.lower():
            failures.append(f"Current {description} SHA256 does not match sidecar")
    result["source_content_verified"] = not failures
    result["verification"] = "; ".join(failures) if failures else "Current audio and output LAB SHA256 match the extension sidecar"
    return result
