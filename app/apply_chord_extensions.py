"""Apply a trusted local seventh-head artifact to audio and an upstream LAB.

Example: python app/apply_chord_extensions.py song.mp3 upstream.lab
         --model head.joblib -o song_extensions.lab
Then pass the output to analyze_with_madmom.py --madmom-lab.
The artifact's frozen threshold is used; this command does not tune a model.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import tempfile

import joblib
import mir_eval
import numpy as np

from chord_extension_model import FEATURE_DIMENSIONS, extend_rows, features_for_rows, normalize_feature_schema
from chord_recognizer import SAMPLE_RATE, decode_audio
from evaluate_chords import TimedChord, harte_chord
from madmom_bridge import file_identity


def read_upstream(path):
    raw = path.read_bytes()
    lines = raw.decode("utf-8-sig").splitlines(keepends=True)
    rows, indices = [], []
    previous_end = 0.
    for index, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"LAB line {index+1} must contain start end chord")
        start, end = float(fields[0]), float(fields[1])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start or start < previous_end-1e-9:
            raise ValueError(f"LAB line {index+1} has invalid, unordered or overlapping times")
        rows.append(TimedChord(start,end,fields[2]))
        indices.append(index)
        previous_end = end
    if not rows:
        raise ValueError("Upstream LAB contains no chord intervals")
    return raw, lines, rows, indices


def verify_preservation(before, after):
    if len(before) != len(after):
        raise ValueError("Extension head changed the number of upstream intervals")
    for source, result in zip(before,after):
        if (source.start,source.end) != (result.start,result.end):
            raise ValueError("Extension head changed an upstream time boundary")
        if source.chord == result.chord:
            continue
        left, right = harte_chord(source.chord), harte_chord(result.chord)
        if not mir_eval.chord.root([left],[right])[0] == 1 or not mir_eval.chord.triads([left],[right])[0] == 1:
            raise ValueError("Extension head changed an upstream root or triad")
        if "/" in source.chord:
            lr, _, lb = mir_eval.chord.encode(left)
            rr, _, rb = mir_eval.chord.encode(right)
            if "/" not in result.chord or (int(lr)+int(lb))%12 != (int(rr)+int(rb))%12:
                raise ValueError("Extension head changed an explicit upstream bass")


def atomic_write(path, content):
    """Replace one completed file atomically, never expose a partial LAB/JSON."""
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(descriptor,"wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def same_file(left, right):
    return left.resolve() == right.resolve() or (left.exists() and right.exists() and os.path.samefile(left,right))


def apply_model(audio, upstream, model, output):
    audio, upstream, model, output = (Path(path).resolve() for path in (audio,upstream,model,output))
    evidence_path = output.with_name(output.name+".evidence.json")
    provenance_path = output.with_name(output.name+".provenance.json")
    for source in (audio,upstream,model):
        if not source.is_file():
            raise ValueError(f"Input file not found: {source}")
        if any(same_file(target,source) for target in (output,evidence_path,provenance_path)):
            raise ValueError("Outputs must not overwrite an input audio, LAB or model")
    signatures = {name:file_identity(path) for name,path in [("audio",audio),("upstream_lab",upstream),("model",model)]}
    code_names = ["apply_chord_extensions.py","chord_extension_model.py","acoustic_chord_model.py",
                  "audio_features.py","chord_recognizer.py","chord_studio.py","evaluate_chords.py"]
    code_signatures = [file_identity(Path(__file__).with_name(name)) for name in code_names]
    raw, lines, rows, line_indices = read_upstream(upstream)
    artifact = joblib.load(model)  # Explicit user-supplied trusted local artifact.
    try:
        estimator_code = Path(inspect.getfile(type(artifact.get("model")))).resolve()
        if estimator_code.is_file() and str(estimator_code) not in {item["path"] for item in code_signatures}:
            code_signatures.append(file_identity(estimator_code))
    except (TypeError, OSError):
        pass
    schema = normalize_feature_schema(artifact.get("feature_schema"))
    threshold = float(artifact["threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Artifact threshold must be finite and in [0, 1]")
    decoded_duration = None
    if threshold == 1:
        features = np.zeros((len(rows),FEATURE_DIMENSIONS[schema]),dtype=np.float32)
    else:
        samples = decode_audio(audio,sample_rate=SAMPLE_RATE,offset=0)
        decoded_duration = len(samples)/SAMPLE_RATE
        # madmom LAB uses 100ms frames and may round the last endpoint upward.
        # Preserve that source coordinate rather than trim its raw timeline.
        if rows[-1].end > decoded_duration+.100001:
            raise ValueError("Upstream LAB extends beyond the supplied audio")
        features, _ = features_for_rows(samples,rows,feature_schema=schema)
    result, evidence = extend_rows(rows,features,artifact,feature_schema=schema)
    verify_preservation(rows,result)
    changes = sum(left.chord != right.chord for left,right in zip(rows,result))
    content = raw
    if changes:
        for index, before, after in zip(line_indices,rows,result):
            if before.chord == after.chord:
                continue
            old = lines[index]
            fields = old.split()
            ending = "\r\n" if old.endswith("\r\n") else "\n" if old.endswith("\n") else ""
            lines[index] = f"{fields[0]} {fields[1]} {after.chord}{ending}"
        content = "".join(lines).encode("utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8")
    for name, path in [("audio",audio),("upstream_lab",upstream),("model",model)]:
        if file_identity(path)["sha256"] != signatures[name]["sha256"]:
            raise ValueError(f"{name} changed during application")
    for signature in code_signatures:
        if file_identity(Path(signature["path"]))["sha256"] != signature["sha256"]:
            raise ValueError("Extension implementation changed during application")
    provenance = dict(schema="fixed_root_seventh_application_v1",**signatures,
        feature_schema=schema,feature_dimension=FEATURE_DIMENSIONS[schema],threshold=threshold,
        audio_offset=0,sample_rate=SAMPLE_RATE,decoded_duration=decoded_duration,
        source_end_rounding_tolerance_seconds=.100001,
        source_end_beyond_audio_seconds=max(0,rows[-1].end-decoded_duration) if decoded_duration is not None else None,
        audio_decoded=threshold<1,mode="identity" if threshold==1 else "extension_head",
        score_kind="uncalibrated_extension_evidence",reference_labels_used=False,
        interval_count=len(rows),change_count=changes,root_triad_times_explicit_bass_preserved=True,
        output_lab={"path":str(output),"bytes":len(content),"sha256":hashlib.sha256(content).hexdigest()},
        code=code_signatures,
        evidence_file=str(evidence_path))
    encoded = lambda value:(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode("utf-8")
    evidence_bytes = encoded({"feature_schema":schema,"threshold":threshold,
                              "score_kind":"uncalibrated_extension_evidence","intervals":evidence})
    provenance["evidence_sha256"] = hashlib.sha256(evidence_bytes).hexdigest()
    provenance_bytes = encoded(provenance)
    atomic_write(evidence_path,evidence_bytes)
    atomic_write(provenance_path,provenance_bytes)
    atomic_write(output,content)
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio",type=Path)
    parser.add_argument("upstream",type=Path,help="Original upstream LAB; never overwritten")
    parser.add_argument("--model",type=Path,required=True,help="Trusted local joblib artifact with a frozen threshold")
    parser.add_argument("-o","--output",type=Path,required=True,help="New LAB output; provenance and evidence are written beside it")
    args = parser.parse_args(argv)
    result = apply_model(args.audio,args.upstream,args.model,args.output)
    print(f"LAB: {result['output_lab']['path']} | {result['change_count']} changes | {result['mode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
