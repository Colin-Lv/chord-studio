"""Apply an explicitly selected v2 experimental head to cached inputs.

No reference, training, extraction fallback, or workbench update is supported.
The artifact fixes the selection policy; the allowed-first policy is a separate
v1-score diagnostic and is not silently combined with this semantic experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from candidate_data_v2 import load_inputs
from experiment_protocol_v2 import signature, write_new_json
from multisource_features_v2 import automatic_candidate_features
from multisource_model_v2 import load_artifact, score_candidates
from multisource_root_guard_v2 import frozen_v1_full_head_from_scores
from multisource_root_guard import preserve_raw_root


def apply_cached(row, model_path, output):
    output=Path(output)
    sidecar=output.with_suffix(output.suffix+".provenance.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Choose a new experiment prediction path")
    started=time.perf_counter()
    artifact,contract=load_artifact(model_path)
    if artifact.get("selection_policy")!="legacy_postselection_root_guard":
        raise ValueError("Unknown v2 selection policy")
    song=load_inputs(row,references=False,expected_nnls=contract["nnls_contract"])
    features,groups=automatic_candidate_features(song["raw"],song["sources"],song["evidence"])
    full=frozen_v1_full_head_from_scores(song["raw"],groups,score_candidates(artifact,features),artifact["margin"])
    predicted,decisions=preserve_raw_root(song["raw"],full)
    output.parent.mkdir(parents=True,exist_ok=True)
    # JSON sidecar preserves missing/null labels; the LAB interchange uses X
    # for missing labels because its grammar has no separate missing token.
    with output.open("x",encoding="utf-8") as stream:
        stream.write("".join(f"{r.start!r}\t{r.end!r}\t{r.chord if r.chord is not None else 'X'}\n" for r in predicted))
    record=dict(schema="cached_multisource_prediction_v2",model=signature(model_path),
        contract=signature(Path(model_path).with_suffix(".contract.json")),prediction=signature(output),
        inputs=song["input_identities"],events=[dict(start=r.start,end=r.end,chord=r.chord) for r in predicted],
        selection_policy=artifact["selection_policy"],guard=decisions,
        reference_labels_read=False,fit_calls=0,acoustic_inference_calls=0,
        diagnostic_only=artifact["diagnostic_only"],calibrated=artifact["calibrated"],
        source_identity_limitation="Input hashes verify bytes; legacy LAB generation/audio pairing is not fully authenticated",
        elapsed_seconds=time.perf_counter()-started)
    write_new_json(sidecar,record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("audio","raw","chordino","largevocab","notes","model","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    row=dict(id=args.audio.stem,audio=args.audio,raw_madmom=args.raw,chordino=args.chordino,
             largevocab=args.largevocab,notes=args.notes)
    result=apply_cached(row,args.model,args.output)
    print(json.dumps(dict(prediction=result["prediction"],diagnostic_only=result["diagnostic_only"]),ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
