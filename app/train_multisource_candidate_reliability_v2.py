"""Train masked v2 HGB only after grouped development/calibration are frozen.

This entry never reads sealed-test annotations or changes frozen v1. It emits
an experimental, uncalibrated artifact. A separate calibrated selection stage
is required before any default deployment decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from candidate_data_v2 import load_inputs, standard_reference
from candidate_diagnostics_v2 import field_metrics, change_diagnostics, summarize_song_metrics, candidate_diagnostics
from candidate_targets_v2 import ReferenceCoveragePolicy, candidate_targets, aggregate_utility, annotation_coverage
from evaluate_chords import evaluate
from experiment_protocol_v2 import load_split_protocol, validate_configuration, signature, write_new_json
from multisource_features_v2 import automatic_candidate_features
from multisource_model_v2 import fit_hgb, save_artifact, load_artifact, score_candidates
from multisource_root_guard_v2 import frozen_v1_full_head_from_scores
from apply_multisource_candidates import load_frozen_artifact, validate_note_contract
from multisource_root_guard import preserve_raw_root
from train_multisource_candidate_reliability import automatic_candidate_features as v1_features, predict_song as v1_predict


def masked_training_arrays(songs, weights, policy):
    features,targets,sample_weights=[],[],[]
    for song in songs:
        local_weights,local_y=[],[]
        for event,group in zip(song["raw"],song["groups"]):
            target=candidate_targets(event,group["pool"],song["reference"],policy)
            utility=aggregate_utility(target,weights)
            local_y.extend(utility.values)
            # Documented reliability exposure, not a negative label: each song
            # has equal total weight, intervals scale by observed field duration.
            local_weights.extend((event.end-event.start)*utility.coverage/max(1,len(utility.values)))
        y,w=np.asarray(local_y),np.asarray(local_weights)
        keep=np.isfinite(y)&(w>0)
        if not keep.any():
            continue
        features.append(song["features"][keep]); targets.append(y[keep]); sample_weights.append(w[keep]/w[keep].sum())
    if not features:
        raise ValueError("No covered reference fields available for supervised training")
    return np.concatenate(features),np.concatenate(targets),np.concatenate(sample_weights)


def selection_utility(fields, weights):
    observed=sum(weights[head]*value["valid_seconds"] for head,value in fields.items())
    return dict(value=sum(weights[head]*value["correct_seconds"] for head,value in fields.items())/observed
                if observed else None,weighted_observed_seconds=observed,
                definition="Sum of weighted field-correct seconds / weighted field-valid seconds")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--configuration",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--baseline-v1-model",type=Path,required=True,
                        help="Unmodified frozen v1 model plus contract, for the paired fifth baseline")
    args=parser.parse_args()
    protocol=load_split_protocol(args.protocol)
    config=validate_configuration(json.loads(args.configuration.read_text(encoding="utf-8")),args.protocol)
    policy=ReferenceCoveragePolicy(**config["reference_coverage_policy"])
    baseline_v1,baseline_contract=load_frozen_artifact(args.baseline_v1_model)
    baseline_seen={row["sha256"] for row in baseline_contract["training_audio"]}
    if any(row["split"] in {"development","calibration","sealed_test"} and
           row["audio_sha256"] in baseline_seen for row in protocol["rows"]):
        raise ValueError("Frozen v1 baseline saw an evaluation recording; use an independent split")
    if args.output.exists():
        raise FileExistsError("Use a new v2 experiment directory")
    args.output.mkdir(parents=True)
    write_new_json(args.output/"protocol.json",dict(schema="masked_hgb_experiment_v2",split=signature(args.protocol),
        configuration=signature(args.configuration),target="Masked harmonic lower targets + explicit bass + degree extensions; not standard metric replacement",
        frozen_v1=signature(args.baseline_v1_model),
        selection_objective="Configured weighted masked field accuracy; compare standard root/triad/tetrad/inversion and regressions separately",
        calibration_status="Reserved; no probability calibration or threshold search performed by this trainer",
        sample_weight="Observed weighted-field duration/candidate_count, normalized per song",
        selection_policy="Unchanged v1 post-selection root guard; E3 allowed-first is a separate factor",
        sealed_reference_labels_parsed=False,acoustic_inference=False))
    start=time.perf_counter(); cpu=time.process_time(); songs=[]; contract=None
    for row in protocol["rows"]:
        if row["split"] not in {"fit","development"}:
            continue
        song=load_inputs(row,references=True,expected_nnls=contract)
        validate_note_contract(song["evidence"],baseline_contract,song["input_identities"]["audio"]["sha256"])
        contract=song["nnls_contract"]
        song["features"],song["groups"]=automatic_candidate_features(song["raw"],song["sources"],song["evidence"])
        song["split"]=row["split"]
        songs.append(song)
    fit=[song for song in songs if song["split"]=="fit"]
    x,y,w=masked_training_arrays(fit,config["weights"],policy)
    model=fit_hgb(x,y,w)
    artifact_path=args.output/"masked_head_v2.joblib"
    save_artifact(artifact_path,model,objective=dict(schema="masked_candidate_targets_v2",weights=config["weights"],policy=policy.to_dict()),
        margin=config["margin"],nnls_contract=contract,selection_config=config,
        provenance=dict(split=signature(args.protocol),configuration=signature(args.configuration),trainer=signature(__file__),
            inputs=[dict(id=s["id"],**s["input_identities"]) for s in fit]),diagnostic_only=True)
    artifact,_=load_artifact(artifact_path)
    scored=[]
    for song in songs:
        if song["split"]!="development":
            continue
        full=frozen_v1_full_head_from_scores(song["raw"],song["groups"],score_candidates(artifact,song["features"]),artifact["margin"])
        prediction,_=preserve_raw_root(song["raw"],full)
        old_x,old_groups=v1_features(song["raw"],[song["sources"]["chordino"],song["sources"]["largevocab"]],song["evidence"])
        old_prediction=v1_predict(dict(raw=song["raw"],features=old_x,groups=old_groups),baseline_v1["method"],baseline_v1["model"],baseline_v1["margin"])
        guarded,_=preserve_raw_root(song["raw"],old_prediction)
        systems=dict(raw=song["raw"],chordino=song["sources"]["chordino"],largevocab=song["sources"]["largevocab"],frozen_v1=guarded,masked_v2=prediction)
        fields={key:field_metrics(song["reference"],value,policy) for key,value in systems.items()}
        prediction_dir=args.output/"predictions"; prediction_dir.mkdir(exist_ok=True)
        predictions={}
        for name,values in (("frozen_v1",guarded),("masked_v2",prediction)):
            path=prediction_dir/(song["id"]+"_"+name+".lab")
            with path.open("x",encoding="utf-8") as stream:
                stream.write("".join(f"{r.start!r}\t{r.end!r}\t{r.chord}\n" for r in values))
            predictions[name]=signature(path)
        row=dict(id=song["id"],metrics={key:evaluate(standard_reference(song["reference"]),value) for key,value in systems.items()},
            field_metrics=fields,selection_objective={name:selection_utility(value,config["weights"]) for name,value in fields.items()},
            predictions=predictions,candidates=candidate_diagnostics(song["raw"],song["sources"],song["reference"],song["groups"],policy),
            changes=change_diagnostics(song["reference"],song["raw"],prediction,policy),
            changes_vs_v1=change_diagnostics(song["reference"],guarded,prediction,policy),
            annotation_coverage=annotation_coverage(song["reference"],policy),inputs=song["input_identities"])
        scored.append(row)
    peak=None
    try:
        import psutil
        peak=getattr(psutil.Process().memory_info(),"peak_wset",None)
    except ImportError:
        pass
    write_new_json(args.output/"report.json",dict(status="trained_not_calibrated_or_promoted",model=signature(artifact_path),
        training_rows=len(x),training_songs=len(fit),songs=scored,summary=summarize_song_metrics(scored),
        training_coverage=[dict(id=s["id"],coverage=annotation_coverage(s["reference"],policy)) for s in fit],
        regressions=[dict(id=s["id"],metric=m,delta=s["metrics"]["masked_v2"][m]-s["metrics"]["frozen_v1"][m])
            for s in scored for m in ("triad_accuracy","tetrad_accuracy","tetrad_inv_accuracy")
            if s["metrics"]["masked_v2"][m]<s["metrics"]["frozen_v1"][m]-1e-12],
        cpu_seconds=time.process_time()-cpu,peak_working_set_bytes=peak,numeric_threads=2,
        elapsed_seconds=time.perf_counter()-start,acoustic_inference_calls=0,fit_calls=1,
        frozen_v1=signature(args.baseline_v1_model),
        unverified=["sealed test","calibration","default workbench performance"]))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
