"""Read-only frozen harmonic-key adapter. No audio decoding or chord inference."""
import argparse
import json
from pathlib import Path

from app.web.storage import atomic_json, signature


def contributions(prediction):
    """Arithmetic explanation of the frozen scores; never changes ranking."""
    if not prediction.get('key'):return {}
    policy=prediction['policy'];result={}
    for tonic,candidates in prediction['mode_candidates_by_tonic'].items():
        centres={};modes={}
        for row in candidates:
            f=row['components']
            common={name:policy[weight]*f[name] for name,weight in dict(root_fit='root_weight',quality_fit='quality_weight',cadence='cadence_weight',pair='pair_weight',loop='loop_weight',ending='ending_weight',phrase_ending='phrase_ending_weight',entry='entry_weight').items()}
            centre={**common,'tonic_duration':policy['tonic_duration_weight']*f['tonic_root_duration']}
            mode={**common,'tonic_quality_alignment':policy['mode_tonic_quality_weight']*f['tonic_quality_alignment']}
            if abs(sum(centre.values())-row['centre_score'])>1e-10 or abs(sum(mode.values())-row['score'])>1e-10:
                raise ValueError('Contribution adapter differs from frozen full-stage score')
            centres[row['mode']]=centre;modes[row['mode']]=mode
        centre_mode=max(centres,key=lambda mode:sum(centres[mode].values()))
        local=policy['local_weight']*prediction.get('window_evidence',{}).get('tonic_shares',{}).get(tonic,0)
        score=sum(centres[centre_mode].values())+local
        original=next(row['score'] for row in prediction['tonic_candidates'] if row['tonic']==tonic)
        if abs(score-original)>1e-10:raise ValueError('Tonic score reconstruction mismatch')
        result[tonic]=dict(centre_by_mode=centres,conditional_mode_by_mode=modes,centre_mode=centre_mode,
                          local_tonic_contribution=local,tonic_score=score,
                          note='Conditional mode uses its own ranking; centre maximum may use a different mode. All scores uncalibrated.')
    return result


def produce(audio, raw, neutral, contract, output):
    # Import in the existing model interpreter, never the HTTP interpreter.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_harmonic_key_v2 import load_contract
    from harmonic_key_v2 import Policy, analyze
    from key_fusion import read_raw_lab
    frozen = load_contract(contract)
    grid = json.loads(Path(neutral).read_text(encoding="utf-8"))
    identities = {name: signature(path) for name, path in
                  dict(audio=audio, raw=raw, neutral=neutral, contract=contract).items()}
    provenance = json.loads(Path(str(raw) + ".provenance.json").read_text(encoding="utf-8"))
    if provenance["audio"]["sha256"] != identities["audio"]["sha256"] or provenance["lab"]["sha256"] != identities["raw"]["sha256"]:
        raise ValueError("Raw LAB/audio provenance mismatch")
    if provenance.get("audio_offset", 0) != 0 or provenance.get("offset", 0) != 0:
        raise ValueError("Nonzero raw offset requires an explicit window adapter")
    if abs(grid["duration"] - provenance["duration"]) > 1 / 22050 + 1e-6:
        raise ValueError("Neutral/raw time support mismatch")
    prediction = analyze(read_raw_lab(raw), grid["duration"], grid.get("bars", []), Policy(**frozen["policy"]))
    atomic_json(output, dict(schema="harmonic_key_review_evidence_v1", inputs=identities,
                            adapter=signature(__file__), prediction=prediction,
                            weighted_contributions=contributions(prediction),
                            score_kind="uncalibrated", automatic_override_enabled=False,
                            new_acoustic_inference=False, new_training=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audio", "raw", "neutral", "contract", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    produce(**vars(args))
