#!/usr/bin/env python
"""Song-held-out, acoustic-only candidate-stage ablation (oracle segmentation).

Candidate tables can contain human-corrected outputs and all-data phrase priors.
This benchmark excludes both, and compares free exact selection with keeping the
acoustic top-1 root. Existing candidate segmentation/key are still supplied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from evaluate_candidate_reranker import evaluate, group_key
from train_candidate_reranker import build_pipeline, filter_evaluable_rows, load_rows, positive_class_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    original = pd.read_csv(args.candidate_csv, encoding="utf-8-sig", low_memory=False)
    # Reference fields remain targets only; symbolic features may embed a prior
    # learned from all songs and must be rebuilt within a fold before use.
    allowed = original.candidate_source.str.contains(r"topn_rank\d+$|chordino", regex=True)
    rejected_sources = original.loc[~allowed, "candidate_source"].value_counts().to_dict()
    rejected_features = [name for name in original if name.startswith(("phrase_", "harm_", "melody_"))]
    clean = original.loc[allowed].drop(columns=rejected_features)
    with tempfile.TemporaryDirectory() as folder:
        filtered = Path(folder) / "acoustic.csv"
        clean.to_csv(filtered, index=False, encoding="utf-8-sig")
        df = load_rows(filtered).reset_index(drop=True)
    free = np.empty(len(df), dtype=float)
    folds = []
    for song in sorted(df.song_id.astype(str).unique()):
        train = filter_evaluable_rows(df[df.song_id != song], "target_exact_chord")
        test = df[df.song_id == song]
        pipeline, numeric, categorical = build_pipeline(train, "random_forest")
        targets = train.target_exact_chord.astype(int).to_numpy()
        pipeline.fit(train, targets)
        free[test.index] = positive_class_scores(pipeline, test)
        folds.append({"test_song": song, "train_songs": sorted(train.song_id.astype(str).unique()),
                      "numeric_features": numeric, "categorical_features": categorical})
    baseline = np.where(df.candidate_source.str.contains("topn_rank1", regex=False), 2.,
                        np.where(df.candidate_source.str.contains("chordino", regex=False), 1., 0.))
    constrained = free.copy()
    groups = pd.Series([group_key(row) for _, row in df.iterrows()], index=df.index)
    for _, group in df.groupby(groups):
        top = group.iloc[int(np.argmax(baseline[group.index]))]
        # Diagnostic with supplied segment groups. Deployment must group on its
        # own proposed intervals and does not have reference interval identities.
        mask = group.candidate_chord_root.fillna("").astype(str) != str(top.candidate_chord_root)
        constrained[group.index[mask]] = -1
    output = {"candidate_csv": str(args.candidate_csv.resolve()),
              "protocol": "leave-one-song-out exact-chord model; acoustic candidates only; no phrase/harmonic/human features; supplied segmentation and key, not end-to-end",
              "excluded_candidate_sources": rejected_sources, "excluded_feature_columns": rejected_features,
              "rows": len(df), "folds": folds, "modes": {}}
    for mode, scores in (("acoustic_top1", baseline), ("exact_model_free", free), ("exact_model_keep_acoustic_root", constrained)):
        output["modes"][mode] = {}
        for target in ("target_exact_chord", "target_same_root", "target_same_roman_root"):
            test = filter_evaluable_rows(df, target)
            output["modes"][mode][target], _ = evaluate(test, scores[test.index], target)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({mode: {target: data["selector_accuracy"] for target, data in results.items()}
                      for mode, results in output["modes"].items()}, indent=2))


if __name__ == "__main__":
    main()
