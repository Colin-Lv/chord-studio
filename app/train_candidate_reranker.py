#!/usr/bin/env python
"""Train/evaluate a small candidate reranker from candidate training rows."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import re
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURE_SCHEMA = "candidate_intervals_v2"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df = exclude_reference_candidates(df)
    numeric_columns = (
        "duration",
        "candidate_confidence",
        "candidate_score",
        "candidate_rank",
        "start",
        "end",
        "bar",
        *optional_audio_feature_columns(),
    )
    for column in numeric_columns:
        if column not in df:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "reference_label_strength" not in df:
        df["reference_label_strength"] = ""
    if "reference_root_locked" not in df:
        df["reference_root_locked"] = 0
    df["reference_label_strength"] = df["reference_label_strength"].fillna("").astype(str)
    df["reference_root_locked"] = pd.to_numeric(df["reference_root_locked"], errors="coerce").fillna(0).astype(int)
    df["reference_root_only"] = (
        df["reference_label_strength"].eq("root_only") | df["reference_root_locked"].astype(bool)
    )
    # Learned imputation belongs inside the fitted pipeline, separately per fold.
    df["candidate_rank"] = df["candidate_rank"].fillna(1)
    df["has_position"] = df["position"].fillna("").astype(str).ne("")
    df["source_family"] = df["candidate_source"].fillna("").astype(str).map(
        lambda value: re.sub(r"_rank\d+$", "", value)
    )
    df["source_kind"] = df["candidate_source"].fillna("").astype(str).map(source_kind)
    df["bar_mod_2"] = (df["bar"].fillna(0).astype(int) % 2).astype(str)
    df["bar_mod_4"] = (df["bar"].fillna(0).astype(int) % 4).astype(str)
    if "target_same_chord_family" not in df:
        df["target_same_chord_family"] = (
            df["candidate_chord_family"].fillna("").astype(str)
            == df["reference_chord_family"].fillna("").astype(str)
        ).astype(int)
    df = add_consensus_features(df)
    return add_groupwise_features(df)


def exclude_reference_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Human/reference labels may supervise a row, never enter its candidate pool."""
    sources = df["candidate_source"].fillna("").astype(str)
    excluded = sources.str.contains(r"(?:^|_)(?:corrected|manual|human|reference|groundtruth|ground_truth)(?:_|$)",
                                   case=False, regex=True)
    counts = sources[excluded].value_counts().to_dict()
    if counts:
        warnings.warn(f"Excluded reference-derived candidate sources: {counts}", RuntimeWarning, stacklevel=2)
    clean = df.loc[~excluded].copy()
    clean.attrs["excluded_reference_candidate_sources"] = counts
    return clean


def filter_evaluable_rows(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target != "target_exact_chord" or "reference_root_only" not in df:
        return df
    return df[~df["reference_root_only"].astype(bool)].copy()


def optional_audio_feature_columns() -> list[str]:
    return [
        "audio_root_energy",
        "audio_third_energy",
        "audio_fifth_energy",
        "audio_triad_mean",
        "audio_flat7_energy",
        "audio_maj7_energy",
        "audio_9_energy",
        "audio_6_energy",
        "audio_4_energy",
        "audio_flat7_over_triad",
        "audio_maj7_over_triad",
        "audio_9_over_triad",
        "audio_dom7_vs_maj7",
        "audio_bass_root_energy",
        "audio_bass_flat7_energy",
        "audio_candidate_has_7",
        "audio_candidate_has_maj7",
        "audio_candidate_has_9",
        "audio_candidate_has_sus4",
        "audio_candidate_has_m7b5",
        "stem_fused_root_energy",
        "stem_fused_third_energy",
        "stem_fused_fifth_energy",
        "stem_fused_triad_mean",
        "stem_fused_flat7_energy",
        "stem_fused_maj7_energy",
        "stem_fused_9_energy",
        "stem_fused_flat7_over_triad",
        "stem_fused_maj7_over_triad",
        "stem_fused_9_over_triad",
        "stem_fused_dom7_vs_maj7",
        "stem_bass_root_energy",
        "melody_voiced_seconds",
        "melody_chord_tone_ratio",
        "melody_scale_tone_ratio",
        "melody_chromatic_ratio",
        "melody_chromatic_chord_tone_ratio",
        "melody_sustained_chromatic_seconds",
        "melody_candidate_contains_chromatic",
        "harm_candidate_is_functional_dom7",
        "harm_next_acoustic_matches_resolution",
        "harm_prev_acoustic_same_root",
        "harm_next_acoustic_same_root",
        "harm_candidate_is_chromatic_to_key",
        "harm_kb_functional_dom7_prob",
        "harm_kb_next_acoustic_resolution_prob",
        "harm_kb_supports_next_acoustic",
        "harm_candidate_is_sus9",
        "harm_sus9_next_acoustic_matches_resolution",
        "harm_sus9_is_plain_v_to_i",
        "phrase_1bar_forward_score",
        "phrase_1bar_backward_score",
        "phrase_2bar_score",
        "phrase_4bar_score",
        "phrase_8bar_score",
        "phrase_completion_score",
        "phrase_any_score",
    ]


def source_kind(value: str) -> str:
    text = value.lower()
    if "_topn_rank" in text or text.endswith("_topn"):
        return "topn"
    if "boundary" in text:
        return "boundary"
    if "chordino" in text or "madmom" in text or "external" in text:
        return "external_acr"
    if "functional" in text:
        return "functional"
    if "sus9" in text or "9sus" in text:
        return "sus9"
    if "half_dim" in text or "m7b5" in text:
        return "half_dim"
    if "borrowed" in text:
        return "borrowed_color"
    if "slash" in text or "bass" in text:
        return "bass_line"
    if "key_timeline" in text:
        return "key_timeline"
    if "base" in text or "regularized" in text or "fixedgrid" in text or "refined" in text:
        return "base_analysis"
    if "current" in text:
        return "current"
    return "other"


def row_overlap(left: pd.Series, right: pd.Series) -> float:
    return max(0.0, min(float(left.end), float(right.end)) - max(float(left.start), float(right.start)))


def add_consensus_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["same_roman_root_votes"] = 0
    df["same_chord_votes"] = 0
    df["overlap_source_count"] = 0
    group_cols = group_columns(df)

    # Candidate tables can become large after adding functional and symbolic
    # alternatives. A pairwise overlap scan per bar is too slow, so use the
    # observable candidate intervals as the consensus unit. Reference boundaries
    # must not alter model inputs, and predicted bar IDs differ across sources.
    source_counts = df.groupby(group_cols, dropna=False)["candidate_source"].transform("nunique")
    roman_source_counts = df.groupby(group_cols + ["candidate_roman_root"], dropna=False)[
        "candidate_source"
    ].transform("nunique")
    chord_source_counts = df.groupby(group_cols + ["candidate_chord"], dropna=False)[
        "candidate_source"
    ].transform("nunique")
    df["overlap_source_count"] = (source_counts - 1).clip(lower=0).astype(int)
    df["same_roman_root_votes"] = (roman_source_counts - 1).clip(lower=0).astype(int)
    df["same_chord_votes"] = (chord_source_counts - 1).clip(lower=0).astype(int)
    return df


def group_columns(df: pd.DataFrame) -> list[str]:
    return ["song_id", "start", "end"]


def add_groupwise_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ("reference_start", "reference_end"):
        if column not in df:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")
    groups = group_columns(df)
    grouped = df.groupby(groups, dropna=False)
    for column in ("candidate_score", "candidate_confidence"):
        maximum = grouped[column].transform("max")
        minimum = grouped[column].transform("min")
        df[f"{column}_delta_from_max"] = maximum - df[column]
        df[f"{column}_range_norm"] = (df[column] - minimum) / (maximum - minimum + 1e-9)
        df[f"{column}_rank_in_group"] = grouped[column].rank(ascending=False, method="min")
    df["candidate_rank_reciprocal"] = 1.0 / np.maximum(df["candidate_rank"], 1.0)
    df["group_size"] = grouped["candidate_chord"].transform("count")
    return df


def build_pipeline(df: pd.DataFrame, model_name: str) -> tuple[Pipeline, list[str], list[str]]:
    numeric = [
        "duration",
        "candidate_confidence",
        "bar",
        "same_roman_root_votes",
        "same_chord_votes",
        "overlap_source_count",
        "candidate_rank",
        "candidate_score",
        "candidate_score_delta_from_max",
        "candidate_score_range_norm",
        "candidate_score_rank_in_group",
        "candidate_confidence_delta_from_max",
        "candidate_confidence_range_norm",
        "candidate_confidence_rank_in_group",
        "candidate_rank_reciprocal",
        "group_size",
        *optional_audio_feature_columns(),
    ]
    categorical = [
        "source_kind",
        "unit",
        "local_key",
        "candidate_roman_root",
        "candidate_chord_family",
        "has_position",
        "bar_mod_2",
        "bar_mod_4",
    ]
    numeric = [column for column in numeric if column in df.columns and not df[column].isna().all()]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=4,
        )
    else:
        model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    return Pipeline([("features", preprocessor), ("model", model)]), numeric, categorical


def song_cross_validation_splits(df: pd.DataFrame):
    """Hold out whole songs; never fall back to splitting bars/candidates."""
    groups = df["song_id"].fillna("").astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if "" in unique_groups or len(unique_groups) < 2:
        raise ValueError("Song-held-out validation requires at least two nonempty song IDs; use --skip-cv for fitting only.")
    return list(GroupKFold(n_splits=min(5, len(unique_groups))).split(df, groups=groups))


def positive_class_scores(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    classes = np.asarray(pipeline.classes_)
    matches = np.flatnonzero(classes == 1)
    return pipeline.predict_proba(df)[:, matches[0]] if len(matches) else np.zeros(len(df))


def cross_validation_predictions(df: pd.DataFrame, target: str, pipeline: Pipeline,
                                 weights: np.ndarray | None = None) -> np.ndarray:
    y = df[target].astype(int).to_numpy()
    scores = np.empty(len(df), dtype=float)
    for train_indices, test_indices in song_cross_validation_splits(df):
        if len(np.unique(y[train_indices])) == 1:
            scores[test_indices] = float(y[train_indices][0])
            continue
        fitted = clone(pipeline)
        kwargs = {} if weights is None else {"model__sample_weight": np.asarray(weights)[train_indices]}
        fitted.fit(df.iloc[train_indices], y[train_indices], **kwargs)
        scores[test_indices] = positive_class_scores(fitted, df.iloc[test_indices])
    return scores


def metrics_for(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    predictions = (scores >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )
    output = {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "average_precision": round(float(average_precision_score(y_true, scores)), 4),
    }
    if len(np.unique(y_true)) >= 2:
        output["roc_auc"] = round(float(roc_auc_score(y_true, scores)), 4)
    return output


def source_summary(df: pd.DataFrame, target: str) -> list[dict[str, object]]:
    rows = []
    for source, group in df.groupby("candidate_source"):
        rows.append(
            {
                "candidate_source": source,
                "rows": int(len(group)),
                "positive_rate": round(float(group[target].astype(int).mean()), 4),
            }
        )
    return rows


def domain_for_song(song_id: object) -> str:
    text = str(song_id).lower()
    if text.startswith("isophonics") or "the_beatles" in text:
        return "beatles"
    return "user"


def sample_weights(df: pd.DataFrame, beatles_weight: float, user_weight: float) -> np.ndarray:
    weights = df["song_id"].map(lambda value: beatles_weight if domain_for_song(value) == "beatles" else user_weight)
    return weights.astype(float).to_numpy()


def weight_summary(df: pd.DataFrame, weights: np.ndarray) -> list[dict[str, object]]:
    temp = df[["song_id"]].copy()
    temp["domain"] = temp["song_id"].map(domain_for_song)
    temp["weight"] = weights
    rows = []
    for domain, group in temp.groupby("domain"):
        rows.append(
            {
                "domain": domain,
                "rows": int(len(group)),
                "songs": int(group["song_id"].nunique()),
                "weight": round(float(group["weight"].iloc[0]), 4),
                "effective_rows": round(float(group["weight"].sum()), 2),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_csv", type=Path)
    parser.add_argument(
        "--target",
        choices=["target_exact_chord", "target_same_root", "target_same_roman_root", "target_same_chord_family"],
        default="target_same_roman_root",
    )
    parser.add_argument("-o", "--output", type=Path, default=REPO_ROOT / "benchmarks" / "models" / "candidate_reranker.pkl")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "candidate_reranker_report.json")
    parser.add_argument("--model", choices=["logistic", "random_forest"], default="logistic")
    parser.add_argument("--beatles-weight", type=float, default=1.0)
    parser.add_argument("--user-weight", type=float, default=1.0)
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip cross-validation on the training CSV. Useful when a separate eval CSV is available.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    training_csv = resolve_path(args.training_csv)
    df = load_rows(training_csv)
    df = filter_evaluable_rows(df, args.target)
    pipeline, numeric, categorical = build_pipeline(df, args.model)
    y = df[args.target].astype(int).to_numpy()
    weights = sample_weights(df, args.beatles_weight, args.user_weight)
    scores = None if args.skip_cv else cross_validation_predictions(df, args.target, pipeline, weights)
    pipeline.fit(df, y, model__sample_weight=weights)

    args.output = resolve_path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump({"pipeline": pipeline, "target": args.target, "numeric": numeric, "categorical": categorical,
                     "training_csv": str(training_csv), "training_song_ids": sorted(df["song_id"].astype(str).unique()),
                     "feature_schema": FEATURE_SCHEMA,
                     "evaluation_protocol": "song-held-out; reference-aligned candidate diagnostic"}, handle)

    report = {
        "training_csv": str(training_csv),
        "rows": int(len(df)),
        "target": args.target,
        "model": args.model,
        "sample_weight_summary": weight_summary(df, weights),
        "positive_rate": round(float(np.mean(y)), 4),
        "cross_validation": None if scores is None else metrics_for(y, scores),
        "cross_validation_group": "song_id",
        "cross_validation_folds": 0 if args.skip_cv else min(5, int(df["song_id"].nunique())),
        "evaluation_scope": "reference-aligned candidates; not end-to-end chord recognition",
        "excluded_reference_features": ["harm_sus9_next_reference_matches_resolution"],
        "feature_schema": FEATURE_SCHEMA,
        "excluded_reference_candidate_sources": df.attrs.get("excluded_reference_candidate_sources", {}),
        "source_summary": source_summary(df, args.target),
        "model_path": str(args.output),
    }
    if scores is not None:
        from evaluate_candidate_reranker import evaluate as evaluate_selector
        report["cross_validation_selector"], _ = evaluate_selector(df, scores, args.target)
    args.report = resolve_path(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
