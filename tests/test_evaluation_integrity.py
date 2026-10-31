"""Regression checks for evaluation coverage, labels and split integrity."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from evaluate_chords import TimedChord, convert_annotation_chord, evaluate, harte_chord, read_reference
from train_candidate_reranker import (add_consensus_features, exclude_reference_candidates,
                                     optional_audio_feature_columns, song_cross_validation_splits)
from evaluate_candidate_reranker import evaluate as evaluate_candidates
from evaluate_chord_boundaries import match_boundaries


class EvaluationIntegrityTests(unittest.TestCase):
    def test_truncated_prediction_does_not_shrink_denominator(self):
        metrics = evaluate([TimedChord(0, 10, "C")], [TimedChord(4, 5, "C")])
        self.assertAlmostEqual(metrics["root_accuracy"], 0.1)
        self.assertAlmostEqual(metrics["prediction_coverage"], 0.1)
        self.assertAlmostEqual(metrics["missing_prediction_seconds"], 9)

    def test_empty_and_disjoint_predictions_are_scored(self):
        for prediction in ([], [TimedChord(20, 30, "C")]):
            self.assertEqual(evaluate([TimedChord(0, 10, "C")], prediction)["root_accuracy"], 0)

    def test_missing_output_does_not_receive_credit_for_reference_silence(self):
        reference = [TimedChord(0, 2, "N"), TimedChord(2, 4, "C")]
        metrics = evaluate(reference, [TimedChord(2, 4, "C")])
        self.assertEqual(metrics["tetrad_accuracy"], 1)
        self.assertEqual(metrics["tetrad_wcsr"], .5)
        self.assertEqual(metrics["no_chord_accuracy"], 0)
        self.assertEqual(evaluate(reference, [TimedChord(0, 2, "N"), TimedChord(2, 4, "C")])["tetrad_wcsr"], 1)

    def test_short_errors_are_duration_weighted_not_frame_aliased(self):
        reference = [TimedChord(0, 0.003, "C"), TimedChord(0.003, 1, "G")]
        pred = [TimedChord(0, 1, "G")]
        for frame in (0.001, 0.1, 5):
            self.assertAlmostEqual(evaluate(reference, pred, frame)["triad_accuracy"], 0.997)

    def test_unknown_is_excluded_and_no_chord_is_scored(self):
        reference = [TimedChord(0, 2, "X"), TimedChord(2, 3, "N"), TimedChord(3, 4, "C")]
        metrics = evaluate(reference, [TimedChord(0, 4, "C")])
        self.assertEqual(metrics["root_accuracy"], 1)
        self.assertEqual(metrics["root_wcsr"], 0.5)
        self.assertEqual(metrics["unknown_reference_seconds"], 2)

    def test_major_seventh_bass_and_suspension_are_preserved(self):
        self.assertEqual(convert_annotation_chord("C:maj7/3"), "Cmaj7/E")
        self.assertEqual(convert_annotation_chord("A:min7/b3"), "Am7/C")
        self.assertEqual(convert_annotation_chord("X"), "X")
        self.assertEqual(harte_chord("Cmaj7/E"), "C:maj7/3")
        metrics = evaluate([TimedChord(0, 1, "C:maj7")], [TimedChord(0, 1, "C7")])
        self.assertEqual(metrics["triad_accuracy"], 1)
        self.assertEqual(metrics["tetrad_accuracy"], 0)
        self.assertEqual(evaluate([TimedChord(0, 1, "Csus2")], [TimedChord(0, 1, "Csus4")])["triad_accuracy"], 0)

    def test_enharmonic_and_inversions(self):
        metrics = evaluate([TimedChord(0, 1, "C#:min7/b3")], [TimedChord(0, 1, "Dbm7/E")])
        self.assertEqual(metrics["tetrad_inv_accuracy"], 1)

    def test_unsorted_input_and_unknown_gaps(self):
        metrics = evaluate([TimedChord(2, 3, "G"), TimedChord(0, 1, "C")],
                           [TimedChord(2, 3, "G"), TimedChord(0, 1, "C")])
        self.assertEqual(metrics["triad_accuracy"], 1)
        self.assertEqual(metrics["evaluated_seconds"], 2)
        self.assertEqual(metrics["unknown_reference_seconds"], 1)

    def test_overlap_and_invalid_times_fail_explicitly(self):
        for rows in ([TimedChord(0, 2, "C"), TimedChord(1, 3, "G")],
                     [TimedChord(0, float("nan"), "C")], [TimedChord(2, 1, "C")]):
            with self.assertRaises(ValueError):
                evaluate(rows, [])

    def test_csv_and_lab_accept_display_minor_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.lab"
            path.write_text("0 1 Am\n1 2 Cmaj7\n2 3 X\n", encoding="utf-8")
            self.assertEqual([row.chord for row in read_reference(path)], ["Am", "Cmaj7", "X"])

    def test_song_cv_never_splits_a_song(self):
        df = pd.DataFrame({"song_id": np.repeat(["a", "b", "c"], 12), "bar": list(range(12)) * 3})
        seen = []
        for train, test in song_cross_validation_splits(df):
            self.assertFalse(set(df.iloc[train].song_id) & set(df.iloc[test].song_id))
            seen.extend(test)
        self.assertEqual(sorted(seen), list(range(len(df))))
        with self.assertRaises(ValueError):
            song_cross_validation_splits(df[df.song_id == "a"])

    def test_reference_resolution_is_never_a_model_feature(self):
        self.assertFalse(any("reference" in column for column in optional_audio_feature_columns()))

    def test_millisecond_annotation_rounding_is_repaired(self):
        metrics = evaluate([TimedChord(0, 1.001, "C"), TimedChord(1, 2, "G")],
                           [TimedChord(0, 1, "C"), TimedChord(1, 2, "G")])
        self.assertEqual(metrics["triad_accuracy"], 1)
        self.assertAlmostEqual(metrics["reference_rounding_overlap_seconds"], 0.001)

    def test_candidate_bar_ids_cannot_duplicate_reference_duration(self):
        row = dict(song_id="a", reference_start=0, reference_end=4, candidate_source="a",
                   candidate_chord="C", candidate_roman="I", reference_chord="C", reference_roman="I",
                   target_exact_chord=1)
        data = pd.DataFrame([{**row, "bar": 1}, {**row, "bar": 2}])
        metrics, _ = evaluate_candidates(data, [0.9, 0.8], "target_exact_chord")
        self.assertEqual(metrics["groups"], 1)
        self.assertEqual(metrics["reference_group_seconds"], 4)

    def test_consensus_features_ignore_reference_boundaries(self):
        df = pd.DataFrame(dict(song_id=["a", "a"], start=[0, 0], end=[4, 4],
                               candidate_source=["first", "second"], candidate_roman_root=["I", "I"],
                               candidate_chord=["C", "C"], bar=[1, 2], reference_start=[0, 1], reference_end=[2, 3]))
        features = add_consensus_features(df)
        self.assertEqual(features.same_chord_votes.tolist(), [1, 1])
        df.reference_start = [100, 200]
        pd.testing.assert_series_equal(features.same_chord_votes, add_consensus_features(df).same_chord_votes)

    def test_boundary_matching_maximizes_valid_pairs(self):
        # Nearest-first consumes .06 for reference .04 and misses reference .1;
        # the valid optimum is .04->0 and .1->.06.
        reference = [(0.04, 1, "C"), (0.1, 1, "G")]
        prediction = [(0, 1, "C"), (0.06, 1, "G")]
        self.assertEqual(match_boundaries(reference, prediction, 0.05)[0], 2)

    def test_reference_answers_are_not_candidates(self):
        data = pd.DataFrame({"candidate_source": ["corrected_grid", "manual_chord", "acoustic_topn_rank1", "chordino"]})
        with self.assertWarns(RuntimeWarning):
            clean = exclude_reference_candidates(data)
        self.assertEqual(clean.candidate_source.tolist(), ["acoustic_topn_rank1", "chordino"])

    def test_attribute_evaluation_integrates_duration_and_honors_root_only(self):
        from evaluate_chord_attributes import compare_attrs, evaluate as evaluate_attributes
        self.assertEqual(compare_attrs("C9", "C7")["exact"], 0)
        self.assertEqual(compare_attrs("Cdim7", "Cm7b5")["seventh"], 0)
        self.assertEqual(compare_attrs("Dbm7/E", "C#m7/E")["exact"], 1)
        with tempfile.TemporaryDirectory() as folder:
            reference = Path(folder) / "ref.json"
            prediction = Path(folder) / "pred.json"
            reference.write_text(json.dumps({"bars": [{"bar": 1, "spans": [
                {"start": 0, "end": 9, "chord": "C", "root_locked": True},
                {"start": 9, "end": 10, "chord": "G"}]}]}), encoding="utf-8")
            prediction.write_text(json.dumps({"bars": [{"bar": 1, "spans": [
                {"start": 0, "end": 10, "chord": "Cmaj7"}]}]}), encoding="utf-8")
            _, report = evaluate_attributes(reference, prediction)
            self.assertEqual(report["metrics"]["root"], 0.9)
            self.assertEqual(report["metrics"]["exact"], 0)
            self.assertEqual(report["metric_denominator_seconds"]["exact"], 1)


if __name__ == "__main__":
    unittest.main()
