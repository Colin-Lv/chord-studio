"""Behavioral regressions for conservative music-structure analysis."""

import copy
import unittest

import numpy as np

from song_structure import (
    analyze_feature_structure,
    apply_manual_sections,
    detect_harmonic_loops,
    valid_bars,
)


def payload_for(n, chords=("C", "G", "Am", "F"), durations=None):
    durations = durations or [2.0] * n
    starts = np.r_[0.0, np.cumsum(durations)]
    bars = []
    for index in range(n):
        start, end = float(starts[index]), float(starts[index + 1])
        bars.append({"bar": index + 1, "start": start, "end": end, "spans": [{"start": start, "end": end, "chord": chords[index % len(chords)]}]})
    return {"bars": bars, "duration": float(starts[-1]), "meter_confidence": 0.95}


def features_for(n):
    chroma = np.zeros((n, 12))
    chroma[:, [0, 4, 7]] = 1
    return {"chroma": chroma, "timbre": np.ones((n, 13)), "rhythm": np.ones((n, 8)), "energy": np.full((n, 1), -20.0)}


class SongStructureTests(unittest.TestCase):
    def test_harmonic_cycle_does_not_invent_chorus(self):
        payload = payload_for(32)
        before = copy.deepcopy(payload)
        result = analyze_feature_structure(payload, features_for(32))
        self.assertEqual(payload, before)
        self.assertEqual(len(result["sections"]), 1)
        self.assertEqual(result["sections"][0]["role"], "unknown")
        self.assertEqual(result["loops"][0]["period_bars"], 4)
        self.assertEqual(result["loops"][0]["repetitions"], 8)

    def test_same_harmony_different_timbre_gets_distinct_sections(self):
        payload = payload_for(48)
        features = features_for(48)
        features["timbre"][16:32] = 30
        result = analyze_feature_structure(payload, features)
        self.assertEqual([section["start_bar"] for section in result["sections"]], [1, 17, 33])
        self.assertEqual([section["group"] for section in result["sections"]], ["A", "B", "A"])
        self.assertTrue(all(section["role"] == "unknown" for section in result["sections"]))

    def test_variable_tempo_keeps_actual_grid_times(self):
        payload = payload_for(48, durations=list(np.linspace(1.5, 2.5, 48)))
        features = features_for(48)
        features["timbre"][16:32] = 30
        result = analyze_feature_structure(payload, features)
        for section in result["sections"]:
            self.assertEqual(section["start"], payload["bars"][section["start_bar"] - 1]["start"])
            self.assertEqual(section["end"], payload["bars"][section["end_bar"] - 1]["end"])

    def test_sustained_chord_and_no_chord_are_not_loops(self):
        self.assertEqual(detect_harmonic_loops(payload_for(32, ("C",))["bars"]), [])
        self.assertEqual(detect_harmonic_loops(payload_for(32, ("N", "G"))["bars"]), [])

    def test_manual_sections_are_validated_and_retain_automatic_result(self):
        payload = payload_for(16)
        automatic = analyze_feature_structure(payload, features_for(16))
        manual = [{"start_bar": 1, "end_bar": 8, "name": "主歌", "role": "verse"}, {"start_bar": 9, "end_bar": 16, "name": "副歌", "role": "chorus"}]
        result = apply_manual_sections(automatic, manual, payload)
        self.assertEqual(result["sections"][1]["source"], "manual")
        self.assertEqual(result["sections"][1]["start"], 16)
        self.assertEqual(result["automatic_sections"], automatic["sections"])
        with self.assertRaises(ValueError):
            apply_manual_sections(automatic, manual[:1], payload)
        with self.assertRaises(ValueError):
            apply_manual_sections(automatic, [{"start_bar": 1, "end_bar": 9}, {"start_bar": 9, "end_bar": 16}], payload)

    def test_invalid_grid_and_nan_are_rejected(self):
        payload = payload_for(16)
        payload["bars"][1]["start"] = 0
        with self.assertRaises(ValueError):
            valid_bars(payload)
        features = features_for(16)
        features["chroma"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            analyze_feature_structure(payload_for(16), features)

    def test_empty_and_single_bar(self):
        self.assertEqual(analyze_feature_structure({"bars": []}, {})["sections"], [])
        self.assertEqual(len(analyze_feature_structure(payload_for(1), features_for(1))["sections"]), 1)


if __name__ == "__main__":
    unittest.main()
