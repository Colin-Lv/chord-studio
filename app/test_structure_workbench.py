from copy import deepcopy
import unittest

from build_structure_workbench import canonical_events, event_hash, expert_grid_review


class StructureWorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.payload = {"source": "clip.wav", "audio_file": "clip.wav", "audio_offset": 30.0,
            "duration": 5.0, "meter": "unknown", "beat_source": "sheetsage2", "display_title": "Song",
            "events": [{"start": .25, "end": 1.25, "chord": "G/B", "chord_display": "G/B",
                "raw_chord": "G:maj/3", "local_key": "Am", "roman": "VII", "roman_display": "VII/ii",
                "confidence": 0., "confidence_status": "not_exported", "source": "sheetsage2",
                "custom": {"must": "survive"}, "bar": 0, "beat_start": None, "beat_end": None}],
            "bars": [{"bar": 0, "start": 0., "end": 5., "beats": [], "spans": []}],
            "structure": {"sections": [{"id": "s1", "start": 0., "end": 5., "start_bar": 0, "end_bar": 0,
                "name": "Verse suggestion", "role": "verse", "source": "allin1"}], "warnings": [],
                "expert_grid": {"beats": [.5 + i * .5 for i in range(10)], "beat_positions": [i % 4 + 1 for i in range(10)], "applied": False}}}

    def test_review_changes_only_display_grid(self):
        original = deepcopy(self.payload)
        result = expert_grid_review(self.payload)
        self.assertEqual(self.payload, original)
        self.assertEqual(canonical_events(result), canonical_events(original))
        self.assertEqual(event_hash(result), event_hash(original))
        self.assertEqual(result["audio_offset"], 30.)
        self.assertEqual(result["source"], "clip.wav")
        self.assertTrue(result["structure"]["grid_review_required"])
        self.assertTrue(result["structure"]["expert_grid"]["applied_to_display"])
        self.assertFalse(result["structure"]["expert_grid"]["confirmed"])
        self.assertEqual(result["structure"]["sections"][0]["start"], 0.)
        self.assertEqual(result["structure"]["sections"][0]["end"], 5.)
        self.assertEqual(result["events"][0]["local_key"], "Am")

    def test_mismatched_expert_arrays_fail(self):
        self.payload["structure"]["expert_grid"]["beat_positions"].pop()
        with self.assertRaises(ValueError):
            expert_grid_review(self.payload)

    def test_outside_final_marker_does_not_move_source_event(self):
        self.payload["structure"]["expert_grid"]["beats"][-1] = 5.01
        result = expert_grid_review(self.payload)
        self.assertEqual(result["events"][0]["start"], .25)
        self.assertEqual(result["events"][0]["end"], 1.25)
        self.assertEqual(result["duration"], 5.)
        self.assertEqual(result["retime_provenance"]["observed_markers"][-1]["time"], 4.5)


if __name__ == "__main__":
    unittest.main()
