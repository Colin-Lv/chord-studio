"""Boundary preservation and honest confidence for external fusion."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from chord_studio import BeatInterval, ExternalChordHint
from downbeat_external_fusion import EdgeFillCandidate, external_harmonic_intervals, make_payload, smooth_chords
from validate_analysis import audit_analysis


class ExternalFusionTimingTests(unittest.TestCase):
    def setUp(self):
        self.beats = [BeatInterval(1, index + 1, float(index), float(index + 1)) for index in range(4)]

    def test_offbeat_and_same_root_quality_changes_survive(self):
        hints = [ExternalChordHint(0, 1.25, "C"), ExternalChordHint(1.25, 2.7, "Cmaj7"),
                 ExternalChordHint(2.7, 4, "G")]
        cells = external_harmonic_intervals(self.beats, hints)
        self.assertEqual([(c.start, c.end) for c in cells], [(0, 1.25), (1.25, 2.7), (2.7, 4)])
        self.assertEqual((cells[1].beat_start, cells[1].beat_end), (2, 3))

    def test_snapping_is_local_and_opt_in(self):
        hints = [ExternalChordHint(0, 1.04, "C"), ExternalChordHint(1.04, 2.3, "F"),
                 ExternalChordHint(2.3, 4, "G")]
        cells = external_harmonic_intervals(self.beats, hints, 0.08)
        self.assertEqual([(c.start, c.end) for c in cells], [(0, 1), (1, 2.3), (2.3, 4)])

    def test_beat_display_uses_time_overlap_and_selected_confidence(self):
        cells = external_harmonic_intervals(self.beats, [ExternalChordHint(0, 0.1, "G"), ExternalChordHint(0.1, 4, "C")])
        payload = make_payload({}, self.beats, cells, ["G", "C"], [{"G": 0.2, "C": 0.9}, {"C": 1}],
                               [(None, 0), (None, 0)], "C", "4/4", 4, 1, "external")
        self.assertEqual(payload["events"][0]["confidence"], 0.2)
        self.assertEqual(payload["bars"][0]["beats"][0]["chord"], "C")
        self.assertEqual(payload["beat_source"], "external")
        self.assertEqual(payload["meter_confidence"], 0)

    def test_empty_decoder_has_no_index_error(self):
        self.assertEqual(smooth_chords([], "C"), [])

    def test_explicit_source_inversion_is_not_replaced_by_template_bass(self):
        cells = external_harmonic_intervals(self.beats, [ExternalChordHint(0, 4, "C/E")])
        payload = make_payload({}, self.beats, cells, ["C/E"], [{"C/E": 1}], [(7, .99)],
                               "G", "4/4", 4, 1, "external")
        assert payload["events"][0]["chord_display"] == "C/E"
        assert payload["events"][0]["roman_display"] == "IV/VI"

    def test_edge_chords_are_editable_unplaced_regions_without_fake_beats(self):
        beats = [BeatInterval(1, index + 1, 1 + index * .75, 1 + (index + 1) * .75) for index in range(4)]
        cells = external_harmonic_intervals(beats, [ExternalChordHint(1, 4, "C")])
        edges = [EdgeFillCandidate(0, .4, "C", 1), EdgeFillCandidate(.4, 1, "G", 1),
                 EdgeFillCandidate(4, 5, "Am", 1)]
        payload = make_payload({"duration": 5}, beats, cells, ["C"], [{"C": 1}], [(None, 0)],
                               "C", "4/4", 4, .75, "external", edge_fills=edges)
        self.assertTrue(audit_analysis(payload)["valid"])
        self.assertEqual([(event["start"], event["end"], event["chord"]) for event in payload["events"]],
                         [(0, .4, "C"), (.4, 1, "G"), (1, 4, "C"), (4, 5, "Am")])
        self.assertEqual([bar["bar"] for bar in payload["bars"]], [0, 1, 2])
        self.assertEqual(payload["bars"][0]["beats"], [])
        self.assertEqual(payload["bars"][-1]["beats"], [])
        self.assertEqual(payload["bars"][-1]["grid_status"], "uncovered")


if __name__ == "__main__":
    unittest.main()
