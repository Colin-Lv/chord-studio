"""Check neural-expert integration independently of optional model packages."""
import copy
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from song_structure import analyze_feature_structure, apply_manual_sections
from structure_expert import HARMONIX_LABELS, _convert_window, compare_beat_grids, import_expert_structure, shift_expert_timebase, write_expert_beats
from test_song_structure import features_for, payload_for


class StructureExpertTests(unittest.TestCase):
    def setUp(self):
        self.payload = payload_for(16)
        self.external = {"segments": [{"start": 0, "end": 16, "label": "verse"}, {"start": 16, "end": 32, "label": "chorus"}], "beats": [0, 0.5, 1], "beat_positions": [1, 2, 3], "downbeats": [0], "activation_fps": 10}
        labels = np.zeros((10, 320))
        labels[HARMONIX_LABELS.index("verse"), :160] = 0.9
        labels[HARMONIX_LABELS.index("chorus"), 160:] = 0.8
        self.activations = {"label": labels, "segment": np.full(320, 0.25)}

    def test_expert_keeps_chords_and_exposes_separate_beat_grid(self):
        original = copy.deepcopy(self.payload)
        result = import_expert_structure(self.payload, self.external, self.activations)
        self.assertEqual(self.payload, original)
        self.assertEqual(result["sections"][1]["role"], "chorus")
        self.assertAlmostEqual(result["sections"][1]["role_confidence"], 0.8)
        self.assertEqual(result["sections"][1]["start_bar"], 9)
        self.assertFalse(result["expert_grid"]["applied"])

    def test_missing_or_low_role_activations_abstain(self):
        result = import_expert_structure(self.payload, self.external)
        self.assertTrue(all(section["role"] == "unknown" for section in result["sections"]))
        self.assertEqual(result["sections"][0]["role_candidate"], "verse")
        self.activations["label"] *= 0.4
        result = import_expert_structure(self.payload, self.external, self.activations)
        self.assertTrue(all(section["role"] == "unknown" for section in result["sections"]))

    def test_same_function_does_not_imply_same_acoustic_group(self):
        self.external["segments"][1]["label"] = "verse"
        result = import_expert_structure(self.payload, self.external, self.activations)
        self.assertNotEqual(result["sections"][0]["group"], result["sections"][1]["group"])

    def test_manual_structure_survives_new_expert(self):
        auto = analyze_feature_structure(self.payload, features_for(16))
        manual = [{"start_bar": 1, "end_bar": 16, "role": "bridge", "name": "已确认桥段"}]
        self.payload["structure"] = apply_manual_sections(auto, manual, self.payload)
        result = import_expert_structure(self.payload, self.external, self.activations)
        self.assertEqual(len(result["sections"]), 1)
        self.assertEqual(result["sections"][0]["name"], "已确认桥段")
        self.assertEqual(len(result["automatic_sections"]), 2)

    def test_overlapping_expert_sections_and_invalid_grid_are_rejected(self):
        self.external["segments"][1]["start"] = 15
        with self.assertRaises(ValueError):
            import_expert_structure(self.payload, self.external)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "grid.beats"
            write_expert_beats(destination, self.external)
            self.assertEqual(destination.read_text().splitlines()[0], "0.000000\t1")
            self.external["beat_positions"].pop()
            with self.assertRaises(ValueError):
                write_expert_beats(destination, self.external)

    def test_manual_name_edit_preserves_boundary_inside_shared_bar(self):
        self.external["segments"][0]["end"] = 15.5
        self.external["segments"][1]["start"] = 15.5
        original = import_expert_structure(self.payload, self.external, self.activations)
        self.assertEqual(original["sections"][0]["end_bar"], original["sections"][1]["start_bar"])
        manual = copy.deepcopy(original["sections"])
        manual[1]["name"] = "已确认副歌"
        edited = apply_manual_sections(original, manual, self.payload)
        self.payload["structure"] = edited
        result = import_expert_structure(self.payload, self.external, self.activations)
        self.assertEqual(result["sections"][1]["start"], 15.5)
        self.assertEqual(result["sections"][1]["name"], "已确认副歌")
        manual[1]["start"] = 15.0
        with self.assertRaises(ValueError):
            apply_manual_sections(original, manual, self.payload)

    def test_grid_comparison_distinguishes_beat_and_bar_phase(self):
        for bar in self.payload["bars"]:
            bar["beats"] = [{"start": bar["start"] + beat * 0.5, "beat": beat + 1} for beat in range(4)]
        proposal = {"beats": list(np.arange(0, 32, 0.5)), "downbeats": list(np.arange(0.5, 32, 2))}
        result = compare_beat_grids(self.payload["bars"], proposal)
        self.assertEqual(result["beat_agreement"], 1.0)
        self.assertEqual(result["downbeat_agreement"], 0.0)
        self.assertTrue(result["requires_review"])
        self.assertFalse(result["reference_is_ground_truth"])

    def test_uncovered_regions_are_not_downbeat_evidence(self):
        bars = [{"bar": 0, "start": 0, "end": 1, "beats": []},
                {"bar": 1, "start": 1, "end": 3, "beats": [{"start": 1, "beat": 3}, {"start": 2, "beat": 4}]}]
        result = compare_beat_grids(bars, {"beats": [0, 1, 2], "downbeats": [0, 1]})
        self.assertIsNone(result["downbeat_agreement"])

    def test_verified_decoder_shift_preserves_activation_correspondence(self):
        original = copy.deepcopy(self.external)
        shifted = shift_expert_timebase(self.external, -1., 31.)
        self.assertEqual(self.external, original)
        self.assertEqual(shifted["segments"][1]["start"], 15.)
        self.assertEqual(shifted["segments"][0]["start"], 0.)
        result = import_expert_structure(self.payload, shifted, self.activations)
        self.assertEqual(result["sections"][1]["role"], "chorus")
        self.assertAlmostEqual(result["sections"][1]["role_confidence"], .8)

    def test_mp3_clip_matches_decoded_full_audio_samples(self):
        """MP3 input seeking can corrupt warmup samples at the clip boundary."""
        import soundfile as sf
        from chord_recognizer import find_ffmpeg
        sample_rate, offset, duration = 44100, .731, 1.2
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            generator = np.random.default_rng(1234)
            source = generator.normal(0, .08, (sample_rate * 3, 2))
            sf.write(folder / "source.wav", source, sample_rate)
            subprocess.run([find_ffmpeg(), "-v", "error", "-y", "-i", str(folder / "source.wav"), "-codec:a", "libmp3lame", str(folder / "source.mp3")], check=True)
            _convert_window(folder / "source.mp3", folder / "full.wav", 0, 3)
            _convert_window(folder / "source.mp3", folder / "clip.wav", offset, duration)
            full, _ = sf.read(folder / "full.wav")
            clip, _ = sf.read(folder / "clip.wav")
            start = round(offset * sample_rate)
            np.testing.assert_array_equal(clip, full[start:start + len(clip)])


if __name__ == "__main__":
    unittest.main()
