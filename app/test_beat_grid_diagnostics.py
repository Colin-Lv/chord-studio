import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np

from beat_grid_diagnostics import diagnose
from downbeat_bridge import run_madmom_downbeats


def payload(times, positions, count=4, duration=None):
    return {"source": "synthetic", "duration": duration or float(times[-1] + 1), "beats_per_bar": count,
            "beat_source": "synthetic", "bars": [{"bar": 1, "beats": [
                {"start": float(time), "beat": int(position)} for time, position in zip(times, positions)]}]}


def synthetic_activations(times, positions, duration):
    values = np.full((int(round(duration * 100)), 2), .01)
    half_width = max(1, round(float(np.median(np.diff(times))) * 100 / 32))
    for time, position in zip(times, positions):
        frame = round(time * 100)
        values[max(0, frame-half_width):frame+half_width+1] = [.005, .99] if position == 1 else [.99, .005]
    return values


class BeatGridDiagnosticTests(unittest.TestCase):
    def test_phase_error_is_ranked_from_acoustic_evidence(self):
        times = .5 + np.arange(40) * .5
        truth = np.arange(40) % 4 + 1
        original = (np.arange(40) + 1) % 4 + 1
        item = payload(times, original, duration=21)
        result = diagnose(item, synthetic_activations(times, truth, 21))
        self.assertEqual(result["candidates"][0]["id"], "detected-phase1")
        self.assertFalse(result["applied"])
        self.assertEqual(item["bars"][0]["beats"][0]["beat"], 2)

    def test_double_density_is_ranked_lower_than_supported_half(self):
        times = .5 + np.arange(80) * .5
        source = payload(times, np.arange(80) % 4 + 1, duration=41)
        truth_times = times[::2]
        result = diagnose(source, synthetic_activations(truth_times, np.arange(len(truth_times)) % 4 + 1, 41))
        self.assertEqual(result["candidates"][0]["id"], "half0-phase0")

    def test_timing_without_evidence_does_not_select_correction(self):
        times = .5 + np.arange(40) * .5
        result = diagnose(payload(times, np.arange(40) % 4 + 1))
        self.assertIsNone(result["score_kind"])
        self.assertFalse(result["phase_verified"])
        self.assertTrue(all("acoustic_log_likelihood_per_frame" not in item for item in result["candidates"]))

    def test_flat_acoustics_cannot_disambiguate(self):
        times = .5 + np.arange(40) * .5
        result = diagnose(payload(times, np.arange(40) % 4 + 1, duration=21), np.full((2100, 2), 1/3))
        scores = [item["acoustic_log_likelihood_per_frame"] for item in result["candidates"]]
        self.assertLess(max(scores) - min(scores), 1e-12)

    def test_missing_position_suppresses_tempo_reinterpretation(self):
        times = .5 + np.arange(40) * .5
        numbers = np.arange(40) % 4 + 1
        numbers[10] = 1
        result = diagnose(payload(times, numbers))
        self.assertFalse(result["beat_positions_continuous"])
        self.assertTrue(all(item["tempo_variant"] == "detected" for item in result["candidates"]))

    def test_activation_window_mismatch_is_rejected(self):
        times = .5 + np.arange(40) * .5
        with self.assertRaisesRegex(ValueError, "duration"):
            diagnose(payload(times, np.arange(40) % 4 + 1, duration=21), np.full((1000, 2), .01))

    def test_rnn_clock_cannot_be_changed_only_in_decoder(self):
        with self.assertRaisesRegex(ValueError, "100 fps"):
            run_madmom_downbeats(Path("missing.wav"), [4], fps=50)

    def test_activation_export_keeps_original_return_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            source.write_bytes(b"placeholder")
            target = Path(directory) / "evidence" / "activation.npy"
            with patch("downbeat_bridge.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout='{"beats":[{"time":0.5,"beat":1}]}')) as process:
                rows = run_madmom_downbeats(source, [4], madmom_python=source, activations_output=target)
            self.assertEqual(rows, [(0.5, 1)])
            self.assertEqual(process.call_args.args[0][-1], str(target.resolve()))
            self.assertTrue(target.parent.is_dir())


if __name__ == "__main__":
    unittest.main()
