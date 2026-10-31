"""Lightweight bridge invariants; no pretrained model or torch allocation."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from sheetsage2_bridge import audit_windows, configure_context, shift_lab


class SheetSage2BridgeTests(unittest.TestCase):
    def model(self):
        return SimpleNamespace(config=SimpleNamespace(input_audio_length=300),
                               hparams=SimpleNamespace(input_audio_length=300),
                               tokenizer=SimpleNamespace(vocab_fingerprint="frozen", n_tokens=31678))

    def test_short_context_updates_both_padding_controls_without_tokenizer_change(self):
        model = self.model()
        tokenizer = model.tokenizer
        report = configure_context(model, 30)
        self.assertEqual(model.config.input_audio_length, 30)
        self.assertEqual(model.hparams.input_audio_length, 30)
        self.assertIs(model.tokenizer, tokenizer)
        self.assertEqual(model.tokenizer.n_tokens, 31678)
        self.assertTrue(report["experimental_short_context"])

    def test_invalid_context_is_rejected(self):
        for value in (0, -1, 301, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                configure_context(self.model(), value)

    def test_absolute_timestamps_preserve_duration_and_beat_number(self):
        self.assertEqual(shift_lab("1 2 C:maj7/3\n", 30), "31.000000\t32.000000\tC:maj7/3\n")
        self.assertEqual(shift_lab("1 2 4 4\n", 30, True), "31.000000\t2\t4\t4\n")
        self.assertEqual(shift_lab("", 30), "")

    def test_window_audit_handles_tail_overlap_without_double_counting(self):
        windows = [dict(start=0, accept_start=0, accept_end=30, tokens=230),
                   dict(start=25, accept_start=30, accept_end=55, tokens=280)]
        report = audit_windows(windows, 55, 1024)
        self.assertTrue(report["complete"])
        self.assertEqual(report["accepted_seconds"], 55)

    def test_token_limit_or_audio_gap_is_not_a_complete_song(self):
        report = audit_windows([dict(accept_start=0, accept_end=30, tokens=1025)], 30, 1024)
        self.assertFalse(report["complete"])
        self.assertEqual(report["token_limit_windows"], [0])
        with self.assertRaises(ValueError):
            audit_windows([dict(accept_start=1, accept_end=30, tokens=200)], 30, 1024)


if __name__ == "__main__":
    unittest.main()
