"""Comparison distinguishes byte formatting from actual event changes."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from benchmark_madmom_cache_reproducibility import compare_labs


def test_cache_comparison_preserves_byte_text_and_event_distinctions(tmp_path):
    old, fresh = tmp_path / "old.lab", tmp_path / "fresh.lab"
    old.write_bytes(b"0.000 1.000 C:maj\r\n1.000 2.000 N\r\n")
    fresh.write_bytes(b"0.000 1.000 C:maj\n1.000 2.000 N\n")
    result = compare_labs(old, fresh)
    assert not result["byte_identical"]
    assert result["text_identical"] and result["parsed_events_identical"]
    fresh.write_text("0.000 1.010 C:maj\n1.010 2.000 N\n")
    result = compare_labs(old, fresh)
    assert not result["parsed_events_identical"]
    assert result["differing_event_positions"] == 2
    assert result["first_differences"][0]["fresh"]["end"] == 1.01
