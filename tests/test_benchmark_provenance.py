import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from run_benchmark import evaluate_row, summarize


def test_legacy_grid_cannot_silently_masquerade_as_automatic(tmp_path):
    reference = tmp_path / "reference.beats"
    reference.write_text("".join(f"{i*.501:.6f} {i%4+1}\n" for i in range(40)))
    prediction = tmp_path / "prediction.json"
    prediction.write_text(json.dumps(dict(beat_source="madmom", bars=[dict(beats=[
        dict(start=i*.501, beat=i%4+1) for i in range(40)])])))
    result = evaluate_row(dict(id="legacy", prediction=str(prediction), beat_reference=str(reference)), tmp_path, .1, .07)
    assert result["reference_grid_exact_fraction"] == 1
    assert "grid_matches_reference_verify_provenance" in result["evaluation_context"]
    assert "provenance_note" in summarize([result])


def test_declared_reference_grid_and_alignment_are_preserved(tmp_path):
    prediction = tmp_path / "prediction.json"
    prediction.write_text(json.dumps(dict(beat_source="reference_grid", evaluation_context={"reference_key": True})))
    result = evaluate_row(dict(id="controlled", prediction=str(prediction), alignment_method="prediction_optimized_time_shift"), tmp_path, .1, .07)
    assert result["evaluation_context"].split(";") == ["prediction_optimized_reference_alignment", "reference_beat_grid", "reference_key"]
