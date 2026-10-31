import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import aggregate_largevocab_posteriors_v2 as aggregate
from largevocab_posteriors_v2 import HEAD_DIMENSIONS, make_evidence, save_posteriors


def setup_evidence(tmp_path):
    def file(name):
        path = tmp_path / name
        path.write_bytes(name.encode())
        return aggregate.signature(path)
    identity = dict(source_audio=file("audio"), recognizer=dict(
        models=[file(f"weight{i}") for i in range(5)], template=file("template"),
        implementation=file("original_bridge"), audio_decoder=file("decoder"),
        audio_decoder_implementation=file("decoder_code"), vendor_implementations=[file("vendor_code")]),
        bridge_implementation=file("bridge_v2"), posterior_implementation=file("posterior_v2"),
        requested_duration_seconds=None)
    data = make_evidence({name: np.eye(size, dtype=np.float32)[[0, 1, 2]]
                          for name, size in HEAD_DIMENSIONS.items()}, identity, decoded_samples=1280)
    path = tmp_path / "heads.npz"
    save_posteriors(path, data)
    return path, identity


def test_raw_boundary_default_and_union_preserve_acoustic_source(tmp_path):
    cache, identity = setup_evidence(tmp_path)
    raw = tmp_path / "raw.lab"
    raw.write_text("0 0.03 C\n0.03 0.08 X\n0.08 0.09 N\n")
    lv = tmp_path / "large.lab"
    lv.write_text("0 0.02 C:maj7\n0.02 0.09 C:maj\n")
    output = tmp_path / "raw.json"
    assert aggregate.main(["--posteriors", str(cache), "--raw-lab", str(raw), "--output", str(output)]) == 0
    current = json.loads(output.read_text())
    assert current["intervals"] == [[0, .03], [.03, .08], [.08, .09]]
    assert current["source_identity"] == identity
    assert current["zero_coverage_intervals"] == 1
    assert all(value is None for value in current["probabilities"]["triad"][-1])
    union = aggregate.run_aggregation(cache, raw, tmp_path / "union.json", anchors="union", largevocab_lab=lv)
    assert union["intervals"] == [[0, .02], [.02, .03], [.03, .08], [.08, .09]]
    assert union["source_identity"] == identity == union["metadata"]["source_identity"]
    assert union["anchor_source"]["experimental"] is True
    assert union["anchor_source"]["source_audio_identity_verified_for_anchor"] is False
    assert current["anchor_source"]["mode"] == "raw"
    assert current["reference_inputs"] is False and current["fitting"] is False


@pytest.mark.parametrize("which", ["audio", "weight0", "posterior_v2"])
def test_changed_audio_weight_or_frontend_rejects_cached_evidence(tmp_path, which):
    cache, identity = setup_evidence(tmp_path)
    (tmp_path / which).write_bytes(b"changed")
    raw = tmp_path / "raw.lab"
    raw.write_text("0 0.03 C\n")
    with pytest.raises(ValueError, match="Stale"):
        aggregate.run_aggregation(cache, raw, tmp_path / "output.json")
    assert not (tmp_path / "output.json").exists()


@pytest.mark.parametrize("text", ["", "0 1\n", "nan 1 C\n", "-1 2 C\n", "1 1 C\n", "0 1 C\n.5 2 G\n"])
def test_invalid_anchor_lab_is_rejected(tmp_path, text):
    path = tmp_path / "bad.lab"
    path.write_text(text)
    with pytest.raises(ValueError):
        aggregate.read_anchor_lab(path)


def test_anchor_generation_does_not_use_labels_or_round_boundaries():
    raw = [(0., .123456789, "C"), (.123456789, .3, "X")]
    changed_labels = [(row[0], row[1], "N") for row in raw]
    assert aggregate.boundary_anchors(raw) == aggregate.boundary_anchors(changed_labels)
    assert aggregate.boundary_anchors(raw)[0][1] == .123456789
    with pytest.raises(ValueError):
        aggregate.boundary_anchors(raw, mode="union")
