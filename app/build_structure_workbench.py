"""Build native and explicitly provisional expert-grid views of the same events."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from chord_studio import write_html, write_json
from audio_waveform import attach_waveform
from retime_analysis_to_downbeats import retime_payload
from song_structure import detect_harmonic_loops, grid_quality, valid_bars
from validate_analysis import audit_analysis


DISPLAY_GRID_FIELDS = {"bar", "beat_start", "beat_end", "position", "grid_status", "grid_placement", "retime_original_grid"}


def canonical_events(payload):
    """Preserve every event field except placement on a displayed beat grid."""
    return [{key: value for key, value in item.items() if key not in DISPLAY_GRID_FIELDS} for item in payload.get("events", [])]


def event_hash(payload):
    return hashlib.sha256(json.dumps(canonical_events(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def expert_grid_review(payload):
    expert = payload.get("structure", {}).get("expert_grid", {})
    beats, positions = expert.get("beats", []), expert.get("beat_positions", [])
    if not beats or len(beats) != len(positions):
        raise ValueError("No complete independent expert beat suggestion is available")
    duration = float(payload["duration"])
    # The model's 100 fps quantization may place a final point just beyond audio.
    # Keep only actually in-window markers; do not shift any point or event.
    rows = [(float(time), int(position)) for time, position in zip(beats, positions) if 0 <= float(time) <= duration]
    reviewed = retime_payload(payload, rows, "auto", beat_source="allin1_expert_review")
    if canonical_events(reviewed) != canonical_events(payload):
        raise ValueError("Expert display projection changed a canonical event")
    structure = reviewed["structure"]
    structure["grid_review_required"] = True
    structure["expert_grid"]["applied_to_display"] = True
    structure["expert_grid"]["applied"] = True
    structure["expert_grid"]["confirmed"] = False
    structure["expert_grid"]["comparison_reference"] = "original_model_grid"
    structure["grid_quality"] = grid_quality(reviewed, valid_bars(reviewed))
    structure["loops"] = detect_harmonic_loops(valid_bars(reviewed))
    structure.setdefault("warnings", []).insert(0, "当前小节排版采用专家拍点建议，尚需人工核对；原始和弦事件时间、标签与局部调性完全保留。")
    reviewed["grid_review_required"] = True
    reviewed["display_title"] = str(payload.get("display_title") or payload.get("audio_file") or "歌曲") + " · 专家拍格建议"
    reviewed["review_grid_provenance"] = {"status": "unconfirmed_model_suggestion", "reference_grid_used": False,
        "canonical_event_sha256": event_hash(payload), "canonical_events_unchanged": True,
        "native_beat_source": payload.get("beat_source"), "native_meter": payload.get("meter"),
        "audio_offset_unchanged": True, "audio_duration_unchanged": True}
    return reviewed


def build_workbenches(payload, output_base, display_title=None, include_expert_grid=True):
    payload = deepcopy(payload)
    attach_waveform(payload)
    if display_title:
        payload["display_title"] = display_title
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    native_json = output_base.with_name(output_base.name + "_native_analysis.json")
    native_html = output_base.with_name(output_base.name + "_native.html")
    native_audit = audit_analysis(payload)
    if not native_audit["valid"]:
        raise ValueError("Native payload failed interval audit")
    write_json(native_json, payload)
    write_html(native_html, payload)
    report = {"source": payload.get("source"), "original_source": payload.get("original_source"),
        "audio_offset": payload.get("audio_offset", 0), "duration": payload["duration"],
        "canonical_event_sha256": event_hash(payload), "event_count": len(payload.get("events", [])),
        "native_json": str(native_json), "native_html": str(native_html), "native_audit": native_audit,
        "role_section_count": len(payload.get("structure", {}).get("sections", [])),
        "expert_grid_is_ground_truth": False}
    if include_expert_grid:
        review = expert_grid_review(payload)
        review_audit = audit_analysis(review)
        if not review_audit["valid"]:
            raise ValueError("Expert grid review failed interval audit")
        review_json = output_base.with_name(output_base.name + "_expert_grid_review_analysis.json")
        review_html = output_base.with_name(output_base.name + "_expert_grid_review.html")
        write_json(review_json, review)
        write_html(review_html, review)
        report.update(review_json=str(review_json), review_html=str(review_html), review_audit=review_audit,
            canonical_events_unchanged=canonical_events(review) == canonical_events(payload),
            expert_review_bar_count=len(review["bars"]), expert_review_meter=review["meter"])
    output_base.with_name(output_base.name + "_integrity.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("-o", "--output-base", type=Path, required=True)
    parser.add_argument("--title")
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.analysis.read_text(encoding="utf-8-sig"))
    print(json.dumps(build_workbenches(payload, args.output_base, args.title, not args.native_only), ensure_ascii=False))


if __name__ == "__main__":
    main()
