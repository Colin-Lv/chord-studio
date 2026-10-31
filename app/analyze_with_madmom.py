#!/usr/bin/env python3
"""One-command madmom chord recognition with Chord Studio HTML output."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from chord_studio import normalize_key_label
from extension_provenance import read_extension_provenance
from madmom_bridge import DEFAULT_MADMOM_PYTHON, cache_matches_audio


def default_output_base(audio: Path) -> Path:
    return audio.with_name(f"{audio.stem}_madmom_studio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Input audio file.")
    parser.add_argument("-o", "--output", type=Path, help="Output base path.")
    parser.add_argument("--key", help="Optional known key, e.g. E, F#m, D:aeolian.")
    parser.add_argument("--key-resolver", type=Path, help="Calibrated disagreement resolver JSON for experimental fused mode.")
    parser.add_argument("--key-source", choices=["functional", "madmom", "fused"], default="madmom",
                        help="Global key estimator when --key is omitted; madmom uses its dedicated key CNN.")
    parser.add_argument("--dominant-seventh-threshold", type=float, default=1.01,
                        help="Experimental template seventh inheritance: 1.01 disables it (default); 0.45 reproduces the legacy fusion.")
    parser.add_argument("--bass-root-weight", type=float, default=0.0,
                        help="Optional template bass-root bonus; default 0 leaves root decisions to chord evidence.")
    parser.add_argument("--infer-slash-bass", action=argparse.BooleanOptionalAction, default=False,
                        help="Opt into legacy inferred bass inversions; default preserves explicit source slash labels only.")
    parser.add_argument("--edge-fill", action=argparse.BooleanOptionalAction, default=True,
                        help="Preserve source chords outside the detected beat grid (default); disable for legacy clipping.")
    parser.add_argument("--edge-fill-include-no-chord", action=argparse.BooleanOptionalAction, default=True,
                        help="Preserve explicit source N labels at the audio edges; disable for legacy missing-N coverage.")
    parser.add_argument("--meter", default="auto", help="Meter, e.g. auto, 4/4, 6/8.")
    parser.add_argument("--beat-source", choices=["librosa", "madmom"], default="madmom")
    parser.add_argument("--bpm", type=float, help="Optional fixed BPM.")
    parser.add_argument(
        "--harmonic-unit",
        choices=["auto", "bar", "half-bar", "compound-half", "beat", "external"],
        default="auto",
        help="Grid unit for external chord aggregation.",
    )
    parser.add_argument(
        "--madmom-lab",
        type=Path,
        help="Reuse an existing madmom .lab file instead of generating one for this output name.",
    )
    parser.add_argument("--display", choices=["name", "roman"], default="roman")
    parser.add_argument("--force", action="store_true", help="Regenerate the madmom .lab file.")
    parser.add_argument("--keep-base-html", action="store_true", help="Keep the intermediate base HTML.")
    parser.add_argument("--open", action="store_true", help="Open final HTML in the default browser.")
    parser.add_argument("--madmom-python", type=Path, default=DEFAULT_MADMOM_PYTHON)
    parser.add_argument(
        "--progression-prior",
        type=Path,
        help="Optional learned Roman-numeral transition prior JSON.",
    )
    parser.add_argument("--progression-prior-strength", type=float, default=0.0,
                        help="Explicit prior weight. Default 0 avoids imposing a learned progression.")
    parser.add_argument("--structure", choices=["lightweight", "allin1", "none"], default="lightweight")
    parser.add_argument("--structure-python", type=Path,
                        default=Path(__file__).resolve().parent.parent / ".tmp" / "structure_env" / "Scripts" / "python.exe")
    parser.add_argument("--structure-model", default="harmonix-fold0")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.bass_root_weight) or args.bass_root_weight < 0:
        print("--bass-root-weight must be finite and nonnegative.", file=sys.stderr)
        return 2
    if not math.isfinite(args.dominant_seventh_threshold) or args.dominant_seventh_threshold < 0:
        print("--dominant-seventh-threshold must be finite and nonnegative.", file=sys.stderr)
        return 2
    if not math.isfinite(args.progression_prior_strength) or args.progression_prior_strength < 0:
        print("--progression-prior-strength must be finite and nonnegative.", file=sys.stderr)
        return 2
    audio = args.audio.resolve()
    if not audio.exists():
        print(f"Input audio not found: {audio}", file=sys.stderr)
        return 2

    output_base = (args.output.resolve() if args.output else default_output_base(audio).resolve())
    output_base.parent.mkdir(parents=True, exist_ok=True)
    app_dir = Path(__file__).resolve().parent
    bridge = app_dir / "madmom_bridge.py"
    studio = app_dir / "chord_studio.py"
    fusion = app_dir / "downbeat_external_fusion.py"
    key = normalize_key_label(args.key)
    key_estimation = None
    key_fusion = None
    progression_prior = args.progression_prior
    if progression_prior is not None and not progression_prior.is_file():
        print(f"Progression prior not found: {progression_prior}", file=sys.stderr)
        return 2
    if args.structure == "allin1" and not args.structure_python.is_file():
        print("Run setup_structure_env.ps1 or pass --structure-python before using --structure allin1.", file=sys.stderr)
        return 2

    lab_path = (
        args.madmom_lab.resolve()
        if args.madmom_lab
        else output_base.with_name(output_base.name + "_madmom.lab")
    )
    base_output = output_base.with_name(output_base.name + "_base_grid")
    base_analysis = base_output.with_name(base_output.name + "_analysis.json")

    if args.madmom_lab and not lab_path.exists():
        print(f"madmom lab not found: {lab_path}", file=sys.stderr)
        return 2
    cache_valid, cache_reason = cache_matches_audio(lab_path, audio) if lab_path.is_file() else (False, "missing LAB")
    if args.madmom_lab:
        print(f"Using supplied madmom lab: {lab_path}")
    elif args.force or not cache_valid:
        if lab_path.is_file() and not args.force:
            print(f"Regenerating automatic LAB cache: {cache_reason}")
        run(
            [
                args.python,
                str(bridge),
                str(audio),
                "-o",
                str(lab_path),
                "--madmom-python",
                str(args.madmom_python.resolve()),
            ]
        )
    else:
        print(f"Using existing madmom lab: {lab_path}")

    if key is None and args.key_source in {"madmom", "fused"}:
        from key_bridge import run_madmom_key
        key_estimation = run_madmom_key(audio, args.madmom_python)
        print(f"CNN evidence: {key_estimation['key']}", flush=True)
        if args.key_source == "madmom":
            key = key_estimation["key"]

    studio_command = [
        args.python,
        str(studio),
        str(audio),
        "-o",
        str(base_output),
        "--analysis-model",
        "template",
        "--mode",
        "seventh",
        "--feature-mode",
        "fast",
        "--beat-source",
        args.beat_source,
        "--madmom-python",
        str(args.madmom_python.resolve()),
        "--meter",
        args.meter,
        "--display",
        args.display,
        "--no-html",
        "--no-structure",
    ]
    if key:
        studio_command += ["--key", key]
    if args.bpm is not None:
        studio_command += ["--bpm", f"{args.bpm:.6f}"]
    if args.key_source == "fused" and not args.key:
        from key_fusion import run_neutral_and_fuse
        key_fusion = run_neutral_and_fuse(studio_command, base_analysis, lab_path, key_estimation, run,
                                          use_resolver=True, resolver_model=args.key_resolver)
        key = key_fusion["key"]
        print(f"Experimental key: {key} ({key_fusion['status']}); review={key_fusion['needs_review']}", flush=True)
    else:
        run(studio_command)
    # A model-derived key passed into the grid tool is still an automatic
    # estimate, not an explicit human key hint.
    if key_estimation is not None and key_fusion is None:
        base_payload = json.loads(base_analysis.read_text(encoding="utf-8"))
        base_payload["key_source"] = "madmom_cnn_key"
        base_payload["key_estimation"] = key_estimation
        base_analysis.write_text(json.dumps(base_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fusion_command = [
        args.python,
        str(fusion),
        str(base_analysis),
        str(base_analysis),
        str(lab_path),
        "-o",
        str(output_base),
        "--meter",
        args.meter,
        "--harmonic-unit",
        args.harmonic_unit,
        "--beat-grid-from-base",
        "--dominant-seventh-threshold",
        str(args.dominant_seventh_threshold),
        "--bass-root-weight",
        str(args.bass_root_weight),
        "--infer-slash-bass" if args.infer_slash_bass else "--no-infer-slash-bass",
        "--edge-fill" if args.edge_fill else "--no-edge-fill",
        "--edge-fill-include-no-chord" if args.edge_fill_include_no_chord else "--no-edge-fill-include-no-chord",
        "--edge-fill-mode", "segments",
        "--edge-fill-min-duration", "0.001",
        "--edge-fill-min-score", "0",
    ]
    if args.key or key_fusion is not None:
        fusion_command += ["--key", key]
    if progression_prior:
        fusion_command += [
            "--progression-prior",
            str(progression_prior.resolve()),
            "--progression-prior-strength",
            f"{args.progression_prior_strength:.6f}",
        ]
    run(fusion_command)

    html_path = output_base.with_name(output_base.name + "_studio.html")
    final_json = output_base.with_name(output_base.name + "_analysis.json")
    final_payload = json.loads(final_json.read_text(encoding="utf-8"))
    cache_valid, cache_reason = cache_matches_audio(lab_path, audio)
    final_payload["external_chord_cache"] = {"path": str(lab_path), "explicitly_supplied": bool(args.madmom_lab),
        "source_content_verified": cache_valid, "verification": cache_reason}
    if args.madmom_lab:
        extension_record = read_extension_provenance(audio, lab_path)
        if extension_record is not None:
            final_payload["extension_provenance"] = extension_record
    final_json.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    from chord_studio import write_html
    write_html(html_path, final_payload)
    if key_fusion is not None:
        from key_fusion import attach_key_metadata
        attach_key_metadata(final_payload, key_fusion, key_estimation)
        final_json.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        write_html(html_path, final_payload)
    elif key_estimation is not None:
        final_payload = json.loads(final_json.read_text(encoding="utf-8"))
        final_payload["key_source"] = "madmom_cnn_key"
        final_payload["key_estimation"] = key_estimation
        final_json.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        from chord_studio import write_html
        write_html(html_path, final_payload)
    if args.structure == "lightweight":
        from chord_studio import write_html, write_json
        from song_structure import analyze_song_structure
        payload = json.loads(final_json.read_text(encoding="utf-8"))
        payload["structure"] = analyze_song_structure(payload)
        write_json(final_json, payload)
        write_html(html_path, payload)
    elif args.structure == "allin1":
        run([str(args.structure_python.resolve()), str(app_dir / "structure_expert.py"),
             str(final_json), "-o", str(final_json), "--model", args.structure_model])
        from chord_studio import write_html
        write_html(html_path, json.loads(final_json.read_text(encoding="utf-8")))
    if args.open:
        import webbrowser

        webbrowser.open(html_path.as_uri())
    if not args.keep_base_html:
        for suffix in ("_base_grid_bar_chords.csv", "_base_grid_bar_chords.txt"):
            try:
                output_base.with_name(output_base.name + suffix).unlink()
            except OSError:
                pass
    print(f"Final HTML: {html_path}")
    print(f"Final JSON: {output_base.with_name(output_base.name + '_analysis.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
