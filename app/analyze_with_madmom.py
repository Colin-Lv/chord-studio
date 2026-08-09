#!/usr/bin/env python3
"""One-command madmom chord recognition with Chord Studio HTML output."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from chord_studio import normalize_key_label
from madmom_bridge import DEFAULT_MADMOM_PYTHON


def default_output_base(audio: Path) -> Path:
    return audio.with_name(f"{audio.stem}_madmom_studio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Input audio file.")
    parser.add_argument("-o", "--output", type=Path, help="Output base path.")
    parser.add_argument("--key", help="Optional known key, e.g. E, F#m, D:aeolian.")
    parser.add_argument("--meter", default="auto", help="Meter, e.g. auto, 4/4, 6/8.")
    parser.add_argument("--beat-source", choices=["librosa", "madmom"], default="madmom")
    parser.add_argument("--bpm", type=float, help="Optional fixed BPM.")
    parser.add_argument(
        "--harmonic-unit",
        choices=["auto", "bar", "half-bar", "compound-half", "beat"],
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
    parser.add_argument("--progression-prior-strength", type=float, default=0.045)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
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
    progression_prior = args.progression_prior
    if progression_prior is None:
        candidate_prior = app_dir.parent / "benchmarks" / "reports" / "isophonics_beatles_progression_prior.json"
        if candidate_prior.exists():
            progression_prior = candidate_prior

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
    if args.madmom_lab:
        print(f"Using supplied madmom lab: {lab_path}")
    elif args.force or not lab_path.exists():
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
        "--meter",
        args.meter,
        "--display",
        args.display,
        "--no-html",
    ]
    if key:
        studio_command += ["--key", key]
    if args.bpm is not None:
        studio_command += ["--bpm", f"{args.bpm:.6f}"]
    run(studio_command)

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
    ]
    if key:
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
