"""Experimental full-vocabulary harmony recognition from one audio file.

Runs three acoustic recognizers followed by a frozen candidate reliability
model. The result is a timed Harte LAB plus provenance; no musical reference
or annotated beat grid is accepted. Existing LABs can be supplied explicitly.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from madmom_bridge import DEFAULT_MADMOM_PYTHON, cache_matches_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", type=Path, required=True,
                        help="Frozen selected_train66_head.joblib with its .contract.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-lab", type=Path)
    parser.add_argument("--chordino-lab", type=Path)
    parser.add_argument("--largevocab-lab", type=Path)
    parser.add_argument("--note-cache", type=Path, default=Path(".tmp/nnls_seventh_features/notes"))
    parser.add_argument("--madmom-python", type=Path, default=DEFAULT_MADMOM_PYTHON)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--preserve-root", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep raw root decisions (default); --no-preserve-root enables the unrestricted experimental head")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    audio, output = args.audio.resolve(), args.output_dir.resolve()
    if not audio.is_file() or not args.model.is_file():
        parser.error("Audio and frozen model must exist")
    for path in (args.raw_lab, args.chordino_lab, args.largevocab_lab):
        if path is not None and not path.is_file():
            parser.error(f"Explicit source LAB does not exist: {path}")
    output.mkdir(parents=True, exist_ok=True)
    prediction = output / "harmony.lab"
    if prediction.exists():
        parser.error("harmony.lab already exists; use a new output directory")
    app = Path(__file__).resolve().parent
    env = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[variable] = str(args.threads)

    state = dict(status="running", stage="validate_model", completed_stages=[], experimental=True,
                 audio=str(audio), model=str(args.model.resolve()))

    def record_state():
        temporary = output / "result.json.tmp"
        temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(output / "result.json")

    def run_command(stage, command, cwd=None):
        state.update(status="running", stage=stage)
        record_state()
        try:
            subprocess.run(command, check=True, env=env, cwd=cwd)
        except Exception as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            record_state()
            raise
        state["completed_stages"].append(stage)
        record_state()

    def run(script, *arguments):
        run_command(script, [sys.executable, str(app / script), *map(str, arguments)])

    # Validate in a short-lived process before allocating acoustic networks;
    # keep the orchestration process small on machines with limited RAM.
    run_command("validate_model", [sys.executable, "-c",
        "import sys\nfrom apply_multisource_candidates import load_frozen_artifact\nload_frozen_artifact(sys.argv[1])",
        str(args.model.resolve())], cwd=app)

    manifest = output / "audio_only.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "audio"])
        writer.writeheader()
        writer.writerow(dict(id="audio", audio=str(audio)))
    raw = args.raw_lab.resolve() if args.raw_lab else output / "raw_madmom.lab"
    if not args.raw_lab:
        if raw.exists():
            valid, reason = cache_matches_audio(raw, audio)
            if not valid:
                state.update(status="failed", stage="raw_cache", error=reason)
                record_state()
                raise ValueError(f"Existing raw LAB cannot be reused: {reason}")
        else:
            run("madmom_bridge.py", audio, "--output", raw, "--madmom-python", args.madmom_python.resolve())
    chordino = args.chordino_lab.resolve() if args.chordino_lab else output / "chordino/labs/audio.lab"
    if not args.chordino_lab:
        run("generate_audio_only_chordino_labs.py", manifest, "--output", output / "chordino", "--expected-count", 1)
    largevocab = args.largevocab_lab.resolve() if args.largevocab_lab else output / "largevocab/labs/audio.lab"
    if not args.largevocab_lab:
        run("generate_audio_only_largevocab_labs.py", "--manifest", manifest, "--output-dir",
            output / "largevocab", "--expected-count", 1, "--threads", args.threads)
    apply_arguments = ["--audio", audio, "--raw", raw, "--chordino", chordino,
        "--largevocab", largevocab, "--model", args.model.resolve(), "--output", prediction,
        "--cache", args.note_cache.resolve()]
    if args.preserve_root:
        apply_arguments.append("--preserve-root")
    run("apply_multisource_candidates.py", *apply_arguments)
    result = dict(status="complete", experimental=True, audio=str(audio), output=str(prediction),
                  provenance=str(prediction.with_name(prediction.name + ".provenance.json")),
                  model=str(args.model.resolve()),
                  preserve_root=args.preserve_root,
                  source_labs=dict(raw=str(raw), chordino=str(chordino), largevocab=str(largevocab)),
                  timing="Original raw intervals are preserved; candidate model changes labels only",
                  reference_inputs=False, fitting=False)
    state.update(result, stage="complete")
    record_state()
    print(f"Harmony LAB: {prediction}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
