#!/usr/bin/env python
"""Render an existing analysis in the current offline workbench."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import webbrowser

from chord_studio import write_html
from validate_analysis import audit_analysis
from audio_waveform import attach_waveform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path, help="Existing *_analysis.json; preserved unchanged")
    parser.add_argument("-o", "--output", type=Path, help="Standalone HTML output")
    parser.add_argument("--audio", type=Path, help="Reconnect the matching original audio after moving files")
    parser.add_argument("--open", action="store_true", help="Open the exported page in the default browser")
    args = parser.parse_args()
    source = args.analysis.resolve()
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if args.audio:
        if not args.audio.is_file():
            parser.error(f"Audio file not found: {args.audio}")
        payload["source"] = str(args.audio.resolve())
        payload["audio_file"] = args.audio.name
        payload.pop("audio_waveform", None)
    output = args.output.resolve() if args.output else source.with_name(source.stem.removesuffix("_analysis") + "_workbench.html")
    if output == source:
        parser.error("The HTML output cannot overwrite the analysis JSON")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = audit_analysis(payload)
    try:
        attach_waveform(payload)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Audio waveform unavailable: {exc}")
    write_html(output, payload)
    print(f"Workbench: {output}")
    if not report["valid"]:
        print(f"Timeline audit found {len(report['issues'])} issues; review them in the workbench.")
    if args.open:
        webbrowser.open(output.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
