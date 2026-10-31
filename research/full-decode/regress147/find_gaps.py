"""Re-run the regression cases whose published key fell back to the acoustic key.

Some span timelines pool into a single segment, so the chord->key evidence cannot run and
the worker publishes the madmom key. The first pass recorded no key there; this pass re-reads
those cases with the fallback recorded so the paired comparison covers all 147.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    cases = []
    for path in sorted((HERE / "cases").glob("*/result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data["stable"].get("final_key") or not data["span"].get("final_key"):
            cases.append(data["case"])
    print(f"{len(cases)} cases need a re-read: {cases}")
    for case in cases:
        data = json.loads((HERE / "cases" / case / "result.json").read_text(encoding="utf-8"))
        print(case, "stable rows", data["stable"]["rows"], "span rows", data["span"]["rows"],
              "evidence keys", list(data["stable"]["evidence"])[:4],
              "|", list(data["span"]["evidence"])[:4])


if __name__ == "__main__":
    main()
