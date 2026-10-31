"""Read-only audit of chord/bar timelines before review, training or scoring."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def audit_analysis(payload: dict) -> dict:
    issues = []
    duration = float(payload.get("duration", 0))
    if not math.isfinite(duration) or duration <= 0:
        issues.append({"severity": "error", "kind": "invalid_duration"})

    def check(rows, label):
        valid = []
        for index, row in enumerate(rows):
            try:
                start, end = float(row["start"]), float(row["end"])
            except (KeyError, TypeError, ValueError):
                issues.append({"severity": "error", "kind": "missing_interval", "timeline": label, "index": index})
                continue
            if not math.isfinite(start + end) or start < 0 or end <= start or end > duration + .01:
                issues.append({"severity": "error", "kind": "invalid_interval", "timeline": label,
                               "index": index, "start": start, "end": end})
                continue
            valid.append((start, end, index, row))
        if [row[0] for row in valid] != sorted(row[0] for row in valid):
            issues.append({"severity": "warning", "kind": "unsorted", "timeline": label})
        ordered = sorted(valid, key=lambda item: (item[0], item[1]))
        active = []
        for start, end, index, row in ordered:
            active = [item for item in active if item[1] > start + .003]
            for before in active:
                issues.append({"severity": "error", "kind": "overlap", "timeline": label,
                               "indices": [before[2], index], "start": start, "end": min(before[1], end),
                               "labels": [before[3].get("chord"), row.get("chord")]})
            active.append((start, end, index, row))
        return ordered

    events = check(payload.get("events", []), "events")
    bars = check(payload.get("bars", []), "bars")
    spans = []
    for _, _, _, bar in bars:
        if bar.get("grid_status") == "uncovered":
            issues.append({"severity": "warning", "kind": "uncovered_grid_region", "bar": bar.get("bar"),
                           "start": bar["start"], "end": bar["end"]})
        for span in bar.get("spans", []):
            spans.append(dict(span, bar=bar.get("bar")))
            if float(span.get("start", 0)) < float(bar["start"]) - .003 or float(span.get("end", 0)) > float(bar["end"]) + .003:
                issues.append({"severity": "error", "kind": "span_outside_bar", "bar": bar.get("bar")})
        check(bar.get("beats", []), f"bar_{bar.get('bar')}_beats")
    span_rows = check(spans, "bar_spans")
    if "bars" in payload and (events or span_rows):
        endpoints = sorted({value for start, end, _, _ in events + span_rows for value in (start, end)})
        mismatched_seconds = 0.
        for start, end in zip(endpoints, endpoints[1:]):
            mid = (start + end) / 2
            event_labels = {row.get("chord", "N") for a, b, _, row in events if a <= mid < b}
            span_labels = {row.get("chord", "N") for a, b, _, row in span_rows if a <= mid < b}
            if event_labels != span_labels:
                mismatched_seconds += end - start
        if mismatched_seconds > .01:
            issues.append({"severity": "error", "kind": "events_bars_disagree", "seconds": mismatched_seconds})
    return {"valid": not any(issue["severity"] == "error" for issue in issues), "issues": issues,
            "event_count": len(payload.get("events", [])), "bar_count": len(payload.get("bars", []))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path, nargs="+")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    reports = [{"path": str(path), **audit_analysis(json.loads(path.read_text(encoding="utf-8-sig")))} for path in args.analysis]
    content = json.dumps(reports, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content)
    return int(any(not report["valid"] for report in reports))


if __name__ == "__main__":
    raise SystemExit(main())
