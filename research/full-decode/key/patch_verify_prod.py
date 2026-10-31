"""Update the production verifier: the full decoder is now the default source."""
from pathlib import Path

PATH = Path(r"D:\ChordStudio\research\full-decode\verify_prod_full.py")

OLD = '''    # production wiring: the app must publish the source, keep span the default, and record
    # that the second tier is derived and not the default.
    analysis_path = run / "analysis.json"
    wired = dict(source_published=False, default_is_span=None, provenance_applied=None,
                 panel_rows=None)
    if analysis_path.is_file():
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        ids = [s["id"] for s in analysis.get("sources", [])]
        wired["source_published"] = "retained10_full" in ids
        wired["default_is_span"] = analysis.get("default_source") == "retained10_span"
        prov = (analysis.get("provenance") or {}).get("full_decode") or {}
        wired["provenance_applied"] = bool(prov.get("applied"))
        row = next((s for s in analysis["sources"] if s["id"] == "retained10_full"), None)
        wired["panel_rows"] = len(row["rows"]) if row else None
    print(f"  published as switchable source : {'PASS' if wired['source_published'] else 'FAIL'}")
    print(f"  default source still span      : "
          f"{'PASS' if wired['default_is_span'] else 'FAIL'}")
'''

NEW = '''    # production wiring: the full decoder is the published default, bar pooling stays
    # available as a switchable source, and the stage records that it ran.
    analysis_path = run / "analysis.json"
    wired = dict(source_published=False, default_is_full=None, span_available=None,
                 provenance_applied=None, panel_rows=None)
    if analysis_path.is_file():
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        ids = [s["id"] for s in analysis.get("sources", [])]
        wired["source_published"] = "retained10_full" in ids
        wired["default_is_full"] = analysis.get("default_source") == "retained10_full"
        wired["span_available"] = "retained10_span" in ids
        prov = (analysis.get("provenance") or {}).get("full_decode") or {}
        wired["provenance_applied"] = bool(prov.get("applied"))
        row = next((s for s in analysis["sources"] if s["id"] == "retained10_full"), None)
        wired["panel_rows"] = len(row["rows"]) if row else None
    print(f"  published as a source          : {'PASS' if wired['source_published'] else 'FAIL'}")
    print(f"  default source is the decoder  : "
          f"{'PASS' if wired['default_is_full'] else 'FAIL'}")
    print(f"  bar pooling still selectable   : "
          f"{'PASS' if wired['span_available'] else 'FAIL'}")
'''

text = PATH.read_text(encoding="utf-8")
count = text.count(OLD)
print(f"occurrences: {count}")
if count != 1:
    raise SystemExit("expected exactly one")
PATH.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("patched")
