"""Give the fused source the same fields as every other source (lab path, sha, bytes)."""
from __future__ import annotations

from pathlib import Path

ENGINE = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\fusion_engine.py")
text = ENGINE.read_text(encoding="utf-8")

OLD = '''def apply_to_analysis(run_dir: Path, segments, meta: dict, source_id: str = "stem_fusion",
                      label: str = "分轨融合（实验）", make_default: bool = True) -> dict:
    """Add the fused timeline to analysis.json as an extra source (never replaces others)."""
    path = run_dir / "analysis.json"
    result = read_json(path)
    sources = [source for source in result["sources"] if source["id"] != source_id]
    sources.append(dict(id=source_id, label=label, rows=segments))
    result["sources"] = sources
    result.setdefault("stem_fusion", {}).update(meta)
    if make_default:
        result["default_source"] = source_id
    atomic_json(path, result)
    return result'''
NEW = '''def apply_to_analysis(run_dir: Path, segments, meta: dict, source_id: str = "stem_fusion",
                      label: str = "分轨融合（实验）", make_default: bool = True) -> dict:
    """Add the fused timeline to analysis.json as a real source (lab file + sha), like the others."""
    import hashlib

    path = run_dir / "analysis.json"
    result = read_json(path)
    lab = run_dir / f"{source_id}.lab"
    lab.write_text("".join(f"{row['start']:.17g} {row['end']:.17g} {row['label']}\\n"
                           for row in segments), encoding="utf-8", newline="\\n")
    raw = lab.read_bytes()
    sources = [source for source in result["sources"] if source["id"] != source_id]
    sources.append(dict(id=source_id, name=label, path=lab.name,
                        sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), rows=segments))
    result["sources"] = sources
    result.setdefault("stem_fusion", {}).update(meta)
    if make_default:
        result["default_source"] = source_id
    atomic_json(path, result)
    return result'''
assert text.count(OLD) == 1
text = text.replace(OLD, NEW)

OLD_TAIL = '''    apply_to_analysis(run_dir, segments, meta, make_default=make_default)
    (run_dir / "stem_fusion.lab").write_text(
        "".join(f"{row['start']:.17g} {row['end']:.17g} {row['label']}\\n" for row in segments),
        encoding="utf-8")
    return meta'''
NEW_TAIL = '''    apply_to_analysis(run_dir, segments, meta, make_default=make_default)
    return meta'''
assert text.count(OLD_TAIL) == 1
text = text.replace(OLD_TAIL, NEW_TAIL)
ENGINE.write_text(text, encoding="utf-8")
print("fusion_engine: source now carries lab path + sha")
