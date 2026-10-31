#!/usr/bin/env python
"""Sample and audit Guitar Pro files inside a supported archive.

Windows tar can list/extract the local Arobas `.rar` archive in this workspace.
This script wraps that flow and feeds the extracted sample through the existing
Guitar Pro importer and quality filter.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from filter_guitarpro_manifest_quality import classify, lab_stats, write_manifest, write_quality
from import_guitarpro import find_gp_files, write_error_report, write_import


REPO_ROOT = Path(__file__).resolve().parent.parent
GP_SUFFIXES = {".gp", ".gp3", ".gp4", ".gp5", ".gpx"}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def run_tar(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["tar", *args],
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def list_archive(archive: Path) -> list[str]:
    completed = run_tar(["-tf", str(archive)])
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def gp_entries(entries: list[str]) -> list[str]:
    return [entry for entry in entries if Path(entry).suffix.lower() in GP_SUFFIXES]


def select_entries(entries: list[str], pattern: str | None, limit: int, offset: int) -> list[str]:
    selected = gp_entries(entries)
    if pattern:
        regex = re.compile(pattern, re.IGNORECASE)
        selected = [entry for entry in selected if regex.search(entry)]
    if offset > 0:
        selected = selected[offset:]
    if limit > 0:
        selected = selected[:limit]
    return selected


def extract_entries_chunked(archive: Path, entries: list[str], output_dir: Path, chunk_size: int) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for start in range(0, len(entries), chunk_size):
        chunk = entries[start : start + chunk_size]
        list_file = output_dir / f"_selected_paths_{start:06d}.txt"
        list_file.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        try:
            run_tar(["-xf", str(archive), "-C", str(output_dir), "-T", str(list_file)])
        except subprocess.CalledProcessError as exc:
            if chunk_size <= 1:
                errors.append(
                    {
                        "source_file": chunk[0] if chunk else "",
                        "error_type": "TarExtractError",
                        "error": (exc.stderr or str(exc)).replace("\r", " ").replace("\n", " "),
                    }
                )
                continue
            errors.extend(extract_entries_chunked(archive, chunk, output_dir, max(1, chunk_size // 5)))
    return errors


def extract_entries(archive: Path, entries: list[str], output_dir: Path, chunk_size: int = 200) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    list_file = output_dir / "_selected_paths.txt"
    list_file.write_text("\n".join(entries) + "\n", encoding="utf-8")
    try:
        run_tar(["-xf", str(archive), "-C", str(output_dir), "-T", str(list_file)])
        return []
    except subprocess.CalledProcessError:
        return extract_entries_chunked(archive, entries, output_dir, max(1, chunk_size))


def import_directory(source_dir: Path, dataset: str, output_root: Path) -> tuple[list[object], list[dict[str, str]]]:
    results = []
    errors: list[dict[str, str]] = []
    for path in find_gp_files(source_dir):
        try:
            results.append(write_import(path, dataset, output_root))
        except Exception as exc:
            errors.append(
                {
                    "source_file": str(path),
                    "error_type": type(exc).__name__,
                    "error": str(exc).replace("\r", " ").replace("\n", " "),
                }
            )
    return results, errors


def write_import_manifest(results: list[object], manifest_path: Path, dataset: str) -> None:
    fieldnames = [
        "id",
        "dataset",
        "split",
        "artist",
        "title",
        "key",
        "meter",
        "audio",
        "prediction",
        "chord_reference",
        "beat_reference",
        "notes",
    ]
    rows = []
    for item in results:
        rows.append(
            {
                "id": item.item_id,
                "dataset": dataset,
                "split": "symbolic",
                "artist": item.artist,
                "title": item.title,
                "key": item.key,
                "meter": item.meter,
                "audio": "",
                "prediction": "",
                "chord_reference": (
                    str(item.chords_lab.relative_to(REPO_ROOT / "benchmarks")).replace("\\", "/")
                    if item.chords_lab
                    else ""
                ),
                "beat_reference": (
                    str(item.beats_file.relative_to(REPO_ROOT / "benchmarks")).replace("\\", "/")
                    if item.beats_file
                    else ""
                ),
                "notes": item.notes,
            }
        )
    write_manifest(manifest_path, rows, fieldnames)


def split_quality(
    manifest_path: Path,
    accepted_path: Path,
    needs_review_path: Path,
    quality_path: Path,
    min_events: int,
    min_unique: int,
    max_duration: float,
) -> dict[str, int]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        manifest_rows = list(csv.DictReader(handle))
    accepted: list[dict[str, str]] = []
    needs_review: list[dict[str, str]] = []
    quality_rows = []
    for row in manifest_rows:
        ref = row.get("chord_reference", "")
        lab = resolve_path(Path("benchmarks") / ref) if ref else Path("__missing__")
        stats = lab_stats(lab)
        status, reason = classify(stats, min_events, min_unique, max_duration)
        (accepted if status == "accepted" else needs_review).append(row)
        quality_rows.append(
            {
                "id": row.get("id", ""),
                "title": row.get("title", ""),
                "artist": row.get("artist", ""),
                "key": row.get("key", ""),
                "meter": row.get("meter", ""),
                "chord_events": stats["chord_events"],
                "unique_chords": stats["unique_chords"],
                "total_labeled": f"{float(stats['total_labeled']):.3f}",
                "max_duration": f"{float(stats['max_duration']):.3f}",
                "status": status,
                "reason": reason,
                "unique_list": stats["unique_list"],
            }
        )
    fieldnames = list(manifest_rows[0].keys()) if manifest_rows else []
    write_manifest(accepted_path, accepted, fieldnames)
    write_manifest(needs_review_path, needs_review, fieldnames)
    write_quality(quality_path, quality_rows)
    return {
        "manifest_rows": len(manifest_rows),
        "accepted": len(accepted),
        "needs_review": len(needs_review),
        "with_chords": sum(int(row["chord_events"]) > 0 for row in quality_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--dataset", default="arobas_audit")
    parser.add_argument("--pattern", help="Case-insensitive regex matched against archive paths.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / ".tmp" / "guitarpro_archive_audit")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "benchmarks" / "raw" / "arobas_audit")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "benchmarks" / "manifest.arobas_audit.csv")
    parser.add_argument("--accepted", type=Path, default=REPO_ROOT / "benchmarks" / "manifest.arobas_audit_accepted.csv")
    parser.add_argument("--needs-review", type=Path, default=REPO_ROOT / "benchmarks" / "manifest.arobas_audit_needs_review.csv")
    parser.add_argument("--quality-report", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "arobas_audit_quality.csv")
    parser.add_argument("--error-report", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "arobas_audit_errors.csv")
    parser.add_argument("--min-events", type=int, default=8)
    parser.add_argument("--min-unique", type=int, default=3)
    parser.add_argument("--max-duration", type=float, default=80.0)
    parser.add_argument("--extract-chunk-size", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive = resolve_path(args.archive)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = resolve_path(args.work_dir) / f"{args.dataset}_{stamp}"
    entries = list_archive(archive)
    selected = select_entries(entries, args.pattern, args.limit, args.offset)
    if not selected:
        print(json.dumps({"selected": 0, "pattern": args.pattern}, ensure_ascii=False, indent=2))
        return 1
    extract_errors = extract_entries(archive, selected, work_dir, args.extract_chunk_size)
    results, errors = import_directory(work_dir, args.dataset, resolve_path(args.output_root))
    errors.extend(extract_errors)
    manifest = resolve_path(args.manifest)
    write_import_manifest(results, manifest, args.dataset)
    write_error_report(errors, resolve_path(args.error_report))
    quality = split_quality(
        manifest,
        resolve_path(args.accepted),
        resolve_path(args.needs_review),
        resolve_path(args.quality_report),
        args.min_events,
        args.min_unique,
        args.max_duration,
    )
    report = {
        "archive": str(archive),
        "pattern": args.pattern,
        "limit": args.limit,
        "offset": args.offset,
        "selected": len(selected),
        "imported": len(results),
        "errors": len(errors),
        **quality,
        "work_dir": str(work_dir),
        "manifest": str(manifest),
        "quality_report": str(resolve_path(args.quality_report)),
        "error_report": str(resolve_path(args.error_report)) if errors else "",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
