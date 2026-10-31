"""Conservative timeline stabilization for retained10 predictions.

The frozen model (v80 weights, v88 audio entry) emits one label per madmom cell.
Nothing here changes the model, the weights, the acoustic frontend or the
per-cell posteriors: it only re-decodes the cell sequence and nudges segment
boundaries onto the bar grid, using the posteriors the model already produced.

Two rules, both deliberately conservative:

1. A segment is merged only when it is short **relative to its own neighbours**
   (so a passage that genuinely changes every beat keeps its chords). Within
   that, a same-root fragment (melody adds a 9th/11th to the current chord) is
   merged freely; a different-root fragment is merged only when the model's own
   posterior is barely above the runner-up, i.e. it was not confident anyway.
2. A boundary moves only onto a **downbeat**, and only within 250 ms. Nothing
   else is quantized, so the timeline stays as close to the audio as possible.

The frozen ``prediction.lab`` is never modified; the caller writes a new file.

Usage:
    python chord_timeline_stabilize.py --lab prediction.lab \
        --posterior posterior.npz --cells cells.json --beats beats.json \
        --output stabilized.lab
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITIES = ["maj", "min", "7", "maj7", "min7", "9", "maj9", "min9", "sus2", "sus4"]
N_STATE = 120
EPS = 1e-12

DEFAULTS = dict(
    rel_threshold=0.50,      # candidate if shorter than 50% of its local neighbours
    max_short=0.40,          # never touch a segment longer than this
    min_absolute=0.12,       # never touch a segment shorter than this (frame noise)
    window=1,                # neighbours considered on each side
    uncertain_margin=2.0,    # different-root merge only when top1-top2 < this (nats)
    downbeat_tolerance=0.25, # boundary snap radius around a downbeat
    min_length=0.12,         # never create a segment shorter than this
    same_root_max=0.50,      # collapse same-root quality churn below this duration
)


def read_lab(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        start, end, label = line.split(maxsplit=2)
        rows.append(dict(start=float(start), end=float(end), chord=label.strip()))
    return rows


def write_lab(segments: list[dict], path: Path) -> None:
    Path(path).write_text(
        "".join(f'{s["start"]:.6f} {s["end"]:.6f} {s["chord"]}\n' for s in segments),
        encoding="utf-8")


def label_state(label: str) -> int:
    if label == "N":
        return N_STATE
    root, _, quality = label.split("/")[0].partition(":")
    if root in ROOTS and quality in QUALITIES:
        return ROOTS.index(root) * 10 + QUALITIES.index(quality)
    return N_STATE


def root_of(label: str) -> str | None:
    root, _, _ = label.split("/")[0].partition(":")
    return root if root in ROOTS else None


def segments_from_cells(cells: list[dict], chord: np.ndarray, nprob: np.ndarray,
                        slash: list[int | None]) -> list[dict]:
    """Reproduce the frozen decode(), then group cells into segments."""
    segments: list[dict] = []
    for index, cell in enumerate(cells):
        if nprob[index] >= 0.5:
            label = "N"
        else:
            root, quality = divmod(int(np.argmax(chord[index])), 10)
            label = f"{ROOTS[root]}:{QUALITIES[quality]}"
            if slash[index] is not None:
                label += "/" + ROOTS[slash[index]]
        if segments and segments[-1]["chord"] == label and abs(segments[-1]["end"] - cell["start"]) < 1e-9:
            segments[-1]["end"] = cell["end"]
            segments[-1]["cells"].append(index)
        else:
            segments.append(dict(start=cell["start"], end=cell["end"], chord=label, cells=[index]))
    return segments


def merge_short(segments: list[dict], chord: np.ndarray, nprob: np.ndarray,
                params: dict) -> list[dict]:
    logp = np.log(np.maximum(chord, EPS))
    nlog = np.log(np.maximum(nprob, EPS))
    segments = [dict(s) for s in segments]
    changed = True
    while changed and len(segments) > 1:
        changed = False
        for i in range(len(segments)):
            seg = segments[i]
            duration = seg["end"] - seg["start"]
            neighbours = [s for j, s in enumerate(segments) if j != i and abs(j - i) <= params["window"]]
            if not neighbours:
                continue
            local = float(np.median([s["end"] - s["start"] for s in neighbours]))
            if duration >= max(params["min_absolute"], params["rel_threshold"] * local):
                continue
            if duration > params["max_short"]:
                continue

            def score(label: str) -> float:
                state = label_state(label)
                cells = seg["cells"]
                if state == N_STATE:
                    return float(np.mean([nlog[c] for c in cells]))
                return float(np.mean([logp[c, state] + nlog[c] for c in cells]))

            options = []
            if i > 0:
                options.append((i - 1, score(segments[i - 1]["chord"]), "left"))
            if i + 1 < len(segments):
                options.append((i + 1, score(segments[i + 1]["chord"]), "right"))
            if not options:
                continue
            index, value, side = max(options, key=lambda item: item[1])
            same_root = root_of(seg["chord"]) is not None and root_of(segments[index]["chord"]) == root_of(seg["chord"])
            if not same_root:
                if value > score(seg["chord"]) + 0.5:
                    pass  # a clearly better neighbour is enough
                else:
                    cells = seg["cells"]
                    top2 = np.sort(chord[cells], axis=1)[:, -2:]
                    confidence = float(np.mean(np.log(top2[:, 1] + EPS) - np.log(top2[:, 0] + EPS)))
                    if confidence >= params["uncertain_margin"]:
                        continue
            target = segments[index]
            if side == "left":
                target["end"] = seg["end"]
                target["cells"] = target["cells"] + seg["cells"]
            else:
                target["start"] = seg["start"]
                target["cells"] = seg["cells"] + target["cells"]
            segments.pop(i)
            changed = True
            break
    merged: list[dict] = []
    for seg in segments:
        if merged and merged[-1]["chord"] == seg["chord"] and abs(merged[-1]["end"] - seg["start"]) < 1e-9:
            merged[-1]["end"] = seg["end"]
            merged[-1]["cells"] = merged[-1]["cells"] + seg["cells"]
        else:
            merged.append(seg)
    return merged


def consolidate_same_root(segments: list[dict], max_short: float = 0.50) -> list[dict]:
    """Collapse short quality churn that never leaves the same root.

    A run of adjacent segments sharing one root (``A:maj9 / A:sus2 / A:maj``)
    is consolidated into the run's longest segment; only members shorter than
    ``max_short`` are absorbed, so a sustained quality change survives.
    """
    if not segments or max_short <= 0:
        return segments
    out: list[dict] = []
    i = 0
    while i < len(segments):
        j = i
        root = root_of(segments[i]["chord"])
        while j + 1 < len(segments) and root is not None and root_of(segments[j + 1]["chord"]) == root:
            j += 1
        run = segments[i:j + 1]
        if len(run) == 1:
            out.append(run[0])
            i = j + 1
            continue
        longest = max(range(len(run)), key=lambda k: run[k]["end"] - run[k]["start"])
        keep = [k for k in range(len(run))
                if k != longest and run[k]["end"] - run[k]["start"] >= max_short]
        if not keep:
            anchor = dict(run[longest])
            anchor["start"] = run[0]["start"]
            anchor["end"] = run[-1]["end"]
            anchor["cells"] = [c for seg in run for c in seg["cells"]]
            out.append(anchor)
        else:
            current = dict(run[0])
            current_keep = 0 in keep or 0 == longest
            for k in range(1, len(run)):
                seg = run[k]
                if k == longest or k in keep:
                    if current_keep:
                        out.append(current)
                    current = dict(seg)
                    current_keep = True
                else:
                    current["end"] = seg["end"]
                    current["cells"] = current["cells"] + seg["cells"]
            if current_keep:
                out.append(current)
        i = j + 1
    merged: list[dict] = []
    for seg in out:
        if merged and merged[-1]["chord"] == seg["chord"] and abs(merged[-1]["end"] - seg["start"]) < 1e-9:
            merged[-1]["end"] = seg["end"]
            merged[-1]["cells"] = merged[-1]["cells"] + seg["cells"]
        else:
            merged.append(seg)
    return merged


def snap_to_downbeats(segments: list[dict], downbeats: list[float],
                      tolerance: float, min_length: float) -> list[dict]:
    if not downbeats or len(segments) < 2:
        return segments
    downbeats = sorted(downbeats)
    out = [dict(s) for s in segments]
    for i in range(1, len(out)):
        boundary = out[i]["start"]
        nearest = min(downbeats, key=lambda b: abs(b - boundary))
        if abs(nearest - boundary) > tolerance:
            continue
        if nearest - out[i - 1]["start"] < min_length or out[i]["end"] - nearest < min_length:
            continue
        out[i - 1]["end"] = nearest
        out[i]["start"] = nearest
    return out


def stabilize(cells: list[dict], chord: np.ndarray, nprob: np.ndarray,
              slash: list[int | None] | None = None,
              downbeats: list[float] | None = None,
              params: dict | None = None) -> list[dict]:
    params = {**DEFAULTS, **(params or {})}
    slash = slash or [None] * len(cells)
    segments = segments_from_cells(cells, chord, nprob, slash)
    segments = consolidate_same_root(segments, params["same_root_max"])
    segments = merge_short(segments, chord, nprob, params)
    segments = snap_to_downbeats(segments, downbeats or [], params["downbeat_tolerance"], params["min_length"])
    for seg in segments:
        seg.pop("cells", None)
    return segments


def load_inputs(lab_path: Path, posterior_path: Path, cells_path: Path) -> tuple[list[dict], np.ndarray, np.ndarray, list[dict]]:
    cells = json.loads(Path(cells_path).read_text(encoding="utf-8"))
    with np.load(posterior_path, allow_pickle=False) as z:
        chord = np.asarray(z["chord"], dtype=np.float64)
        nprob = np.asarray(z["N"], dtype=np.float64)
    if chord.shape[0] != len(cells):
        raise ValueError("posterior rows do not match the cell grid")
    return read_lab(lab_path), chord, nprob, cells


def load_downbeats(beats_path: Path | None) -> list[float]:
    if beats_path is None or not Path(beats_path).exists():
        return []
    rows = json.loads(Path(beats_path).read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("beats", [])
    return [float(r["time"]) for r in rows if int(r.get("beat", 0)) == 1]


def stabilize_lab(lab_path: Path, posterior_path: Path, cells_path: Path,
                  output_path: Path, downbeats: list[float] | None = None,
                  params: dict | None = None) -> list[dict]:
    """Derive a stabilized .lab from a frozen run; the frozen file is untouched.

    Called by ``retained_worker`` after the beats stage so available downbeats
    can pull boundaries onto the bar grid. Pure functions, numpy only.
    """
    if Path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    _, chord, nprob, cells = load_inputs(lab_path, posterior_path, cells_path)
    segments = stabilize(cells, chord, nprob, downbeats=downbeats or [], params=params)
    write_lab(segments, output_path)
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", type=Path, required=True, help="frozen prediction.lab")
    parser.add_argument("--posterior", type=Path, required=True, help="posterior.npz from the run")
    parser.add_argument("--cells", type=Path, required=True, help="cells.json from the audio cache")
    parser.add_argument("--beats", type=Path, help="beats.json (madmom downbeats); optional")
    parser.add_argument("--output", type=Path, required=True, help="new .lab to write")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    _, chord, nprob, cells = load_inputs(args.lab, args.posterior, args.cells)
    downbeats = load_downbeats(args.beats)
    segments = stabilize(cells, chord, nprob, downbeats=downbeats)
    write_lab(segments, args.output)
    print(json.dumps(dict(output=str(args.output), segments=len(segments),
                          cells=len(cells), downbeats=len(downbeats)), ensure_ascii=False))


if __name__ == "__main__":
    main()
