"""Shared LAB parsing for result creation and editor provenance checks."""
import math
from pathlib import Path


def canonical_rows(rows, duration=None, complete=False):
    """Validate generated events without rounding away real short chords.

    Numeric JSON and LAB use the same round-trip float representation. Only
    floating-point noise (1 ns) is snapped; musical gaps are never filled here.
    """
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError('Invalid audio duration')
    out=[]
    for original in rows:
        row=dict(original)
        a,b=float(row['start']),float(row['end'])
        if not math.isfinite(a) or not math.isfinite(b) or a<0 or b<=a:
            raise ValueError('Invalid chord interval: nonfinite, negative or nonpositive duration')
        if out:
            previous=out[-1]['end']
            if abs(a-previous)<=1e-9:a=previous
            if a<previous or (complete and a!=previous):
                raise ValueError('Invalid chord timeline: overlap or gap')
        elif complete and a!=0:
            raise ValueError('Incomplete chord timeline: missing start')
        if duration is not None and b>duration:
            if b-duration<=1e-9:b=float(duration)
            else:raise ValueError('Chord interval exceeds audio duration')
        if b<=a:raise ValueError('Chord interval collapsed at audio boundary')
        label=row.get('label')
        if not isinstance(label,str) or not label.strip() or any(c.isspace() for c in label):
            raise ValueError('Invalid chord label')
        row.update(id=len(out),start=a,end=b,start_text=repr(a),end_text=repr(b))
        out.append(row)
    if not out:raise ValueError('Empty chord timeline')
    if complete and duration is not None and abs(out[-1]['end']-duration)>1e-9:
        raise ValueError('Incomplete chord timeline: missing end')
    return out


def serialize_rows(rows, duration=None, complete=False):
    rows=canonical_rows(rows,duration,complete)
    return rows,''.join(f"{r['start_text']} {r['end_text']} {r['label']}\n" for r in rows)


def source_for_audio(source_id, name, path, run_dir, duration):
    """Import a model source, preserving raw output and clipping frame padding.

    External frame-based tools may emit a final frame past the audio end. Keep
    that original LAB immutable; publish a separately identified bounded file.
    More than 250 ms is treated as a clock mismatch, not assumed to be padding.
    """
    from .storage import signature,atomic_text
    path,run_dir=Path(path),Path(run_dir)
    rows=lab_rows(path);meta={}
    if rows[-1]['end']>duration:
        if rows[-1]['end']-duration>.25:
            raise ValueError('Model source exceeds audio by more than a frame-padding allowance')
        raw=dict(path=str(path.relative_to(run_dir)),**signature(path))
        bounded=[dict(r,end=min(r['end'],duration)) for r in rows if r['start']<duration]
        rows,content=serialize_rows(bounded,duration)
        path=path.with_name(path.stem+'_audio_bounds.lab')
        atomic_text(path,content)
        meta['audio_bounds']=dict(original=raw,policy='clip_external_frame_padding',duration=duration)
    canonical_rows(rows,duration)
    return dict(id=source_id,name=name,path=str(path.relative_to(run_dir)),**signature(path),rows=rows,**meta)


def lab_rows(path):
    rows, previous = [], 0.0
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        left, right, label = line.split(maxsplit=2)
        start, end = float(left), float(right)
        if not math.isfinite(start + end) or start < previous or end <= start or start < 0:
            raise ValueError(f"Invalid LAB timeline: {path}")
        rows.append(dict(id=len(rows), start=start, end=end, start_text=left, end_text=right, label=label))
        previous = end
    if not rows:
        raise ValueError(f"Empty LAB: {path}")
    return rows
