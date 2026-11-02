"""Canonical clocks and recovery of sustained chords lost by coarse decoding."""
import numpy as np
from .full_decode import NAMES,QUALITIES
from .stem_harmony import pool


def canonical_cuts(rows, downbeats, duration, tolerance=.020):
    """Preserve chord edges; a nearby grid edge is an alias, not a new event.

    The tolerance addresses clock quantization, not minimum chord duration.
    Two genuine chord boundaries are never merged with one another.
    """
    from .source_rows import canonical_rows
    rows=canonical_rows(rows)
    if not np.isfinite(duration) or duration<=0:raise ValueError('Invalid audio duration')
    anchors=sorted({0.,float(duration),*[min(float(r[k]),duration) for r in rows for k in ('start','end')]})
    if anchors[0]<0 or any(float(r['end'])>duration+.05 for r in rows):
        raise ValueError('原和弦时间轴超出音频')
    cuts=set(anchors);snaps=[]
    for value in downbeats:
        t=float(value)
        if not np.isfinite(t):raise ValueError('Invalid beat time')
        if not 0<t<duration:continue
        index=int(np.searchsorted(anchors,t))
        nearest=min(anchors[max(0,index-1):min(len(anchors),index+1)],key=lambda a:abs(a-t))
        if abs(nearest-t)<=tolerance:
            if abs(nearest-t)>1e-8:snaps.append(dict(grid_time=t,chord_time=nearest))
        else:cuts.add(t)
    return sorted(cuts),snaps


def recover_sustained_chords(rows,posterior,nprob,bounds,beats,coarse_label,bass_at):
    """Recover a frozen winner sustained for a beat, backed by independent bass.

    Only interior events with a beat of context on both sides are recovered.
    Edge candidates belong to boundary placement, not swallowed-chord repair.
    A 10:1 pooled label-evidence ratio is required to overturn the coarse label.
    This is not an accuracy probability. Qualities come from the frozen winner;
    no seventh or extension is invented from the requested reference.
    """
    positive=np.diff(np.asarray(beats,dtype=float));positive=positive[positive>0]
    if not len(positive):return list(rows),[]
    minimum=float(np.median(positive));winners=np.asarray(posterior).argmax(1)
    winners=np.where(np.asarray(nprob).reshape(-1)>=.5,-1,winners)
    groups=[];left=0
    for stop in range(1,len(winners)+1):
        if stop<len(winners) and winners[stop]==winners[left] and abs(bounds[stop,0]-bounds[stop-1,1])<1e-6:continue
        a,b=float(bounds[left,0]),float(bounds[stop-1,1]);k=int(winners[left]);left=stop
        if k>=0 and b-a>=minimum-1e-6:groups.append((a,b,k))
    out=[];repairs=[]
    for row in rows:
        candidates=[(a,b,k) for a,b,k in groups
                    if a-row['start']>=minimum-1e-6 and row['end']-b>=minimum-1e-6
                    and NAMES[k//10]!=row['label'].split(':')[0]]
        if not candidates or row['label'] in ('N','X'):
            out.append(dict(row));continue
        current=coarse_label(row);root=current.split(':')[0]
        try:
            note,quality=current.split('/')[0].split(':');current_index=NAMES.index(note)*10+QUALITIES.index(quality)
        except ValueError:
            out.append(dict(row));continue
        cursor=row['start']
        for a,b,k in candidates:
            if NAMES[k//10]==root:continue
            p=pool(posterior,bounds,a,b)
            if int(p.reshape(12,10).sum(1).argmax())!=k//10:continue
            ratio=float(p[k]/max(p[current_index],1e-9))
            if ratio<10:continue
            bass=bass_at(a,b)
            if bass.get('note')!=NAMES[k//10]:continue
            label=f'{NAMES[k//10]}:{QUALITIES[k%10]}'
            if a>cursor:out.append(dict(row,start=cursor,end=a))
            out.append(dict(start=a,end=b,label=label));cursor=b
            repairs.append(dict(start=a,end=b,previous=row['label'],coarse_output=current,label=label,
                                evidence_ratio=ratio,bass_note=bass['note'],minimum_seconds=minimum))
        if cursor<row['end']:out.append(dict(row,start=cursor))
    return out,repairs
