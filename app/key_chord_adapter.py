"""Strict full-pitch chord semantics adapted to the experimental key interface.

Reuses the existing Harte/mir_eval semantics; no longest-prefix guessing.
This is a symbolic-label adapter, not evidence that all implied tones sound.
"""
import math
from dataclasses import dataclass

if __package__:
    from .chord_semantics_v3 import parse_chord
else:
    from chord_semantics_v3 import parse_chord


@dataclass
class KeyChord:
    start: float
    end: float
    label: str
    root: int
    kind: str
    third: object
    fifth: object
    seventh: object
    bass: object
    colors: list
    semantics: object

    @property
    def dur(self):
        return self.end-self.start


def adapt(label):
    """Return chord plus a diagnostic; preserve unknown vs silence vs missing."""
    value=parse_chord(label)
    if value.state!='chord':
        return None,dict(state=value.state,error=value.parse_error)
    # mir_eval's representation implies a root. Do not claim a rootless voicing.
    if any(d.number==1 and d.alteration==0 and d.omitted for d in value.degrees):
        return None,dict(state='X',error='Explicit root omission cannot be represented by this key interface')
    full={i for i,p in enumerate(value.chord_bitmap) if p}
    structural={i for i,p in enumerate(value.tetrad_bitmap) if p}
    quality=value.quality or ''
    # A #9 is a colour, not evidence that a dominant seventh is a minor triad.
    if quality.startswith(('min','dim','hdim')):third=3 if 3 in structural else None
    elif quality.startswith(('sus','5')):third=None
    elif quality:third=4 if 4 in structural else None
    else:third=4 if 4 in structural else 3 if 3 in structural else None
    fifth=next((x for x in (7,6,8) if x in structural),None)
    seventh=next((x for x in (11,10,9) if x in structural and x not in (third,fifth)),None)
    if third==3 and fifth==6:kind='dim'
    elif third==4 and fifth==8:kind='aug'
    elif third==3:kind='min'
    elif third==4:kind='dom' if seventh==10 else 'maj'
    elif quality.startswith('sus'):kind='sus'
    elif quality=='5':kind='power'
    else:kind='other'
    used={x for x in (0,third,fifth,seventh) if x is not None}
    colors=[(off,.35 if kind=='sus' and off in (2,5) else .20) for off in sorted(full-used)]
    return KeyChord(0,0,str(label),value.root_pc,kind,third,fifth,seventh,value.bass_pc,colors,value),dict(state='chord',error=None)


def parse_chord_label(label):
    return adapt(label)[0]


def parse_rows(rows):
    result=[];counts=dict(tokens=0,unparsed=0,N=0,X=0,missing=0);errors=[]
    for row in rows:
        start,end,label=(row['start'],row['end'],row.get('label')) if isinstance(row,dict) else row
        start,end=float(start),float(end)
        if not math.isfinite(start) or not math.isfinite(end) or start<0 or end<=start:
            raise ValueError('Invalid chord span')
        counts['tokens']+=1
        chord,diagnostic=adapt(label)
        if chord is None:
            counts[diagnostic['state']]+=1
            if diagnostic['state']=='X':
                counts['unparsed']+=1
                errors.append(dict(start=start,end=end,label=label,error=diagnostic['error']))
            continue
        chord.start,chord.end=start,end
        result.append(chord)
    return result,dict(**counts,errors=errors)
