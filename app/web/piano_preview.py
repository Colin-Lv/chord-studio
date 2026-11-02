"""Render a preview with the existing local SoundFont sampler (not ground truth)."""
from pathlib import Path
import sys
import numpy as np
import pretty_midi
import soundfile as sf


def render(source, target):
    from . import preview_sampler as renderer
    from app.runtime_paths import soundfont_path
    sampler=renderer.make_sampler_class(renderer.sf2)(soundfont_path())
    midi=pretty_midi.PrettyMIDI(str(source))
    # SoundFont sampler has no CC engine: extend only the audition copies of
    # held notes. MIDI itself retains independent key release and pedal events.
    for inst in midi.instruments:
        intervals=[];start=None
        for cc in sorted(inst.control_changes,key=lambda c:c.time):
            if cc.number!=64:continue
            if cc.value>=64 and start is None:start=cc.time
            elif cc.value<64 and start is not None:intervals.append((start,cc.time));start=None
        notes=sorted(inst.notes,key=lambda n:n.start)
        for i,n in enumerate(notes):
            for a,b in intervals:
                if a<=n.end<b:n.end=b;break
            following=next((x.start for x in notes[i+1:] if x.pitch==n.pitch),None)
            if following is not None:n.end=min(n.end,following)
    audio=renderer.render_pm(midi,sampler,sr=22050)
    peak=float(np.max(abs(audio))) if len(audio) else 0
    if peak>.98:audio*=.98/peak
    sf.write(str(target),audio,22050,subtype='PCM_16')


if __name__=='__main__':
    render(Path(sys.argv[1]),Path(sys.argv[2]))
