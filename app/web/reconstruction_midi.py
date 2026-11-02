"""Lossless-in-time common MIDI clock, provenance and import-back checks."""
from pathlib import Path
import math
from .piano_midi import BeatMap, PPQ


def read_track(path,track_id,label,program,duration):
    import pretty_midi
    source=pretty_midi.PrettyMIDI(str(path))
    tracks=[]
    for index,inst in enumerate(source.instruments):
        notes=[];discarded=0;clipped=0
        for n in inst.notes:
            if not all(math.isfinite(x) for x in [n.start,n.end]) or n.start>=duration or n.end<=0 or n.end<=n.start:
                discarded+=1;continue
            start=max(0.,float(n.start));end=min(duration,float(n.end))
            clipped+=int(start!=n.start or end!=n.end)
            notes.append(dict(pitch=n.pitch,velocity=n.velocity,start=start,end=end))
        if not notes:continue
        tracks.append(dict(id=track_id if len(source.instruments)==1 else f'{track_id}_{index}',label=label,
                           program=int(inst.program if program is None else program),is_drum=bool(inst.is_drum),
                           model_program=int(inst.program),model_instrument_name=str(inst.name),
                           notes=notes,controls=[dict(number=c.number,value=c.value,time=float(c.time)) for c in inst.control_changes if 0<=c.time<=duration],
                           bends=[dict(pitch=b.pitch,time=float(b.time)) for b in inst.pitch_bends if 0<=b.time<=duration],
                           bounds_check=dict(discarded_outside_audio=discarded,clipped_to_audio=clipped)))
    return tracks


def write_midi(tracks,path,beats,duration):
    import mido
    grid=BeatMap(beats,duration)
    mf=mido.MidiFile(type=1,ticks_per_beat=PPQ)
    meta=[(tick,0,mido.MetaMessage('set_tempo',tempo=value)) for tick,value in grid.tempos()]
    if not beats or beats[0]['time']>0:meta.append((0,1,mido.MetaMessage('marker',text='Unmetered introduction')))
    meter=None
    for bar in grid.bar_rows:
        tick=round(bar['quarter']*PPQ)
        if meter!=bar['numerator']:
            meta.append((tick,1,mido.MetaMessage('time_signature',numerator=bar['numerator'],denominator=4)));meter=bar['numerator']
        meta.append((tick,2,mido.MetaMessage('marker',text='Bar '+str(bar['number']))))
    def add(events):
        dest=mido.MidiTrack();mf.tracks.append(dest);last=0
        for tick,_,msg in sorted(events,key=lambda e:(e[0],e[1])):
            dest.append(msg.copy(time=tick-last));last=tick
        dest.append(mido.MetaMessage('end_of_track',time=max(0,round(grid.quarter(duration)*PPQ)-last)))
    add(meta)
    melodic=[i for i in range(16) if i!=9];pitched=0
    for track in tracks:
        if not track['is_drum'] and pitched>=15:raise ValueError('超过 15 个有音高轨道，需要分端口导出')
        ch=9 if track['is_drum'] else melodic[pitched]
        if not track['is_drum']:pitched+=1
        events=[(0,-2,mido.MetaMessage('track_name',name=track['id'])),(0,-1,mido.Message('program_change',channel=ch,program=track['program']))]
        for n in track['notes']:
            start=round(grid.quarter(n['start'])*PPQ);end=max(start+1,round(grid.quarter(n['end'])*PPQ))
            events.extend([(start,2,mido.Message('note_on',channel=ch,note=n['pitch'],velocity=n['velocity'])),
                           (end,0,mido.Message('note_off',channel=ch,note=n['pitch'],velocity=0))])
        for c in track['controls']:
            events.append((round(grid.quarter(c['time'])*PPQ),1,mido.Message('control_change',channel=ch,control=c['number'],value=c['value'])))
        for b in track['bends']:
            events.append((round(grid.quarter(b['time'])*PPQ),1,mido.Message('pitchwheel',channel=ch,pitch=b['pitch'])))
        add(events)
    mf.save(str(path))


def verify_roundtrip(tracks,path):
    import pretty_midi
    midi=pretty_midi.PrettyMIDI(str(path))
    if len(midi.instruments)!=len(tracks):raise ValueError('导出后的轨道数量不一致')
    maximum=0.
    for track,inst in zip(tracks,midi.instruments):
        if inst.program!=track['program'] or inst.is_drum!=track['is_drum']:raise ValueError('导出改变乐器或鼓通道')
        a=sorted(track['notes'],key=lambda n:(n['pitch'],n['start'],n['end']))
        b=sorted(inst.notes,key=lambda n:(n.pitch,n.start,n.end))
        if len(a)!=len(b):raise ValueError('导出后的音符数量不一致')
        for x,y in zip(a,b):
            if (x['pitch'],x['velocity'])!=(y.pitch,y.velocity):raise ValueError('导出改变了音高或力度')
            maximum=max(maximum,abs(x['start']-y.start),abs(x['end']-y.end))
        if len(track['controls'])!=len(inst.control_changes) or len(track['bends'])!=len(inst.pitch_bends):raise ValueError('导出丢失控制事件')
        for x,y in zip(sorted(track['controls'],key=lambda c:c['time']),inst.control_changes):
            if (x['number'],x['value'])!=(y.number,y.value):raise ValueError('导出改变控制值')
            maximum=max(maximum,abs(x['time']-y.time))
        for x,y in zip(sorted(track['bends'],key=lambda b:b['time']),inst.pitch_bends):
            if x['pitch']!=y.pitch:raise ValueError('导出改变弯音值')
            maximum=max(maximum,abs(x['time']-y.time))
    if maximum>.002:raise ValueError(f'导出后时序误差过大：{maximum:.6f} 秒')
    return dict(notes=sum(len(t['notes']) for t in tracks),tracks=len(tracks),maximum_time_error_ms=maximum*1000,
                pitch_velocity_inventory_preserved=True,control_counts_preserved=True)
