"""Performance-preserving MIDI export and conservative beat-relative annotation.

Pitch/velocity/note inventory come exclusively from the transcription model.
No chord-template notes are added and no non-chord tones are removed.
"""
from fractions import Fraction
import math
from pathlib import Path
import numpy as np

PPQ = 6720  # exact binary, triplet, quintuplet and septuplet subdivisions


def explicit_grid(bpm, first_downbeat, beats_per_bar, duration):
    """A user-supplied constant grid, never presented as an acoustic estimate."""
    if not (20<=bpm<=300 and 0<=first_downbeat<duration and 2<=beats_per_bar<=12):
        raise ValueError('手动节拍参数超出范围，第一小节起点必须在音频结束之前')
    step=60./bpm
    return [dict(time=float(t),beat=i%beats_per_bar+1)
            for i,t in enumerate(np.arange(first_downbeat,duration,step))]


class BeatMap:
    def __init__(self, rows, duration):
        self.rows = [dict(time=float(r['time']), beat=int(r['beat'])) for r in rows]
        self.duration = float(duration)
        if not self.rows:
            # A failed beat tracker must not fabricate bars or prevent MIDI
            # export. A fixed MIDI clock preserves seconds, with no quantizing.
            self.period=.5;self.meter=4;self.first_q=0
            self.times=np.array([0.,max(.001,self.duration)])
            self.quarters=self.times*2;self.bar_rows=[]
            return
        times = np.array([r['time'] for r in self.rows])
        if len(times) < 3 or not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0):
            raise ValueError('拍点不足或顺序不正确，无法生成带小节的 MIDI')
        if any(r['beat'] < 1 or r['beat'] > 16 for r in self.rows):
            raise ValueError('拍号位置无效')
        self.period = float(np.median(np.diff(times)))
        self.meter = max(r['beat'] for r in self.rows)
        # Explicit unmetered lead-in. Preserve audio t=0, while placing the first
        # tracked downbeat on a DAW bar boundary. Never crop the introduction.
        q0 = math.ceil(times[0] / self.period / self.meter) * self.meter + self.rows[0]['beat'] - 1
        if times[0] == 0:
            q0 = 0
        self.first_q = q0
        self.times = times
        self.quarters = np.arange(len(times), dtype=float) + q0
        if times[0] > 0:
            self.times = np.r_[0., times]
            self.quarters = np.r_[0., self.quarters]
        self.bar_rows = []
        for index, r in enumerate(self.rows):
            if r['beat'] == 1:
                following = next((j for j in range(index+1, len(self.rows)) if self.rows[j]['beat'] == 1), len(self.rows))
                count = max(self.rows[j]['beat'] for j in range(index, following))
                if following == len(self.rows):
                    count = self.bar_rows[-1]['numerator'] if self.bar_rows else self.meter
                self.bar_rows.append(dict(number=len(self.bar_rows)+1, start=r['time'],
                                         quarter=float(q0+index), numerator=count, denominator=4))

    def quarter(self, seconds):
        if seconds > self.times[-1]:
            return float(self.quarters[-1]+(seconds-self.times[-1])/self.period)
        return float(np.interp(seconds, self.times, self.quarters))

    def seconds(self, quarter):
        if quarter > self.quarters[-1]:
            return float(self.times[-1]+(quarter-self.quarters[-1])*self.period)
        return float(np.interp(quarter, self.quarters, self.times))

    def position(self, seconds):
        if not self.rows:
            return dict(bar=0, beat=None, fraction=None, in_tracked_region=False)
        index = int(np.searchsorted([r['time'] for r in self.rows], seconds, side='right'))-1
        if index < 0:
            return dict(bar=0, beat=None, fraction=None, in_tracked_region=False)
        bar = next((r for r in reversed(self.bar_rows) if r['start'] <= seconds), None)
        q = self.quarter(seconds)
        within = q - (bar['quarter'] if bar else self.first_q)
        return dict(bar=bar['number'] if bar else 0, beat=math.floor(within)+1,
                    fraction=float(within % 1), in_tracked_region=seconds <= self.rows[-1]['time'])

    def tempos(self):
        entries = []
        for i in range(len(self.times)-1):
            usec = round(1e6*(self.times[i+1]-self.times[i])/(self.quarters[i+1]-self.quarters[i]))
            if not 0 < usec <= 0xffffff:
                raise ValueError('拍点间距超出 MIDI 速度范围')
            entries.append((round(self.quarters[i]*PPQ), usec))
        entries.append((round(self.quarters[-1]*PPQ), round(self.period*1e6)))
        return entries


def swing_regions(notes, grid):
    """Estimate a repeated offbeat phase; no assumption that swing equals 2:1."""
    result = []
    bars = grid.bar_rows
    for b, bar in enumerate(bars):
        # Use this bar, not its neighbours: a following triplet passage must
        # not impose swing on a preceding straight passage. Require a paired
        # beat/offbeat pattern; quintuplets/septuplets are not swing evidence.
        end = bars[b+1]['start'] if b+1 < len(bars) else grid.rows[-1]['time']
        phases = {}
        for n in notes:
            q = grid.quarter(n['start'])
            if abs(q-round(q)) <= .055:
                q = float(round(q))
            if bar['quarter'] <= q < grid.quarter(end):
                phases.setdefault(math.floor(q), []).append(q % 1)
        values=[]
        for events in phases.values():
            groups=[]
            for f in sorted(events):
                if groups and f-groups[-1][0] <= .055:groups[-1].append(f)
                else:groups.append([f])
            if len(groups)==2 and np.median(groups[0])<=.055 and .55<=np.median(groups[1])<=.76:
                values.append(float(np.median(groups[1])))
        if len(values) >= 3:
            center = float(np.median(values))
            spread = float(np.median(abs(np.asarray(values)-center)))
            if spread <= .035:
                result.append(dict(bar=bar['number'], start=bar['start'], end=end,
                                   offbeat_phase=center, ratio=center/(1-center),
                                   supporting_beats=len(values), phase_mad=spread,
                                   interpretation='swing_or_repeated_triplet_offbeat'))
    return result


def propose_grid(seconds, grid, swing):
    pos = grid.position(seconds)
    if not pos['in_tracked_region']:
        return seconds, dict(kind='unmetered', snapped=False)
    q = grid.quarter(seconds)
    # Prefer the closest supported subdivision. Simplicity only breaks a
    # sub-millisecond tie: a genuine septuplet must not become a nearby sextuplet.
    # Do not force every note
    # onto a lattice: grace notes, rolls and expressive timing stay untouched.
    candidates = []
    for divisor in (1, 2, 3, 4, 6, 8, 5, 7):
        target = round(q*divisor)/divisor
        error = abs(grid.seconds(target)-seconds)
        if error <= min(.04, .08*grid.period):
            candidates.append((divisor, error, target, 'subdivision'))
    region = next((r for r in swing if r['bar'] == pos['bar']), None)
    if region:
        target = math.floor(q)+region['offbeat_phase']
        error = abs(grid.seconds(target)-seconds)
        if error <= min(.04, .08*grid.period):
            candidates.insert(0, (0, error, target, 'swing'))
    if not candidates:
        return seconds, dict(kind='free', snapped=False)
    _, error, target, kind = min(candidates,key=lambda c:(round(c[1]*1000),c[0]))
    return grid.seconds(target), dict(kind=kind, snapped=error > 1e-9,
                                      grid_quarter=target,
                                      subdivision=str(Fraction(target % 1).limit_denominator(6720)),
                                      shift_ms=(grid.seconds(target)-seconds)*1000)


def export(raw_path, output, beats, duration, chord_rows=()):
    import pretty_midi
    import mido
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI(str(raw_path))
    notes = sorted([dict(pitch=int(n.pitch), velocity=int(n.velocity), start=float(n.start), end=float(n.end))
                    for inst in midi.instruments for n in inst.notes], key=lambda n:(n['start'],n['pitch']))
    controls = [dict(number=int(c.number), value=int(c.value), time=float(c.time))
                for inst in midi.instruments for c in inst.control_changes]
    grid = BeatMap(beats, duration)
    swings = swing_regions(notes, grid)
    annotations, quantized = [], []
    for index, n in enumerate(notes):
        start, info = propose_grid(n['start'], grid, swings)
        _, end_info = propose_grid(n['end'], grid, swings)
        # Key release is articulation, not necessarily a notated grid boundary.
        # Preserve held duration when moving an onset; pedals remain on the
        # original clock. Offset subdivision is annotation only.
        end = start+(n['end']-n['start'])
        end = min(duration, end)
        if end <= start:
            start, end = n['start'], n['end']
        quantized.append(dict(n, start=start, end=end))
        chord = next((r.get('label') for r in chord_rows if r['start'] <= n['start'] < r['end']), None)
        annotations.append(dict(id=index, **n, **grid.position(n['start']), rhythm=info,
                                release_rhythm=end_info, chord_context=chord,
                                held_quarters=grid.quarter(n['end'])-grid.quarter(n['start'])))
    # Keep repeated notes distinguishable. Never merge/delete two re-attacks
    # merely because quantization maps them to the same tick.
    for pitch in range(128):
        indexes = [i for i,n in enumerate(notes) if n['pitch']==pitch]
        for a,b in zip(indexes,indexes[1:]):
            if quantized[b]['start'] <= quantized[a]['start']:
                for i in (a,b):
                    quantized[i]=dict(notes[i])
                    annotations[i]['rhythm']=dict(kind='repeat_preserved',snapped=False)
        for a,b in zip(indexes,indexes[1:]):
            if notes[a]['end'] <= notes[b]['start'] and quantized[a]['end'] > quantized[b]['start']:
                quantized[a]['end'] = quantized[b]['start']
    for i,n in enumerate(quantized):
        annotations[i].update(grid_start=n['start'], grid_end=n['end'])

    def write(name, sequence):
        mf=mido.MidiFile(type=1,ticks_per_beat=PPQ)
        tempo=mido.MidiTrack();mf.tracks.append(tempo)
        meta=[(tick,0,mido.MetaMessage('set_tempo',tempo=value)) for tick,value in grid.tempos()]
        if not grid.rows or grid.rows[0]['time']>0:
            meta.append((0,1,mido.MetaMessage('marker',text='Unmetered introduction')))
        previous=None
        for bar in grid.bar_rows:
            tick=round(bar['quarter']*PPQ)
            if bar['numerator'] != previous:
                meta.append((tick,1,mido.MetaMessage('time_signature',numerator=bar['numerator'],denominator=4)))
                previous=bar['numerator']
            meta.append((tick,2,mido.MetaMessage('marker',text='Bar '+str(bar['number']))))
        for r in chord_rows:
            meta.append((round(grid.quarter(r['start'])*PPQ),3,mido.MetaMessage('text',text=str(r['label']))))
        track=mido.MidiTrack();mf.tracks.append(track)
        events=[(0,-2,mido.MetaMessage('track_name',name='Piano')), (0,-1,mido.Message('program_change',program=0))]
        for n in sequence:
            start=round(grid.quarter(n['start'])*PPQ)
            end=max(start+1,round(grid.quarter(n['end'])*PPQ))
            events.extend([(start,2,mido.Message('note_on',note=n['pitch'],velocity=n['velocity'])),
                           (end,0,mido.Message('note_off',note=n['pitch'],velocity=0))])
        for c in controls:
            events.append((round(grid.quarter(c['time'])*PPQ),1,
                           mido.Message('control_change',control=c['number'],value=c['value'])))
        for dest,source in ((tempo,meta),(track,events)):
            last=0
            for tick,_,msg in sorted(source,key=lambda r:(r[0],r[1])):
                msg.time=tick-last;last=tick;dest.append(msg)
            dest.append(mido.MetaMessage('end_of_track',time=max(0,round(grid.quarter(duration)*PPQ)-last)))
        mf.save(str(output/name))
    write('performance.mid',notes)
    write('grid.mid',quantized)
    return dict(notes=annotations, controls=controls, bars=grid.bar_rows, swing=swings,
                duration=duration, tempo_source='beat_tracker', quarter_note_bpm=60/grid.period,
                tracked_start=grid.rows[0]['time'] if grid.rows else None,
                tracked_end=grid.rows[-1]['time'] if grid.rows else None,
                count=len(notes), quantized_count=len(quantized),
                snapped_onsets=sum(abs(n['start']-q['start'])>1e-9 for n,q in zip(notes,quantized)),
                maximum_onset_shift_ms=max([abs(n['start']-q['start'])*1000 for n,q in zip(notes,quantized)] or [0]),
                time_signature_note='Beat positions use quarter-note units; compound-meter notation remains ambiguous.',
                harmony_policy='context markers only; no pitch or note-count modifications')
