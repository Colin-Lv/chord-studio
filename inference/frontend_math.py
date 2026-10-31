"""Frozen frontend math collected verbatim; no annotation readers or training imports."""
import csv,io,math
from dataclasses import dataclass
from typing import NamedTuple
from types import SimpleNamespace
import numpy as np
from chord_semantics_v2 import parse_chord
SAMPLE_RATE=44100
BLOCK_SIZE=16384
HOP_LENGTH=2048
MIDI_NOTES=np.arange(21,105,dtype=np.int16)
FEATURE_DIMENSION=97
SUPPORT_THRESHOLD=.10
DIMENSION=149
BASIC=('maj','min','sus2','sus4')
MAX_CELL_SECONDS=.25
TRANSFORM='@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n@prefix vamp: <http://purl.org/ontology/vamp/> .\n@prefix : <#> .\n:transform a vamp:Transform ;\n vamp:plugin <http://vamp-plugins.org/rdf/plugins/nnls-chroma#nnls-chroma> ;\n vamp:step_size "2048"^^xsd:int ;\n vamp:block_size "16384"^^xsd:int ;\n vamp:sample_rate "44100"^^xsd:float ;\n vamp:plugin_version "5" ;\n vamp:parameter_binding [ vamp:parameter [ vamp:identifier "chromanormalize" ] ; vamp:value "0"^^xsd:float ] ;\n vamp:parameter_binding [ vamp:parameter [ vamp:identifier "rollon" ] ; vamp:value "0"^^xsd:float ] ;\n vamp:parameter_binding [ vamp:parameter [ vamp:identifier "s" ] ; vamp:value "0.7"^^xsd:float ] ;\n vamp:parameter_binding [ vamp:parameter [ vamp:identifier "tuningmode" ] ; vamp:value "0"^^xsd:float ] ;\n vamp:parameter_binding [ vamp:parameter [ vamp:identifier "useNNLS" ] ; vamp:value "1"^^xsd:float ] ;\n vamp:parameter_binding [ vamp:parameter [ vamp:identifier "whitening" ] ; vamp:value "1"^^xsd:float ] ;\n vamp:output <http://vamp-plugins.org/rdf/plugins/nnls-chroma#nnls-chroma_output_semitonespectrum> .\n'
@dataclass(frozen=True)
class TimedChord:
    start: float
    end: float
    chord: str


class NoteEvidence(NamedTuple):
    times: np.ndarray
    activations: np.ndarray
    metadata: dict


def _validate_evidence(evidence):
    times, activations, metadata = evidence
    if (times.ndim != 1 or activations.shape != (84, len(times))
            or not np.all(np.isfinite(times)) or not np.all(np.isfinite(activations))
            or np.any(activations < 0) or np.any(times < 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("NNLS evidence requires finite increasing times and nonnegative (84,T) activations")
    if len(times) > 1 and not np.allclose(np.diff(times), HOP_LENGTH / SAMPLE_RATE, rtol=0, atol=1e-10):
        raise ValueError("Unexpected NNLS frame clock")
    duration = float(metadata["duration_seconds"])
    if not np.isfinite(duration) or duration < 0 or (len(times) and times[-1] >= duration + HOP_LENGTH / SAMPLE_RATE):
        raise ValueError("NNLS frames exceed decoded audio support")
    expected_positions = np.arange(0, int(metadata["decoded_samples"]), HOP_LENGTH, dtype=np.int64)
    if times.shape != expected_positions.shape or not np.allclose(times, expected_positions / SAMPLE_RATE, rtol=0, atol=1e-10):
        raise ValueError("NNLS frame clock must cover the entire decoded audio from sample zero")


def _read_csv(text):
    samples, frames = [], []
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        if len(row) != 85:
            raise ValueError(f"Expected timestamp and 84 NNLS values, got {len(row)} columns")
        position = float(row[0])
        if not np.isfinite(position) or position != round(position):
            raise ValueError("NNLS sample timestamp must be an integer")
        samples.append(int(position))
        frames.append([float(value) for value in row[1:]])
    activations = np.asarray(frames, dtype=np.float32).reshape(-1, 84).T
    return np.asarray(samples, dtype=np.int64), activations


def _longest_support(support):
    longest = np.zeros(12, dtype=np.float64)
    current = np.zeros(12, dtype=np.float64)
    for frame in support.T:
        current = np.where(frame, current + 1, 0)
        longest = np.maximum(longest, current)
    return longest / max(1, support.shape[1])


def features_for_intervals(evidence, rows):
    """97 current-interval features, rotated to each upstream chord's root.

    Groups: mean/q25/q75/support; low(A0..B2), mid(C3..B4), high(C5..G#7)
    means; longest support; base minor flag. Spectral mass is L1-normalized
    jointly across all notes in each frame, so register strengths stay honest.
    All quantiles include silence. N/unsupported qualities return zero rows.
    Sub-hop intervals use only the closest frame centre to their midpoint.
    """
    from chord_extension_model import chord_attributes

    _validate_evidence(evidence)
    times, notes, metadata = evidence
    output = np.zeros((len(rows), FEATURE_DIMENSION), dtype=np.float32)
    normalized = notes / np.maximum(notes.sum(axis=0, keepdims=True), 1e-12)
    registers = []
    for mask in (np.ones(84, dtype=bool), MIDI_NOTES < 48,
                 (MIDI_NOTES >= 48) & (MIDI_NOTES < 72), MIDI_NOTES >= 72):
        chroma = np.zeros((12, len(times)), dtype=np.float32)
        np.add.at(chroma, MIDI_NOTES[mask] % 12, normalized[mask])
        registers.append(chroma)
    for index, row in enumerate(rows):
        start, end = float(row.start), float(row.end)
        if not np.isfinite(start + end) or start < 0 or end <= start:
            raise ValueError("Intervals require finite nonnegative start and positive duration")
        if end > metadata["duration_seconds"] + 1 / SAMPLE_RATE:
            raise ValueError("Interval exceeds decoded audio duration; do not silently extrapolate")
        attributes = chord_attributes(row.chord)
        if attributes is None:
            continue
        output[index, -1] = attributes[1]
        if not len(times):
            continue
        left, right = np.searchsorted(times, [start, end], side="left")
        if left == right:
            left = int(np.argmin(np.abs(times - (start + end) / 2)))
            right = left + 1
        order = (attributes[0] + np.arange(12)) % 12
        current = registers[0][order, left:right]
        support = current >= SUPPORT_THRESHOLD
        output[index, :12] = current.mean(axis=1)
        output[index, 12:36] = np.quantile(current, [.25, .75], axis=1).reshape(-1)
        output[index, 36:48] = support.mean(axis=1)
        for register in range(3):
            output[index, 48 + 12 * register:60 + 12 * register] = registers[register + 1][order, left:right].mean(axis=1)
        output[index, 84:96] = _longest_support(support)
    return output


def make_cells(raw,duration):
    if not np.isfinite(duration) or duration<=0:raise ValueError('Invalid audio duration')
    cells=[];previous=-math.inf
    for event in raw:
        if not np.isfinite(event.start+event.end) or event.start<previous or event.end<=event.start or event.start<0:
            raise ValueError('Raw anchors must be finite ordered disjoint intervals')
        if event.end>duration+.100001:raise ValueError('Raw anchor exceeds supported source-frame rounding')
        previous=event.end;start,end=event.start,min(event.end,duration)
        if start>=end:continue
        for index in range(math.ceil((end-start)/MAX_CELL_SECONDS-1e-10)):
            left=start+index*MAX_CELL_SECONDS;right=min(end,left+MAX_CELL_SECONDS)
            cells.append(TimedChord(left,right,event.chord))
    return cells


def register_means(notes,groups):
    """Low/mid/high each L1-normalized per frame; silence stays zero, no labels."""
    _validate_evidence(notes)
    registers=[]
    for mask in [MIDI_NOTES<48,(MIDI_NOTES>=48)&(MIDI_NOTES<72),MIDI_NOTES>=72]:
        selected=notes.activations[mask]
        normalized=selected/np.maximum(selected.sum(axis=0,keepdims=True),1e-12)
        chroma=np.zeros((12,len(notes.times)),np.float32)
        np.add.at(chroma,MIDI_NOTES[mask]%12,normalized)
        registers.append(chroma)
    output=np.zeros((groups[-1]['end'] if groups else 0,36),np.float32)
    for group in groups:
        cell=group['cell'];semantic=parse_chord(cell.chord)
        if semantic.state!='chord' or semantic.parse_error:continue
        middle=(cell.start+cell.end)/2
        lo=max(0,middle-probe.CONTEXT_SECONDS/2);hi=min(notes.metadata['duration_seconds'],middle+probe.CONTEXT_SECONDS/2)
        left,right=np.searchsorted(notes.times,[lo,hi],side='left')
        if left==right:
            if not len(notes.times):continue
            left=int(np.argmin(np.abs(notes.times-middle)));right=left+1
        order=(semantic.root_pc+np.arange(12))%12
        value=np.concatenate([r[order,left:right].mean(axis=1) for r in registers])
        output[group['start']:group['end']]=value
    if not np.isfinite(output).all():raise ValueError('Nonfinite normalized evidence')
    return output


def root_basic(label):
    """Map only supported chord identities; N/unknown are not training classes."""
    s = parse_chord(label)
    if s.state != 'chord' or s.parse_error:
        return None
    bits = s.triad_bitmap
    if bits[4] and bits[7] and not bits[3]:
        quality = 0
    elif bits[3] and bits[7] and not bits[4]:
        quality = 1
    elif bits[2] and bits[7] and not any(bits[i] for i in (3, 4, 5)):
        quality = 2
    elif bits[5] and bits[7] and not any(bits[i] for i in (2, 3, 4)):
        quality = 3
    else:
        return None
    return int(s.root_pc), quality


def build_absolute_features(anchors, notes):
    """Same audio-only cells/windows as quality baseline; always absolute PCs."""
    duration = notes.metadata['duration_seconds']
    cells = probe.make_cells(anchors, duration)
    windows = []
    dummy_groups = []
    for i, cell in enumerate(cells):
        middle = (cell.start + cell.end) / 2
        windows.append(TimedChord(max(0., middle - probe.CONTEXT_SECONDS / 2),
                                  min(duration, middle + probe.CONTEXT_SECONDS / 2), 'C:maj'))
        dummy_groups.append(dict(start=i, end=i + 1,
                                 cell=TimedChord(cell.start, cell.end, 'C:maj')))
    global96 = features_for_intervals(notes, windows)[:, :96]
    register36 = register_means(notes, dummy_groups)
    flags = np.zeros((len(cells), 17), np.float32)
    for i, cell in enumerate(cells):
        current = root_basic(cell.chord)
        if current is None:
            flags[i, 16] = 1
        else:
            root, basic = current
            flags[i, root] = 1
            flags[i, 12 + basic] = 1
    x = np.concatenate((global96, register36, flags), axis=1)
    if x.shape != (len(cells), DIMENSION) or not np.isfinite(x).all():
        raise ValueError('Invalid absolute NNLS features')
    return x, cells


def energy_features(audio,rate,cells):
 power=np.mean(np.square(audio.astype('float64')),axis=1);prefix=np.r_[0.,np.cumsum(power)];n=len(power)
 def rms(start,end):
  left=max(0,min(n-1,round(start*rate)));right=max(left+1,min(n,round(end*rate)));return np.sqrt(max(0.,prefix[right]-prefix[left])/(right-left))
 frame_rms=np.array([rms(max(0,t-.01),min(n/rate,t+.01)) for t in np.arange(0,n/rate,.01)]);scale=max(float(np.quantile(frame_rms,.95)),1e-6)
 def level(v):return np.clip(20*np.log10(max(v,1e-10)/scale),-80,12)/20
 result=[]
 for c in cells:
  middle=(c['start']+c['end'])/2;a=level(rms(middle-.04,middle+.04));b=level(rms(c['start'],c['end']));d=level(rms(middle-.25,middle+.25));result.append([a,b,d,a-d])
 return np.asarray(result,dtype='float32')

probe=SimpleNamespace(make_cells=make_cells,CONTEXT_SECONDS=1.)
