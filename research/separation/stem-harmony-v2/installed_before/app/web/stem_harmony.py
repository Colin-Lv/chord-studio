"""Stem-conditioned chord evidence, independent of key and human annotations.

V2 keeps the released segment decoder as the boundary proposal. It re-scores
all 120 chord hypotheses from accompaniment notes and a separate bass view.
No song-specific coefficients, annotation input, or silent decoder fallback.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from .storage import ROOT, atomic_json, atomic_text, now, read_json, signature
from .full_decode import INTERVALS, NAMES, QUALITIES

VERSION = 'stem-evidence-v2.0'
SOURCE = 'stem_evidence_v2'
# Explicit engineering defaults, fixed before the evaluation; not learned confidence.
PARAMS = dict(baseline_bonus=1.5, tone_weight=2.0, bass_weight=0.5)
LABELS = [f'{root}:{quality}' for root in NAMES for quality in QUALITIES]
MIDI = np.arange(21, 105)


def validate_cells(cells, count, duration):
    bounds = np.asarray([[c['start'], c['end']] for c in cells], dtype=float)
    if (bounds.shape != (count, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 1] <= bounds[:, 0]) or abs(bounds[0, 0]) > 0.002
            or abs(bounds[-1, 1] - duration) > 0.05
            or not np.allclose(bounds[1:, 0], bounds[:-1, 1], atol=1e-7)):
        raise ValueError('和弦后验时间轴不完整或与音频不一致，请重新分析')
    return bounds


def pool(values, bounds, start, end):
    """Integrate by overlap duration, never by array index or invented uniform cells."""
    weights = np.maximum(0, np.minimum(bounds[:, 1], end) - np.maximum(bounds[:, 0], start))
    if weights.sum() <= 0:
        raise ValueError('和弦区间不在声学证据覆盖范围内')
    return np.average(values, axis=0, weights=weights)


def run(command, env=None):
    result = subprocess.run(command, capture_output=True, env=env, timeout=900,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if result.returncode:
        raise RuntimeError('分轨音符提取失败：' + result.stderr.decode('utf8', errors='replace')[-1500:])
    return result.stdout


def extract_notes(audio, folder):
    """Use the installed official NNLS plugin; cache includes audio/tool/transform identity."""
    import ast
    from scipy.io import wavfile

    runtime = read_json(ROOT / 'inference/runtime.local.json')
    # Read the frozen transform literal without modifying or importing its entry module.
    tree = ast.parse((ROOT / 'inference/frontend_math.py').read_text(encoding='utf8'))
    transform = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'TRANSFORM' for t in node.targets))
    identity = dict(audio=signature(audio)['sha256'], transform=transform,
                    tools={k: signature(runtime[k]['path'])['sha256']
                           for k in ('ffmpeg', 'sonic_annotator', 'nnls_plugin')})
    cache = folder / hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    receipt_path = cache / 'complete.json'
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt['identity'] != identity or signature(cache / 'notes.npz')['sha256'] != receipt['sha256']:
            raise ValueError('分轨音符缓存校验失败')
    else:
        raw = run([runtime['ffmpeg']['path'], '-v', 'error', '-i', str(audio),
                   '-ac', '1', '-ar', '44100', '-f', 'f32le', 'pipe:1'])
        y = np.frombuffer(raw, dtype=np.float32)
        if not len(y) or not np.isfinite(y).all() or len(y) > 600 * 44100:
            raise ValueError('分轨音频为空、损坏或超过十分钟')
        wav = cache / 'input.wav'
        try:
            wavfile.write(wav, 44100, np.pad(y, (0, 16384)))
            atomic_text(cache / 'nnls.ttl', transform)
            env = os.environ.copy()
            env['VAMP_PATH'] = str(Path(runtime['nnls_plugin']['path']).parent)
            raw = run([runtime['sonic_annotator']['path'], '-q', '-t', str(cache / 'nnls.ttl'),
                       '-w', 'csv', '--csv-stdout', '--csv-omit-filename', '--csv-sample-timing',
                       '--csv-digits', '9', str(wav)], env)
            rows = np.asarray(list(csv.reader(io.StringIO(raw.decode('utf8')))), dtype=float)
            if rows.ndim != 2 or rows.shape[1] != 85 or not np.isfinite(rows).all():
                raise ValueError('NNLS 输出无效')
            rows = rows[(rows[:, 0] >= 0) & (rows[:, 0] < len(y))]
            if not np.array_equal(rows[:, 0], np.arange(0, len(y), 2048)) or np.any(rows[:, 1:] < 0):
                raise ValueError('NNLS 音符时钟或数值无效')
            times = rows[:, 0] / 44100
            # Absolute energy stops quiet separation residue being promoted to evidence.
            energy = np.array([np.sqrt(np.mean(y[int(s):min(int(s)+2048, len(y))]**2)) for s in rows[:, 0]])
            np.savez_compressed(cache / 'notes.npz', times=times, activations=rows[:, 1:].T,
                                energy=energy, duration=len(y)/44100)
            atomic_json(receipt_path, dict(identity=identity, sha256=signature(cache / 'notes.npz')['sha256']))
        finally:
            wav.unlink(missing_ok=True)
    with np.load(cache / 'notes.npz', allow_pickle=False) as z:
        return {k: z[k] for k in z.files}, str(cache)


def build_views(base, manifest, duration, output):
    import soundfile as sf
    ids = {t['id'] for t in manifest['tracks']}
    harmonic = [s for s in ('other', 'guitar', 'piano') if s in ids]
    if not harmonic and 'instrumental' in ids:
        harmonic = ['instrumental']
    if not harmonic:
        raise ValueError('当前分轨没有可用于和弦识别的伴奏轨')
    groups = {'accompaniment': harmonic}
    if 'bass' in ids:
        groups['bass'] = ['bass']
    inputs, views = {}, {}
    output.mkdir(parents=True, exist_ok=True)
    for role, names in groups.items():
        total, rate = None, None
        for name in names:
            path = base / 'wav' / f'{name}.wav'
            inputs[name] = signature(path)['sha256']
            audio, sr = sf.read(path, always_2d=True, dtype='float32')
            if not len(audio) or not np.isfinite(audio).all() or abs(len(audio)/sr-duration) > 0.05:
                raise ValueError('分轨与原音频时长不一致：' + name)
            if total is not None and (rate != sr or total.shape != audio.shape):
                raise ValueError('分轨采样率、长度或声道不一致：' + name)
            total = audio if total is None else total + audio
            rate = sr
        # Preserve source levels. Float WAV avoids clipping and arbitrary RMS boosts.
        path = output / (role + '.wav')
        sf.write(path, total, rate, subtype='FLOAT')
        views[role] = path
    return views, inputs, groups


def pitch_profile(evidence, start, end, bass=False):
    times, act = evidence['times'], evidence['activations']
    bounds = np.column_stack([times, np.minimum(times + 2048/44100, float(evidence['duration']))])
    mask = (MIDI < 60) if bass else (MIDI >= 48)
    pc = np.zeros((len(times), 12))
    for note in np.where(mask)[0]:
        pc[:, MIDI[note] % 12] += act[note]
    pc /= np.maximum(pc.sum(axis=1, keepdims=True), 1e-9)
    activity = np.clip((evidence['energy'] - 1e-4) / 0.002, 0, 1)
    profile = pool(pc * activity[:, None], bounds, start, end)
    reliability = float(pool(activity, bounds, start, end))
    return profile / max(profile.sum(), 1e-9), reliability


def score_chords(prior, original, tones, bass, tone_activity, bass_activity, params=None):
    p = dict(PARAMS, **(params or {}))
    score = np.log(np.maximum(prior, 1e-9))
    if original in LABELS:
        # The baseline includes contextual decoding that raw frame probabilities
        # do not. With no new evidence this layer must be an identity operation.
        score[LABELS.index(original)] = score.max() + p['baseline_bonus']
    support = tones / max(float(tones.max()), 1e-9)
    for index, label in enumerate(LABELS):
        root, quality = divmod(index, len(QUALITIES))
        pcs = [(root + interval) % 12 for interval in INTERVALS[QUALITIES[quality]]]
        fit = np.log(0.15 + 0.85 * support[pcs]).mean() - 2 * (1 - tones[pcs].sum())
        # Inversions and pedal notes are legal: bass is supporting evidence, never a root veto.
        compatible = 0.15 + 0.55 * bass[root] + 0.30 * bass[pcs].sum()
        score[index] += p['tone_weight'] * tone_activity * fit
        score[index] += p['bass_weight'] * bass_activity * np.log(compatible)
    return score


def recognize(run_dir, base, output, publish=lambda message: None):
    from .fusion_engine import full_decode_segments, segments_from
    started = time.perf_counter()
    analysis = read_json(run_dir / 'analysis.json')
    prediction = read_json(run_dir / 'retained10/prediction.json')
    duration = float(analysis['duration'])
    if not np.isfinite(duration) or not 0 < duration <= 600:
        raise ValueError('分轨和弦支持十分钟以内的有效音频')
    manifest = read_json(base / 'manifest.json')
    if manifest['audio_sha256'] != prediction['input_sha256']:
        raise ValueError('分轨与和弦分析来自不同音频')
    cache = Path(prediction['cache_directory'])
    with np.load(run_dir / 'retained10/posterior.npz') as z:
        posterior, nprob = z['chord'], z['N'].reshape(-1)
    if (posterior.ndim != 2 or posterior.shape[1] != 120 or not np.isfinite(posterior).all()
            or np.any(posterior < 0) or not np.allclose(posterior.sum(axis=1), 1, atol=1e-3)):
        raise ValueError('和弦后验格式错误')
    bounds = validate_cells(read_json(cache / 'cells.json'), len(posterior), duration)
    if len(nprob) != len(posterior) or not np.isfinite(nprob).all() or np.any((nprob < 0) | (nprob > 1)):
        raise ValueError('无和弦后验长度或数值错误')
    publish('准备伴奏与低音证据')
    views, identities, groups = build_views(base, manifest, duration, output)
    evidence, caches = {}, {}
    for role, audio in views.items():
        publish('提取伴奏构成音' if role == 'accompaniment' else '提取独立低音证据')
        evidence[role], caches[role] = extract_notes(audio, output / 'notes-cache')
    publish('结合分轨证据识别和弦')
    # Require a real decoder result. No hidden change of decoder upon failure.
    beats = read_json(run_dir / 'beats.json')
    boundary_policy = 'released full decoder; complete edge coverage'
    if sum(int(b.get('beat', 0)) == 1 for b in beats) < 3:
        baseline = [dict(start=r['start'], end=r['end'], label=r['chord']) for r in prediction['segments']]
        boundary_policy = 'short clip: original model segments (fewer than three downbeats)'
    else:
        baseline = full_decode_segments(run_dir, posterior)
    if not baseline:
        raise ValueError('完整段解码没有产出，请检查拍点分析')
    # Complete edge coverage with the original mixed-audio predictions, rather than
    # scoring only the convenient interior. Main boundaries remain identical.
    intervals = sorted({0., duration, *[float(r[k]) for r in baseline for k in ('start', 'end')]})
    diagnostics, labels, cells = [], [], []
    for start, end in zip(intervals, intervals[1:]):
        if end-start < 1e-6:
            continue
        prior = pool(posterior, bounds, start, end)
        original = next((r['label'] for r in baseline if r['start'] <= (start+end)/2 < r['end']), LABELS[int(prior.argmax())])
        tones, active = pitch_profile(evidence['accompaniment'], start, end)
        bass, bass_active = (pitch_profile(evidence['bass'], start, end, bass=True)
                             if 'bass' in evidence else (np.zeros(12), 0.))
        scores = score_chords(prior, original, tones, bass, active, bass_active)
        order = np.argsort(scores)[::-1][:3]
        label = LABELS[int(order[0])]
        # Only the original mix can propose N; removing vocals cannot invent silence.
        if float(pool(nprob, bounds, start, end)) >= 0.5:
            label = 'N'
        if active == 0 and bass_active == 0 and label != 'N':
            label = original
        diagnostics.append(dict(start=start, end=end, baseline=original, label=label,
                                accompaniment_activity=active, bass_activity=bass_active,
                                tones=tones.tolist(), bass=bass.tolist(),
                                alternatives=[dict(label=LABELS[int(i)], score=float(scores[i])) for i in order]))
        labels.append(label);cells.append((start, end))
    segments = segments_from(labels, cells)
    meta = dict(version=VERSION, created_at=now(), params=PARAMS, model=manifest['model'],
                source_audio_sha256=prediction['input_sha256'], input_stems=identities,
                groups=groups, evidence_caches=caches, weights_sha256=prediction['weights_sha256'],
                cells=len(bounds), segments=len(segments), seconds=round(time.perf_counter()-started, 2),
                key_used=False, reference_inputs=False, fitting=False,
                boundary_policy=boundary_policy,
                score_kind='uncalibrated hypothesis score, not accuracy',
                limitations=['10 chord qualities; no slash bass output', 'Boundary proposals inherited from full decoder',
                             'Separation may remove useful chord tones; improvement is not guaranteed'])
    atomic_json(output / 'diagnostics.json', dict(meta=meta, segments=diagnostics))
    atomic_json(output / 'result.json', dict(meta=meta, rows=segments, baseline=baseline))
    return segments, meta


def publish_result(run_dir, segments, meta, make_default=False):
    path = run_dir / 'analysis.json'
    result = read_json(path)
    # Content-addressed source avoids invalidating a user's already saved correction.
    content = ''.join(f"{r['start']:.6f} {r['end']:.6f} {r['label']}\n" for r in segments)
    digest = hashlib.sha256(content.encode()).hexdigest()
    source_id = SOURCE + '_' + digest[:12]
    lab = run_dir / (source_id + '.lab')
    atomic_text(lab, content)
    sources = [s for s in result['sources'] if s['id'] != source_id]
    sources.append(dict(id=source_id, name='分轨构成音和弦 · V2', path=lab.name,
                        **signature(lab), rows=segments))
    result.update(sources=sources, stem_harmony_v2=dict(meta, source_id=source_id))
    if make_default:
        result['default_source'] = source_id
    atomic_json(path, result)
    return source_id
