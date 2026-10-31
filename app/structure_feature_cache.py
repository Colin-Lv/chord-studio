"""Content-bound structure feature caches; legacy unbound files remain untouched."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np

SCHEMA = 'structure_features_content_bound_v2'
WIDTHS = {'chroma': 12, 'timbre': 13, 'rhythm': 8, 'energy': 1}


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def array_digest(values) -> str:
    array = np.ascontiguousarray(values, dtype='<f8')
    result = hashlib.sha256(json.dumps(array.shape).encode('ascii'))
    result.update(array.tobytes())
    return result.hexdigest()


def cache_identity(audio_sha256, y, sr, offset, duration, grid, extractor):
    if not np.isfinite([offset, duration]).all() or offset < 0 or duration <= 0:
        raise ValueError('Invalid structure analysis window')
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError('Invalid decoded structure audio')
    if not np.isfinite(sr) or sr <= 0:
        raise ValueError('Invalid structure sample rate')
    if grid.ndim != 2 or grid.shape[1] != 2 or not np.isfinite(grid).all():
        raise ValueError('Invalid structure feature grid')
    audio_duration = len(y) / sr
    # Match the established production endpoint tolerance, but never reuse
    # the last spectral frame for a bar wholly outside the decoded window.
    if len(grid) and (np.any(grid[:, 0] >= audio_duration) or np.any(grid[:, 1] > audio_duration + 0.1)):
        raise ValueError('The bar grid extends beyond the supplied audio window')
    return dict(schema=SCHEMA, audio_sha256=audio_sha256, decoded_pcm_sha256=array_digest(y),
                decoded_samples=len(y), sample_rate=int(sr), offset_seconds=float(offset), requested_duration_seconds=float(duration),
                grid_sha256=array_digest(grid), grid_rows=len(grid),
                extractor_sha256=hashlib.sha256(inspect.getsource(extractor).encode('utf8')).hexdigest(),
                runtime={name: version(name) for name in ['numpy', 'librosa', 'scipy']})


def cache_path(directory: Path, identity: dict) -> Path:
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()
    return directory / 'v2' / (digest + '.npz')


def validate_features(features, count):
    if set(features) != set(WIDTHS):
        raise ValueError('Incomplete structure feature groups')
    for name, width in WIDTHS.items():
        values = np.asarray(features[name])
        if values.shape != (count, width) or not np.isfinite(values).all():
            raise ValueError('Invalid structure feature shape or values: ' + name)


def load_features(path, identity, grid):
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            metadata = json.loads(str(cached['metadata']))
            if metadata['identity'] != identity or not np.array_equal(cached['grid'], grid):
                return None
            features = {name: cached[name].copy() for name in WIDTHS}
        validate_features(features, len(grid))
        if any(array_digest(features[name]) != metadata['feature_sha256'][name] for name in WIDTHS):
            return None
        return features
    except (ValueError, KeyError, OSError, EOFError, TypeError, zipfile.BadZipFile):
        return None


def save_features(path, identity, grid, features):
    """Atomic own-cache update, retaining any invalid predecessor byte-for-byte."""
    validate_features(features, len(grid))
    path = path.resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(identity=identity, feature_sha256={name: array_digest(values) for name, values in features.items()})
    backup = None
    if path.exists():
        digest = file_digest(path); backup = path.with_name(path.stem + '.invalid-' + digest + '.npz')
        if not backup.exists():
            with backup.open('xb') as stream:
                stream.write(path.read_bytes())
        elif file_digest(backup) != digest:
            raise ValueError('Existing invalid-cache backup identity mismatch')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='structure-', suffix='.npz', delete=False) as stream:
            temporary = Path(stream.name).resolve()
            np.savez_compressed(stream, grid=grid, metadata=json.dumps(metadata, sort_keys=True, allow_nan=False), **features)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.parent == path.parent:
            temporary.unlink(missing_ok=True)
    return backup
