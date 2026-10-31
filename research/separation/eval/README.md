# Stem separation evaluation harness

A small, deterministic, dependency-light evaluator for the WAV stems produced by
ChordStudio. It has two independent modes:

- **Reference mode** (`--truth`): SDR and SI-SDR per stem when ground-truth
  stems are available.
- **Reference-free mode** (`--mix`): reconstruction residual, per-stem level and
  activity, and a cross-stem envelope-correlation matrix when they are not.

No GPU, model download, network access, or real multitrack data is required.
The only runtime dependencies are `numpy` and `soundfile`; tests also use
`pytest`.

All paths below are inside:

```text
D:\ChordStudio\research\separation\eval\
```

---

## Quick start

Use the Python interpreter that already has the required packages:

```powershell
$py = "D:\ChordStudio\Dialogue\.tmp\structure_env\Scripts\python.exe"
$eval = "D:\ChordStudio\research\separation\eval"

& $py "$eval\demo.py"
& $py -m pytest -q "$eval\tests"
```

The demo creates three synthetic stems and a mix in a temporary folder, runs
both modes, and prints one JSON object.

### Reference-free mode

```powershell
& "D:\ChordStudio\Dialogue\.tmp\structure_env\Scripts\python.exe" `
  "D:\ChordStudio\research\separation\eval\stem_eval.py" `
  --mix "D:\path\to\mix.wav" `
  --stems "D:\ChordStudio\data\stems\<audio_sha>\htdemucs\wav"
```

Point `--stems` at the folder that directly contains `vocals.wav`,
`drums.wav`, etc. (normally the model's `wav` subfolder, not the `htdemucs`
parent). Output is JSON on stdout; diagnostics go to stderr.

### Reference mode

```powershell
& "D:\ChordStudio\Dialogue\.tmp\structure_env\Scripts\python.exe" `
  "D:\ChordStudio\research\separation\eval\stem_eval.py" `
  --truth "D:\path\to\ground_truth_stems" `
  --stems "D:\path\to\estimated_stems"
```

`--mix` and `--truth` are mutually exclusive. Both modes require `--stems`.
The exit code is `0` on success, `2` for input/evaluation errors.

---

## Input contract

- Stems are matched by **canonical filename**, case-insensitively:
  `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`, `guitar.wav`,
  `piano.wav`.
- Only `.wav` files directly inside the given folder are considered. The scan
  is not recursive. Other filenames and other extensions are ignored.
- At least one supported stem must be present. In reference mode, the truth
  and estimate folders must contain the same set of supported stems; a missing
  or extra recognized stem is an error.
- All signals used together must have exactly the same sample rate, channel
  count, and number of frames. The harness deliberately **does not resample,
  trim, pad, align, or downmix**; do that upstream so it cannot silently hide a
  pipeline mistake.
- WAVs are loaded as float64 via `soundfile`; dBFS assumes full scale `1.0`.
  Integer PCM, FLOAT and DOUBLE WAVs are supported by the loader.

---

## Metrics

All JSON dB values are rounded to 4 decimal places. Correlations are rounded to
6 decimal places. Silences and mathematically undefined values are handled as
described below; the CLI always emits strict JSON (no `NaN` or `Infinity`).

### SI-SDR (reference mode)

Each channel is mean-centred, then all channels are flattened into one vector.
The estimate is projected onto the reference:

```text
alpha   = <s_hat, s> / ||s||^2
target  = alpha * s
SI-SDR  = 10 log10( ||target||^2 / ||s_hat - target||^2 )
```

Higher is better. SI-SDR is invariant to a global gain and polarity flip. It is
reported for each matched stem and averaged **unweighted** over stems with a
defined score. The score is clipped to `[-200, 200]` dB for JSON safety; an
exactly perfect estimate therefore shows `200.0`, and a silent reference gives
`null`.

Important: scale invariance is a feature of the metric, not proof of quality.
A gain-mangled or polarity-inverted copy can score extremely well.

### SDR (reference mode)

This is the **plain, unfiltered SDR**:

```text
SDR = 10 log10( sum(s^2) / sum((s - s_hat)^2) )
```

Higher is better. It is not BSS Eval v4: no time-invariant distortion filter is
allowed, no window is optimized, and no source permutation is searched. Gain
and phase mismatches are therefore penalized, and these numbers should not be
compared directly with published BSS Eval/museval scores. The same
`[-200, 200]` dB clipping and unweighted mean rule applies.

### Reconstruction residual (reference-free mode)

For the set of estimated stems:

```text
residual            = mix - sum(estimated stems)
residual_db_rel_mix = 10 log10( sum(residual^2) / sum(mix^2) )
```

More negative is better. An exactly reconstructing set of stems reports the
`-200.0` dB floor. A silent mix reports `null` because the relative ratio is
undefined. Also reported:

- `residual_rms_dbfs` — absolute residual level;
- `stem_sum_rms_dbfs` — level of the sum of estimates;
- `mix_rms_dbfs` — level of the original mix.

This metric checks consistency, not correctness: four swapped or mislabelled
stems can reconstruct the mix perfectly. See limitations below.

### RMS dBFS (reference-free mode)

```text
RMS_dBFS = 20 log10( RMS(samples) / 1.0 )
```

Computed over every frame and channel of the stem. Digital silence is reported
at the `-200 dBFS` floor. This is level, not loudness: it is not LUFS, not
K-weighted, and not loudness-normalized.

### Active ratio (reference-free mode)

Frames are 20 ms long with a 10 ms hop. A frame is active when its RMS is:

```text
frame_dbfs >= max(-60 dBFS, stem_peak_frame_dbfs - 40 dB)
```

The result is the fraction of frames that are active, in `[0, 1]`. The
stem-relative threshold makes loud and quiet stems comparable; the absolute
floor keeps numerical noise from counting. Quiet bleed, room noise, and residual
artifacts above the threshold still count as activity, so this is a useful
heuristic rather than a semantic instrument detector.

### Cross-stem envelope correlation (reference-free mode)

Each stem's amplitude envelope is a short-time RMS sequence (50 ms frames,
25 ms hop, channel power averaged). Every pair gets a zero-lag Pearson
correlation; the diagonal is `1.0` when the envelope has non-zero variance.
Values are in `[-1, 1]`. `null` means the envelope is silent or perfectly
constant and the correlation is undefined.

Identical stems in two filename slots produce approximately `1.0`. Correlation
does not establish that two stems are correctly assigned; two stems sharing a
beat grid can correlate even when their content is wrong. A fixed timing offset
also lowers zero-lag correlation.

---

## JSON output shape

Reference mode:

```json
{
  "mode": "reference",
  "sample_rate": 44100,
  "channels": 2,
  "duration_sec": 6.0,
  "n_stems": 2,
  "n_scored_stems": 2,
  "stems": {
    "vocals": {"si_sdr_db": 18.42, "sdr_db": 15.10},
    "drums": {"si_sdr_db": 200.0, "sdr_db": 200.0}
  },
  "mean_si_sdr_db": 109.21,
  "mean_sdr_db": 107.55
}
```

Reference-free mode:

```json
{
  "mode": "reference_free",
  "sample_rate": 44100,
  "channels": 2,
  "duration_sec": 6.0,
  "n_stems": 2,
  "stems": {
    "vocals": {"rms_dbfs": -24.65, "active_ratio": 0.92},
    "drums": {"rms_dbfs": -22.67, "active_ratio": 0.71}
  },
  "reconstruction": {
    "residual_db_rel_mix": -23.1,
    "residual_rms_dbfs": -41.5,
    "stem_sum_rms_dbfs": -18.4,
    "mix_rms_dbfs": -18.4
  },
  "envelope_correlation": {
    "frame_ms": 50.0,
    "hop_ms": 25.0,
    "labels": ["vocals", "drums"],
    "matrix": [[1.0, 0.13], [0.13, 1.0]]
  }
}
```

Stem order is canonical: vocals, drums, bass, other, guitar, piano. All
supported files present are included, not only a fixed four-stem set.

---

## Tests

The test suite covers both the arithmetic and the CLI:

```powershell
& "D:\ChordStudio\Dialogue\.tmp\structure_env\Scripts\python.exe" `
  -m pytest -q "D:\ChordStudio\research\separation\eval\tests"
```

Synthetic exact-answer coverage:

1. A perfect estimate gets SI-SDR and SDR of `200.0` (well above 60 dB).
2. Noise added at a known 20 dB SNR gives SI-SDR and SDR within 0.5 dB of
   20 dB.
3. Adding another stem at 50% amplitude produces the theoretical ~6.02 dB
   SI-SDR/SDR, not a perfect score.
4. Identical content in two stem slots gives envelope correlation `1.0`.
5. Additional guards cover silent estimates, active ratio, RMS level, shape
   mismatches, CLI JSON validity, and missing-stem errors.

Current result:

```text
12 passed
```

---

## Honest limitations

- **Arithmetic is not perception.** The synthetic tests prove that formulas are
  implemented correctly on signals with exact answers. They do not show that a
  metric tracks what a listener hears.
- **SI-SDR ignores many audible artifacts.** It is invariant to gain and
  polarity, and it can rate gated, phasey, spectrally holey, or locally doubled
  stems highly if their global waveform projection is close. It also punishes
  even a small fixed time offset heavily, because no alignment search is
  performed; a listener may be more tolerant than the score suggests.
- **Plain SDR is not BSS Eval v4.** No distortion filtering, windowing,
  permutation matching, or missing-source accounting is performed. Values are
  internally comparable under this harness, not directly comparable with
  published BSS Eval numbers.
- **Reconstruction residual does not prove correct assignment.** Any exact
  partition of the mix reconstructs it, even if stems are swapped, duplicated,
  or musically nonsensical. Reference-free metrics cannot replace ground truth.
- **Envelope correlation is not source identity.** It uses zero lag and linear
  amplitude, is sensitive to timing offsets, and cannot distinguish common
  rhythmic structure from correct separation. It is only a diagnostic.
- **Active ratio uses an arbitrary threshold.** Bleed, noise, and artifacts
  above the threshold count as active. It is not an annotation, onset detector,
  or perceptual measure.
- **Reference-free metrics are proxies.** Without ground-truth stems there is
  no reference SDR/SI-SDR in this harness. Do not report reconstruction quality
  as separation accuracy.
- **Global metrics only.** Scores cover the whole file; a few seconds of severe
  failure can disappear in a long track. There are no per-window scores,
  confidence intervals, or significance tests.
- **No resampling/alignment.** Inputs must match exactly in rate, channels, and
  length. A pipeline with a one-sample offset or a sample-rate mismatch will
  be rejected rather than silently scored.
- **Filename identity is assumed.** A correct-sounding file named `vocals.wav`
  is compared against `vocals.wav`; mislabelled stems are scored as
  mislabelled. Supported extras must match by name in reference mode.
- **Clipping hides extremes.** Perfect scores become `200.0`, catastrophic
  scores bottom out at `-200.0`, and digital silence is `-200 dBFS`. These are
  JSON-safe sentinels, not measurements of an actual 200 dB.
- **Levels are not loudness.** dBFS RMS does not account for frequency
  weighting, perceived loudness, or playback level.
- **No real ground truth was available in this session.** The harness is
  validated on synthetic signals only; its usefulness on real music depends on
  acquiring aligned, correctly named reference stems.

---

## Files

```text
stem_eval.py       CLI and metric implementation
demo.py            synthetic end-to-end demonstration, no real audio needed
README.md          this document
tests/
  conftest.py      import path setup
  test_metrics.py  exact synthetic metric checks
  test_cli.py      both CLI modes and error handling
```
