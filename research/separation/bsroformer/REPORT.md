# BS-RoFormer on a 6 GB RTX 4050 — measured results

Date: 2026-11-01. Work folder: `D:\ChordStudio\research\separation\bsroformer\`
GPU env: `D:\ChordStudio\runtime\gpu\Scripts\python.exe` (Python 3.13.12, torch 2.14.1+cu126, torchaudio 2.11.0+cu126)
Test audio: `clip20.wav` (20 s), `virtual.wav` (238 s), both 44.1 kHz stereo.

**Headline: BS-RoFormer runs fine on 6 GB — 1.59 GB peak VRAM — but it is ~11x slower than
htdemucs on the full song. It produced 2 stems (vocals/instrumental), not 4.**

---

## 1. Install: worked, torch untouched

`audio-separator 0.48.0` declares `torch>=2.3,<3`, which the installed `2.14.1+cu126` already
satisfies, so pip would not have replaced torch. I still used `--no-deps` and added dependencies
manually so the resolver never had a chance to touch torch/numpy.

Exact commands that worked (in this order):

```powershell
$py="D:\ChordStudio\runtime\gpu\Scripts\python.exe"

# 1) package itself, no deps
& $py -m pip download audio-separator --no-deps --dest D:\ChordStudio\research\separation\bsroformer\wheel
& $py -m pip install --no-deps D:\ChordStudio\research\separation\bsroformer\wheel\audio_separator-0.48.0-py3-none-any.whl

# 2) everything the Roformer path actually imports (skipping demucs/onnx-conversion extras)
& $py -m pip install "beartype>=0.18.5,<0.19" ml_collections "rotary-embedding-torch>=0.6.1,<0.7.0" `
    pydub pyyaml requests six audioop-lts librosa onnxruntime

# 3) one import iteration was needed: spec_utils -> audioread
& $py -m pip install --no-deps audioread
```

Verification that the GPU/torch environment is unchanged (run after the installs):

```
torch 2.14.1+cu126 | torchaudio 2.11.0+cu126
cuda avail True | device NVIDIA GeForce RTX 4050 Laptop GPU | cuda ver 12.6
matmul on cuda ok True
```

A dry run before installing showed the resolver would install **no** torch and **no** numpy
candidate (`Would install Cython ... librosa-1.0.0 llvmlite-0.50.0 numba-0.68.1 onnxruntime-1.31.1
scikit-learn-1.9.1 soxr-1.1.0 ...`), i.e. numpy 2.5.4 stayed put.

`pip check` still lists 5 declared-but-absent packages: `diffq-fixed`, `onnx-weekly`,
`onnx2torch-py313`, `resampy`, `samplerate`. They are only used by the vendored Demucs / MDX
paths; `from audio_separator.separator import Separator` and the whole Roformer path work without
them. I deliberately did not install them.

Pre-existing, unrelated issue: `torchaudio.load()` fails in this env because `torchcodec` cannot
find its FFmpeg DLLs (`libtorchcodec_core*.dll`). This predates the install and does not affect
audio-separator, which uses the **ffmpeg CLI 7.1** it finds on PATH.

## 2. Model and weights

| item | value |
|---|---|
| checkpoint | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` |
| URL | `https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt` |
| size | 639,331,213 bytes (609.7 MiB) |
| download time | **59.8 s (10.20 MB/s)** |
| config | `model_bs_roformer_ep_317_sdr_12.9755.yaml` (2,273 B), from `https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/model_bs_roformer_ep_317_sdr_12.9755.yaml` |
| architecture | BS-Roformer, `dim=512`, `depth=12`, stereo, `num_stems=1`, 62 freq bands, `stft_n_fft=2048`, `hop=441`, `chunk_size=352800` (8 s) |
| stems | **Vocals + Instrumental (2 stems only)** — the config's `training.instruments` is `[Vocals, Instrumental]`; this is the 2-stem vocals checkpoint, not the 4-stem MUSDB one |

Gotcha: `audio-separator 0.48.0` no longer lists this file in its bundled `models.json` (it now
ships 84 RoFormer entries, mostly Mel-Band), so `load_model()` raises
`ValueError: Model file ... not found in supported model files`. `run_roformer.py` patches that
single lookup **in memory only** and points it at the already-downloaded file + yaml; no file
outside this folder was modified.

## 3. Run commands

```powershell
$py="D:\ChordStudio\runtime\gpu\Scripts\python.exe"; $h="D:\ChordStudio\research\separation\bsroformer"
# fp16 via autocast, one GPU job at a time
& $py "$h\run_roformer.py" --input "D:\ChordStudio\research\separation\clip20.wav"  --outdir "$h\out_clip20_autocast"  --mode autocast
& $py "$h\run_roformer.py" --input "D:\ChordStudio\research\separation\virtual.wav" --outdir "$h\out_virtual_autocast" --mode autocast
```

Driver options: `--mode autocast|native_fp16|fp32` (`use_autocast` / `use_native_fp16` of the
`Separator`), `--chunk-duration`. Stems are written with the default `normalization_threshold`
overridden to `1.0` (see §5).

## 4. Timings and VRAM (measured, `torch.cuda.max_memory_allocated`)

| input | duration | load | separation | total | peak VRAM alloc | peak VRAM reserved | stems |
|---|---|---|---|---|---|---|---|
| clip20.wav | 20 s | 2.08 s | **11.15 s** | 13.23 s | **1587.2 MB** | 2346.0 MB | 2 |
| virtual.wav | 238.1 s | 2.12 s | **167.31 s** | 169.43 s | **1587.2 MB** | 2346.0 MB | 2 |

Peak VRAM is **identical** for both files: the model always works on 8 s chunks
(`chunk_size=352800`), so memory is independent of input length. 1.59 GB of 6 GB is used —
lots of headroom, no chunking/segmentation tricks were needed.

Speed vs the existing demucs baselines (given): htdemucs 7.5 s / 15.4 s, htdemucs_ft 42 s for
virtual.wav.

| | clip20 | virtual.wav |
|---|---|---|
| htdemucs | 7.5 s | 15.4 s |
| **BS-RoFormer ep317** | **11.15 s** | **167.3 s** |
| slowdown vs htdemucs | 1.5x | **10.9x** |

Real-time factor on the full song: 238 s of audio in 167 s = **0.70x real time**.

## 5. Outputs and levels

Files (`out_*_autocast/`), e.g. `virtual_(Vocals)_model_bs_roformer_ep_317_sdr_12.wav`,
`virtual_(Instrumental)_...wav`. 16-bit PCM at 44.1 kHz (bit depth follows the input).

| track | stem | RMS dBFS | peak dBFS | active ratio* |
|---|---|---|---|---|
| clip20 | Vocals | -17.50 | -4.72 | 0.958 |
| clip20 | Instrumental | -16.70 | -5.08 | 1.000 |
| virtual | Vocals | -20.91 | -3.13 | 0.671 |
| virtual | Instrumental | -15.34 | 0.00 | 0.965 |

\* fraction of 50 ms frames above -40 dBFS.

Reconstruction residual, `rms(original - sum(stems)) / rms(original)`:

| system | clip20 | virtual |
|---|---|---|
| BS-RoFormer ep317 (2 stems) | **-76.8 dB** | **-22.5 dB** raw → **-79.0 dB** gain-corrected |
| htdemucs (4 stems) | -34.7 dB | -25.5 dB |

**Why the virtual number looked bad, and why it is not model error.** The instrumental's written
peak is exactly 1.0. A least-squares gain fit gives `vocals x1.00013`, `instrumental x1.09113`;
with that single scalar the residual drops to **-79.0 dB** (joint 2-parameter fit: -80.3 dB).
So the raw instrumental peaked at 1.091, and audio-separator's peak normalisation
(`normalize(max_peak=normalization_threshold)`) scaled that one stem down by 1/1.091 to fit full
scale. Only 2 samples out of 21,002,418 were actually clipped, so this is a pure gain change, not
clipping. Expect it whenever a stem exceeds full scale.

Caveat on interpreting the residual: with `num_stems=1` audio-separator derives the instrumental
as `input - vocals` in the time domain, so a near-zero residual is **by construction** and says
nothing about separation quality. It is a good integrity/plumbing check, nothing more.

Cross-stem envelope correlation (50 ms RMS, same track): roformer vocals vs instrumental
**-0.027** (clip20) / **0.040** (virtual) — essentially uncorrelated, i.e. no gross
"same envelope in both stems" collapse; htdemucs' own cross-stem values are in the same range
(-0.43 … +0.28).

## 6. Cross-system agreement (the closest thing to a quality check I can do)

BS-RoFormer vs htdemucs on clip20, compared as like-for-like (roformer instrumental vs
drums+bass+other):

| metric | value |
|---|---|
| vocals 50 ms envelope correlation | **0.998** (null test, 2 s circular shift: -0.130) |
| instrumental 50 ms envelope correlation | **0.998** (null -0.037) |
| vocals RMS | -17.50 dBFS (roformer) vs -17.55 dBFS (demucs) |
| instrumental RMS | -16.70 vs -16.79 dBFS |
| vocals share of energy | 45.4 % (roformer) vs 45.7 % (demucs) |

On this ballad clip the two independent systems agree almost perfectly at the envelope level and
put almost exactly the same amount of energy in "vocals". That is evidence they are both doing
something sane and consistent; it is **not** evidence that either one is better, because envelope
correlation cannot see artifacts, spectral leakage or high-frequency detail.

## 7. Verdict

**Usable on 6 GB? Yes, comfortably.** 1.59 GB peak allocated / 2.35 GB reserved, constant for a
20 s clip and a 4-minute song. No fp16 tricks beyond `use_autocast=True` and no manual chunking
were required; the 8 s internal chunk size does all the work.

**Plausibly better than demucs? Cannot be determined from these measurements — and I cannot
listen to the output.** What is measurable: it is 10.9x slower than htdemucs on the full song
(167 s vs 15.4 s), it gives 2 stems instead of 4, and its vocals agree with htdemucs to 0.998
envelope correlation with a near-identical energy split. There is no ground-truth multitrack
here, so no SDR/SI-SDR is reported and none is invented. A real answer needs a MUSDB-style
reference or a listening test; the residual numbers in §5 must not be read as quality.

Practical note for the app: at 0.70x real time, a 4-minute song costs ~3 minutes of GPU time,
versus ~15 s for htdemucs. That is the real trade-off to weigh, not VRAM.

## 8. Drum parts — NOT RUN (time-box)

Cutting the excerpt worked, but the drum model was **not executed** because the time-box was
reached. What is already in place for a follow-up:

* excerpt cut OK: `drums20.wav` (3,528,078 bytes, 20 s from 96 s of `virtual.wav`,
  44.1 kHz stereo, via `ffmpeg -y -ss 96 -t 20 -i ...\virtual.wav -ac 2 -ar 44100 ...\drums20.wav`)
* model located: `MDX23C-DrumSep-aufr33-jarredou.ckpt` (437,652,699 B) +
  `config_drumsep_mdx23c.yaml` (2,417 B). Unlike ep317 this one **is** in the bundled
  `mdx23c_download_list`, so the plain `Separator.load_model("MDX23C-DrumSep-aufr33-jarredou.ckpt")`
  path should work with no shim. Note the 404s: both files come from the
  `nomadkaraoke/python-audio-separator/releases/download/model-configs` mirror, not from
  `TRvlvr/model_repo` (both returned 404 there).
* expected extra work: the drumsep config yields multiple drum stems; `Separator` needs
  `mdxc_params` defaults, and VRAM/time are unmeasured.

## 9. Files in this folder

| file | role |
|---|---|
| `run_roformer.py` | driver: loads ep317 via an in-memory lookup patch, runs one file, prints `RESULT {...}` JSON with load/separate time and peak VRAM |
| `analyze_stems.py` | residual / per-stem RMS / active ratio / cross-stem envelope correlation, prints `METRICS {...}` |
| `compare_systems.py` | BS-RoFormer vs htdemucs agreement incl. shift-null test, prints `COMPARE {...}` |
| `models/` | checkpoint + config yaml |
| `out_clip20_autocast/`, `out_virtual_autocast/` | separated stems |
| `drums20.wav` | drum-heavy excerpt for the unrun drum task |
| `wheel/`, `src/` | downloaded wheel and its extracted sources (for inspection) |
