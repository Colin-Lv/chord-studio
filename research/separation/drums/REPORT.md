# MDX23C-DrumSep on a 6 GB RTX 4050 — measured results

Date: 2026-11-01. Work folder: `D:\ChordStudio\research\separation\drums\`
GPU env: `D:\ChordStudio\runtime\gpu\Scripts\python.exe` (torch 2.14.1+cu126, CUDA 12.6,
audio-separator 0.48.0). No package was installed or upgraded for this task; torch untouched.
Every write is inside the folder above.

**Headline: the drum model runs easily on 6 GB (725 MB peak) and is fast (238 s of audio in
37.5 s), and it emits 6 genuine, distinct stems. But it is a drum-part *splitter*, not a drum
*extractor*: fed a full mix it passes the non-drum content through at unity gain (measured),
so it only makes sense as a second stage after a drum stem exists.**

---

## 1. Model acquisition

| item | value |
|---|---|
| checkpoint | `MDX23C-DrumSep-aufr33-jarredou.ckpt` |
| URL | `https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/MDX23C-DrumSep-aufr33-jarredou.ckpt` |
| size | **437,652,699 B (417.4 MiB)**, HTTP 200 |
| download time | **41.27 s (10.1 MiB/s)**, `curl -L` |
| MD5 | `7D6FD78B22B29F3CFE8147F632B6B91F` |
| config | `config_drumsep_mdx23c.yaml` (2,417 B, 0.91 s) from the same `model-configs` release |
| architecture | MDX23C / TFC-TDF, 6 outputs, `dim_f=1024`, `dim_t=256`, `n_fft=2048`, `hop=512`, 4 subbands, `chunk=130560` (2.96 s), inference overlap 4, `num_channels=2`, 44.1 kHz |
| declared stems | `kick, snare, toms, hh, ride, crash` (config `training.instruments`); `target_instrument: null` |

Exact download commands:

```powershell
$dir='D:\ChordStudio\research\separation\drums\models'
$base='https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs'
curl.exe -L --fail -o "$dir\config_drumsep_mdx23c.yaml" "$base/config_drumsep_mdx23c.yaml"
curl.exe -L --fail -o "$dir\MDX23C-DrumSep-aufr33-jarredou.ckpt" "$base/MDX23C-DrumSep-aufr33-jarredou.ckpt"
```

**No `models.json` shim was needed.** Unlike the BS-RoFormer case, this checkpoint *is* in the
bundled `mdx23c_download_list` (`"MDX23C Model: MDX23C DrumSep by aufr33-jarredou"`), so
`Separator.load_model("MDX23C-DrumSep-aufr33-jarredou.ckpt")` resolves it normally and simply
skips the download because both files already exist in `drums\models\`. `run_drumsep.py` is a
plain driver (no monkey-patch).

Note: the app's own model folder `D:\ChordStudio\data\models\drumsep\` already held a
**byte-identical** checkpoint (same MD5), so this download was only needed for the isolated
experiment; the app itself needs no new file.

## 2. Test material

| file | what it is |
|---|---|
| `drums20.wav` | 20 s (3,528,078 B), cut exactly as instructed: `ffmpeg -y -ss 96 -t 20 -i ...\virtual.wav -ac 2 -ar 44100` — i.e. a **full mix** excerpt |
| `D:\ChordStudio\data\stems\d4867...\htdemucs_ft\wav\drums.wav` | the existing htdemucs_ft drum stem, 238.1 s, 44.1 kHz stereo |
| `synth.wav` + `synth_sources\*.wav` | 20 s **synthetic** drum mix with known ground-truth parts (kick / snare / hats / crash / toms), see §6 — added because the real song's drum stem turned out to be kick-only |

Run commands (one GPU job at a time):

```powershell
$py="D:\ChordStudio\runtime\gpu\Scripts\python.exe"; $h="D:\ChordStudio\research\separation\drums"
& $py "$h\run_drumsep.py" --input "$h\drums20.wav" --outdir "$h\out_20s"
& $py "$h\run_drumsep.py" --input "D:\ChordStudio\data\stems\d4867...\htdemucs_ft\wav\drums.wav" --outdir "$h\out_full"
& $py "$h\make_synth.py"; & $py "$h\run_drumsep.py" --input "$h\synth.wav" --outdir "$h\out_synth"
```

`run_drumsep.py` sets `normalization_threshold=1.0` and `amplification_threshold=0.0` so the
written stems are not rescaled (keeps the sum test meaningful), otherwise defaults are used.

## 3. Timings and VRAM (measured, `torch.cuda.max_memory_allocated`)

| input | duration | load | separation | total | peak alloc | peak reserved | stems |
|---|---|---|---|---|---|---|---|
| `drums20.wav` (full mix) | 20.0 s | 2.19 s | **5.03 s** | 7.22 s | **724.8 MB** | 1036.0 MB | 6 |
| `drums.wav` (drum stem) | 238.1 s | 1.52 s | **37.50 s** | 39.03 s | **724.8 MB** | 1036.0 MB | 6 |
| `synth.wav` | 20.0 s | 1.64 s | 4.42 s | 6.05 s | 724.8 MB | 1036.0 MB | 6 |

Peak VRAM is **identical for 20 s and 238 s**. That is not an accident and not a measurement
artefact: audio-separator 0.48.0 only keeps the overlap-add accumulation buffers on the GPU for
MPS (`should_accumulate_on_device()` returns true only for `mps`), so on CUDA the [6, 2, N]
float32 accumulator lives in **host RAM** (`accumulation_device = ... else torch.device("cpu")`)
and only 3 s chunks plus the model are on the GPU. Memory therefore does not grow with song
length — but it also means a long song pays host↔device copies per chunk.

Speed: **9.45 s of GPU per minute of audio**, real-time factor **0.158x**.
A 4-minute song: **≈ 38 s** separation + ~1.5 s model load ≈ **40 s**.
For comparison (prior report): htdemucs_ft took 42 s for the same 238 s song, so a
"drum stem → drum parts" pipeline costs roughly **80 s per 4-minute song** on this laptop.

Output volume: 6 stems × 42.0 MB = **252 MB per 4-minute song** (16-bit 44.1 kHz stereo, full length).

## 4. Stems produced

Exactly 6 files per run, named `<input>_(<stem>)_MDX23C-DrumSep-aufr33-jarredou.wav`:
**kick, snare, toms, hh, ride, crash**. Full length, no "other"/"instrumental" residual stem —
by design the 6 stems reconstruct the input.

### Full 238 s drum stem — per-stem measurements

| stem | RMS dBFS | peak dBFS | active ratio* | active ratio rel.** | energy share |
|---|---|---|---|---|---|
| kick | **-24.95** | -0.23 | 0.216 | 0.290 | **86.13 %** |
| snare | -33.56 | -0.89 | 0.025 | 0.043 | 11.87 % |
| hh | -47.59 | -11.15 | 0.021 | 0.248 | 0.47 % |
| toms | -47.85 | -18.84 | 0.041 | 0.315 | 0.44 % |
| crash | -62.62 | -23.24 | 0.000 | 0.112 | 0.01 % |
| ride | -72.73 | -39.76 | 0.000 | 0.524 | 0.00 % |

\* fraction of 50 ms frames above **-40 dBFS absolute**. \*\* same threshold but relative to that
stem's own loudest 50 ms frame — the more useful number for sparse stems.

### 20 s full-mix excerpt — per-stem measurements (input RMS -14.48 dBFS, peak -0.43)

| stem | RMS dBFS | peak dBFS | active rel. | energy share |
|---|---|---|---|---|
| kick | -19.47 | 0.00 | 0.882 | 35.3 % |
| snare | -21.89 | -5.19 | 1.000 | 20.2 % |
| toms | -21.97 | -6.59 | 0.998 | 19.8 % |
| ride | -28.49 | -8.09 | 0.985 | 4.4 % |
| hh | -37.94 | -11.38 | 0.917 | 0.5 % |
| crash | -45.88 | -16.07 | 0.775 | 0.1 % |

## 5. Does the sum reconstruct the input? — yes, and that is exactly the problem

Residual = `rms(input - sum(stems)) / rms(input)`:

| input | sum vs input corr | residual rel. | note |
|---|---|---|---|
| `drums20.wav` (full mix) | 0.9988 | **-23.0 dB** | 6 stems reproduce the *mix*, not the drums |
| `drums.wav` (drum stem, 238 s) | 0.9994 | **-28.0 dB** | plumbing check, passes |
| `synth.wav` | 0.9996 | **-30.0 dB** | plumbing check, passes |

Two consequences, both measured:

1. **On a drum stem, near-exact reconstruction is a plumbing check, not a quality claim** — the
   6 outputs subtract back to the input almost perfectly, but that says nothing about whether the
   split is musically right.
2. **On a full mix, the reconstruction is a red flag.** Compared against the htdemucs_ft stems for
   the same 96–116 s region:

| metric (96–116 s, from the full-mix excerpt) | value |
|---|---|
| corr(drumsep sum, full mix) | 0.9988 |
| corr(drumsep sum, htdemucs drums) | 0.6223 |
| corr(drumsep sum, htdemucs non-drum = vocals+bass+other) | 0.7872 |
| **gain applied to non-drum content** | **1.003** |
| share of the sum's energy explained by non-drum content | **62.0 %** |
| sum RMS vs drums-stem RMS | **+5.28 dB** |
| `htdemucs_ft` 4-stem sum vs mix (sanity of the reference) | -15.5 dB |

The non-drum content comes through at **unity gain (1.003)**. The model does not suppress
anything; it partitions whatever it is given into six drum-labelled outputs. Feed it a mix and
you get vocals/bass/synths back, relabelled as toms/snare/ride. **It must be run on a drum stem.**

## 6. The test song's drum stem is kick-only — which limits what can be validated

The htdemucs_ft `drums.wav` for this song is spectrally almost pure low end (energy shares,
whole 238 s): 40.4 % at 0–60 Hz, 54.8 % at 60–150 Hz, 3.9 % at 150–400 Hz, 0.79 % at 400–2 kHz,
**0.06 % at 2–6 kHz, 0.01 % above 6 kHz**. A drum stem with essentially no cymbal band cannot
exercise snare/toms/hats/cymbals. The model's output is consistent with that input (kick 97.8 % of
its energy below 150 Hz, 86 % of total energy; everything else 8–48 dB down), so on this song the
"separation" is real but nearly trivial, and **the near-silence of `hh`/`ride`/`crash` here is a
property of the material, not proof the model is broken**.

Because of that, a controlled **synthetic** drum mix (`synth.wav`, 20 s, 60 BPM: kick / snare /
8th-note hats / two crashes / tom fills, peak-normalised to 0.7, gain recorded in
`synth_gain.json`) was run through the same model to measure actual routing. `alpha` is the
least-squares amplitude of a true part found in an output stem (1.0 = the whole part at the
mixture's level; the analysis multiplies the ground truth by the mixture's normalisation gain).

| true part | kick stem | snare stem | toms stem | hh stem | ride stem | crash stem |
|---|---|---|---|---|---|---|
| kick | **0.99** | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 |
| snare | 0.00 | **0.94** | 0.20 | 0.15 | 0.00 | 0.00 |
| hats | 0.00 | 0.02 | 0.00 | **0.78** | 0.01 | 0.01 |
| crash | 0.00 | 0.01 | 0.00 | **0.89** | 0.05 | 0.00 |
| toms | 0.00 | 0.00 | **0.38** | 0.00 | 0.00 | 0.00 |

Read as: kick → `kick` (98 % of its energy), snare → `snare` (89 %), hats → `hh` (60 %),
**crash → `hh` (80 %, the `crash` stem stays empty)**, **toms split roughly 1/3 into `kick` and
1/3 into `toms`**, ride → almost nothing anywhere. The routing is clearly real and
instrument-specific (not a fake split), but the cymbal and tom heads are weak.

Caveat: this synthetic mix is out of distribution (sine-sweep kick, noise snare, filtered-noise
cymbals) and 20 s long. It shows the model *can* separate these parts, not how it will do on real
acoustic drums.

## 7. Verdict

**Usable at 6 GB — comfortably.** 725 MB allocated / 1.04 GB reserved, flat across 20 s and
238 s, no fp16 tricks, no chunking, no OOM. It is also cheaper than expected: 238 s of audio in
37.5 s, i.e. ~38 s for a 4-minute song, and roughly 80 s for the full
`htdemucs_ft → drumsep` two-stage pipeline.

**What is solid:** the checkpoint downloads, loads and runs unmodified through audio-separator
(no registry shim); it yields six distinct, correctly *labelled* stems whose sum reconstructs the
input to -28 dB; on clean synthetic drums kick/snare/hats land in the right stems at 0.78–0.99
amplitude.

**What is clearly unreliable / unproven — stated honestly, because I cannot listen to any of it:**

* **It is not a drum extractor.** On a full mix the non-drum content passes through at unity gain
  (1.003) and explains 62 % of the output energy. Using it on a mix without a prior drum-stem
  stage produces six confidently-named stems that are mostly *not* drums.
* **`ride` never activated** in either test (0.0 % of the real song's energy, 0.01 % on synthetic).
  Assume this head is effectively non-functional until proven otherwise.
* **Crash/hats are not separated from each other**: in the synthetic test 80 % of the crash landed
  in `hh` and the `crash` stem stayed empty. Expect "cymbals" lumped into `hh`.
* **Toms are split with kick** (0.33 / 0.38), so `toms` is not a clean tom bus.
* **On the one real song available, 4 of 6 stems are near-silent**, because its drum stem is 95 %
  below 150 Hz. That is a property of the material; it means I have **no evidence at all** about
  real-acoustic-kit snare/tom/cymbal quality, phase coherence, or artifacts. No SDR/SI-SDR is
  reported and none is invented — there is no ground-truth multitrack here.
* The synthetic routing test is out-of-distribution by construction and cannot substitute for a
  listening test or a MUSDB-style reference.

**Recommendation for Chord Studio:** the measured behaviour matches what the app already does.
The installed app (`C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\`) already
has `drums_worker.py` + `drumsep_runner.py`, and it runs this model **on the already separated
drums stem** (`stems_dir/sha/model/wav/drums.wav`), with `normalization_threshold=1.0` and
`amplification_threshold=0.0` — i.e. the two-stage design this report says is required. Its model
folder `D:\ChordStudio\data\models\drumsep\` already contains a **byte-identical** copy of the
checkpoint (same MD5 `7D6FD78B…`, 437,652,699 B), so no download or integration work is needed.
What the app should act on: budget ~80 s per 4-minute song and 252 MB of extra stems for the
`htdemucs_ft → 6 drum parts` pipeline, and treat `ride`/`crash` (and, on this song, hi-hat) as
unreliable rows — the worker already carries a per-row `level_factor` for exactly this reason.

## 8. Files in this folder

| file | role |
|---|---|
| `run_drumsep.py` | driver: loads the model by its registry name, runs one file, prints `RESULT {...}` (times, peak VRAM, output names) |
| `analyze_drums.py` | per-stem RMS/peak/active-ratio/energy share, sum residual, optional reference comparison; prints `METRICS {...}` |
| `compare_drums.py` | same-region full-mix vs drum-stem comparison, non-drum leakage, band shares, cross-stem correlations; prints `COMPARE {...}` |
| `profile_drums.py` | band energy + 5 s level profiles of the input stem and the 6 outputs; prints `PROFILE {...}` |
| `make_synth.py`, `analyze_synth.py` | synthetic ground-truth drum mix and its routing matrix; prints `ROUTING {...}` |
| `models\` | checkpoint (437,652,699 B) + config yaml |
| `out_20s\`, `out_full\`, `out_synth\` | separated stems |
| `drums20.wav`, `synth.wav`, `synth_sources\` | test audio and synthetic ground truth |
| `log_20s.txt`, `log_full.txt`, `metrics_*.txt`, `compare.txt`, `profile.txt`, `routing.txt` | raw run logs and measured output |
