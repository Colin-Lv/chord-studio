# synth6 — does the six-stem model actually separate piano from guitar?

**Question.** With piano and guitar *both* playing, does `BS-Roformer-SW.ckpt` put each
instrument in its own output file, or does it dump both into one stem?

**Answer.** It separates them, cleanly, on this material. Every part's best-matching
output stem is its own stem by a margin of 29–47 dB, and 88–98 % of each part's energy
lands in the file with its name on it. The model does not dump piano+guitar into one
stem, and it does not hide them in `other` (`other` and `vocals` come back as digital
silence). The residual problem is not confusion but **loss**: the piano stem is the
weakest of the four (SI-SDR 8.89 dB, 11.4 % of the piano's energy missing), which is a
fidelity problem, not a routing problem.

Built and scored 2026-11-01 on the local RTX 4050. Everything below lives in
`D:\ChordStudio\research\separation\synth6\`.

---

## 1. Material

### 1.1 Arrangement (hand-written with pretty_midi)

105 BPM, 4/4, 16 bars = **36.57 s** of music, harmony `C – Am – F – G` × 4.
Four independent `PrettyMIDI` objects, one per part, each saved as its own `.mid`.

| part | GM program | notes | pitch range | rhythm |
|---|---|---|---|---|
| piano | 0 Acoustic Grand | 128 | **41–60** (F2–C4) | block chord, struck on beat 1 and re-attacked softer on beat 3 |
| guitar | 25 Acoustic Steel | 128 | **62–72** (D4–C5) | 8th-note arpeggio, 8 plucks/bar, chord-tone pattern `0-1-2-1-0-2-1-2` |
| bass | 33 Electric Finger | 48 | **29–48** (F1–C3) | root on 1 and 3, octave on the "and" of 3 |
| drums | ch.9 kit, `is_drum=True` | 192 | 36 / 38 / 42 | kick on 1+3, snare on 2+4, closed hats on 8ths |

Registers were deliberately kept apart (piano tops out at C4, guitar starts at D4) so the
test is about *the model's routing decision*, not about whether the material is
impossible. Same chords in both parts, different rhythm and different octave — the
easiest possible honest version of the question.

### 1.2 Renderer, and the one deviation

Renderer = `D:\ChordStudio\research\boundary-dataset\render.py`
(`find_soundfont` / `load_sf2_module` / `make_sampler_class`), soundfont
`D:\ChordStudio\assets\soundfonts\FluidR3_GM.sf2`, `sr=48000`, mono, one sampler
instance shared by all four parts.

**Deviation (one line).** `render_pm` hardcodes `bank=0` for every instrument. The GM
percussion kit lives at **bank 128**, so with `bank=0` a drum hit on note 36 renders as
a *piano* note — which would have silently contaminated the exact stem this experiment
is about. `build_synth6.py` therefore carries a copy of `render_pm` that is verbatim
except for `bank = 128 if inst.is_drum else 0`. The copy is asserted bit-identical to
`render.render_pm` on the piano part at build time
(`render_pm_copy_matches_project: true` in `manifest.json`), and bank 128 preset 0 is
present in the soundfont (149 zones, keys 27–87) — verified with `probe_sampler.py`.

### 1.3 Rendered stems — duration and level

Each stem is peak-normalised to 0.89 by `render_pm`, as the project always does.

| stem | duration | peak | RMS | notes |
|---|---|---|---|---|
| piano.wav | 37.31 s | −1.01 dBFS | **−17.91 dBFS** | 128 |
| guitar.wav | 37.33 s | −1.01 dBFS | **−15.64 dBFS** | 128 |
| bass.wav | 37.31 s | −1.01 dBFS | **−12.69 dBFS** | 48 |
| drums.wav | 37.20 s | −1.01 dBFS | **−20.99 dBFS** | 192 |

The 8.3 dB RMS spread is a direct consequence of per-part peak normalisation: the
sustained bass has a low crest factor and ends up loudest, the transient-heavy drums
end up quietest. That is the mix balance the model was given; it was not re-balanced.

### 1.4 mix.wav

Plain sample sum of the four rendered stems, then **one** scale factor:

```
sum peak before scaling : 1.9091
scale factor            : 0.466195   (= −6.63 dB)
peak after scaling      : 0.8900  (−1.00 dBFS)
mix RMS                 : −16.40 dBFS
```

Written as 48 kHz mono float32 (the model's loader reported `Input audio subtype: FLOAT`,
`Detected input bit depth: 32-bit`).

---

## 2. Model runs

| run | command | device | load | separate | peak VRAM | outputs |
|---|---|---|---|---|---|---|
| six-stem | `run_any_model.py --model BS-Roformer-SW.ckpt` | cuda (RTX 4050) | 2.77 s | **8.78 s** | 1590 MB | `mix_(bass|drums|guitar|other|piano|vocals)_BS-Roformer-SW.wav` |
| two-stem | `bsroformer\run_roformer.py` | cuda | 1.80 s | **21.91 s** | 1587 MB | `mix_(Instrumental|Vocals)_model_bs_roformer_ep_317_sdr_12.wav` |

Both ran one at a time on `mix.wav`. The six-stem model produced all six named stems
from the `inference.instruments` list in its YAML (`bass, drums, other, vocals, guitar,
piano`); nothing had to be renamed or guessed.

**Evaluation rate.** Model outputs are 44.1 kHz stereo; renders are 48 kHz mono. Scoring
was done at **44.1 kHz** (truth downsampled with `resample_poly`), because the FluidR3
samples are themselves 44.1 kHz — so going 48→44.1 discards essentially nothing, whereas
upsampling the model output would have injected avoidable interpolation error into the
estimates. Model stereo was downmixed to mono. Everything trimmed to a common
**37.200 s / 1 640 520 samples**.

---

## 3. The confusion matrix (key result)

`SI-SDR(truth part, model output stem)` in dB, computed with the project harness's own
`si_sdr_db` from `eval\stem_eval.py`. **Rows = ground truth, columns = one model output
file.** Read a row as: *where did this instrument's content actually go?*

| truth \ model stem | 6: **piano** | 6: **guitar** | 6: bass | 6: drums | 6: vocals | 6: other | 2: instrumental | no-separation baseline (the mix itself) |
|---|---|---|---|---|---|---|---|---|
| **piano**  | **8.89** | −20.34 | −46.45 | −30.34 | −35.37 | −21.23 | −7.43 | −7.43 |
| **guitar** | −22.96 | **14.04** | −62.99 | −31.07 | −50.70 | −51.43 | −4.22 | −4.21 |
| **bass**   | −28.47 | −48.41 | **16.78** | −50.00 | −68.81 | −33.24 | −0.16 | −0.16 |
| **drums**  | −62.10 | −34.12 | −39.08 | **12.83** | −67.03 | −54.25 | −10.59 | −10.59 |

Bold = the diagonal. The nearest wrong answer for every row is 29–47 dB worse than the
right one; there is no ambiguity about assignment. For reference, if the model had
dumped everything into one file, every entry would look like the last column (−7.4 to
−0.2 dB) — that is what "no separation" scores on this material.

The two "other" columns are the control:

* `6: other` = **−97.6 dBFS RMS**, `6: vocals` = **−105.9 dBFS RMS** — i.e. digital
  silence. Their cells are garbage-level scores against a silent signal (−21 to −68 dB),
  not leakage. **Nothing was dumped into `other`.** This was the specific failure mode
  worth ruling out, and it is ruled out.
* `2: instrumental` is numerically identical to the unseparated mix column (max
  difference 0.01 dB) — see §5.

Harness reference-mode output (`score_six.json`, truth = 4 stems, stems = the 4
six-stem files renamed to `piano/guitar/bass/drums`):

```
piano   SI-SDR  8.8873 dB   SDR 4.1204 dB
guitar  SI-SDR 14.0434 dB   SDR 5.0551 dB
bass    SI-SDR 16.7777 dB   SDR 5.1177 dB
drums   SI-SDR 12.8302 dB   SDR 5.0321 dB
mean    SI-SDR 13.1346 dB   SDR 4.8313 dB
```

The plain `SDR` column is *not* evidence of a bad separation. The model writes its stems
in **mix space**: every stem comes back at 0.419–0.461× the truth amplitude, and the mix
scale factor is 0.466. Plain SDR has no gain correction, so it is pinned near
`−20·log10(1−0.466) ≈ 5.2 dB` by that constant alone. SI-SDR removes the gain and is the
only one of the two worth reading here.

---

## 4. Where each part's energy actually went

SI-SDR says "how good is this waveform"; this says "how much of the instrument is in this
file". Least-squares projection of every model stem onto the four truth stems
(`cond(Gram)=6.8`, truth stems mutually uncorrelated at |r| ≤ 0.033, so the decomposition
is well-posed).

**(a) Composition of each model stem** — share of the stem's energy attributable to each part:

| model stem | from piano | from guitar | from bass | from drums | R² (4 truth stems) |
|---|---|---|---|---|---|
| 6: piano | **0.886** | 0.003 | 0.002 | −0.000 | 0.891 |
| 6: guitar | 0.006 | **0.960** | −0.000 | 0.000 | 0.966 |
| 6: bass | 0.000 | 0.000 | **0.981** | 0.000 | 0.981 |
| 6: drums | 0.001 | 0.000 | −0.000 | **0.950** | 0.951 |
| 6: vocals | 0.000 | 0.000 | −0.000 | 0.000 | 0.000 |
| 6: other | 0.007 | 0.000 | 0.000 | 0.000 | 0.008 |
| 2: instrumental | 0.154 | 0.267 | 0.501 | 0.078 | **1.000** |
| baseline: mix | 0.154 | 0.267 | 0.501 | 0.078 | 1.000 |

**(b) Recovery of each part** — share of the part's energy found in the six stems
(the last column answers "was it lost, or just put somewhere else?"):

| truth part | 6: piano | 6: guitar | 6: bass | 6: drums | 6: vocals | 6: other | R² all six | off-own-stem share |
|---|---|---|---|---|---|---|---|---|
| piano | **0.886** | 0.000 | −0.000 | 0.001 | −0.000 | 0.001 | 0.887 | **0.001** |
| guitar | −0.002 | **0.965** | 0.000 | 0.000 | −0.000 | −0.000 | 0.963 | **−0.002** |
| bass | −0.000 | 0.000 | **0.980** | 0.000 | −0.000 | 0.000 | 0.980 | **−0.000** |
| drums | −0.000 | −0.000 | 0.000 | **0.951** | −0.000 | −0.000 | 0.951 | **0.000** |

Two things fall out:

1. **Routing is correct and exclusive.** Guitar leaking into the piano stem is 0.3 % of
   the piano stem's energy; piano leaking into the guitar stem is 0.6 %. The off-own-stem
   share of every part is ≤ 0.1 %. This is the "does it dump both into one stem" question,
   and the answer is no.
2. **What goes wrong is attenuation, not misrouting.** The piano's total recovery (0.887)
   equals its own-stem share (0.886) — the missing 11 % is *nowhere* in the six outputs,
   it was simply not reconstructed. Corroborating evidence: the model's piano stem is
   spectrally duller than the truth (centroid 1929 Hz → 1680 Hz, −13 %), and its gain is
   0.9 dB low in mix space while the other three are within 0.25 dB. The piano — the
   densest, most sustained part — is being smeared/softened, not confused with the guitar.

---

## 5. Per-stem RMS dBFS (is a stem simply empty?)

Measured at 44.1 kHz over the common 37.200 s.

| stem | truth (as rendered) | truth × mix scale 0.466 | six-stem output | Δ vs mix-space truth | active frames |
|---|---|---|---|---|---|
| piano | −17.89 | −24.52 | **−25.45** | −0.93 dB | 98.4 % |
| guitar | −15.62 | −22.25 | **−22.42** | −0.17 dB | 99.9 % |
| bass | −12.68 | −19.31 | **−19.54** | −0.23 dB | 84.1 % |
| drums | −20.99 | −27.62 | **−27.72** | −0.10 dB | 57.1 % |
| vocals | — | — | **−105.87** | — | **0.0 %** |
| other | — | — | **−97.61** | — | **0.0 %** |
| *mix* | — | — | −16.40 | — | — |

No output stem is empty, and the two that are supposed to be empty are empty to the
noise floor (`other` max |sample| = 3.97e−4, `vocals` max |sample| = 7.63e−5).
No stem is a copy of another: per-stem gains track mix-space truth within 1 dB, and the
peak correlation between any truth part and a *wrong* model stem is 0.20 (§7).

---

## 6. Two-stem model, for reference

`model_bs_roformer_ep_317_sdr_12.9755.ckpt` (vocals / instrumental) on the same `mix.wav`:

| | result |
|---|---|
| harness `--truth truth2_eval --stems stems2` | instrumental **SI-SDR 60.05 dB** (truth = the mix), vocals n/a (truth silent) |
| vocals stem RMS | **−240.00 dBFS** — bit-exact digital silence |
| instrumental composition | piano 0.154 / guitar 0.267 / bass 0.501 / drums 0.078, **R² = 1.000 vs the mix** |
| instrumental vs unseparated-mix column | identical to 0.01 dB |

The two-stem model is a pass-through on this material: it maps the entire mix to
`instrumental` and writes silence to `vocals`. Every instrument's SI-SDR against
`instrumental` (−7.4, −4.2, −0.2, −10.6 dB) is the no-separation baseline, not a
separated result. If the product path only runs the 2-stem model, piano and guitar are
inseparable **by construction**, and these numbers are what that looks like.

---

## 7. Sanity checks that could have invalidated the table

| check | result |
|---|---|
| time alignment (a lag silently eats SI-SDR) | best lag **0.000 ms** for piano, guitar, drums; **−0.023 ms** for bass. Peak correlations 0.941 / 0.981 / 0.990 / 0.975 |
| cross-stem correlation (would a "dump" show up?) | truth piano vs guitar stem 0.107, truth guitar vs piano stem 0.078; the largest wrong-pair correlation anywhere is 0.20 (truth piano vs bass stem), vs 0.94–0.99 on the correct pairs |
| truth stems linearly independent | max \|r\| between truth parts 0.032, Gram condition number 6.8 |
| stems are not the mix | `6: other` 0.008 R², `6: vocals` 0.000 R² |
| stems sum back to the mix | reference-free harness: residual **−35.17 dB** relative to the mix (residual RMS −51.57 dBFS, stem sum −16.408 vs mix −16.400 dBFS) |
| envelope correlation caveat | piano↔bass envelope correlation is 0.657 despite zero waveform leakage — both land on beats 1 and 3. Envelope correlation is *not* a leakage metric on rhythmic material. |

---

## 8. Caveats

* **One example, one checkpoint.** 37 s, one synthetic arrangement, one model file. This
  measures what this model does to this material; it is not a MUSDB benchmark.
* **The material is the easy case.** Non-overlapping registers, two distinct rhythms, four
  clean FluidR3 presets, no vocals, no cymbals wash, no reverb, no distortion. If a
  six-stem model can separate piano from guitar at all, it should succeed here. A pass
  does not license "piano/guitar separation is solved" — a fail would have been damning.
* **`other` being silent is model-specific behaviour** on a mix with nothing outside the
  four trained classes. On a real track with strings/synth it would not be.
* **My renderer deviation** (drum bank 128) is documented and asserted above; every
  melodic part is byte-identical to `render.render_pm`.
* **SI-SDR on synthetic renders flatters everyone.** Truth and estimate come from the same
  sampler, so timbre mismatch that would hurt on real recordings is absent.

---

## 9. Verdict

On this material the six-stem model's piano/guitar separation is **trustworthy at the
routing level and mediocre at the fidelity level**. Piano and guitar are genuinely split,
not dumped together: each one's best-matching output file is its own by 29 dB (piano) and
37 dB (guitar), 96 % of the guitar and 89 % of the piano land in the correctly named stem,
cross-leakage into the other instrument's stem is under 1 % of energy, nothing is smuggled
into `other` (−97.6 dBFS, i.e. silence), and the six outputs sum back to the mix within
35 dB. Those are not the numbers of a model that gives up and puts both keyboards and
guitars in one file. But the piano stem is the weakest of the four (SI-SDR 8.89 dB vs
12.8–16.8 dB for the rest) and its 11 % shortfall is *lost content, not misplaced
content* — the model returns a slightly dulled, 0.9 dB-quiet piano rather than a
contaminated one. Practically:
you can trust `(piano)` and `(guitar)` to tell you *which instrument is which*, and you
should not trust either to be a transparent, phase-accurate isolation — which matters if
the downstream task is chord/note detection on the separated stem rather than routing.
Two caveats keep this from being a stronger claim: the material was built to be separable
(strictly disjoint registers, two clearly different rhythms), so this is the best case
rather than an average one; and the two-stem model, by contrast, does no separation at
all — it returns the mix as `instrumental` with bit-exact silent `vocals`, so any product
path that relies on it cannot separate piano from guitar under any circumstances.

---

## 10. Files and reproduction

```
D:\ChordStudio\research\separation\synth6\
  build_synth6.py              arrangement + render + mix           (CPU, 1.6 s)
  analyze_synth6.py            align + score + confusion matrix    (CPU, ~20 s)
  probe_sampler.py             soundfont/drum-bank probe
  parts\{piano,guitar,bass,drums}.mid
  truth\{piano,guitar,bass,drums}.wav        48 kHz mono float32, as rendered
  mix.wav                                    48 kHz mono float32, peak −1.00 dBFS
  manifest.json                              durations, levels, scale factor
  model6\                                    6 raw model outputs
  model2\                                    2 raw model outputs
  stems6\  stems6_eval\  stems2\  truth_eval\  truth2_eval\  mix_eval.wav
  score_six.json  score_two.json  score_six_reference_free.json  confusion.json
  model6_run.log  model2_run.log
```

```bat
rem 1. build
D:\ChordStudio\Dialogue\.tmp\structure_env\Scripts\python.exe ^
  D:\ChordStudio\research\separation\synth6\build_synth6.py

rem 2. six-stem (GPU)
D:\ChordStudio\runtime\gpu\Scripts\python.exe ^
  D:\ChordStudio\research\separation\run_any_model.py ^
  --input  D:\ChordStudio\research\separation\synth6\mix.wav ^
  --outdir D:\ChordStudio\research\separation\synth6\model6 ^
  --model  BS-Roformer-SW.ckpt

rem 3. two-stem (GPU)
D:\ChordStudio\runtime\gpu\Scripts\python.exe ^
  D:\ChordStudio\research\separation\bsroformer\run_roformer.py ^
  --input  D:\ChordStudio\research\separation\synth6\mix.wav ^
  --outdir D:\ChordStudio\research\separation\synth6\model2

rem 4. align + score (writes stems6_eval\, truth_eval\ and runs the harness)
D:\ChordStudio\Dialogue\.tmp\structure_env\Scripts\python.exe ^
  D:\ChordStudio\research\separation\synth6\analyze_synth6.py
```

The harness call that produced `score_six.json` (invoked from `analyze_synth6.py`):

```bat
D:\ChordStudio\Dialogue\.tmp\structure_env\Scripts\python.exe ^
  D:\ChordStudio\research\separation\eval\stem_eval.py ^
  --truth D:\ChordStudio\research\separation\synth6\truth_eval ^
  --stems D:\ChordStudio\research\separation\synth6\stems6_eval
```

`truth_eval\` holds `piano.wav guitar.wav bass.wav drums.wav`; `stems6_eval\` holds the
model's `mix_(piano)_BS-Roformer-SW.wav` etc. renamed to those four names, 44.1 kHz mono,
trimmed to the common 1 640 520 samples. Nothing outside `synth6\` was created or
modified.
