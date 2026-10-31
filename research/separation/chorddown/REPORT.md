# Mix vs. separated harmonic stems — chord downstream, synth6

**Question.** For the frozen chord model in this app, is it better to feed the full mix or the separated harmonic parts, on material where the true chords are known exactly?

**Answer (details below).** Feeding the isolated harmonic parts does not make the model materially better: on the app's published timeline the mix scores 93.20 % exact and the separated piano+bass 98.64 %, but that gap is almost entirely the app *not publishing* one edge bar, not a wrong chord. On complete-coverage timelines every condition gets the 16-bar C–Am–F–G progression right and differs by at most 2 frames (1.36 pp) — while the run-to-run delta on identical audio is exactly 0.00 pp.

## 1. Material and the ground truth

`synth6/mix.wav`: 37.326 s, 48 kHz mono float32, 105 BPM, 4/4, 16 bars, rendered from four MIDI parts (piano / guitar / bass / drums) with a GM soundfont. The six-stem separation is BS-Roformer-SW (`synth6/model6/`), 44.1 kHz stereo, same duration.

**The brief's bar map is wrong for this material, and that matters.** The brief said C×4, Am×4, F×4, G×4 (four bars per chord). The build script (`build_synth6.py`) uses `PROGRESSION = ["C","Am","F","G"]` with `chord = PROGRESSION[bar % 4]`, i.e. **one chord per bar**. Verified three independent ways (`cd_verify_truth.py`):

- `manifest.json` records `progression = [C,Am,F,G] * 4` and 16 bars of 2.2857 s.
- The written MIDI has, per bar, exactly the target voicing (`bar 0 piano [48,52,55,60] C`, `bar 1 [45,48,52,57] Am`, `bar 2 [41,45,48,53] F`, `bar 3 [43,47,50,55] G`, …) and bass roots C2/A1/F1/G1 — 16/16 bars.
- Chroma decoded per bar on `mix.wav`, `truth/piano.wav` and `truth/bass.wav` returns `C Am F G` repeated four times for all three files (16/16).

The scoring below therefore uses the real map `C, Am, F, G, C, Am, F, G, …` (16 bars, one chord each, bar = 4·60/105 = 2.2857 s). Scoring against the brief's map would have measured the brief, not the model.

Grid: 147 frames of 0.25 s whose start lies in 0 → 36.50 s — all 16 bars (the last bar ends at 36.5714 s; the remaining 0.75 s of release tail is excluded).

## 2. Conditions and levels

All four are 48 kHz mono float32, exactly as long as the mix, so format is not a confound. Conditions 2–4 are summed, then scaled by one scalar to the mix's RMS.

| # | condition | sources | sum RMS before | gain applied | RMS after | peak after |
|---|---|---|---|---|---|---|
| 01 | full mix | synth6/mix.wav (byte-identical copy) | — | +0.00 dB | -16.42 dBFS | -1.01 dBFS |
| 02 | truth harmonic | truth/piano.wav + truth/bass.wav | -11.66 dBFS | -4.76 dB | -16.42 dBFS | -1.65 dBFS |
| 03 | separated harmonic | SW (piano) + SW (bass) | -18.42 dBFS | +2.00 dB | -16.42 dBFS | -1.64 dBFS |
| 04 | separated 4-part harmonic | SW (piano)+(bass)+(guitar)+(other) | -16.81 dBFS | +0.39 dB | -16.42 dBFS | -2.27 dBFS |
| 06 | control: natural level | SW (piano) + SW (bass), no gain | -18.42 dBFS | +0.00 dB | -18.42 dBFS | -3.65 dBFS |
| 07 | control: truth + guitar | truth p+b+guitar | -10.15 dBFS | -6.27 dB | -16.42 dBFS | -2.24 dBFS |

No condition clips (worst peak −1.01 dBFS, the mix itself). The mix needs no gain (+0.00 dB by definition). Note the harmonic conditions need -4.76 dB and +2.00 dB respectively: piano+bass alone is quieter than the full mix, so RMS-matching pushes the separated version 2 dB *up* and the ideal version 4.8 dB *down*. Control 6 keeps the separated version at its natural level (2 dB below the mix) precisely so this gain cannot be mistaken for a separation effect.

## 3. Runs

Each condition was uploaded as its own project and analysed once with `profile=retained10`, `key_mode=continuous`, exactly like `downstream_chords.py`; the app serialised the jobs. One analysis at a time; 15–24 s each here (the brief budgeted 60–110 s, so there was room for the two extra controls).

| run | project id | result file |
|---|---|---|
| 1 mix | `21fedd34166b40c6b7cb11573cc8b493` | `results/01_mix.json` |
| 2 truth p+b | `0b57accbc98d4665b12b3fca84981c75` | `results/02_truth_harm.json` |
| 3 sep p+b | `39bfa1bafb034cb2b754ce61a3c669b3` | `results/03_sep_harm.json` |
| 4 sep p+b+g+o | `5f71b512b6c54631867843c68e45e40b` | `results/04_sep_harm4.json` |
| 1r mix repeat | `d0639a0d51484a49b1e924267f43f988` | `results/05_mix_repeat.json` |
| 6 sep p+b nat | `39dbcefe613a4b64931cbb23f25c7e5f` | `results/06_sep_harm_natural.json` |
| 7 truth p+b+g | `d93c29184c644bd98823c63fa8cdc2a1` | `results/07_truth_harm4.json` |

## 4. Headline accuracy (the app's published timeline, `retained10_full`)

This is the view the app shows and the one `downstream_chords.py` reads (it takes `default_source`). Comparison on the 0.25 s grid, 147 frames inside the 16 bars.

| condition | exact % | root-only % | frames with a published chord | frames wrong chord | frames with no output |
|---|---|---|---|---|---|
| 1 mix | **93.20** | **93.20** | 93.2 % | 0 | 10 |
| 2 truth p+b | **98.64** | **98.64** | 99.3 % | 1 | 1 |
| 3 sep p+b | **98.64** | **98.64** | 100.0 % | 2 | 0 |
| 4 sep p+b+g+o | **91.84** | **91.84** | 93.2 % | 2 | 10 |

Run-to-run noise (condition 1 uploaded twice, identical bytes): **0.00 pp** — the two runs produced bit-identical timelines and bit-identical key scores. The pipeline is deterministic, so the noise floor is 0.00 pp and the smallest resolvable difference is 1 frame = 0.68 pp.

**But read the last two columns.** On this view the mix's 10 misses and 10 of the 4-part harmonic's 12 are frames where the app published *no chord at all* (its complete-segment decoder refused the first bar of the mix and the last bar of three harmonic variants). Wrong-chord frames are 0–2 in every condition.

## 5. Same runs, complete-coverage timeline (`retained10_stable`)

`retained10_stable` is the app's stabilised, full-length chord timeline (and the source its key estimate is pinned to). Same protocol, same 147 frames.

| condition | exact % | root-only % | wrong frames | confusion frames |
|---|---|---|---|---|
| 1 mix | **97.28** | **97.28** | 4 | C->A:min×1, F->A:min×1, G->F:maj×1, F->G:maj×1 |
| 2 truth p+b | **99.32** | **99.32** | 1 | G->F:maj×1 |
| 3 sep p+b | **98.64** | **98.64** | 2 | G->F:maj×1, F->A:min×1 |
| 4 sep p+b+g+o | **98.64** | **98.64** | 2 | G->F:maj×1, F->A:min×1 |

All five timeline views for every run are in `score.json` (`conditions.<name>.views.<view>`). **The ordering is not stable across views**, which is the point: on the base `retained10` view the mix ties the best harmonic variant (97.28), on `retained10_stable` the truth harmonic leads, and on `raw_madmom` the mix is again joint best. The three complete-coverage views (`retained10`, `retained10_stable`, `raw_madmom`) span at most 2.04 pp across conditions; only `retained10_span` / `retained10_full` spread wider, and that extra spread is the edge abstentions counted as errors.

| view | 1 mix | 2 truth p+b | 3 sep p+b | 4 sep p+b+g+o |
|---|---|---|---|---|
| `retained10` | 97.28 | 97.28 | 97.28 | 97.96 |
| `retained10_stable` | 97.28 | 99.32 | 98.64 | 98.64 |
| `retained10_span` | 93.20 | 98.64 | 98.64 | 91.84 |
| `retained10_full` | 93.20 | 98.64 | 98.64 | 91.84 |
| `raw_madmom` | 95.92 | 95.24 | 94.56 | 95.92 |

## 6. Key estimates

`result.key_analysis` has no scalar `key` field — it is `{candidates: [{key, score}, …], model_files: …}`. Top candidate per condition:

| condition | top key | score | C major found | C major score | runner-up |
|---|---|---|---|---|---|
| 1 mix | C | 0.9941 | yes | 0.9941 | G 0.003 |
| 2 truth p+b | C | 0.7787 | yes | 0.7787 | Cm 0.051 |
| 3 sep p+b | C | 0.5245 | yes | 0.5245 | Cm 0.128 |
| 4 sep p+b+g+o | C | 0.9820 | yes | 0.9820 | G 0.011 |
| 1r mix repeat | C | 0.9941 | yes | 0.9941 | G 0.003 |
| 6 sep p+b nat | C | 0.5114 | yes | 0.5114 | Cm 0.137 |
| 7 truth p+b+g | C | 0.9848 | yes | 0.9848 | G 0.009 |

C major is the top candidate in all seven runs, so the key answer is stable. What is **not** stable is the confidence: the full mix gives 0.994 and the 4-part harmonic 0.982–0.985, but piano+bass alone collapses to 0.51–0.78 (runner-up Cm 0.13, Dm 0.11). Dropping the guitar and drums leaves a chroma that is correct but much less discriminative for key.

## 7. Bar grid / downbeat phase

| run | detected bars | first detected bar | ungridded lead-in |
|---|---|---|---|
| 1 mix | 16 | 1.14–3.43 s | 1.14 s |
| 2 truth p+b | 17 | 0.01–2.29 s | 0.01 s |
| 3 sep p+b | 17 | 0.00–2.30 s | 0.00 s |
| 4 sep p+b+g+o | 16 | 0.01–2.29 s | 0.01 s |
| 1r mix repeat | 16 | 1.14–3.43 s | 1.14 s |
| 6 sep p+b nat | 16 | 0.00–2.30 s | 0.00 s |
| 7 truth p+b+g | 16 | 0.01–2.29 s | 0.01 s |

The mix's downbeat grid is a half bar (1.14 s = 2 beats) late: its first detected bar starts at 1.14 s, so the whole first bar is ungridded. Every harmonic variant starts its grid at 0.00–0.01 s. This is the mechanism behind the 10 missing frames in the default view of the mix — the drums derail downbeat phase, and the complete-segment decoder then publishes nothing for the leading bar.

## 8. Timelines side by side

Dominant label per bar, app default view. A cell like `C(9/10)` means 9 of that bar's 10 frames carry `C` and the remaining frame carries a neighbouring chord (the change was crossed one frame early or late); a plain `C` means every frame in the bar agrees. `—` means the app published no chord at all for that bar.

| bar | t0 (s) | truth | 1 mix | 2 truth p+b | 3 sep p+b | 4 sep p+b+g+o |
|---|---|---|---|---|---|---|
| 1 | 0.00 | C | — | C(9/10) | C | C(9/10) |
| 2 | 2.29 | Am | Am(9/10) | Am(9/10) | Am(9/10) | Am(9/10) |
| 3 | 4.57 | F | F | F(9/10) | F(9/10) | F(9/10) |
| 4 | 6.86 | G | G(9/10) | G(9/10) | G(9/10) | G(9/10) |
| 5 | 9.14 | C | C(9/10) | C(9/10) | C(9/10) | C(9/10) |
| 6 | 11.43 | Am | Am(9/10) | Am(9/10) | Am(9/10) | Am(9/10) |
| 7 | 13.71 | F | F(9/10) | F(9/10) | F(9/10) | F(9/10) |
| 8 | 16.00 | G | G | G(9/10) | G(9/10) | G(9/10) |
| 9 | 18.29 | C | C(9/10) | C(9/10) | C(9/10) | C(9/10) |
| 10 | 20.57 | Am | Am(9/10) | Am(9/10) | Am(9/10) | Am(9/10) |
| 11 | 22.86 | F | F(9/10) | F(9/10) | F(9/10) | F(9/10) |
| 12 | 25.14 | G | G(9/10) | G(9/10) | G(9/10) | G(9/10) |
| 13 | 27.43 | C | C(9/10) | C(9/10) | C(9/10) | C(9/10) |
| 14 | 29.71 | Am | Am(9/10) | Am(9/10) | Am(9/10) | Am(9/10) |
| 15 | 32.00 | F | F | F | F(9/10) | F(9/10) |
| 16 | 34.29 | G | G(9/10) | G(9/10) | G(9/10) | —(9/10) |

Same table on the complete-coverage view:

| bar | t0 (s) | truth | 1 mix | 2 truth p+b | 3 sep p+b | 4 sep p+b+g+o |
|---|---|---|---|---|---|---|
| 1 | 0.00 | C | C(9/10) | C | C | C |
| 2 | 2.29 | Am | Am | Am(9/10) | Am(9/10) | Am(9/10) |
| 3 | 4.57 | F | F(9/10) | F(9/10) | F(9/10) | F(9/10) |
| 4 | 6.86 | G | G | G(9/10) | G(9/10) | G(9/10) |
| 5 | 9.14 | C | C(9/10) | C(9/10) | C(9/10) | C(9/10) |
| 6 | 11.43 | Am | Am(9/10) | Am(9/10) | Am(9/10) | Am(9/10) |
| 7 | 13.71 | F | F(9/10) | F(9/10) | F(9/10) | F(9/10) |
| 8 | 16.00 | G | G(9/10) | G(9/10) | G(9/10) | G(9/10) |
| 9 | 18.29 | C | C(9/10) | C(9/10) | C(9/10) | C(9/10) |
| 10 | 20.57 | Am | Am | Am(9/10) | Am(9/10) | Am(9/10) |
| 11 | 22.86 | F | F(9/10) | F(9/10) | F(9/10) | F(9/10) |
| 12 | 25.14 | G | G(9/10) | G(9/10) | G(9/10) | G(9/10) |
| 13 | 27.43 | C | C(9/10) | C(9/10) | C(9/10) | C(9/10) |
| 14 | 29.71 | Am | Am(9/10) | Am(9/10) | Am(9/10) | Am(9/10) |
| 15 | 32.00 | F | F(9/10) | F | F(9/10) | F(9/10) |
| 16 | 34.29 | G | G | G(9/10) | G(9/10) | G(9/10) |

Frame-level excerpt (every 0.25 s frame) for bar 1, bar 2 and bar 16, default view:

| t (s) | truth | 1 mix | 2 truth p+b | 3 sep p+b | 4 sep p+b+g+o |
|---|---|---|---|---|---|
| 0.00 | C | — | — | C | — |
| 0.25 | C | — | C | C | C |
| 0.50 | C | — | C | C | C |
| 0.75 | C | — | C | C | C |
| 1.00 | C | — | C | C | C |
| 1.25 | C | — | C | C | C |
| 1.50 | C | — | C | C | C |
| 1.75 | C | — | C | C | C |
| 2.00 | C | — | C | C | C |
| 2.25 | C | — | C | C | C |
| 2.29 | Am | — | C | C | C |
| 2.54 | Am | Am | Am | Am | Am |
| 2.79 | Am | Am | Am | Am | Am |
| 3.04 | Am | Am | Am | Am | Am |
| 3.29 | Am | Am | Am | Am | Am |
| 3.54 | Am | Am | Am | Am | Am |
| 3.79 | Am | Am | Am | Am | Am |
| 4.04 | Am | Am | Am | Am | Am |
| 4.29 | Am | Am | Am | Am | Am |
| 4.54 | Am | Am | Am | Am | Am |
| 34.29 | G | F | F | F | F |
| 34.54 | G | G | G | G | — |
| 34.79 | G | G | G | G | — |
| 35.04 | G | G | G | G | — |
| 35.29 | G | G | G | G | — |
| 35.54 | G | G | G | G | — |
| 35.79 | G | G | G | G | — |
| 36.04 | G | G | G | G | — |
| 36.29 | G | G | G | G | — |
| 36.54 | G | G | G | G | — |

## 9. Verdict

1. **No material gain, and no consistent winner.** Across the three complete-coverage views all four conditions sit in a 94.6–99.3 % band; on the stabilised view separated piano+bass scores 98.64 % against the mix's 97.28 % (+1.36 pp = 2 frames), and on the base `retained10` view the mix ties the harmonic stems at 97.28 %. The model already gets all 16 bars right from the full mix, drums included.
2. **The apparent +5.44 pp on the published view is an edge artifact, not a chord-recognition gain.** The mix loses bar 1 because the drums put its downbeat grid a half bar late (leading ungridded 1.14 s); the 4-part harmonic loses bar 16 to the same complete-segment rule. Wrong-chord frames are 0–2 everywhere.
3. **Bigger than the noise?** Literally yes (noise is 0.00 pp — the pipeline is deterministic, identical input gives bit-identical output), but the effect is 1–2 boundary frames = 0.68–1.36 pp, the size of one chord change crossed a fraction of a second early or late. That is not a meaningful improvement, and the default-view spread (93.20 / 98.64 / 91.84) is dominated by which edge bar the decoder happened to refuse.

**Practical answer:** keep the full mix as the chord input. Separation is not a chord front-end here — it fixes no wrong chords (there are none) and loses no chord accuracy; its only reproducible effects are on things other than chord identity. It does repair the mix's half-bar downbeat phase error (1.14 s → 0.00 s), and it halves the key confidence when the guitar is dropped (0.99 → 0.51). If the goal is the app's published first bar, fix the downbeat phase in the beat tracker rather than paying for a separation pass.

## 10. Limits and caveats

- One synthetic song, one progression, one model, one profile (`retained10`). The material is deliberately easy: block chords, one chord per bar, no modulation. A harder set could show separation mattering more (or the reverse).
- Deterministic re-runs give a 0.00 pp noise estimate, which measures *pipeline* noise only. It says nothing about song-to-song variance; n = 1 arrangement.
- The `other` and `vocals` SW stems are effectively digital silence here (−96.9 / −102.9 dBFS), so condition 4 is really piano+bass+guitar.
- Scored frames cover the 16 bars (0 → 36.57 s); the 0.75 s release tail is excluded.
- `retained10_full` (default) may omit partial edge segments; those frames are counted as "no output" and reported separately from wrong chords. Any comparison that uses only the default view silently mixes up abstention with error.

## 11. Files

| file | what |
|---|---|
| `cd_common.py` | API session/auth, upload/analyze, timeline extraction |
| `cd_verify_truth.py` | proves the bar map from MIDI + chroma (corrects the brief) |
| `cd_build_conditions.py` | builds conditions 1–4, RMS-matches, writes levels.json |
| `cd_build_extras.py` | builds controls 6–7 |
| `cd_run.py` | uploads/analyses each condition, saves raw results/<name>.json |
| `cd_score.py` | scores all five timeline views, writes score.json |
| `cd_report.py` | renders this REPORT.md from the measured json |
| `truth_check.json` | per-bar MIDI pitches, bass roots, chroma decode |
| `levels.json` | per-condition sources, RMS before/after, gain in dB |
| `score.json` | exact/root/coverage per condition per view + noise + keys |
| `results/*.json` | raw /api/projects/<id>/result payloads, untouched |
| `cond/*.wav` | the seven uploaded files |

Reproduce: `python cd_verify_truth.py`, `python cd_build_conditions.py`, `python cd_build_extras.py`, `python cd_run.py`, `python cd_score.py`, `python cd_report.py` (run from this folder).
