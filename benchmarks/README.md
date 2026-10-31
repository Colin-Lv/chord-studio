# Benchmark Workspace

This folder is the local evaluation workspace for beat/downbeat tracking, chord recognition, and harmonic progression experiments.

Audio is not committed here. Many benchmark datasets contain copyrighted recordings, so keep only annotations, manifests, and local paths to legally obtained audio.

## Recommended Data Mix

| Dataset | Primary Use | Suggested First Split |
| --- | --- | --- |
| McGill Billboard | chord recognition, transition statistics | 50 validation + 100 test |
| Isophonics | chord + beat/bar + key + structure | 20 Beatles + 20 mixed albums |
| Ballroom | beat, tempo, meter | 50 rhythm-diverse tracks |
| SALAMI | section/progression structure | 30 tracks with matching audio |
| RWC-Pop | pop generalization | add later if you have access |

## Directory Layout

> **2026-10-30 归档**：`audio/`、`raw/`、`external_key_data/`、`predictions/` 等语料目录已移出项目树，
> 现在各目录里只剩 `README_ARCHIVED.md` 指路，实际数据在
> `D:\ChordStudio\_archive\2026-10-30\benchmarks\`。需要跑数据集评测时按 README 里的命令移回即可。
> `reports/` 与 `models/` 只保留被代码引用的部分，索引见同目录的 `README_INDEX.md`、`README_CURRENT.md`。

```text
benchmarks/
  manifest*.csv           当前清单（保留在原处）
  references/             评测参考标签（保留）
  models/                 1 个被引用的模型 + _legacy_202608/（旧模型）
  reports/                18 个实验目录 + _legacy_tables_202608/（旧散表）
  audio/  raw/  predictions/  external_key_data/   → 已归档，目录内只有指路 README
```

Use `manifest.example.csv` as the schema. Paths are relative to the manifest file unless absolute.

## Manifest Columns

- `id`: stable item id, for example `isophonics_beatles_001`
- `dataset`: `billboard`, `isophonics`, `ballroom`, `salami`, `rwc_pop`, or `local`
- `split`: `dev`, `test`, `train`, or `holdout`
- `artist`, `title`: human-readable metadata
- `audio`: optional local audio path
- `prediction`: Chord Studio `*_analysis.json`
- `chord_reference`: `.lab`, `.txt`, or `.csv` timed chord reference
- `beat_reference`: `.txt`, `.beats`, or `.json` beat/downbeat reference
- `key`, `meter`: optional known metadata
- `notes`: free text

## Running Reports

```powershell
python .\app\run_benchmark.py .\benchmarks\manifest.csv -o .\benchmarks\reports\latest
```

Chord metrics:

- `root_accuracy`
- `triad_accuracy`

Beat metrics:

- `beat_f1`
- `downbeat_f1`
- median absolute timing error in milliseconds

## Dataset Sources

- McGill Billboard Project: https://ddmal.ca/research/The_McGill_Billboard_Project_%28Chord_Analysis_Dataset%29/
- Billboard annotation mirror/code: https://github.com/DDMAL/billboard
- Isophonics datasets: https://isophonics.net/datasets
- Ballroom annotations via mirdata/CPJKU: https://github.com/CPJKU/BallroomAnnotations
- SALAMI annotations: https://github.com/DDMAL/salami-data-public
- RWC Music Database: https://staff.aist.go.jp/m.goto/RWC-MDB/

## Importing Annotations

Use the importer to normalize downloaded annotation files into `.lab` and `.beats` references:

```powershell
python .\app\import_annotations.py ".\benchmarks\raw\billboard" --dataset billboard --chord-glob "**/*.lab" --manifest ".\benchmarks\manifest.billboard.csv"
python .\app\import_annotations.py ".\benchmarks\raw\isophonics" --dataset isophonics --chord-glob "**/*.lab" --beat-glob "**/*.txt" --manifest ".\benchmarks\manifest.isophonics.csv"
python .\app\import_annotations.py ".\benchmarks\raw\ballroom" --dataset ballroom --chord-glob "**/*.none" --beat-glob "**/*.beats" --manifest ".\benchmarks\manifest.ballroom.csv"
```

Then collect progression statistics from entries that have `key` filled in:

```powershell
python .\app\progression_stats.py ".\benchmarks\manifest.billboard.csv" -o ".\benchmarks\reports\billboard_progressions.json"
```

## Practical First Benchmark

Start with a small, balanced set:

- 10 Isophonics Beatles tracks with beat + chord annotations
- 25 Billboard tracks with chord annotations
- 20 Ballroom tracks for beat/downbeat only
- 5 local Chinese pop/folk tracks once you have corrected labels

This is enough to prevent overfitting to one song while keeping iteration fast.

## Current Local Manifests

The following annotation-only manifests have been prepared locally:

- `manifest.isophonics_beatles_all.csv`: 180 Beatles entries from the Isophonics `all/` TTL files
- `manifest.isophonics_beatles_20.csv`: 20-entry dev subset requiring chord + beat + key
- `manifest.ballroom.csv`: 698 Ballroom beat annotation entries
- `manifest.ballroom_50.csv`: 50-entry beat/downbeat dev subset
- `manifest.public_dev.csv`: combined 70-entry quick dev benchmark

These manifests do not include audio. Fill the `audio` column with legal local audio paths, generate predictions into the `prediction` paths, then run:

```powershell
python .\app\run_benchmark.py .\benchmarks\manifest.public_dev.csv -o .\benchmarks\reports\public_dev
```
