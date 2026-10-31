#!/usr/bin/env python3
"""Validate the POP909 boundary dataset.

Checks
------
1. **Wrong notes** - every note of the chord track must be a chord tone of the
   chord that is sounding at its onset.  Melody notes must be a chord tone *or*
   an in-key note, and must not be a semitone away from any chord tone (no b9 /
   major-7th clash).
2. **Boundary alignment** - every positive boundary must coincide with a MIDI
   note onset within 30 ms, and every negative point that is marked
   ``has_onset`` must also have an onset there.
3. **Harmonic sanity** - Roman numerals are recomputed relative to the song key
   and the progression is flagged when it starts on V, contains chords whose
   root is outside the key, or uses a non-diatonic quality (secondary dominant /
   modal mixture candidates).
4. **Files** - MIDI, label and audio files exist; audio is 48 kHz mono and at
   least as long as the label timeline.
5. A random sample of 10 progressions is printed for manual inspection.

Usage
-----
    python check.py --dataset .            # validate everything that exists
    python check.py --dataset . --audio    # also require/validate wav files
    python check.py --dataset . --prune    # drop suspicious samples from dataset.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import pretty_midi
import soundfile as sf

HERE = Path(__file__).resolve().parent

NOTE_PC = {
    "C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5, "F": 5, "F#": 6, "Gb": 6, "G": 7,
    "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}
MAJOR_STEPS = {0: "I", 1: "bII", 2: "II", 3: "bIII", 4: "III", 5: "IV",
               6: "#IV", 7: "V", 8: "bVI", 9: "VI", 10: "bVII", 11: "VII"}
MINOR_STEPS = {0: "I", 1: "bII", 2: "II", 3: "III", 4: "#III", 5: "IV",
               6: "#IV", 7: "V", 8: "VI", 9: "#VI", 10: "VII", 11: "#VII"}
SCALE_PCS = {"maj": {0, 2, 4, 5, 7, 9, 11}, "min": {0, 2, 3, 5, 7, 8, 10}}
DIATONIC_QUALITY = {
    "maj": {0: {"maj", "maj7", "maj6", "sus2", "sus4", "7", "aug", "sus4(b7)"},
            2: {"min", "min7", "min6"},
            4: {"min", "min7"},
            5: {"maj", "maj7", "maj6", "sus2", "sus4", "7", "sus4(b7)"},
            7: {"maj", "7", "sus4", "sus2", "sus4(b7)"},
            9: {"min", "min7", "min6"},
            11: {"dim", "hdim7", "dim7", "min7"}},
    "min": {0: {"min", "min7", "min6", "minmaj7", "sus4", "sus2"},
            2: {"dim", "hdim7", "dim7", "min7"},
            3: {"maj", "maj7", "maj6", "sus2", "sus4"},
            5: {"min", "min7", "min6"},
            7: {"min", "min7", "maj", "7", "sus4", "sus2", "sus4(b7)"},
            8: {"maj", "maj7", "maj6", "sus2", "sus4"},
            10: {"maj", "7", "sus4", "sus2", "sus4(b7)"}},
}


def pcs_from_name(name: str) -> int:
    name = name.strip()
    if name in NOTE_PC:
        return NOTE_PC[name]
    if len(name) >= 2 and name[1] in "#b" and name[:2] in NOTE_PC:
        return NOTE_PC[name[:2]]
    raise KeyError(f"cannot parse note name {name!r}")


def roman_of(root_pc: int, quality: str, tonic: int, mode: str) -> str:
    rel = (root_pc - tonic) % 12
    numeral = (MAJOR_STEPS if mode == "maj" else MINOR_STEPS)[rel]
    if quality in ("min", "min7", "min6", "minmaj7", "dim", "dim7", "hdim7"):
        body = numeral.lower()
    else:
        body = numeral.upper()
    if numeral[0] in "b#":
        body = numeral[0] + body[1:]
    return body


def classify_chromatic(root_pc: int, tonic: int, mode: str,
                       quality: str = "maj") -> str | None:
    """Return None if the chord root is diatonic, else how it is explained."""
    rel = (root_pc - tonic) % 12
    if rel in SCALE_PCS[mode]:
        return None
    if quality in ("dim", "dim7", "hdim7", "aug"):
        return "unexplained"
    if mode == "maj" and rel in {1, 3, 8, 10}:
        return "mixture"
    if mode == "min" and rel in {4, 9, 11}:
        return "mixture"
    if ((rel - 7) % 12) in SCALE_PCS[mode]:
        return "secondary_dominant"
    return "unexplained"


def melody_allowed(chord_pcs: set[int], scale_pcs: set[int]) -> set[int]:
    allowed = set(chord_pcs)
    for pc in scale_pcs:
        if pc in allowed:
            continue
        if any((pc - c) % 12 in (1, 11) for c in chord_pcs):
            continue
        allowed.add(pc)
    return allowed


def analyse_sample(sample: dict, ds: Path, require_audio: bool) -> dict:
    sid = sample["sample_id"]
    issues: list[str] = []
    label_path = ds / sample["labels"]
    midi_path = ds / sample["midi"]
    audio_path = ds / sample["audio"]
    if not label_path.is_file():
        return {"sample_id": sid, "fatal": f"missing label {label_path}"}
    if not midi_path.is_file():
        return {"sample_id": sid, "fatal": f"missing midi {midi_path}"}
    lab = json.loads(label_path.read_text(encoding="utf-8"))
    tonic = pcs_from_name(lab["key"].split(":")[0])
    mode = lab["mode"]
    scale = {(tonic + k) % 12 for k in SCALE_PCS[mode]}
    chords = lab["chords"]
    starts = [c["start_beat"] for c in chords]
    bpm = lab["bpm"]

    import bisect

    def chord_at(t: float):
        """Chord sounding at time t (beat-accurate, independent of rounded times)."""
        beat = t * bpm / 60.0
        # MIDI times come back from pretty_midi with ~1e-6 beat error; snap to
        # the quantised grid so that a note exactly on a chord change belongs to
        # the new chord.
        if abs(beat - round(beat)) < 2e-3:
            beat = float(round(beat))
        i = bisect.bisect_right(starts, beat + 1e-6) - 1
        if i < 0:
            i = 0
        if i >= len(chords):
            i = len(chords) - 1
        return chords[i]

    # ---- 1. wrong notes -------------------------------------------------
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    chord_notes = []
    melody_notes = []
    for inst in pm.instruments:
        is_melody = (inst.name or "").upper().startswith("MEL")
        (melody_notes if is_melody else chord_notes).extend(inst.notes)
    wrong_chord = []
    for n in chord_notes:
        c = chord_at(n.start)
        if (n.pitch % 12) not in set(c["pcs"]):
            wrong_chord.append({"time": round(n.start, 4), "pitch": n.pitch,
                                "pc": n.pitch % 12, "chord": c["label"],
                                "chord_pcs": c["pcs"]})
    melody_bad = []
    melody_conflict = []
    for n in melody_notes:
        c = chord_at(n.start)
        cps = set(c["pcs"])
        if (n.pitch % 12) not in melody_allowed(cps, scale):
            kind = "out_of_key" if (n.pitch % 12) not in scale else "clash"
            melody_bad.append({"time": round(n.start, 4), "pitch": n.pitch,
                               "chord": c["label"], "reason": kind})
        elif (n.pitch % 12) not in cps and any(
                ((n.pitch % 12) - pc) % 12 in (1, 11) for pc in cps):
            melody_conflict.append({"time": round(n.start, 4), "pitch": n.pitch,
                                    "chord": c["label"]})
    if wrong_chord:
        issues.append(f"{len(wrong_chord)} wrong chord-tone notes")
    if melody_bad:
        issues.append(f"{len(melody_bad)} melody notes outside key/chord")

    # ---- 2. boundary alignment -----------------------------------------
    onsets = sorted([n.start for n in chord_notes] + [n.start for n in melody_notes])
    max_err = 0.0
    bad_align = []
    for b in lab["boundaries"]:
        t = b["time"]
        j = bisect.bisect_left(onsets, t)
        cand = [onsets[k] for k in (j - 1, j) if 0 <= k < len(onsets)]
        err = min(abs(x - t) for x in cand) if cand else 1e9
        max_err = max(max_err, err)
        if err > 0.030:
            bad_align.append({"time": t, "err_ms": round(err * 1000, 1)})
    neg_bad = []
    for np_ in lab["negative_points"]:
        if not np_.get("has_onset"):
            continue
        t = np_["time"]
        j = bisect.bisect_left(onsets, t)
        cand = [onsets[k] for k in (j - 1, j) if 0 <= k < len(onsets)]
        err = min(abs(x - t) for x in cand) if cand else 1e9
        if err > 0.030:
            neg_bad.append({"time": t, "err_ms": round(err * 1000, 1), "kind": np_.get("kind")})
    if bad_align:
        issues.append(f"{len(bad_align)} boundaries >30 ms from an onset")
    if neg_bad:
        issues.append(f"{len(neg_bad)} negative points >30 ms from an onset")

    # ---- 3. harmonic sanity --------------------------------------------
    starts_on_v = False
    out_of_key = []
    quality_mismatch = []
    for i, c in enumerate(chords):
        root = c["root_pc"]
        rel = (root - tonic) % 12
        stored = c["roman"]
        calc = roman_of(root, c["quality"], tonic, mode)
        if not stored.startswith(calc.rstrip("0123456789+°ø")):
            issues.append(f"roman mismatch {c['label']}: label={stored} calc={calc}")
        if i == 0:
            base = calc.lstrip("b#")
            starts_on_v = base.startswith("V") and not base.startswith("VI") and not base.startswith("VII")
        if rel not in SCALE_PCS[mode]:
            out_of_key.append({"index": i, "label": c["label"], "roman": stored,
                               "rel": rel,
                               "explanation": classify_chromatic(root, tonic, mode,
                                                                c["quality"])})
        elif c["quality"] not in ("sus2", "sus4", "sus4(b7)"):
            # suspended chords do not state a third, so they fit any degree
            allowed = DIATONIC_QUALITY[mode].get(rel, set())
            if c["quality"] not in allowed:
                quality_mismatch.append({"index": i, "label": c["label"],
                                         "roman": stored, "quality": c["quality"]})
    if starts_on_v:
        issues.append("progression starts on V")
    if out_of_key:
        issues.append(f"{len(out_of_key)} chord(s) with out-of-key root")
    if quality_mismatch:
        issues.append(f"{len(quality_mismatch)} non-diatonic quality")

    # ---- 4. files -------------------------------------------------------
    audio_info = None
    if require_audio:
        if not audio_path.is_file():
            issues.append("missing audio")
        else:
            info = sf.info(str(audio_path))
            audio_info = {"sr": info.samplerate, "channels": info.channels,
                          "duration": round(info.duration, 3)}
            if info.samplerate != 48000:
                issues.append(f"audio sample rate {info.samplerate} != 48000")
            if info.channels != 1:
                issues.append(f"audio has {info.channels} channels (expected mono)")
            if info.duration + 0.05 < lab["duration"]:
                issues.append("audio shorter than label timeline")

    return {
        "sample_id": sid,
        "key": lab["key"],
        "texture": lab["texture"],
        "instrument": lab["instrument"]["name"],
        "melody": lab["melody"],
        "duration": lab["duration"],
        "n_chords": len(chords),
        "n_boundaries": len(lab["boundaries"]),
        "n_negatives": len(lab["negative_points"]),
        "n_notes_chord": len(chord_notes),
        "n_notes_melody": len(melody_notes),
        "wrong_chord_notes": len(wrong_chord),
        "wrong_chord_examples": wrong_chord[:5],
        "melody_out_of_key": len(melody_bad),
        "melody_conflicts": len(melody_conflict),
        "boundary_max_err_ms": round(max_err * 1000, 3),
        "boundary_bad": bad_align,
        "negative_bad": neg_bad,
        "starts_on_v": starts_on_v,
        "out_of_key": out_of_key,
        "quality_mismatch": quality_mismatch,
        "progression_roman": lab["progression_roman"],
        "progression_labels": lab["progression_labels"],
        "audio": audio_info,
        "issues": issues,
    }


def suspicion_reasons(r: dict) -> list[str]:
    reasons = []
    if r.get("wrong_chord_notes"):
        reasons.append("wrong_notes")
    if r.get("melody_out_of_key"):
        reasons.append("melody_out_of_key")
    if r.get("boundary_bad"):
        reasons.append("boundary_misaligned")
    if r.get("starts_on_v"):
        reasons.append("starts_on_V")
    if len(r.get("out_of_key", [])) >= 3 or (
            r.get("out_of_key") and len(r["out_of_key"]) / max(1, r["n_chords"]) > 0.25):
        reasons.append("too_many_out_of_key")
    if any(o.get("explanation") == "unexplained" for o in r.get("out_of_key", [])):
        reasons.append("unexplained_chromatic")
    fatal = r.get("fatal")
    if fatal:
        reasons.append(f"fatal:{fatal}")
    return reasons


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(HERE))
    ap.add_argument("--audio", action="store_true", help="also validate the wav files")
    ap.add_argument("--prune", action="store_true",
                    help="drop samples with wrong notes / out-of-key overload / V start")
    ap.add_argument("--n-print", type=int, default=10)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)

    ds = Path(args.dataset)
    manifest = json.loads((ds / "dataset.json").read_text(encoding="utf-8"))
    results = []
    for s in manifest["samples"]:
        r = analyse_sample(s, ds, args.audio)
        r["suspicion"] = suspicion_reasons(r)
        results.append(r)

    ok = [r for r in results if not r.get("fatal")]
    tot_notes = sum(r["n_notes_chord"] + r["n_notes_melody"] for r in ok)
    wrong = sum(r["wrong_chord_notes"] for r in ok)
    mel_bad = sum(r["melody_out_of_key"] for r in ok)
    mel_conf = sum(r["melody_conflicts"] for r in ok)
    max_err = max((r["boundary_max_err_ms"] for r in ok), default=0.0)
    bad_b = sum(len(r["boundary_bad"]) for r in ok)
    bad_n = sum(len(r["negative_bad"]) for r in ok)
    n_v = sum(1 for r in ok if r["starts_on_v"])
    n_oob = sum(1 for r in ok if r["out_of_key"])
    n_nondia = sum(1 for r in ok if r["quality_mismatch"])
    chrom_all = [o for r in ok for o in r["out_of_key"]]
    n_unexpl = sum(1 for o in chrom_all if o.get("explanation") == "unexplained")
    n_secondary = sum(1 for o in chrom_all if o.get("explanation") == "secondary_dominant")
    n_mixture = sum(1 for o in chrom_all if o.get("explanation") == "mixture")
    suspicious = [r for r in ok if r["suspicion"]]
    audio_ok = [r for r in ok if r.get("audio")]
    audio_issues = [r for r in ok if args.audio and any(
        i in ("missing audio", "audio shorter than label timeline")
        or i.startswith("audio sample rate") or i.startswith("audio has")
        for i in r["issues"])]

    report = {
        "dataset": str(ds),
        "samples": len(results),
        "samples_checked": len(ok),
        "notes_total": tot_notes,
        "wrong_chord_notes": wrong,
        "melody_notes_outside_key_or_chord": mel_bad,
        "melody_semitone_conflicts": mel_conf,
        "boundary_alignment": {
            "max_err_ms": round(max_err, 3),
            "boundaries_over_30ms": bad_b,
            "negatives_over_30ms": bad_n,
            "threshold_ms": 30,
        },
        "harmony": {
            "starts_on_V": n_v,
            "samples_with_out_of_key_chords": n_oob,
            "samples_with_non_diatonic_quality": n_nondia,
            "out_of_key_chords_total": len(chrom_all),
            "out_of_key_secondary_dominants": n_secondary,
            "out_of_key_mixture": n_mixture,
            "out_of_key_unexplained": n_unexpl,
        },
        "audio": {
            "validated": bool(args.audio),
            "files": len(audio_ok),
            "sample_rate": sorted({a["audio"]["sr"] for a in audio_ok}) if audio_ok else [],
            "channels": sorted({a["audio"]["channels"] for a in audio_ok}) if audio_ok else [],
        },
        "suspicious": [{"sample_id": r["sample_id"], "reasons": r["suspicion"],
                        "progression": r["progression_roman"],
                        "out_of_key": [o["label"] for o in r["out_of_key"]]}
                       for r in suspicious],
        "results": results,
    }

    rng = random.Random(args.seed)
    sample10 = rng.sample(ok, min(args.n_print, len(ok)))

    lines = []
    lines.append("# Boundary dataset validation\n")
    lines.append(f"- samples: {len(results)} checked ({len(ok)} readable)")
    lines.append(f"- notes: {tot_notes} (chord {sum(r['n_notes_chord'] for r in ok)}, "
                 f"melody {sum(r['n_notes_melody'] for r in ok)})")
    lines.append(f"- **wrong chord-tone notes: {wrong}**")
    lines.append(f"- melody notes outside key/chord: {mel_bad}, "
                 f"melody semitone conflicts: {mel_conf}")
    lines.append(f"- boundary alignment: max |err| = {max_err:.3f} ms, "
                 f"{bad_b} positives / {bad_n} negatives over 30 ms")
    lines.append(f"- progressions starting on V: {n_v}")
    lines.append(f"- samples with out-of-key chord roots: {n_oob} "
                 f"({report['harmony']['out_of_key_chords_total']} chords: "
                 f"{n_secondary} secondary dominant, {n_mixture} modal mixture, "
                 f"**{n_unexpl} unexplained**)")
    lines.append(f"- samples with non-diatonic chord quality: {n_nondia}")
    if args.audio:
        lines.append(f"- audio: {len(audio_ok)} files, sample rates {report['audio']['sample_rate']}, "
                     f"channels {report['audio']['channels']}, "
                     f"{len(audio_issues)} audio issues")
    lines.append("")
    if suspicious:
        lines.append("## Suspicious samples")
        for s in suspicious[:40]:
            lines.append(f"- `{s['sample_id']}` {s['suspicion']} :: "
                         f"{' '.join(s['progression_roman'])}"
                         + (f" | chromatic: "
                            + ", ".join(f"{o['label']}({o['explanation']})"
                                        for o in s["out_of_key"])
                            if s["out_of_key"] else ""))
    else:
        lines.append("## Suspicious samples\n\nNone - every sample starts on a tonic / "
                     "submediant / subdominant chord and has <= 25 % out-of-key roots.")
    lines.append("")
    lines.append(f"## {len(sample10)} random progressions for manual inspection")
    for s in sample10:
        lines.append(f"\n### {s['sample_id']}  ({s['key']}, {s['texture']}, "
                     f"{s['instrument']}, melody={s['melody']})")
        lines.append(f"- labels: {' '.join(s['progression_labels'])}")
        lines.append(f"- roman:  {' '.join(s['progression_roman'])}")

    md = "\n".join(lines) + "\n"
    (ds / "check_report.md").write_text(md, encoding="utf-8")
    (ds / "check_report.json").write_text(
        json.dumps({k: v for k, v in report.items() if k != "results"},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    (ds / "check_details.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print(md)
    if args.prune and suspicious:
        drop = {s["sample_id"] for s in suspicious}
        before = len(manifest["samples"])
        manifest["samples"] = [s for s in manifest["samples"] if s["sample_id"] not in drop]
        (ds / "dataset.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
        print(f"[prune] dropped {before - len(manifest['samples'])} samples: {sorted(drop)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
