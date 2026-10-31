# Boundary dataset validation

- samples: 144 checked (144 readable)
- notes: 33622 (chord 31522, melody 2100)
- **wrong chord-tone notes: 0**
- melody notes outside key/chord: 0, melody semitone conflicts: 0
- boundary alignment: max |err| = 0.050 ms, 0 positives / 0 negatives over 30 ms
- progressions starting on V: 0
- samples with out-of-key chord roots: 24 (27 chords: 3 secondary dominant, 24 modal mixture, **0 unexplained**)
- samples with non-diatonic chord quality: 90
- audio: 144 files, sample rates [48000], channels [1], 0 audio issues

## Suspicious samples

None - every sample starts on a tonic / submediant / subdominant chord and has <= 25 % out-of-key roots.

## 10 random progressions for manual inspection

### pop909-632-arpeggio-mel  (Ab:min, arpeggio, Acoustic Grand Piano, melody=True)
- labels: Ab:min E:maj/3 Eb:min/b3 C#:min/b3 Eb:min/b3 E:maj/3 Eb:min/b3 C#:min/b3 Eb:min/b3 Ab:min F#:maj E:maj B:maj C#:min Ab:min Eb:maj Ab:maj
- roman:  i VI v iv v VI v iv v i VII VI III iv i V I

### pop909-882-strum-mel  (D:maj, strum, Acoustic Guitar (steel), melody=True)
- labels: F#:maj D:maj G:maj G:sus2 G:maj B:maj B:min G:maj E:min G:maj A:maj D:sus2 D:maj B:min G:maj
- roman:  III I IV IVsus2 IV VI vi IV ii IV V Isus2 I vi IV

### pop909-643-strum  (Db:maj, strum, Acoustic Guitar (nylon), melody=False)
- labels: C#:maj Ab:maj Bb:min Ab:min C#:maj F#:maj F:min Eb:min Ab:maj C#:maj A:maj Ab:maj C#:maj Ab:maj Bb:sus2 Ab:maj F#:maj F:min Eb:min Bb:min Ab:maj C#:maj Ab:maj Bb:min C#:7 F#:maj
- roman:  I V vi v I IV iii ii V I bVI V I V VIsus2 V IV iii ii vi V I V vi I7 IV

### pop909-717-arpeggio-mel  (E:maj, arpeggio, Acoustic Grand Piano, melody=True)
- labels: E:maj Ab:min/5 C#:min B:maj A:maj F#:min B:maj Ab:min C#:min Ab:min C#:min F#:min E:maj/5 B:sus4 E:maj Ab:min/5 C#:min B:maj A:maj F#:min B:maj E:maj/5 Ab:min C#:min Ab:min C#:min/5
- roman:  I iii vi V IV ii V iii vi iii vi ii I Vsus4 I iii vi V IV ii V I iii vi iii vi

### pop909-322-block-mel  (E:min, block, Acoustic Guitar (nylon), melody=True)
- labels: G:maj D:maj/3 B:min E:min7 B:min7 E:min7 A:min D:maj G:maj D:maj/3 B:min7 E:min C:maj G:maj D:maj G:maj G:sus4 D:maj G:maj D:maj/3
- roman:  III VII v i7 v7 i7 iv VII III VII v7 i VI III VII III IIIsus4 VII III VII

### pop909-260-strum  (B:maj, strum, Acoustic Grand Piano, melody=False)
- labels: C#:min F#:maj B:maj B:sus2 C#:min7 C#:sus2 Ab:sus2 Ab:min C#:min Eb:sus4 Eb:maj Ab:min B:maj7/3 C#:min F#:maj F#:7 B:maj7 B:sus2 B:maj C#:min7 Ab:min E:maj F#:maj Ab:min F#:maj
- roman:  ii V I Isus2 ii7 IIsus2 VIsus2 vi ii IIIsus4 III vi Imaj7 ii V V7 Imaj7 Isus2 I ii7 vi IV V vi V

### pop909-717-strum  (E:maj, strum, Acoustic Guitar (nylon), melody=False)
- labels: E:maj Ab:min/5 C#:min B:maj A:maj F#:min B:maj Ab:min C#:min Ab:min C#:min F#:min E:maj/5 B:sus4 E:maj Ab:min/5 C#:min B:maj A:maj F#:min B:maj E:maj/5 Ab:min C#:min Ab:min C#:min/5
- roman:  I iii vi V IV ii V iii vi iii vi ii I Vsus4 I iii vi V IV ii V I iii vi iii vi

### pop909-647-arpeggio  (Bb:min, arpeggio, Electric Piano 1, melody=False)
- labels: G:min Bb:maj F#:maj F:maj C#:maj/3 Bb:min F:min F#:maj C#:maj Bb:min7 F:min Eb:maj F:maj Bb:min F:min F#:maj C#:maj Bb:min F:min Eb:maj
- roman:  #vi I VI V III i v VI III i7 v IV V i v VI III i v IV

### pop909-197-block  (C:maj, block, Acoustic Guitar (steel), melody=False)
- labels: B:hdim7 A:min7 C:maj/5 G:maj6 G:sus4(b7) F:maj F:sus2 C:maj/3 C:maj D:min7 G:sus4(b7) G:sus2 A:sus2 D:sus4(b7) C:maj C:sus2 C:maj E:maj/5 B:sus4(b7) A:min7 G:min C:maj G:sus2 F:maj C:maj/3 D:min7 C:sus4 G:maj G:sus4 D:sus4 D:min7 D:sus4(b7) F:maj/3 G:sus4(b7) G:sus4 G:sus4(b7) C:maj C:maj6 C:maj D:min7 C:maj/5 D:min7 F:maj/5
- roman:  viihdim7 vi7 I V6 V7sus4 IV IVsus2 I I ii7 V7sus4 Vsus2 VIsus2 II7sus4 I Isus2 I III VII7sus4 vi7 v I Vsus2 IV I ii7 Isus4 V Vsus4 IIsus4 ii7 II7sus4 IV V7sus4 Vsus4 V7sus4 I I6 I ii7 I ii7 IV

### pop909-632-block  (Ab:min, block, Acoustic Guitar (steel), melody=False)
- labels: Ab:min E:maj/3 Eb:min/b3 C#:min/b3 Eb:min/b3 E:maj/3 Eb:min/b3 C#:min/b3 Eb:min/b3 Ab:min F#:maj E:maj B:maj C#:min Ab:min Eb:maj Ab:maj
- roman:  i VI v iv v VI v iv v i VII VI III iv i V I
