import unittest
from app.key_chord_adapter import adapt, parse_rows, parse_chord_label
from app.key_coherence import pcs, rank


class KeyChordAdapterTests(unittest.TestCase):
    def test_explicit_full_pitch_sets(self):
        cases={'Cmaj9':{0,2,4,7,11},'Dm9':{0,2,4,5,9},'G#add9':{0,3,8,10},
               'C11':{0,2,4,5,7,10},'C13':{0,2,4,5,7,9,10},
               'Cmaj13':{0,2,4,5,7,9,11},'Cm13':{0,2,3,5,7,9,10},
               'CmMaj9':{0,2,3,7,11},'C7sus4':{0,5,7,10},'C9sus4':{0,2,5,7,10},
               'C:9(*9)':{0,4,7,10},'C:min(*5)':{0,3},'C:(b3,b5)':{0,3,6}}
        for label,expected in cases.items():
            with self.subTest(label=label):self.assertEqual(pcs(parse_chord_label(label)),expected)

    def test_aliases_transposition_and_score(self):
        for a,b in [('Dm9','D:min9'),('Cmaj9','C:maj7(9)'),('G#add9','Ab:maj(9)'),('C9','C:7(9)')]:
            x=parse_rows([(0,2,a)])[0];y=parse_rows([(0,2,b)])[0]
            self.assertEqual(pcs(x[0]),pcs(y[0]))
            self.assertEqual([r['score'] for r in rank(x)],[r['score'] for r in rank(y)])
        for pc,label in enumerate(['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']):
            self.assertEqual(pcs(parse_chord_label(label+'maj9')),{(pc+x)%12 for x in [0,2,4,7,11]})

    def test_explicit_bass_is_separate_and_preserved(self):
        chord=parse_chord_label('C/D')
        self.assertEqual(chord.bass,2)
        self.assertEqual(chord.colors,[])
        self.assertEqual(pcs(chord),{0,2,4,7})
        self.assertIsNone(parse_chord_label('C').bass)
        self.assertEqual(parse_chord_label('Cb/Eb').bass,3)

    def test_bad_suffix_is_not_truncated_to_a_known_chord(self):
        for label in ['Cmaj9oops','C:min7BAD','C:maj//E','C:???','C:maj(*1)']:
            self.assertIsNone(adapt(label)[0])
            self.assertEqual(adapt(label)[1]['state'],'X')
        _,stats=parse_rows([(0,1,'N'),(1,2,'X'),(2,3,None),(3,4,'Cmaj9oops')])
        self.assertEqual((stats['N'],stats['X'],stats['missing'],stats['unparsed']),(1,2,1,2))

    def test_sharp_nine_does_not_change_triad_quality(self):
        c=parse_chord_label('C:7(#9)')
        self.assertEqual((c.third,c.fifth,c.seventh,c.kind),(4,7,10,'dom'))
        self.assertIn((3,.20),c.colors)


if __name__=='__main__':unittest.main()
