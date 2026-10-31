import unittest
from types import SimpleNamespace
from app.key_coherence import evidence, native, rank, key_id, name

def chord(root,third=4,start=0,end=1,seventh=None):
    return SimpleNamespace(root=root,third=third,fifth=7,seventh=seventh,
                           colors=[],bass=None,start=start,end=end,dur=end-start,
                           label=f'{root}:{third}')

class CoherenceTests(unittest.TestCase):
    def test_no_evidence_is_not_c_major(self):
        self.assertEqual(rank([]),[])

    def test_harmonic_minor_dominant(self):
        self.assertTrue(native(chord(4,4,seventh=10),9,'minor'))
        self.assertFalse(native(chord(0,4,seventh=8),9,'minor'))

    def test_arrival_requires_correct_target_quality_and_adjacency(self):
        major=[chord(7,4,0,1),chord(0,4,1,2)]
        minor=[chord(7,4,0,1),chord(0,3,1,2)]
        self.assertEqual(evidence(major,0,'major')['authentic_arrivals'],1)
        self.assertEqual(evidence(minor,0,'major')['authentic_arrivals'],0)
        major[1].start=2
        major[1].end=3
        self.assertEqual(evidence(major,0,'major')['authentic_arrivals'],0)

    def test_transposition_equivariance(self):
        for mode in ['major','minor']:
            home_third=4 if mode=='major' else 3
            seq=[chord(7,4,0,1,10),chord(0,home_third,1,4)]
            base=rank(seq)[0]['key']
            for shift in range(12):
                moved=[chord((c.root+shift)%12,c.third,c.start,c.end,c.seventh) for c in seq]
                actual=key_id(rank(moved)[0]['key'])
                pc,m=key_id(base)
                self.assertEqual(actual,((pc+shift)%12,m))

    def test_splitting_held_chord_does_not_multiply_arrivals(self):
        full=[chord(7,4,0,2),chord(0,4,2,6)]
        split=[chord(7,4,0,1),chord(7,4,1,2),chord(0,4,2,3),chord(0,4,3,6)]
        for key in [(0,'major'),(9,'minor'),(7,'major')]:
            a,b=evidence(full,*key),evidence(split,*key)
            for field in ('tonic_share','authentic_arrivals','predominant_returns','outside_pitch_share','log_compatibility'):
                self.assertAlmostEqual(a[field],b[field])

    def test_secondary_dominant_subdivision_preserves_all_scores(self):
        full=[chord(4,4,0,2,10),chord(9,3,2,4)]
        split=[chord(4,4,0,1,10),chord(4,4,1,2,10),chord(9,3,2,4)]
        a={r['key']:r for r in rank(full)}
        b={r['key']:r for r in rank(split)}
        for key in a:
            self.assertAlmostEqual(a[key]['score'],b[key]['score'],msg=key)

    def test_major_seventh_is_not_a_secondary_dominant(self):
        seq=[chord(4,4,0,1,11),chord(9,3,1,2)]
        roles=[x['role'] for x in evidence(seq,0,'major')['exceptions']]
        self.assertNotIn('resolved_secondary_dominant',roles)

    def test_invalid_times_fail_explicitly(self):
        for start,end in [(1,1),(2,1),(0,float('nan')),(0,float('inf')),(-1,1)]:
            with self.subTest(start=start,end=end),self.assertRaises(ValueError):
                rank([chord(0,start=start,end=end)])
        with self.assertRaises(ValueError):
            rank([chord(0,start=0,end=2),chord(7,start=1,end=3)])

    def test_acoustic_contract(self):
        all_keys=[dict(key=name(pc,m),score=1.0) for pc in range(12) for m in ('major','minor')]
        self.assertEqual(len(rank([],all_keys)),24)
        for bad in [all_keys[:-1],all_keys+[dict(key='Db',score=1)],
                    [dict(c,score=float('nan')) for c in all_keys],
                    [dict(c,score=-1) for c in all_keys],
                    [dict(c,score=0) for c in all_keys]]:
            with self.assertRaises(ValueError):rank([],bad)

    def test_gap_is_not_a_resolution_and_input_is_not_mutated(self):
        seq=[chord(4,4,0,1,10),chord(4,4,1,2,10),chord(9,3,3,4)]
        before=[vars(c).copy() for c in seq]
        e=evidence(seq,0,'major')
        self.assertNotIn('resolved_secondary_dominant',[x['role'] for x in e['exceptions']])
        self.assertEqual(before,[vars(c) for c in seq])

if __name__=='__main__':unittest.main()
