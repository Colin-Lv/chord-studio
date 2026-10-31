import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

from app.key_transition_semantics import dominant_arrivals, major_minor_flag, tonic_triad

PARSER=Path('D:/ChordStudio/research/key-audit-20261031-r1/before/key_from_chords.py')
spec=importlib.util.spec_from_file_location('semantic_fixture_parser',PARSER)
parser=importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)


def parse(rows):
    return parser.parse_rows(rows)[0]


class KeyTransitionTests(unittest.TestCase):
    def test_aliases_and_nontriads(self):
        for label in ['Am','A:min','Amin','Am7']:
            self.assertIs(major_minor_flag(parser.parse_chord_label(label)),True)
        for label in ['Adim','Asus4','Aaug','A5']:
            self.assertIsNone(major_minor_flag(parser.parse_chord_label(label)))
            self.assertFalse(tonic_triad(parser.parse_chord_label(label),9,'major'))

    def test_wrong_quality_and_gap(self):
        for rows in [[(0,1,'G'),(1,2,'N'),(2,3,'C')],[(0,1,'Gsus4'),(1,2,'C')],
                     [(0,1,'Gmaj7'),(1,2,'C')],[(0,1,'Gaug'),(1,2,'C')],
                     [(0,1,'G7'),(1,2,'Cm')]]:
            self.assertEqual(dominant_arrivals(parse(rows),0,'major'),[])
        self.assertEqual(len(dominant_arrivals(parse([(0,1,'G7'),(1,2,'Cm')]),0,'minor')),1)

    def test_transposition_and_inversion(self):
        for mode,tonic in [('major','C'),('minor','Cm')]:
            for shift in range(12):
                seq=parse([(0,1,'G7/B'),(1,2,tonic+'/G')])
                for c in seq:
                    c.root=(c.root+shift)%12
                    if c.bass is not None:c.bass=(c.bass+shift)%12
                self.assertEqual(len(dominant_arrivals(seq,shift,mode)),1)

    def test_held_chord_subdivision(self):
        full=parse([(0,2,'G7'),(2,4,'C')])
        split=parse([(0,1,'G7'),(1,2,'G7'),(2,3,'C'),(3,4,'C')])
        self.assertEqual(dominant_arrivals(full,0,'major'),dominant_arrivals(split,0,'major'))

    def test_repaired_integration(self):
        folder=Path('D:/ChordStudio/research/key-stage2-20261101/repaired')
        script='''
import json,sys
sys.path.insert(0,sys.argv[1])
import key_from_chords as p, key_decision as d, key_layers as l
parse=lambda rows:p.parse_rows(rows)[0]
a=d.guard_456(parse([(0,1,'F'),(1,2,'G'),(2,3,'Am')]),'Am')
b=d.guard_456(parse([(0,1,'F'),(1,2,'G'),(2,3,'A:min')]),'Am')
assert a==b
seq=parse([(0,1,'G'),(1,2,'N'),(2,3,'C')])
assert l.score(seq,0,'major')['cadences']==0
assert l.score(parse([(0,1,'G'),(1,2,'Cm')]),0,'major')['cadences']==0
assert l.score(parse([(0,1,'Csus4')]),0,'major')['tonic_share']==0
assert isinstance(l.adjust('C','G',seq)['use_chord'],bool)
print(json.dumps({'aliases_equal':True,'gaps_respected':True,'optional_adjust_works':True}))
'''
        result=subprocess.run([sys.executable,'-c',script,str(folder)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(json.loads(result.stdout)['aliases_equal'])


if __name__=='__main__':unittest.main()
