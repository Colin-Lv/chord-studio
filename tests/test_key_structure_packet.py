import unittest
from types import SimpleNamespace as NS

from app.build_key_structure_packet import parse_structure, phrase_contexts, align_melody


class StructurePacketTests(unittest.TestCase):
    def test_phrase_units_and_nonmelodic_labels(self):
        rows=parse_structure('i1A2o1')
        self.assertEqual([(r['start'],r['end'],r['melodic']) for r in rows],
                         [(0,16,False),(16,48,True),(48,64,False)])
        self.assertEqual(parse_structure('A2',12)[0]['end'],24)
        for invalid in ['A0','A4???','A','']:
            with self.assertRaises(ValueError):parse_structure(invalid)

    def test_sustained_note_is_not_a_terminal_note(self):
        segments=parse_structure('A1B1')
        notes=[dict(start=0,end=20,pitch=60),dict(start=20,end=32,pitch=67)]
        result=phrase_contexts(segments,notes,32,[])
        self.assertEqual(result[0]['status'],'crosses_boundary')
        self.assertEqual(result[1]['status'],'observable')

    def test_no_encoded_tail_does_not_become_observed_silence(self):
        result=phrase_contexts(parse_structure('A2'),[dict(start=0,end=12,pitch=60)],12,[])
        self.assertEqual(result[0]['status'],'tail_not_encoded')

    def test_alignment_rejects_internal_shift(self):
        notes=[dict(start=i*4,end=i*4+2,pitch=40+i) for i in range(24)]
        for drift,expected in [(0,True),(8,False)]:
            raw=[NS(start=(i*4+16+(drift if i>=12 else 0))/4,pitch=40+i) for i in range(24)]
            music=NS(instruments=[NS(name='MELODY',notes=raw)],resolution=100,time_to_tick=lambda t:round(t*100))
            self.assertEqual(align_melody(music,notes)['accepted'],expected)

    def test_same_harmony_different_melodic_endpoint_and_transposition(self):
        harmony=[dict(start=0,end=16,label='C:maj',pcs=[0,4,7],root=0)]
        for shift in range(12):
            pcs=[]
            for ending in [60,69]:
                seq=[dict(start=0,end=8,pitch=67+shift),dict(start=8,end=16,pitch=ending+shift)]
                context=phrase_contexts(parse_structure('A1'),seq,16,harmony)[0]
                pcs.append(context['pc'])
            self.assertEqual(pcs,[shift,(shift+9)%12])


if __name__=='__main__':unittest.main()
