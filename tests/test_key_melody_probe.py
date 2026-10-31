import unittest
from types import SimpleNamespace as Note
from app.probe_key_melody import features

class MelodyProbeTests(unittest.TestCase):
    def test_rest_proxy_ignores_track_end(self):
        notes=[Note(start=0,end=1,pitch=60),Note(start=2,end=3,pitch=67)]
        f=features(notes,[0],[120])
        self.assertEqual(len(f['pre_rest_events']),1)
        self.assertEqual(f['pre_rest_pc'][0],1)
        self.assertEqual(f['final_pc'],7)

    def test_tied_note_subdivision_preserves_evidence(self):
        full=[Note(start=0,end=1,pitch=60),Note(start=2,end=3,pitch=67)]
        split=[Note(start=0,end=.5,pitch=60),Note(start=.5,end=1,pitch=60),full[1]]
        self.assertEqual(features(full,[0],[120]),features(split,[0],[120]))

    def test_transposition(self):
        for shift in range(12):
            notes=[Note(start=0,end=1,pitch=60+shift),Note(start=2,end=3,pitch=67+shift)]
            self.assertEqual(features(notes,[0],[120])['pre_rest_pc'][shift],1)

if __name__=='__main__':unittest.main()
