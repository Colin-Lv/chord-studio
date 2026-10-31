"""Export old/new compatibility and hand-verifiable bias probes before fitting."""
import argparse
import json
from pathlib import Path

import detect_key_timeline as old
import harmonic_key_v2 as new
from evaluate_global_key_fusion import signature


def audit(output):
    table=new.compatibility_table()
    suffixes={'major':'','minor':'m','dim':'dim','aug':'aug','sus2':'sus2','sus4':'sus4','unknown':'unknown'}
    for row in table['entries']:
        chord=row['chord_root']+suffixes[row['chord_quality']]
        try:row['legacy_chord_fit']=old.chord_fit(chord,row['key'])
        except (ValueError,TypeError):row['legacy_chord_fit']=None
    probes=[('Am','Am'),('Am','C'),('Em','Am'),('Em','C'),('E','Am'),('Bdim','Am'),('Bm7b5','Am')]
    table['probes']=[dict(chord=chord,key=key,old=old.chord_fit(chord,key),new=new.compatibility((new.parse_chord_name(chord)[0]-new.parse_chord_name(key)[0])%12,new.quality_of(chord),'minor' if key.endswith('m') else 'major')) for chord,key in probes]
    table['legacy_root_causes']=[
        dict(function='chord_fit',issue='Natural-minor V declares only major quality, so natural v is mismatched'),
        dict(function='chord_fit',issue='Major I/IV/V receive 1.25 and major vi 1.05; corresponding minor fits receive 0.95'),
        dict(function='score_key',issue='Unconditional minor score -= 0.12'),
        dict(function='score_key',issue='Root-duration reward/penalty and entry handling only run for major'),
        dict(function='score_key',issue='Major pair/loop tables reused for minor; loop rotations are incompletely listed'),
        dict(function='chord_family',issue='m7b5 starts with m and is classified minor before diminished quality is considered')]
    table['code']=[signature(Path(__file__).parent/name) for name in ['detect_key_timeline.py','harmonic_key_v2.py','audit_harmonic_compatibility.py']]
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(table,ensure_ascii=False,indent=2),encoding='utf-8')
    print(dict(keys=len(table['theoretical_triads']),chord_quality_entries=len(table['entries']),probes=table['probes']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.output)
