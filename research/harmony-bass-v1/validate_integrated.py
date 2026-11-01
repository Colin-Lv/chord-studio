"""Fresh rendered-MIDI challenge for the deployed bass branch. No weight fitting.

Chord labels are supplied ground truth here; this measures bass/slash assignment,
not the frozen acoustic chord classifier's accuracy. Song checks are separate.
"""
import sys,json,importlib.util
from pathlib import Path
import numpy as np,pretty_midi,soundfile as sf
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from app.web import bass_voice
from app.web.storage import atomic_json,signature
spec=importlib.util.spec_from_file_location('render',ROOT/'research/boundary-dataset/render.py')
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
OUT=HERE/'integrated-challenge';OUT.mkdir(exist_ok=True)
KINDS=['root_motion','fixed_upper','inversions','high_piano','passing_tone','weak_bass','arpeggio','silence']


def main(programs=(3,5,28,29),out=None,seed=2026110300,model_path=None):
    global OUT
    if out is not None:OUT=Path(out);OUT.mkdir(parents=True,exist_ok=True)
    sampler=renderer.make_sampler_class(renderer.load_sf2_module())(renderer.find_soundfont())
    model=bass_voice.load_model(model_path or bass_voice.MODEL);manifest=[];results=[]
    # Programs 3/5/28/29 were not accompaniment presets in the ranker dataset.
    for group,program in enumerate(programs):
        for kind in KINDS:
            rng=np.random.default_rng(seed+group*17+KINDS.index(kind))
            key=int(rng.integers(12));pm=pretty_midi.PrettyMIDI(initial_tempo=120)
            comp=pretty_midi.Instrument(program,name='UPPER_CHORD')
            lower=pretty_midi.Instrument(program if kind=='high_piano' else 35,name='LOWEST_VOICE')
            melody=pretty_midi.Instrument(73,name='LEAD')
            pm.instruments=[comp,lower,melody];events=[]
            for bar in range(4):
                start=.4+bar*2;end=start+2
                root=(key+([0,2,4,5][bar] if kind=='root_motion' else 0))%12
                pc=(root+[0,4,7,4][bar])%12 if kind=='inversions' else (key+[0,2,4,5][bar])%12 if kind in ['fixed_upper','high_piano','arpeggio'] else root
                note=(60 if group<2 else 72)+pc if kind=='high_piano' else 36+pc
                ceiling=note+4
                upper=[60+(root+x)%12 for x in [0,4,7]]
                upper=[n+12*max(0,int(np.ceil((ceiling-n)/12))) for n in upper]
                if kind!='silence':
                    lower.notes.append(pretty_midi.Note(40 if kind=='weak_bass' else 80,note,start,end-.025))
                    for beat in range(8):
                        for pitch in ([upper[beat%3]] if kind=='arpeggio' else upper):
                            comp.notes.append(pretty_midi.Note(int(rng.integers(60,100)),pitch,start+beat*.25,start+(beat+.85)*.25))
                        melody.notes.append(pretty_midi.Note(100, min(119,max(96,max(upper)+5))+int(beat%3),start+beat*.25,start+(beat+.7)*.25))
                    if kind=='passing_tone':
                        lower.notes.append(pretty_midi.Note(115,note-2,start+.6,start+.76))
                label=f'{bass_voice.NAMES[root]}:maj' if kind!='silence' else 'N'
                truth=None if kind=='silence' else bass_voice.NAMES[pc]
                events.append(dict(start=start,end=end,label=label,truth=truth,bass_midi=None if truth is None else note))
            ident=f'{group}-{kind}';midi=OUT/(ident+'.mid');wav=OUT/(ident+'.wav');pm.write(str(midi))
            audio=renderer.render_pm(pm,sampler,sr=22050)
            audio=np.pad(audio,(0,max(0,int(9*22050)-len(audio))))
            sf.write(wav,audio,22050,subtype='FLOAT');evidence=bass_voice.spectrum(audio,22050)
            for e in events:
                found=bass_voice.summarize_beats(model,evidence,e['start'],e['end'],np.arange(e['start'],e['end']+.01,.5))
                predicted=bass_voice.attach_bass(e['label'],found['note'])
                expected=bass_voice.attach_bass(e['label'],e['truth'])
                results.append(dict(group=ident,kind=kind,program=program,**e,predicted_bass=found['note'],
                    predicted_label=predicted,expected_label=expected,score=found['score'],margin=found['margin']))
            manifest.append(dict(id=ident,program=program,events=events,midi_sha256=signature(midi)['sha256'],audio_sha256=signature(wav)['sha256']))
            print('rendered',ident,flush=True)
    def metrics(rows):
        accepted=[r for r in rows if r['predicted_bass'] is not None]
        return dict(n=len(rows),correct_bass=sum(r['predicted_bass']==r['truth'] for r in rows),
            correct_label=sum(r['predicted_label']==r['expected_label'] for r in rows),
            accepted=len(accepted),accepted_correct=sum(r['predicted_bass']==r['truth'] for r in accepted),
            rejected=sum(r['truth'] is not None and r['predicted_bass'] is None for r in rows))
    report=dict(model=signature(model_path or bass_voice.MODEL),fitting=False,scope='bass on rendered accompaniment; chord identity supplied',
                overall=metrics(results),by_kind={kind:metrics([r for r in results if r['kind']==kind]) for kind in KINDS},rows=results)
    atomic_json(OUT/'manifest.json',manifest);atomic_json(OUT/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
