"""Render MIDI counterexamples and train a bass ranker; no real-song fitting."""
import json,sys,importlib.util
from pathlib import Path
import joblib,numpy as np,pretty_midi,soundfile as sf
from sklearn.ensemble import ExtraTreesClassifier
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from app.web.bass_voice import spectrum,feature_rows,estimate,NOTES
from app.web.storage import atomic_json,signature
spec=importlib.util.spec_from_file_location('render',ROOT/'research/boundary-dataset/render.py');renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
OUT=HERE/'training-register-v2';OUT.mkdir(exist_ok=True)

def generate():
    if (OUT/'features.npz').exists():return
    sampler=renderer.make_sampler_class(renderer.load_sf2_module())(renderer.find_soundfont())
    x=[];y=[];groups=[];splits=[];labels=[];manifest=[]
    for split,count,programs in [('train',98,[0,4,24,25,48,80]),('validation',21,[2,26,50]),('test',21,[1,27,49])]:
        for index in range(count):
            rng=np.random.default_rng(20261102+index+{'train':0,'validation':10000,'test':20000}[split])
            group=f'{split}-{index:03d}';pm=pretty_midi.PrettyMIDI(initial_tempo=100)
            prog=int(rng.choice(programs));comp=pretty_midi.Instrument(prog,name='ACCOMPANIMENT')
            bass=pretty_midi.Instrument(int(rng.choice([prog,32,33,34])),name='BASS_VOICE')
            melody=pretty_midi.Instrument(int(rng.choice([0,73,80])),name='MELODY')
            pm.instruments=[comp,bass,melody];events=[];seconds=float(rng.choice([1.5,2,3]))
            key=int(rng.integers(12));kind=index%7
            for bar in range(8):
                a=.4+bar*seconds;b=a+seconds
                root=(key+([0,2,4,5][bar%4] if kind==0 else 0))%12
                intervals=[0,4,7] if kind!=3 else [0,2,7]
                # Fixed upper structure over moving bass, ordinary root changes,
                # inversions, pedals, weak/missing fundamentals, and high voicings.
                bass_pc=(key+[0,2,4,5][bar%4])%12 if kind in (1,4) else (root+intervals[bar%3])%12 if kind==2 else root
                bass_note=36+bass_pc+int(rng.choice([-12,0,12]))
                while bass_note<28:bass_note+=12
                while bass_note>59:bass_note-=12
                if kind==5:bass_note=60+bass_pc  # high-register lowest voice
                upper=[60+(root+iv)%12 for iv in intervals]
                upper=[n+12*((max(0,bass_note+3-n)+11)//12) for n in upper]
                for step in range(4):
                    for note in upper if kind!=4 else [upper[step%3]]:
                        comp.notes.append(pretty_midi.Note(int(rng.integers(45,95)),int(note),a+step*seconds/4,a+(step+.9)*seconds/4))
                bass.notes.append(pretty_midi.Note(int(rng.integers(45,100)),int(bass_note),a,b-.04))
                for step in range(4):
                    pitch=84+(key+int(rng.choice([0,2,4,5,7,9,11])))%12
                    melody.notes.append(pretty_midi.Note(int(rng.integers(35,100)),pitch,a+step*seconds/4,a+(step+.8)*seconds/4))
                events.append(dict(start=a,end=b,bass=int(bass_note%12) if kind!=6 else -1,bass_midi=int(bass_note) if kind!=6 else -1,root=int(root),kind=kind))
            if kind==6:
                for inst in pm.instruments:inst.notes=[]
            folder=OUT/'midi';folder.mkdir(exist_ok=True);pm.write(str(folder/(group+'.mid')))
            audio=renderer.render_pm(pm,sampler,sr=22050)
            audio=np.pad(audio,(0,max(0,int((.4+8*seconds+.8)*22050)-len(audio))))
            # Audible noise/tails plus spectral tilt; these remain synthetic examples.
            if index%3==0:
                from scipy.signal import butter,sosfilt
                audio=sosfilt(butter(2,120,fs=22050,btype='highpass',output='sos'),audio)
            audio=audio+rng.normal(0,float(rng.uniform(0,.002)),len(audio))
            waves=OUT/'audio';waves.mkdir(exist_ok=True);sf.write(waves/(group+'.wav'),audio,22050,subtype='FLOAT')
            e=spectrum(audio,22050)
            for event in events:
                f,_=feature_rows(e,event['start']+.15,event['end']-.1)
                x.append(f);y.append((NOTES==event['bass_midi']).astype(int));labels.append(event['bass'])
                groups.append(group);splits.append(split)
            manifest.append(dict(group=group,split=split,program=prog,events=events,midi_sha256=signature(folder/(group+'.mid'))['sha256'],audio_sha256=signature(waves/(group+'.wav'))['sha256']))
            if index%10==0:print('render',split,index,flush=True)
    atomic_json(OUT/'manifest.json',manifest)
    np.savez_compressed(OUT/'features.npz',x=x,y=y,labels=labels,groups=groups,splits=splits)

def train():
    z=np.load(OUT/'features.npz');x,y=z['x'],z['y'];split=z['splits'];true=z['labels'];reports=[];best=None
    for leaf in [2,8]:
        m=ExtraTreesClassifier(n_estimators=120,min_samples_leaf=leaf,class_weight='balanced',n_jobs=4,random_state=20261102)
        m.fit(x[split=='train'].reshape(-1,x.shape[-1]),y[split=='train'].ravel());m.n_jobs=1
        raw=m.predict_proba(x[split=='validation'].reshape(-1,x.shape[-1]))[:,1].reshape(-1,len(NOTES))
        pc=np.array([raw[:,NOTES%12==i].max(1) for i in range(12)]).T
        valid=true[split=='validation']>=0
        acc=float((pc.argmax(1)[valid]==true[split=='validation'][valid]).mean());reports.append(dict(leaf=leaf,validation_pitched_accuracy=acc))
        if best is None or acc>best[0]:best=(acc,m,leaf)
    m=best[1]
    # Select the abstention operating point on validation only, never the song.
    sel=split=='validation'
    raw=m.predict_proba(x[sel].reshape(-1,x.shape[-1]))[:,1].reshape(-1,len(NOTES))
    pc=np.array([raw[:,NOTES%12==i].max(1) for i in range(12)]).T;order=np.sort(pc,axis=1)
    choices=[]
    for threshold in [.3,.4,.5,.6,.7,.8,.9]:
        for margin in [.05,.1,.2]:
            accepted=(order[:,-1]>=threshold)&((order[:,-1]-order[:,-2])>=margin)
            precision=float((pc.argmax(1)[accepted]==true[sel][accepted]).mean()) if accepted.any() else 0
            if precision>=.99:choices.append((int(accepted.sum()),threshold,margin,precision))
    if not choices:raise ValueError('No validation operating point meets precision requirement')
    chosen=max(choices,key=lambda v:(v[0],v[1],v[2]))
    _,threshold,margin,_=chosen;m.bass_min_score_=threshold;m.bass_min_margin_=margin
    joblib.dump(m,OUT/'bass_voice_v2.joblib')
    result=dict(selection=reports,leaf=best[2],metrics={},thresholds=dict(score=threshold,margin=margin),
                operating_point_validation_candidates=choices)
    for part in ['validation','test']:
        sel=split==part;raw=m.predict_proba(x[sel].reshape(-1,x.shape[-1]))[:,1].reshape(-1,len(NOTES))
        pc=np.array([raw[:,NOTES%12==i].max(1) for i in range(12)]).T;order=np.sort(pc,axis=1)
        accepted=(order[:,-1]>=threshold)&((order[:,-1]-order[:,-2])>=margin);pred=pc.argmax(1);t=true[sel]
        strongest=NOTES[x[sel,:,5].argmax(1)]%12
        result['metrics'][part]=dict(n=int(sel.sum()),pitched_accuracy=float((pred[t>=0]==t[t>=0]).mean()),strongest_bin_baseline=float((strongest[t>=0]==t[t>=0]).mean()),coverage=float(accepted.mean()),accepted_accuracy=float((pred[accepted]==t[accepted]).mean()) if accepted.any() else None,no_low_voice_false_positive=int(accepted[t<0].sum()))
    atomic_json(OUT/'metrics.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':generate();train()
