"""Acoustic timing and independent core-chord proposals at accompaniment attacks.

The alignment path changes only a recovered onset. The attack path identifies
newly sounded core tones without future predictions, and can re-segment a
known swallowed-chord region when its first attack agrees with the anchor.
Neither path snaps every chord to the grid or changes the frozen model.
"""
import numpy as np

POLICY='attack-aware-recovery-v2'


def chord_core(label):
    root,quality=label.split('/')[0].split(':')
    family='min' if quality.startswith('min') else 'maj' if quality in ('maj','7','maj7','9','maj9') else quality
    return root+':'+family


def reconcile_attack(attack,posterior,bass):
    """A pitch-class template alone cannot publish a new chord root.

    A consistently ranked bass can corroborate an independent acoustic chord
    prediction even below the bass acceptance threshold. It cannot alone
    introduce a root or a slash label. Genuine inversions remain legal when
    template and acoustic chord evidence agree on a different root.
    """
    from .full_decode import NAMES,QUALITIES
    p=np.asarray(posterior);k=int(p.argmax())
    frozen=f'{NAMES[k//10]}:{QUALITIES[k%10]}'
    root=NAMES[int(p.reshape(12,10).sum(1).argmax())]
    votes=[b.get('candidate') or b.get('note') for b in bass.get('beats',[])]
    record=dict(template_label=attack['label'],frozen_label=frozen,frozen_root=root,bass_candidates=votes)
    if chord_core(frozen)==chord_core(attack['label']):
        return attack['label'],dict(record,resolution='template_and_frozen_agree')
    # All analysed beats must nominate the same note. A single loud/passing
    # note or an accepted inversion bass is not a root override by itself.
    if len(votes)>=2 and all(v==root for v in votes) and frozen.split(':')[0]==root:
        return frozen,dict(record,resolution='frozen_chord_and_consistent_bass')
    return None,dict(record,resolution='conflicting_chord_evidence')


def onset_evidence(audio,sr):
    """Short windows for timing; keep the trained bass estimator unchanged.

    Its 743 ms windows resolve low pitches but blur actual chord attacks.
    Here 186 ms windows and 23 ms hops are only used to compare textures.
    """
    from math import gcd
    from scipy.signal import resample_poly,stft
    y=np.asarray(audio,dtype=np.float32)
    if y.ndim==2:y=y.mean(axis=1)
    divisor=gcd(int(sr),11025);y=resample_poly(y,11025//divisor,int(sr)//divisor)
    y=np.pad(y,(0,max(0,2048-len(y))))
    frequencies,times,z=stft(y,fs=11025,nperseg=2048,noverlap=1792)
    power=abs(z)**2
    notes=[]
    for midi in range(28,109):
        hz=440*2**((midi-69)/12)
        notes.append(power[abs(frequencies-hz)<=max(11025/2048,hz*.025)].max(axis=0))
    return dict(times=times,notes=np.asarray(notes),energy=power.sum(axis=0),hop=256/11025)


def spectral_profile(evidence,start,end):
    times=np.asarray(evidence['times']);half_hop=evidence.get('hop',1024/11025)/2
    weights=np.maximum(0,np.minimum(times+half_hop,end)-np.maximum(times-half_hop,start))
    if not weights.sum():return None
    notes=np.average(evidence['notes'],axis=1,weights=weights)
    energy=float(np.average(evidence['energy'],weights=weights))
    if energy<1e-9 or notes.sum()<=0:return None
    return dict(notes=notes/notes.sum(),energy=energy)


def similarity(a,b):
    # Hellinger affinity: keeps pitch register, and does not let one loud bin
    # completely hide a sustained accompaniment underneath a melody.
    return float(np.sqrt(a['notes']*b['notes']).sum())


def attack_chord(evidence,time,beat):
    """Independent core-chord proposal from newly sounding accompaniment notes.

    Uses neither the coarse chord nor a future stable label. Seventh/extension
    quality is deliberately not inferred here. Scores are template agreement,
    not calibrated correctness probabilities.
    """
    width=min(.3,beat*.4)
    before=spectral_profile(evidence,time-width-.05,time-.05)
    after=spectral_profile(evidence,time,time+width)
    if after is None:return None
    ratio=after['energy']/max(before['energy'] if before else 0,1e-12)
    if ratio<3:return None
    # Compare absolute note-bin powers, not normalized pitch-class profiles.
    def raw(a,b):
        t=evidence['times'];h=evidence['hop']/2
        w=np.maximum(0,np.minimum(t+h,b)-np.maximum(t-h,a))
        if not w.sum():return np.zeros(evidence['notes'].shape[0])
        return np.average(evidence['notes'],axis=1,weights=w)
    change=np.maximum(0,raw(time,time+width)-raw(time-width-.05,time-.05))
    pitches=np.arange(28,109);pc=np.bincount(pitches%12,weights=change,minlength=12)
    if pc.sum()<=0:return None
    pc/=pc.sum();names=('C','C#','D','D#','E','F','F#','G','G#','A','A#','B')
    candidates=[]
    for root in range(12):
        for quality,intervals in [('maj',(0,4,7)),('min',(0,3,7)),('sus2',(0,2,7)),('sus4',(0,5,7))]:
            ix=(root+np.asarray(intervals))%12
            candidates.append(dict(label=names[root]+':'+quality,root=root,pcs=ix,
                score=float(np.sqrt(pc[ix]).sum()/np.sqrt(3)),coverage=float(pc[ix].sum()),weakest=float(pc[ix].min())))
    candidates.sort(key=lambda x:x['score'],reverse=True);best=candidates[0]
    margin=best['score']-candidates[1]['score']
    # A single pitched note's harmonic series is not a complete played chord.
    # Require the third/suspension within two octaves of the strongest root note.
    root_bins=np.flatnonzero(pitches%12==best['root'])
    root_pitch=int(pitches[root_bins[np.argmax(change[root_bins])]])
    third=int(best['pcs'][1]);mask=(pitches%12==third)&(pitches<root_pitch+24)
    independent_third=float(change[mask].sum()/max(change.sum(),1e-12))
    accepted=best['coverage']>=.8 and best['weakest']>=.03 and margin>=.05 and independent_third>=.03
    return dict(time=float(time),label=best['label'],accepted=bool(accepted),energy_ratio=ratio,
                score=best['score'],coverage=best['coverage'],weakest_tone=best['weakest'],margin=margin,
                independent_third=independent_third,pitch_classes=pc.tolist())


def resolve_recovered_attacks(rows,original,repairs,beats,evidence,no_chord_at,verify_span):
    """Re-segment a swallowed-chord region from independently recognised attacks.

    Limited rollout: requires an existing recovered chord to agree with the
    first attack. It does not rewrite arbitrary bars from templates alone.
    """
    out=[dict(r) for r in rows];decisions=[];beats=np.asarray(beats,dtype=float)
    for repair in repairs:
        parent=next((r for r in original if r['start']<=repair['start'] and r['end']>=repair['end']),None)
        if parent is None:continue
        siblings=[r for r in repairs if parent['start']<=r['start'] and r['end']<=parent['end']]
        if len(siblings)!=1:continue
        candidates=[]
        for t in beats[(beats>=repair['start']-repair['minimum_seconds'])&(beats<parent['end'])]:
            proposal=attack_chord(evidence,float(t),repair['minimum_seconds'])
            if proposal is not None:
                decisions.append(proposal)
                if proposal['accepted'] and no_chord_at(t,min(t+repair['minimum_seconds'],parent['end']))<.5:candidates.append(proposal)
        if not candidates:continue
        first=candidates[0]
        if first['label']!=chord_core(repair['label']) or not parent['start']<first['time']<=repair['start']:continue
        # Confirm texture continuity through the stable anchor; an attack on a
        # preceding, unrelated chord must not pull the anchor backwards.
        anchor=spectral_profile(evidence,repair['start'],repair['end'])
        onset=spectral_profile(evidence,first['time'],min(first['time']+.3,repair['end']))
        if anchor is None or onset is None or similarity(anchor,onset)<.9:continue
        if any(p['time']<repair['end'] and p['label']!=first['label'] for p in candidates[1:]):continue
        after=[p for p in candidates[1:] if p['time']>=repair['end']]
        if not after:continue # No independently supported ending: use onset-only path.
        changes=[dict(start=first['time'],end=after[0]['time'],label=repair['label'])]
        for i,p in enumerate(after):
            end=after[i+1]['time'] if i+1<len(after) else parent['end']
            label,verification=verify_span(p,p['time'],end)
            p['verification']=verification;p['label_used']=label
            # Retain the previous model hypothesis if the new proposal cannot
            # be corroborated. Do not turn a rejected template into a fact.
            changes.append(dict(start=p['time'],end=end,label=label or parent['label']))
        kept=[]
        for r in out:
            if r['end']<=first['time'] or r['start']>=parent['end']:kept.append(r);continue
            if r['start']<first['time']:kept.append(dict(r,end=first['time']))
            if r['end']>parent['end']:kept.append(dict(r,start=parent['end']))
        out=sorted(kept+changes,key=lambda r:r['start'])
        first['applied']=True
        for p in after:
            p['boundary_applied']=True;p['applied']=p['label_used'] is not None
    return out,decisions


def align_recovered_onsets(rows,repairs,beats,profile_at,no_chord_at):
    out=[dict(r) for r in rows];diagnostics=[]
    beats=np.asarray(beats,dtype=float)
    for repair in repairs:
        start,end=repair['start'],repair['end']
        index=next((i for i,r in enumerate(out) if r['start']==start and r['end']==end and r['label']==repair['label']),None)
        if index is None or index==0:continue
        previous=out[index-1]
        if previous['end']!=start or previous['label']!=repair['previous']:continue
        j=int(np.searchsorted(beats,start,side='left'))-1
        if j<1:continue
        candidate=float(beats[j]);context=float(beats[j-1])
        if context<previous['start'] or candidate<=previous['start'] or start-candidate>repair['minimum_seconds']+1e-6:continue
        # Existing near-identical clocks are handled separately. Very short
        # prefixes cannot establish a different spectral context reliably.
        if start-candidate<.15:continue
        record=dict(seed_start=start,seed_end=end,candidate_start=candidate,label=repair['label'],accepted=False)
        diagnostics.append(record)
        if no_chord_at(candidate,start)>=.5:
            record['reason']='no_chord_prefix';continue
        anchor=profile_at(start,end);old=profile_at(context,candidate)
        middle=(candidate+start)/2
        prefixes=[profile_at(candidate,middle),profile_at(middle,start)]
        if anchor is None or old is None or any(p is None for p in prefixes):
            record['reason']='insufficient_audio';continue
        affinities=[similarity(p,anchor) for p in prefixes]
        margins=[affinities[i]-similarity(p,old) for i,p in enumerate(prefixes)]
        record.update(anchor_affinities=affinities,context_margins=margins)
        if any(p['energy']<.1*anchor['energy'] for p in prefixes):
            record['reason']='quiet_prefix';continue
        if min(affinities)<.90 or min(margins)<.10:
            record['reason']='different_or_ambiguous_acoustics';continue
        previous['end']=candidate;out[index]['start']=candidate
        record.update(accepted=True,reason='stable_acoustic_continuation')
    return out,diagnostics
