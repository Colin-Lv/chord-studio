"""Opt-in decoder experiments on cached uncalibrated HGB utility scores.

No references or global key are accepted. The inherited per-interval raw-root
policy is retained. Context evidence only adds finite soft utility adjustments.
"""
import numpy as np
import mir_eval
from evaluate_chords import TimedChord,harte_chord
from analyze_multisource_candidate_potential import chord_identity
from multisource_root_guard_v2 import _validated_inputs
from harmonic_note_features import MIDI_NOTES
from key_profile_baseline import PROFILES


def allowed_indices(event,group):
    baseline=harte_chord(event.chord)
    result=[0] if group['candidates'] else []
    if baseline not in {'N','X'}:
        for i,c in enumerate(group['candidates'][1:],1):
            label=harte_chord(c['label'])
            if label not in {'N','X'} and mir_eval.chord.root([baseline],[label])[0]==1:
                result.append(i)
    return result


def decode(raw,groups,scores,*,transition_costs=None,margin=.02):
    values=_validated_inputs(raw,groups,scores,margin)
    n=len(raw)
    costs=np.zeros(n) if transition_costs is None else np.asarray(transition_costs,dtype=float)
    if costs.shape!=(n,) or not np.isfinite(costs).all() or np.any(costs<0):
        raise ValueError('Finite nonnegative transition cost required per interval')
    output=[];all_choices=[];start=0
    while start<n:
        stop=start+1
        while stop<n and abs(raw[stop].start-raw[stop-1].end)<=1e-9:stop+=1
        states=[];back=[];utilities=None;previous_ids=None
        for i in range(start,stop):
            allowed=allowed_indices(raw[i],groups[i])
            if not allowed:raise ValueError('Empty candidate group cannot be decoded')
            local=values[groups[i]['start']:groups[i]['end']]
            # The strict raw replacement margin is paid for every nonraw state.
            node=np.array([(raw[i].end-raw[i].start)*(local[j]-(margin+1e-12 if j else 0)) for j in allowed])
            identities=[chord_identity(groups[i]['candidates'][j]['label']) for j in allowed]
            if utilities is None:
                utilities=node;back.append(None)
            else:
                matrix=np.array([[u-costs[i]*int(a!=b) for a,u in zip(previous_ids,utilities)] for b in identities])
                parents=np.argmax(matrix,axis=1)
                utilities=node+matrix[np.arange(len(allowed)),parents]
                back.append(parents)
            states.append(allowed);previous_ids=identities
        chosen=int(np.argmax(utilities));choices=[]
        for k in range(len(states)-1,-1,-1):
            choices.append(states[k][chosen])
            if k:chosen=int(back[k][chosen])
        choices.reverse()
        for i,j in zip(range(start,stop),choices):
            output.append(TimedChord(raw[i].start,raw[i].end,groups[i]['candidates'][j]['label']))
        all_choices.extend(choices);start=stop
    return output,all_choices


def switch_costs(raw,beats=None):
    if not raw:return np.empty(0)
    scale=.02*float(np.median([r.end-r.start for r in raw]))
    costs=np.full(len(raw),scale);costs[0]=0
    if beats is not None:
        beats=np.asarray(beats,dtype=float)
        if beats.ndim!=2 or beats.shape[1]!=2 or len(beats)<2 or not np.isfinite(beats).all() or np.any(np.diff(beats[:,0])<=0):
            raise ValueError('Ordered independent beat times and positions required')
        width=.15*float(np.median(np.diff(beats[:,0])))
        down=beats[beats[:,1]==1,0]
        for i,r in enumerate(raw[1:],1):
            beat_alignment=.5*np.exp(-.5*(np.min(abs(beats[:,0]-r.start))/width)**2)
            bar_alignment=.75*np.exp(-.5*(np.min(abs(down-r.start))/width)**2) if len(down) else 0.
            costs[i]*=1-max(beat_alignment,bar_alignment)
    return costs


def context_chroma(evidence,raw):
    """Eight-second, four-bin contexts on NNLS frame centres, no extrapolation."""
    values=evidence.activations/np.maximum(evidence.activations.sum(axis=0,keepdims=True),1e-12)
    chroma=np.zeros((12,len(evidence.times)))
    np.add.at(chroma,MIDI_NOTES%12,values)
    result=np.zeros((len(raw),4,12));coverage=np.zeros((len(raw),4))
    for i,r in enumerate(raw):
        middle=(r.start+r.end)/2
        for b in range(4):
            left=max(0.,middle-4+2*b);right=min(evidence.metadata['duration_seconds'],middle-2+2*b)
            if right<=left:continue
            mask=(evidence.times>=left)&(evidence.times<right)
            if mask.any():result[i,b]=chroma[:,mask].mean(axis=1);coverage[i,b]=(right-left)/2
    return result,coverage


def local_key_adjustment(raw,groups,contexts,coverage):
    templates=[];scales=[]
    for tonic in range(12):
        for mode in ('major','minor'):
            vector=np.roll(np.array(PROFILES[mode]),tonic);vector-=vector.mean()
            templates.append(vector/np.linalg.norm(vector))
            scale=np.zeros(12);scale[(tonic+np.array([0,2,4,5,7,9,11] if mode=='major' else [0,2,3,5,7,8,10]))%12]=1
            scales.append(scale)
    templates=np.array(templates);scales=np.array(scales)
    delta=np.zeros(groups[-1]['end'] if groups else 0);diagnostics=[]
    for i,(event,group) in enumerate(zip(raw,groups)):
        vector=np.average(contexts[i],axis=0,weights=np.maximum(coverage[i],1e-12));centered=vector-vector.mean();norm=np.linalg.norm(centered)
        if norm<1e-12:diagnostics.append(dict(confidence=0.,profile_weights=None));continue
        correlations=templates@(centered/norm)
        order=np.argsort(-correlations,kind='stable');spread=float(correlations.max()-correlations.min())
        confidence=float((correlations[order[0]]-correlations[order[1]])/spread) if spread>0 else 0.
        weights=np.exp(4*(correlations-correlations.max()));weights/=weights.sum()
        compatibility=weights@scales
        def value(label):
            label=harte_chord(label)
            if label in {'N','X'}:return 0.
            root,mask,_=mir_eval.chord.encode(label);pcs=np.flatnonzero(mask)
            return float(np.mean(compatibility[(root+pcs)%12])) if len(pcs) else 0.
        base=value(event.chord)
        for j,c in enumerate(group['candidates'],group['start']):delta[j]=.02*confidence*(value(c['label'])-base)
        diagnostics.append(dict(confidence=confidence,profile_weights=weights.tolist()))
    return delta,diagnostics


def repetition_adjustment(raw,groups,contexts,coverage,seed_labels):
    if len(seed_labels)!=len(raw):raise ValueError('One fixed seed label required per interval')
    vectors=contexts.reshape(len(raw),48)
    norm=np.linalg.norm(vectors,axis=1);vectors=vectors/np.maximum(norm[:,None],1e-12)
    centers=np.array([(r.start+r.end)/2 for r in raw]);similarity=vectors@vectors.T
    complete=np.all(coverage>=1-1e-9,axis=1)
    delta=np.zeros(groups[-1]['end'] if groups else 0);neighbors=[]
    identities=[chord_identity(label) for label in seed_labels]
    for i,group in enumerate(groups):
        eligible=[j for j in range(len(raw)) if complete[i] and complete[j] and norm[i]>1e-12 and norm[j]>1e-12 and abs(centers[j]-centers[i])>=16 and similarity[i,j]>=.98]
        eligible.sort(key=lambda j:(-similarity[i,j],j));eligible=eligible[:5]
        neighbors.append([dict(index=j,similarity=float(similarity[i,j])) for j in eligible])
        if not eligible:continue
        weights=np.array([similarity[i,j] for j in eligible]);weights/=weights.sum()
        def consensus(label):return float(sum(w for w,j in zip(weights,eligible) if chord_identity(label)==identities[j]))
        base=consensus(raw[i].chord)
        for j,c in enumerate(group['candidates'],group['start']):delta[j]=.02*(consensus(c['label'])-base)
    return delta,neighbors


def soft_snap_to_beats(raw,beats):
    """Half-step toward a nearby independent beat; retain labels, edges and gaps.

    Only contiguous internal boundaries within a quarter of median beat spacing
    are eligible. Simultaneous proposals causing collapsed intervals are undone.
    This is a privileged annotated-timing diagnostic when used with beat files.
    """
    beats=np.asarray(beats,dtype=float)
    if beats.ndim!=2 or beats.shape[1]!=2 or len(beats)<2 or not np.isfinite(beats).all() or np.any(np.diff(beats[:,0])<=0):
        raise ValueError('Ordered independent beat times required')
    if not raw:return [],dict(moved_boundaries=0,rejected_collapses=0)
    original=np.array([[r.start,r.end] for r in raw]);proposed=original.copy()
    if not np.isfinite(original).all() or np.any(original[:,0]<0) or np.any(original[:,1]<=original[:,0]) or np.any(original[:-1,1]>original[1:,0]):
        raise ValueError('Ordered disjoint raw intervals required')
    radius=.25*float(np.median(np.diff(beats[:,0])));eligible=[]
    for i in range(1,len(raw)):
        if abs(raw[i].start-raw[i-1].end)>1e-9:continue
        edge=raw[i].start;nearest=beats[np.argmin(abs(beats[:,0]-edge)),0]
        if abs(nearest-edge)<=radius:
            value=(edge+nearest)/2
            proposed[i-1,1]=value;proposed[i,0]=value;eligible.append(i)
    rejected=set()
    while np.any(proposed[:,1]<=proposed[:,0]):
        for j in np.flatnonzero(proposed[:,1]<=proposed[:,0]):
            for i in (int(j),int(j)+1):
                if i in eligible:
                    proposed[i-1,1]=original[i-1,1];proposed[i,0]=original[i,0];rejected.add(i)
    result=[TimedChord(float(a),float(b),r.chord) for r,(a,b) in zip(raw,proposed)]
    return result,dict(moved_boundaries=int(sum(proposed[i,0]!=original[i,0] for i in eligible)),
                       rejected_collapses=len(rejected),radius_seconds=radius,
                       maximum_shift_seconds=float(np.max(abs(proposed-original))))
