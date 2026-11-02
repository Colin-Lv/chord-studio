import numpy as np
import pytest
from app.web.harmony_boundaries import canonical_cuts,recover_sustained_chords


def test_grid_and_decoder_edges_do_not_create_five_millisecond_chords():
    rows=[dict(start=50.82,end=56.825,label='D#:7'),dict(start=56.825,end=59.815,label='G#:maj'),dict(start=59.815,end=61.29,label='C#:maj')]
    cuts,snaps=canonical_cuts(rows,[53.8,56.83,59.81],62)
    assert 56.825 in cuts and 59.815 in cuts
    assert 56.83 not in cuts and 59.81 not in cuts
    assert len(snaps)==2


def test_canonicalization_preserves_real_short_chords():
    rows=[dict(start=0,end=1,label='C:maj'),dict(start=1,end=1.005,label='G:maj'),dict(start=1.005,end=2,label='C:maj')]
    cuts,_=canonical_cuts(rows,[1.002],2)
    assert 1 in cuts and 1.005 in cuts and 1.002 not in cuts


def example(step=.25):
    bounds=np.array([[i*step,(i+1)*step] for i in range(16)])
    post=np.full((16,120),.001)
    post[:,32]=.85 # D#7, coarse background
    post[6:10,32]=.002;post[6:10,10]=.85 # C# major, one second
    post/=post.sum(1,keepdims=True)
    rows=[dict(start=0,end=4,label='D#:7')]
    return rows,post,np.zeros(16),bounds,np.arange(0,4,.5)


def test_sustained_chord_with_independent_bass_is_recovered_without_inventing_seventh():
    args=example();rows,repairs=recover_sustained_chords(*args,lambda row:row['label'],lambda a,b:{'note':'C#'})
    assert [(r['start'],r['end'],r['label']) for r in rows]==[(0,1.5,'D#:7'),(1.5,2.5,'C#:maj'),(2.5,4,'D#:7')]
    assert len(repairs)==1 and repairs[0]['label']=='C#:maj'
    assert args[0]==[dict(start=0,end=4,label='D#:7')]


def test_uncertain_bass_or_short_flash_cannot_force_a_new_chord():
    args=example()
    rows,repairs=recover_sustained_chords(*args,lambda row:row['label'],lambda a,b:{'note':None})
    assert rows==args[0] and not repairs
    args[1][7:10]=args[1][0] # Only 250 ms remains, below a 500 ms beat.
    rows,repairs=recover_sustained_chords(*args,lambda row:row['label'],lambda a,b:{'note':'C#'})
    assert rows==args[0] and not repairs


def test_existing_corrected_root_is_not_split_into_needless_fragments():
    args=example()
    rows,repairs=recover_sustained_chords(*args,lambda row:'C#:maj',lambda a,b:{'note':'C#'})
    assert rows==args[0] and not repairs


def test_uncertain_chord_winner_does_not_overrule_coarse_path():
    args=example();args[1][6:10,32]=.3;args[1][6:10,10]=.4
    rows,repairs=recover_sustained_chords(*args,lambda row:row['label'],lambda a,b:{'note':'C#'})
    assert rows==args[0] and not repairs


def test_silence_never_becomes_a_recovered_chord():
    args=example();args[2][:]=1
    rows,repairs=recover_sustained_chords(*args,lambda row:row['label'],lambda a,b:{'note':'C#'})
    assert rows==args[0] and not repairs


def test_cell_subdivision_does_not_change_recovery():
    rows,post,n,bounds,beats=example()
    split=np.array([[a+(b-a)*k/2,a+(b-a)*(k+1)/2] for a,b in bounds for k in range(2)])
    call=lambda p,np_,bs:recover_sustained_chords(rows,p,np_,bs,beats,lambda r:r['label'],lambda a,b:{'note':'C#'})
    first,second=call(post,n,bounds),call(post.repeat(2,0),n.repeat(2),split)
    assert first[0]==second[0]
    assert first[1][0]['evidence_ratio']==pytest.approx(second[1][0]['evidence_ratio'])


def test_edge_candidate_does_not_create_a_short_tail_or_move_existing_boundary():
    rows,post,n,bounds,beats=example();post[10:]=post[6];bounds[-1,1]=3.98
    result,repairs=recover_sustained_chords(rows,post,n,bounds,beats,lambda r:r['label'],lambda a,b:{'note':'C#'})
    assert result==rows and not repairs


def test_quality_uncertainty_is_not_evidence_for_a_new_root():
    args=example()
    args[1][6:10,10]=.35
    args[1][6:10,30]=.25
    args[1][6:10,31]=.25
    rows,repairs=recover_sustained_chords(*args,lambda row:row['label'],lambda a,b:{'note':'C#'})
    assert rows==args[0] and not repairs


@pytest.mark.parametrize('root',range(12))
@pytest.mark.parametrize('beat',[.25,.5,.75,1.2])
def test_sustained_chord_recovery_across_keys_and_tempos(root,beat):
    from app.web.full_decode import NAMES
    background=(root+2)%12
    bounds=np.array([(i*beat/2,(i+1)*beat/2) for i in range(16)])
    post=np.full((16,120),.0001);post[:,background*10+2]=.9
    for i in range(6,10):
        post[i]=.0001
        post[i,root*10]=.8
        post[i,root*10+3]=.15
    post/=post.sum(1,keepdims=True)
    original=[dict(start=0,end=8*beat,label=NAMES[background]+':7')]
    rows,repairs=recover_sustained_chords(original,post,np.zeros(16),bounds,np.arange(9)*beat,
                                         lambda r:r['label'],lambda a,b:{'note':NAMES[root]})
    assert len(repairs)==1
    assert rows[1]['label']==NAMES[root]+':maj'
    assert rows[1]['start']==pytest.approx(3*beat) and rows[1]['end']==pytest.approx(5*beat)


def test_missing_beat_grid_does_not_guess_a_duration_prior():
    args=list(example());args[-1]=[]
    rows,repairs=recover_sustained_chords(*args,lambda r:r['label'],lambda a,b:{'note':'C#'})
    assert rows==args[0] and not repairs


def test_changing_bass_alone_does_not_create_chord_changes():
    args=example();args[1][:]=args[1][0]
    rows,repairs=recover_sustained_chords(*args,lambda r:r['label'],lambda a,b:{'note':'C#'})
    assert rows==args[0] and not repairs
