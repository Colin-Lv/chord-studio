from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from evaluate_chords import TimedChord, evaluate
from nnls_seventh_model import split_upstream, supervision, decode_windows


def test_subwindows_recover_mixed_quality_supervision_without_using_reference_edges():
    upstream = [TimedChord(.1, 2.1, 'C')]
    reference = [TimedChord(.1, 1.1, 'C'), TimedChord(1.1, 2.1, 'C7')]
    assert supervision(upstream, reference).tolist() == [-1]
    windows, _ = split_upstream(upstream, .5)
    assert [(r.start, r.end) for r in windows] == [(.1, .6), (.6, 1.1), (1.1, 1.6), (1.6, 2.1)]
    assert supervision(windows, reference).tolist() == [0, 0, 1, 1]
    # Root errors are not relabeled as no-seventh negatives.
    assert supervision(windows, [TimedChord(.1, 2.1, 'G7')]).tolist() == [-1]*4


def test_quality_can_change_within_source_but_keeps_root_bass_coverage_and_gaps():
    source = [TimedChord(0, 1.2, 'Db/F'), TimedChord(2, 3, 'N'), TimedChord(3, 4, 'Cadd9')]
    windows, parents = split_upstream(source, .5)
    probability = np.tile([.01, .98, .01], (len(windows), 1))
    probability[0] = [.99, .005, .005]
    result, changes = decode_windows(source, windows, parents, probability, .8)
    assert result == [TimedChord(0, .5, 'Db/F'), TimedChord(.5, 1.2, 'Db7/F'), *source[1:]]
    score = evaluate(source, result)
    assert score['root_accuracy'] == score['triad_accuracy'] == score['prediction_coverage'] == 1
    assert sum(r.end-r.start for r in source) == sum(r.end-r.start for r in result)
    assert len(changes) == 2
    assert decode_windows(source, windows, parents, probability, 1)[0] == source
