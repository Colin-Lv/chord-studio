"""Experimental continuous key scoring exposed to the workbench as a second tier.

It never changes the published key and never overrides the acoustic model: the UI shows the
24-candidate ranking, the evidence terms, the parse statistics and the known regressions, and
the user has to click to adopt a key.

Method: `key_coherence.rank` (continuous score = log acoustic prior + compatibility cost +
tonic occupancy + saturating arrival/return counts) over chords parsed by the strict semantic
adapter (`key_chord_adapter`, i.e. chord_semantics_v3 / Harte-compatible degrees).

Status: experimental. On the exposed 147-case set this scorer reaches 123/147, against 122 for
the acoustic model and 110 for the shipped gate (`decide`) replay; it fixes 3 and breaks 2
relative to the acoustic model. The weights were chosen after the 虚拟 and 147-case evaluations
were already visible, so those numbers are regression evidence, not independent generalisation.
"""
from __future__ import annotations

import math

METHOD = "continuous_key_v1"
STATUS = "experimental_not_default"
SOURCE_ID = "retained10_stable"
TOP_N = 6

# Cases where the scorer is known to be worse than the acoustic model on the exposed set.
KNOWN_REGRESSIONS = (
    dict(case="191347", reference="D#m", experimental="A#m",
         note="反复回归奖励饱和（10 次 -> 封顶 0.8）压过更有支持的候选"),
    dict(case="853834", reference="Em", experimental="E",
         note="发射势阶段已偏向 E 大和弦（0.52 vs 0.06），时间模型只是放大"),
)

VALIDATION = dict(
    set="harmonic_key_v2_20261006 (147 cases, exposed regression set, 131 minor / 16 major)",
    experimental_hits=123, cnn_hits=122, production_hits=110,
    fixed_vs_cnn=["3958755", "4693198", "5417860"],
    broken_vs_cnn=["191347", "853834"],
    note="分数未校准；147 例已反复查看，不是盲测；同一作品的两个版本放同一组",
)

WARNINGS = (
    "连续评分是排序证据，不是已校准的正确概率。",
    "147 例为已暴露回归集：123/147（CNN 122、生产裁决 110），相对 CNN 改对 3 例、改错 2 例。",
    "权重在看过多份评测后设定，尚未在按作品隔离的新集上验证。",
    "不改变默认主调；采用需要人工点击，且与手动改调走同一条保存路径。",
)


def _round(value):
    return round(float(value), 6) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _candidate(row):
    return dict(key=row["key"], score=_round(row["score"]),
                acoustic_term=_round(row.get("acoustic_term")),
                harmonic_term=_round(row.get("harmonic_term")),
                log_compatibility=_round(row.get("log_compatibility")),
                native_share=_round(row.get("native_share")),
                tonic_share=_round(row.get("tonic_share")),
                outside_pitch_share=_round(row.get("outside_pitch_share")),
                authentic_arrivals=row.get("authentic_arrivals"),
                predominant_returns=row.get("predominant_returns"),
                exceptions=(row.get("exceptions") or [])[:6])


def evaluate(rows, acoustic_candidates, published_key=None):
    """Return the experimental block for one run; raises on invalid input.

    `rows` are LAB rows (dicts with start/end/label) in the same timeline that produced the
    published key; `acoustic_candidates` must be the full 24-key distribution from key.json.
    """
    import key_coherence as K
    from key_chord_adapter import parse_rows as parse_strict

    if not acoustic_candidates:
        raise ValueError("缺少 24 候选声学分布，无法计算连续评分")
    if len(acoustic_candidates) != 24:
        raise ValueError(f"声学候选应为 24 个，实际 {len(acoustic_candidates)}")
    chords, stats = parse_strict(rows)
    if len(chords) < 3:
        raise ValueError(f"可解析和弦太少（{len(chords)}），无法做连续评分")
    ranking = K.rank(chords, acoustic_candidates)
    top = ranking[:TOP_N]
    cnn_key = max(acoustic_candidates, key=lambda row: float(row["score"]))["key"]
    gap = float(ranking[0]["score"]) - float(ranking[1]["score"]) if len(ranking) > 1 else None
    predicted = ranking[0]["key"]
    identity = K.key_id(predicted)
    return dict(
        method=METHOD, status=STATUS, source_id=SOURCE_ID,
        key=predicted, runner_up=ranking[1]["key"] if len(ranking) > 1 else None,
        gap=_round(gap) if gap is not None else None,
        published_key=published_key, acoustic_key=cnn_key,
        same_as_published=(K.key_id(published_key) == identity if published_key else None),
        same_as_acoustic=(K.key_id(cnn_key) == identity),
        ranking=[_candidate(row) for row in top],
        candidates=len(ranking), chords=len(chords), parse=stats,
        validation=VALIDATION, known_regressions=list(KNOWN_REGRESSIONS),
        warnings=list(WARNINGS),
    )
