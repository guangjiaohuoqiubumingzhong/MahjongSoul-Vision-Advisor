"""Tile-class acceptance from the top score and its strongest competitor."""
import math


def dominant_candidate_config(value):
    if value is None:
        return None
    if not isinstance(value,dict):
        raise ValueError("领先候选配置必须为对象。")
    floor = value.get("minimum_confidence",.5)
    ratio = value.get("minimum_ratio",3.)
    if type(floor) not in (int,float) or not math.isfinite(floor) or not .5 <= floor < 1:
        raise ValueError("领先候选最低分数应在 0.5（含）至 1（不含）之间。")
    if type(ratio) not in (int,float) or not math.isfinite(ratio) or ratio <= 1:
        raise ValueError("领先候选倍数必须为大于 1 的有限数值。")
    return {"minimum_confidence":float(floor),"minimum_ratio":float(ratio)}


def candidate_acceptance(prediction, confidence, margin, dominant=None):
    """Return the qualifying rule; colours/geometry are checked separately.

    The second score is the largest of all remaining 33 classes, represented
    exactly by the model's top-one minus top-two margin.
    """
    score, gap = prediction["score"],prediction["margin"]
    if gap < margin:
        return None
    if score >= confidence:
        return "confidence_threshold"
    second = max(0.,score-gap)
    if (dominant is not None and score > dominant["minimum_confidence"]
            and score >= dominant["minimum_ratio"]*second):
        return "dominant_candidate"
    return None
