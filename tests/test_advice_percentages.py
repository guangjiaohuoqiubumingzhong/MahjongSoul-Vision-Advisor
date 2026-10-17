"""Relative candidate scores, including negative Q values and strong gaps."""
from copy import deepcopy

import pytest

from app.ui.advice import candidate_percentages


@pytest.mark.parametrize('scores,expected',[
    ([],[]),
    ([3],['100.0%']),
    ([-2,-2,-2,-2,-2],['20.0%']*5),
    ([100,-1,-1,-2,-2],['99.6%','0.1%','0.1%','0.1%','0.1%']),
    ([10000,-10000],['99.9%','0.1%']),
])
def test_relative_percentages_have_a_finite_display_floor_and_total(scores,expected):
    candidates=[{'model_score':score} for score in scores]
    rows=candidate_percentages(candidates)
    assert [percentage for _,percentage in rows]==expected
    if rows:
        assert sum(float(percentage[:-1]) for _,percentage in rows)==pytest.approx(100)


def test_top_five_are_ranked_without_changing_model_data_or_using_omitted_scores():
    candidates=[{'action_id':i,'model_score':score} for i,score in enumerate([-4,2,0,1,-1,-2,-3])]
    original=deepcopy(candidates)
    rows=candidate_percentages(candidates)
    assert [candidate['action_id'] for candidate,_ in rows]==[1,3,2,4,5]
    assert candidates==original
    percentages=[percentage for _,percentage in rows]
    shifted=[{**candidate,'model_score':candidate['model_score']+10000} for candidate in candidates]
    assert [percentage for _,percentage in candidate_percentages(shifted)]==percentages
    assert [percentage for _,percentage in candidate_percentages([candidate for candidate,_ in rows])]==percentages
    assert sum(float(percentage[:-1]) for percentage in percentages)==pytest.approx(100)
