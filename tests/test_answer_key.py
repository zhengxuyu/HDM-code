from litjev._legacy.decision import ChoiceAnswer, NoulAnswer, ScoreAnswer
from litjev.bench.routing_bench import _answer_key


def test_a_choice_answers_with_its_key():
    answer = ChoiceAnswer(type="choice", choice="G", probabilities={"G": 0.8}, confidence=0.8)
    assert _answer_key(answer) == "G"


def test_a_score_answers_with_the_most_probable_level_not_the_weighted_index():
    # The weighted index is 0*0.1 + 1*0.3 + 2*0.6 = 1.5, which rounds to 2 either
    # way; the point is that the key follows the probabilities, not the mean.
    answer = ScoreAnswer(type="score", score=1.5, legend={},
                         probabilities={"0": 0.1, "1": 0.3, "2": 0.6}, confidence=0.6)
    assert _answer_key(answer) == 2
    assert isinstance(_answer_key(answer), int)


def test_a_score_whose_mean_and_mode_disagree_follows_the_mode():
    # Mass at the ends: mean is the middle, which no level was chosen.
    answer = ScoreAnswer(type="score", score=1.0, legend={},
                         probabilities={"0": 0.45, "1": 0.1, "2": 0.45}, confidence=0.45)
    assert _answer_key(answer) in (0, 2)


def test_a_noul_answers_true_above_a_half():
    assert _answer_key(NoulAnswer(type="noul", noul=0.9)) is True
    assert _answer_key(NoulAnswer(type="noul", noul=0.5)) is True
    assert _answer_key(NoulAnswer(type="noul", noul=0.4999)) is False


def test_an_unknown_type_scores_nothing_rather_than_guessing():
    class Odd:
        type = "odd"

    assert _answer_key(Odd()) is None
