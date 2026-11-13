"""The prompt a proposal is made with, which asked for the wrong thing entirely.

`propose` rendered `build_decision_messages`, whose system prompt is "Return
only the option code". Asked that, a model proposing answers to a maths question
returns B, C and A -- correctly, for the question it was asked -- and coverage
came out at 1.4% on 500 questions before anyone looked at what was in the
candidate sets.

The fault was invisible in every unit test, because they all fed proposals in by
hand. These read the prompt instead.
"""
import pytest

from hde._legacy.prompting import (
    build_decision_messages,
    build_proposal_messages,
    build_thinking_messages,
)


def rendered(messages):
    return "\n".join(str(message["content"]) for message in messages)


def test_a_proposal_prompt_mentions_neither_options_nor_codes():
    text = rendered(build_proposal_messages("state", "What is 2 + 2?")).lower()
    for forbidden in ("option", "code", "labeled", "labelled"):
        assert forbidden not in text, f"a proposal prompt must not say {forbidden!r}"


def test_a_proposal_prompt_carries_the_question():
    text = rendered(build_proposal_messages("Answer the question.", "What is 2 + 2?"))
    assert "What is 2 + 2?" in text
    assert "Answer the question." in text


def test_the_decision_prompt_still_asks_for_a_code():
    """The readout needs it; only the proposal must not have it."""
    assert "option code" in rendered(build_decision_messages("state")).lower()


def test_the_three_prompts_are_not_the_same_prompt():
    made = {name: rendered(builder)
            for name, builder in (
                ("proposal", build_proposal_messages("s", "q")),
                ("decision", build_decision_messages("s")),
                ("thinking", build_thinking_messages("s", "q")))}
    assert len(set(made.values())) == 3


def test_a_proposal_asks_for_no_working():
    """A beam with a reasoning trace in it is not a candidate set."""
    text = rendered(build_proposal_messages("s", "q")).lower()
    assert "do not explain" in text


@pytest.mark.parametrize("backend", ["transformers", "sglang"])
def test_both_backends_render_the_proposal_prompt(backend, monkeypatch):
    """Whichever backend proposes, it must send the proposal prompt and not the decision one."""
    seen = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **_):
            seen["text"] = rendered(messages)
            return seen["text"]

    if backend == "transformers":
        from hde.backends.backend import TransformersScorer

        scorer = TransformersScorer.__new__(TransformersScorer)
        scorer.tokenizer = Tokenizer()

        class Model:
            device = "cpu"

        scorer.model = Model()
        with pytest.raises(Exception):  # generation is not mocked; the prompt is
            scorer._propose("Answer the question.", "What is 2 + 2?", 4, 8)
    else:
        from inference.sglang_backend import SGLangScorer, SGLangSettings

        scorer = SGLangScorer(engine=None, tokenizer=Tokenizer(),
                              settings=SGLangSettings(model_id="m"), hidden_size=8)
        with pytest.raises(Exception):
            scorer.propose("Answer the question.", "What is 2 + 2?", width=4, temperature=0.7)

    assert "What is 2 + 2?" in seen["text"], "the question never reached the prompt"
    assert "option code" not in seen["text"].lower(), "the decision prompt was sent"


def test_option_letters_are_recognised_as_not_answers():
    from hde.candidates.proposing import looks_like_codes

    assert looks_like_codes([("B", -0.1), ("C", -0.4), ("A", -0.9)])
    assert not looks_like_codes([("42", -0.1), ("41", -0.4)])
    assert not looks_like_codes([("\\frac{1}{2}", -0.1), ("0.5", -0.4)])


def test_a_run_of_letters_is_stopped_once_there_is_enough_to_say_so():
    from hde.candidates.proposing import watch_proposals

    assert watch_proposals(10, [2] * 10, 10) == (None, None), "too early to judge"
    assert "option letters" in watch_proposals(20, [8] * 20, 15)[0]
    assert watch_proposals(20, [8] * 20, 1) == (None, None)


def test_a_collapsed_candidate_set_is_reported_and_not_aborted():
    """Low temperature collapses the set; the sweep is there to measure where."""
    from hde.candidates.proposing import watch_proposals

    abort, warning = watch_proposals(20, [1] * 12 + [3] * 8, 0)
    assert abort is None, "aborting here would kill the run that measures the collapse"
    assert warning and "collapse" in warning


def test_letters_still_abort_however_many_distinct_there_are():
    from hde.candidates.proposing import watch_proposals

    abort, _ = watch_proposals(20, [8] * 20, 15)
    assert abort and "option letters" in abort


def test_a_caller_can_ask_for_a_minimum_and_get_an_abort():
    from hde.candidates.proposing import watch_proposals

    abort, _ = watch_proposals(20, [2] * 20, 0, minimum_distinct=3)
    assert abort and "asked for" in abort


def test_the_proposal_prompt_asks_for_the_value_alone():
    """An integer benchmark compares strings, so "x = 2" would not match "2"."""
    text = rendered(build_proposal_messages("s", "q")).lower()
    assert "equals sign" in text and "no units" in text


def test_the_sglang_refusal_names_only_the_setting_at_fault():
    """28281 passed `device_map='cuda:0'` with `dtype='bfloat16'`, which is allowed,
    and was told the backend takes neither. A diagnostic that names a setting the
    caller got right sends them to change it."""
    import pytest

    from hde.candidates.proposing import load_proposer

    with pytest.raises(ValueError) as refusal:
        load_proposer("m", backend="sglang", dtype="bfloat16", device_map="cuda:0")
    assert "device_map='cuda:0'" in str(refusal.value)
    assert "dtype" not in str(refusal.value), "it was not the dtype"

    with pytest.raises(ValueError) as both:
        load_proposer("m", backend="sglang", dtype="float16", device_map="cuda:0")
    assert "dtype='float16'" in str(both.value) and "device_map='cuda:0'" in str(both.value)

    with pytest.raises(ValueError) as only_dtype:
        load_proposer("m", backend="sglang", dtype="float32", device_map="auto")
    assert "dtype='float32'" in str(only_dtype.value)
    assert "device_map" not in str(only_dtype.value)
