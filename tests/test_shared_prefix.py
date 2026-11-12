"""The fast and slow prompts must be one render, against the real chat template.

The whole point of deriving both modes from one prefix is that an escalation
continues from the fast pass's cache instead of prefilling the prompt again. A
radix cache matches on TOKENS, so a string prefix is not enough -- these check the
token sequences, against a cached Qwen3 tokenizer rather than the fake one the
rest of the suite uses, because the property lives in the template.

These check the mechanics. They are not enough on their own: an earlier version
cut the prefix out of the thinking-OFF render, every assertion here passed, and
the slow mode wrote `\\n\\n</think>` and an answer instead of reasoning. The
behaviour is checked against a served model, not here.
"""

import pytest

from litjev.core.prompts_utils import (
    CLOSE_THINKING,
    OPEN_THINKING,
    close_thinking,
    open_thinking,
    shared_prefix,
)

MODEL = "Intel/Qwen3.8-Flash-Next-W4A16-AutoRound"


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except (OSError, ValueError) as error:  # not cached on this machine
        pytest.skip(f"{MODEL} is not in the local cache: {error}")


def render(tokenizer, thinking):
    messages = [
        {"role": "system", "content": "Evaluate the state using the question and options."},
        {"role": "user", "content": "A state long enough that its prefill is the cost.\n\nQ?"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )


def test_thinking_off_closes_the_block_so_another_one_cannot_be_opened(tokenizer):
    """Why a prefix cut out of this render makes the slow mode close rather than
    reason: `<think>` here is always followed by the closing sequence."""
    off = render(tokenizer, False)
    assert off.endswith(CLOSE_THINKING)
    assert off.count("<think>") == 1 and off.count("</think>") == 1


def test_thinking_on_injects_a_preamble_the_other_render_does_not_have(tokenizer):
    """And that preamble is why the shared prefix is cut from THIS render.

    Without it the model treats a bare `<think>` as the no-think opener and
    closes it immediately, which is what the slow mode did until this was fixed.
    """
    off, on = render(tokenizer, False), render(tokenizer, True)
    assert "Reasoning effort" in on and "Reasoning effort" not in off
    assert on.endswith(OPEN_THINKING) and not on.endswith(CLOSE_THINKING)


def test_both_prompts_extend_the_shared_prefix_token_for_token(tokenizer):
    on = render(tokenizer, True)
    shared = shared_prefix(on)
    assert open_thinking(shared) == on, "the slow prompt is the render itself"

    ids = tokenizer.encode(shared, add_special_tokens=False)
    for name, prompt, added in (("slow", open_thinking(shared), 1),
                                ("fast", close_thinking(shared), 3)):
        whole = tokenizer.encode(prompt, add_special_tokens=False)
        assert whole[: len(ids)] == ids, (
            f"the {name} prompt must extend the shared prefix token for token, or "
            "an escalation re-prefills instead of continuing from the cache")
        assert len(whole) - len(ids) == added


def test_the_shared_prefix_ends_at_the_open_block_and_keeps_the_preamble(tokenizer):
    shared = shared_prefix(render(tokenizer, True))
    assert shared.endswith("<think>")
    assert "</think>" not in shared
    assert "Reasoning effort" in shared, "the preamble is what makes the slow mode reason"


def test_the_prefix_is_everything_but_the_last_few_tokens(tokenizer):
    """The cost of the split, stated: the state and the question are all shared."""
    fast_ids = tokenizer.encode(close_thinking(shared_prefix(render(tokenizer, True))),
                                add_special_tokens=False)
    shared_ids = tokenizer.encode(shared_prefix(render(tokenizer, True)),
                                  add_special_tokens=False)
    assert len(shared_ids) / len(fast_ids) > 0.85, (
        "on this toy prompt the shared part is already most of it; on a real state "
        "it is all but three tokens"
    )


def test_a_template_without_an_open_block_is_refused_rather_than_guessed_at():
    with pytest.raises(ValueError, match="does not end with an open thinking block"):
        shared_prefix("<|im_start|>assistant\n")
    with pytest.raises(ValueError, match="does not end with an open thinking block"):
        shared_prefix("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_the_sglang_renderer_cuts_both_modes_from_one_string(tokenizer):
    """End to end through `_render`, with the fake tokenizer the suite already has.

    Before this, the two modes used different system text and put the question in
    different places, so they diverged at the first token of the system turn and
    an escalation re-prefilled the whole prompt.
    """
    from inference.sglang_backend import SGLangScorer, SGLangSettings  # noqa: PLC0415
    from litjev.data.schema import Choice, DecisionSchema  # noqa: PLC0415

    engine = SGLangScorer(engine=None, tokenizer=tokenizer,
                          settings=SGLangSettings(model_id="m", feature_layers=(-1,)),
                          hidden_size=8)
    schema = DecisionSchema({"q": Choice(instructions="Pick", criteria={"a": "1", "b": "2"})})
    fast, fast_labels = engine._render("STATE", schema, thinking=False)
    slow, slow_labels = engine._render("STATE", schema, thinking=True)

    assert fast_labels == slow_labels
    for quick, deliberate in zip(fast, slow, strict=True):
        assert quick.startswith(deliberate), (
            "the fast prompt must extend the slow one; they are the same render"
        )
        assert quick[len(deliberate):] == "\n</think>\n\n" + "\nAnswer:"
        assert deliberate.endswith("<think>\n"), "the slow prompt is the thinking render"
        assert "Reasoning effort" in deliberate
