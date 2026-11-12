"""The unified render on the transformers path: the same prompt, not a second one.

The layer ablation has to be measured on the prompt the served heads are read
at, and it cannot be measured where that prompt lives: SGLang returns only the
final hidden layer, so a sweep over layers runs on transformers. That makes the
load-bearing property "these two backends render the same string", and it is the
first test here -- a paraphrase that drifted by a newline would collect features
at a prompt no head is served under, and nothing downstream would say so.

The second property is the index. The fast pass reads one position per row and
the rows are padded, so an off-by-one or a left/right padding mix-up returns a
state from a pad token. That is invisible: a head fits to it, trains, and reports
an AUROC.
"""
import numpy as np
import pytest

from inference.sglang_backend import SGLangScorer, SGLangSettings
from litjev.backends.backend import ModelSettings, TransformersScorer
from litjev.core.prompts_utils import ANSWER_BOUNDARY
from litjev.data.schema import Choice, SystemOneRequest
from test_inference_sglang import _RealEnoughTokenizer

STATE = "A bag holds two red marbles."
SCHEMA = {
    "colour": Choice(instructions="Which colour", criteria={"red": "r", "blue": "b"}),
    "count": Choice(instructions="How many", criteria={"two": "2", "three": "3", "four": "4"}),
}


class PaddingTokenizer(_RealEnoughTokenizer):
    """The SGLang tests' tokenizer plus the pad id `_pad_id` needs.

    Theirs, deliberately: the first test asserts the two backends render the same
    string, and a second tokenizer here would let a difference in the stub pass
    for agreement between the two code paths.
    """

    pad_token_id = 0
    eos_token_id = 0

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class StubModel:
    """Returns logits and hidden states that encode the position they came from.

    So an assertion can say *which* position was read rather than that something
    was read. `logits[row, position, token] = position`, and the hidden state at
    a position is that position repeated, which is what makes a pad-token read
    detectable instead of plausible.
    """

    class Config:
        num_hidden_layers = 4

    # Wide enough to index the tokenizer's boundary code ids, which sit at
    # 10000 + ord(letter); a narrow vocab fails as an IndexError rather than as
    # a wrong reading, but it fails every test that scores anything.
    def __init__(self, width=10_200, layers=5):
        self.config = StubModel.Config()
        self.width = width
        self.layers = layers
        self.calls = []

    def eval(self):
        return self

    def get_input_embeddings(self):
        import torch

        class Embedding:
            weight = torch.zeros(1)

        return Embedding()

    def __call__(self, input_ids=None, attention_mask=None, use_cache=None,
                 output_hidden_states=False, **kwargs):
        import torch

        rows, positions = input_ids.shape
        self.calls.append({"input_ids": input_ids.tolist(),
                           "attention_mask": attention_mask.tolist(),
                           "output_hidden_states": output_hidden_states})
        ramp = torch.arange(positions, dtype=torch.float32)[None, :, None]
        logits = ramp.expand(rows, positions, self.width).clone()
        hidden = None
        if output_hidden_states:
            hidden = tuple(ramp.expand(rows, positions, 4).clone() + layer * 1000.0
                           for layer in range(self.layers))

        class Output:
            pass

        output = Output()
        output.logits = logits
        output.hidden_states = hidden
        return output

    # The readout paths call the base transformer and apply the head only at the
    # positions they read, so the stub has to offer that pair or it tests an
    # interface production no longer uses. The contract is preserved exactly:
    # hidden_states[layer] is position + layer * 1000 as before,
    # last_hidden_state is hidden_states[-1], and lm_head undoes the top layer's
    # offset so a logit still reads back as the position it came from.
    @property
    def model(self):
        def inner(input_ids=None, attention_mask=None, use_cache=None,
                  output_hidden_states=False, **kwargs):
            import torch

            rows, positions = input_ids.shape
            self.calls.append({"input_ids": input_ids.tolist(),
                               "attention_mask": attention_mask.tolist(),
                               "output_hidden_states": output_hidden_states})
            ramp = torch.arange(positions, dtype=torch.float32)[None, :, None]
            hidden = tuple(ramp.expand(rows, positions, 4).clone() + layer * 1000.0
                           for layer in range(self.layers))

            class Base:
                pass

            base = Base()
            base.hidden_states = hidden if output_hidden_states else None
            base.last_hidden_state = hidden[-1]
            return base

        return inner

    @property
    def lm_head(self):
        def head(states):
            return (states[:, :1] - (self.layers - 1) * 1000.0).expand(-1, self.width).clone()

        return head


def transformers_scorer(feature_layers=(), unified=True, code_offset=0):
    return TransformersScorer(
        StubModel(), PaddingTokenizer(), max_input_tokens=4096,
        feature_layers=feature_layers, code_offset=code_offset, unified_prompt=unified)


def request():
    return SystemOneRequest(model="litjev", state=STATE, questions=SCHEMA)


def test_the_two_backends_render_the_same_strings():
    """The property the whole port rests on, for both cuts of the render.

    Same tokenizer on both sides, so a difference is the code's and not the
    template's. Asserted on the strings rather than on token counts: two renders
    differing by a newline can tokenise to the same length.
    """
    schema = request().to_schema()
    theirs = SGLangScorer(engine=None, tokenizer=PaddingTokenizer(),
                          settings=SGLangSettings(model_id="m"), hidden_size=4)
    mine = transformers_scorer()

    for thinking in (False, True):
        their_prompts, their_labels = theirs._render(STATE, schema, thinking=thinking)
        my_prompts, my_labels = mine._render_unified(STATE, schema, thinking=thinking)
        assert my_prompts == their_prompts, f"thinking={thinking}"
        assert my_labels == their_labels


def test_the_fast_cut_closes_the_block_and_the_slow_cut_leaves_it_open():
    """And both extend one shared prefix, which is what "one render" means.

    Without this the first test only says the two backends agree -- they could
    agree on a render where the modes do not share a prefix at all, which is the
    arrangement this replaces.
    """
    schema = request().to_schema()
    mine = transformers_scorer()
    fast, _ = mine._render_unified(STATE, schema, thinking=False)
    slow, _ = mine._render_unified(STATE, schema, thinking=True)

    for quick, slowly in zip(fast, slow, strict=True):
        assert quick.endswith(ANSWER_BOUNDARY)
        shared = slowly[: -len("\n")]
        assert quick.startswith(shared), "the fast prompt does not extend the shared prefix"
        assert slowly.startswith(shared)


def test_a_rotated_code_set_is_refused_rather_than_ignored():
    """`code_offset` has no slots to rotate against here, and SGLang ignores it.

    Honouring it on one path and not the other would make the two renders differ
    for a run that set it, which is exactly what this flag exists to prevent.
    """
    scorer = transformers_scorer(code_offset=1)
    with pytest.raises(ValueError, match="code_offset"):
        scorer._render_unified(STATE, request().to_schema())


def test_each_row_is_read_at_its_own_last_token_not_at_the_padding():
    """The index, with a model that reports which position it was asked for.

    The two questions render to different lengths, so the shorter row is padded.
    `logits[row, position, :] == position` here, so the value read *is* the index
    used -- a read at the padded end would come back as the longest row's length
    for both, and a head fitted to that is fitted to a pad token.
    """
    schema = request().to_schema()
    scorer = transformers_scorer(feature_layers=(-1,))
    prompts, _ = scorer._render_unified(STATE, schema, thinking=False)
    lengths = [len(scorer.tokenizer.encode(text)) for text in prompts]
    assert len(set(lengths)) > 1, "both rows are the same length; this test proves nothing"

    fields = scorer.score(STATE, schema)
    for field, length in zip(fields, lengths, strict=True):
        assert field.provenance["selected_logit_index"] == length - 1
        assert np.allclose(field.logits, float(length - 1))
        assert field.input_tokens == length
        # And the hidden state comes from that same position, not another. The
        # stub offsets each layer by 1000 and `-1` is its last of five, so the
        # expected value carries that offset; what is under test is the
        # `length - 1` added to it.
        assert np.allclose(field.hidden[0], 4000.0 + (length - 1))


def test_every_requested_layer_comes_back_in_order():
    """Which is the whole point of running the ablation on this path.

    The stub offsets each layer by 1000, so the returned stack says which layers
    it holds and in which order -- a silently reordered stack would make the
    sweep report layer 16's AUROC against layer 24's name.
    """
    schema = request().to_schema()
    scorer = transformers_scorer(feature_layers=(-1, 1, 2))
    prompts, _ = scorer._render_unified(STATE, schema, thinking=False)
    first = len(scorer.tokenizer.encode(prompts[0])) - 1

    field = scorer.score(STATE, schema)[0]
    assert field.hidden.shape == (3, 4)
    # layer -1 is the stub's last (index 4), then layers 1 and 2.
    assert np.allclose(field.hidden[0], 4000.0 + first)
    assert np.allclose(field.hidden[1], 1000.0 + first)
    assert np.allclose(field.hidden[2], 2000.0 + first)


def test_the_records_say_which_render_produced_them():
    """`tab:features` was collected under the other render and its records cannot say so.

    So the provenance names it. Without this a future reader has two collections
    from this backend, at the same `slot_format`, with no field distinguishing
    them -- which is the position we are in now.
    """
    schema = request().to_schema()
    field = transformers_scorer().score(STATE, schema)[0]
    assert field.provenance["method"] == "one_forward_unified_prompt"
    assert field.provenance["system"] == "unified"
    assert field.provenance["boundary"] == ANSWER_BOUNDARY.strip()


def test_the_flag_is_off_by_default_so_the_measured_path_is_untouched():
    """Every transformers number to date, `tab:features` included, is the other render."""
    assert ModelSettings().unified_prompt is False
    assert TransformersScorer(StubModel(), PaddingTokenizer()).unified_prompt is False
    # And `score` dispatches on it rather than on anything else.
    scorer = transformers_scorer(unified=False)
    assert scorer.unified_prompt is False


def test_the_readout_reproduces_the_causal_lm_at_the_positions_it_reads():
    """The equivalence the memory saving rests on, as a test rather than a claim.

    `_forward_readout` calls the base transformer and applies `lm_head` to the
    gathered rows instead of letting the causal LM materialise
    `batch x width x vocab`. That is only the same number if the head is the last
    thing the forward does and the gather picks the same position, and neither is
    visible in the output: a wrong index returns a pad token's logits, which score
    something. So both are computed here and compared.
    """
    import torch

    scorer = transformers_scorer()
    rows = [[7, 8, 9, 10], [7, 8, 0, 0]]
    positions = [3, 1]
    ids = torch.tensor(rows)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

    _base, readout = scorer._forward_readout(
        positions, input_ids=ids, attention_mask=mask, use_cache=False)
    whole = scorer.model(input_ids=ids, attention_mask=mask, use_cache=False).logits

    for row, position in enumerate(positions):
        assert torch.equal(readout[row], whole[row, position]), (
            f"row {row} read position {readout[row][0].item()} "
            f"where the causal LM has {whole[row, position][0].item()}")


def test_a_backbone_that_softcaps_its_logits_is_refused_not_quietly_uncapped():
    """Gemma-2/3 and Cohere apply a transform after `lm_head`; Qwen does not.

    Dropping it would be invisible in the place anyone would look: softcapping is
    monotone, so every answer argmax -- and so `fast_correct`, `slow_correct` and
    every accuracy in the paper -- is unchanged, while the softmax over the
    candidate ids moves, and that softmax is what `distribution_stats` reads. An
    equivalence gate on accuracy would pass a backbone whose head features had
    all shifted.
    """
    import torch

    model = StubModel()
    model.config.final_logit_softcapping = 30.0
    scorer = TransformersScorer(model, PaddingTokenizer(), max_input_tokens=4096)

    with pytest.raises(RuntimeError, match="after lm_head"):
        scorer._forward_readout(
            [1], input_ids=torch.tensor([[7, 8]]),
            attention_mask=torch.tensor([[1, 1]]), use_cache=False)

    # And the same model without the declaration is served, so the guard is
    # keyed on the transform rather than on the stub.
    del model.config.final_logit_softcapping
    scorer._forward_readout([1], input_ids=torch.tensor([[7, 8]]),
                            attention_mask=torch.tensor([[1, 1]]), use_cache=False)


def test_the_softcap_guard_reads_the_config_the_forward_reads():
    """Gemma-4 declares it under `get_text_config()`, not as a plain attribute.

    `$H/models/gemma-4-31b-it` is a `Gemma4ForConditionalGeneration` whose
    `final_logit_softcapping = 30.0` sits in the text config, and whose forward
    reads it as `config.get_text_config().final_logit_softcapping`. A guard that
    only looked at the top-level config would pass that checkpoint, which is the
    one such checkpoint actually on the cluster.
    """
    import torch

    class Nested:
        num_hidden_layers = 4
        final_logit_softcapping = 30.0

    class Multimodal:
        num_hidden_layers = 4

        def get_text_config(self):
            return Nested()

    model = StubModel()
    model.config = Multimodal()
    scorer = TransformersScorer(model, PaddingTokenizer(), max_input_tokens=4096)

    with pytest.raises(RuntimeError, match="after lm_head"):
        scorer._forward_readout([1], input_ids=torch.tensor([[7, 8]]),
                                attention_mask=torch.tensor([[1, 1]]), use_cache=False)
