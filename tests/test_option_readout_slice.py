"""The listable-options readout reads the positions it used to read.

That readout stopped materialising `batch x width x vocab` logits: three of them
were alive at once -- the bf16 logits the output still holds, the float32 copy
`.float()` makes, and log_softmax's float32 result -- which for ten options over
a 1000-token prefix on the 27B is 28 GiB. `logits_to_keep` applies here and not
at the coded readouts, because every row is `prefix + option` so the positions
wanted are a common suffix rather than one per row, and the summation is rebased
onto the slice.

Two numbers have to agree for that to be right, and getting either wrong is
silent: the slice must start at `len(prefix) - 1`, and the rebase must map
absolute `len(prefix) - 1 + offset` to slice index `offset`. A shifted slice
reads a neighbouring token's distribution, which is a perfectly plausible
log-probability and still ranks the options.

So the stub makes each position's only non-negligible logit the token that
actually follows it -- teacher forcing, which is what the readout is doing -- and
needs no knowledge of where the prefix ends. A read at the right position scores
~0 for every option; a read one position away looks up position `p`'s
distribution for position `p+1`'s token and scores -1e4.

The first attempt here did try to derive the prefix, as the longest common prefix
of the batch, and that was unsound for the reason this stub avoids: these options
all begin "the marbles are", so the rows agree well past the prefix and the
inferred boundary was wrong. It failed loudly rather than quietly, which is the
only reason it is worth recording.
"""
import torch

from hde.backends.backend import TransformersScorer
from hde.data.schema import Choice
from test_unified_prompt_transformers import PaddingTokenizer

STATE = "A bag holds two red marbles."
QUESTION = Choice(instructions="Which colour is described",
                  criteria={"red": "the marbles are red",
                            "blue": "the marbles are blue",
                            "green": "the marbles are green"})


class OneHotAtTheReadout:
    """Logits that are ~0 only where `summed_logprob` is supposed to look."""

    class Config:
        num_hidden_layers = 4

    def __init__(self, vocab=70_000):
        self.config = OneHotAtTheReadout.Config()
        self.vocab = vocab
        self.kept = []

    def eval(self):
        return self

    def get_input_embeddings(self):
        class Embedding:
            weight = torch.zeros(1)

        return Embedding()

    def __call__(self, input_ids=None, attention_mask=None, logits_to_keep=None,
                 output_hidden_states=False, **kwargs):
        rows, width = input_ids.shape
        self.kept.append(logits_to_keep)
        lengths = attention_mask.sum(dim=1).tolist()

        keep = width if logits_to_keep in (None, 0) else int(logits_to_keep)
        start = width - keep
        logits = torch.full((rows, keep, self.vocab), -1e4)
        for row in range(rows):
            for absolute in range(start, min(width, int(lengths[row]) - 1)):
                # Position `absolute` predicts the token at `absolute + 1`, which
                # is the only thing a correct readout asks it for.
                logits[row, absolute - start, int(input_ids[row, absolute + 1])] = 0.0

        class Output:
            pass

        output = Output()
        output.logits = logits
        output.hidden_states = tuple(
            torch.zeros(rows, width, 4) for _ in range(5)) if output_hidden_states else None
        return output


def scorer(model):
    return TransformersScorer(model, PaddingTokenizer(), max_input_tokens=4096)


def test_each_option_is_scored_at_its_own_tokens_after_the_slice():
    model = OneHotAtTheReadout()
    scores, _rewritten, _hidden = scorer(model).score_options(
        STATE, QUESTION, content_free=False)

    assert len(scores) == 3
    for score in scores:
        # Zero to within log-softmax's own floor: a correct read sums ~0 per
        # token, a read one position away sums about -1e4 per token.
        assert score.total > -1.0, (
            f"option {score.text!r} scored {score.total}, which is the stub's "
            "penalty rather than its readout position")
        assert score.tokens > 0


def test_the_slice_is_asked_for_and_is_the_narrow_one():
    """Or the test above would pass over the full-width tensor it is here to avoid."""
    model = OneHotAtTheReadout()
    scorer(model).score_options(STATE, QUESTION, content_free=False)

    assert model.kept, "the readout made no forward call"
    assert all(k not in (None, 0) for k in model.kept), (
        f"the readout asked for the whole sequence's logits: {model.kept}")
    # The prefix is a chat template plus the question, so the kept window is a
    # small fraction of the width; an exact figure here would pin the tokenizer
    # stub rather than the slice.
    assert all(k < 40 for k in model.kept), model.kept
