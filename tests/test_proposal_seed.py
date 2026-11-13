"""Whether a proposal pass can be repeated, and whether it says which draw it was.

27893 and 27894 were the same configuration and came out at 45% and 35% coverage.
Nothing separated them: the sampler was unseeded, so the gate's number was one
draw of many with nothing in the record saying which.

The seed reaches the engine through `sampling_params`, which `generate` takes as a
plain dict -- so a key this SGLang does not know is dropped in silence, and the run
would report a seed it never used. The name is therefore read off the installed
`SamplingParams` rather than written down here, and these tests are about that
resolution and about the seed arriving, since the engine itself is not importable
on this machine.
"""
import inspect
import sys
import types

import pytest

# Where sglang 0.5.9 keeps SamplingParams, read off the installed package on the
# cluster after 27901 died importing the path this code had guessed.
MEASURED_PATH = "sglang.srt.sampling.sampling_params"


class Tokenizer:
    def apply_chat_template(self, messages, **_):
        return " ".join(part.get("content", "") for part in messages)


class Engine:
    """Records what it was asked for, and returns something `_samples_from` rejects."""

    def __init__(self):
        self.asked = []

    def generate(self, prompt, sampling_params=None, **_):
        self.asked.append(sampling_params)
        return []


def scorer_with(engine):
    from inference.sglang_backend import SGLangScorer, SGLangSettings

    return SGLangScorer(engine=engine, tokenizer=Tokenizer(),
                        settings=SGLangSettings(model_id="m"), hidden_size=8)


def sglang_taking(*names, at=None):
    """A stand-in SamplingParams taking `names`, registered at the module path `at`.

    `at` defaults to where 0.5.9 actually keeps it. The first version of this file
    put the stub at the path the code guessed, so the tests agreed with the guess
    and 27901 died on a ModuleNotFoundError the whole suite had passed over. A
    stub can show that the right name is picked out of a signature; it cannot show
    that the signature is the installed one, and the tests below are written to
    say which of the two they are doing.

    The signature is the whole point, so it is set rather than written out:
    `inspect.signature` reads `__signature__` when there is one.
    """
    class SamplingParams:
        def __init__(self, *arguments, **keywords):
            pass

    SamplingParams.__init__.__signature__ = inspect.Signature(
        [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        + [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
           for name in ("temperature", "n", "regex", "json_schema", *names)])
    modules = {}
    parts = (at or MEASURED_PATH).split(".")
    for depth in range(1, len(parts) + 1):
        modules[".".join(parts[:depth])] = types.ModuleType(".".join(parts[:depth]))
    for depth in range(1, len(parts)):
        setattr(modules[".".join(parts[:depth])], parts[depth],
                modules[".".join(parts[: depth + 1])])
    modules[at or MEASURED_PATH].SamplingParams = SamplingParams
    return modules


@pytest.fixture
def installed(monkeypatch):
    def install(*names, at=None):
        # Nothing under `sglang` survives from another test, so a path that is not
        # installed here fails to import exactly as it would on the cluster.
        for name in list(sys.modules):
            if name == "sglang" or name.startswith("sglang."):
                monkeypatch.delitem(sys.modules, name)
        monkeypatch.setitem(sys.modules, "sglang", types.ModuleType("sglang"))
        for name, module in sglang_taking(*names, at=at).items():
            monkeypatch.setitem(sys.modules, name, module)
    return install


@pytest.mark.parametrize("name", ["sampling_seed", "seed", "random_seed"])
def test_the_seed_parameter_is_read_off_the_installed_class(installed, name):
    """Rather than written down here. SGLang has moved this name between versions,
    and a dict key it does not know is dropped without a word."""
    from inference.sglang_backend import SGLangScorer

    installed(name)
    assert SGLangScorer.seed_parameter() == name


def test_the_first_name_this_version_has_is_the_one_used(installed):
    from inference.sglang_backend import SEED_NAMES, SGLangScorer

    installed("random_seed", "seed", "sampling_seed")
    assert SGLangScorer.seed_parameter() == SEED_NAMES[0]


def test_the_path_this_reads_is_the_one_0_5_9_uses(installed):
    """Not a check of SGLang -- a stub cannot be one -- but of this file against
    the package, since the two disagreeing is what 27901 died of. If this constant
    stops matching the installed layout, the run refuses rather than crashes."""
    from inference.sglang_backend import SEED_MODULES

    assert SEED_MODULES[0] == MEASURED_PATH
    installed("sampling_seed", at=MEASURED_PATH)
    from inference.sglang_backend import SGLangScorer

    assert SGLangScorer.seed_parameter() == "sampling_seed"


def test_an_older_layout_is_still_reached(installed):
    from inference.sglang_backend import SEED_MODULES, SGLangScorer

    installed("seed", at=SEED_MODULES[1])
    assert SGLangScorer.seed_parameter() == "seed"


def test_a_layout_this_does_not_know_is_refused_not_crashed_into(installed):
    """A raw ModuleNotFoundError is what 27901 got, from inside a run that had
    already loaded a 27B model. The caller's question is whether it can be seeded."""
    from inference.sglang_backend import SGLangScorer

    installed("sampling_seed", at="sglang.somewhere.else.entirely")
    with pytest.raises(ValueError) as raised:
        SGLangScorer.seed_parameter()
    assert "per-request sampling seed" in str(raised.value)
    for path in ("sglang.srt.sampling.sampling_params", "sglang.sampling_params"):
        assert path in str(raised.value), "it must say what it tried"


def test_a_version_with_no_seed_at_all_is_refused(installed):
    """Not ignored. The alternative is a run that reports a seed it never used."""
    from inference.sglang_backend import SGLangScorer

    installed("top_p")
    with pytest.raises(ValueError) as raised:
        SGLangScorer.seed_parameter()
    assert "per-request sampling seed" in str(raised.value)
    assert "top_p" in str(raised.value), "it must say what the version does take"


def test_the_seed_reaches_the_engine(installed):
    installed("sampling_seed")
    engine = Engine()
    with pytest.raises(Exception):  # the fake engine returns nothing to read
        scorer_with(engine).propose("Answer the question.", "What is 2 + 2?",
                                    width=4, temperature=0.7, seed=11)
    (asked,) = engine.asked
    assert asked["sampling_seed"] == 11
    assert asked["temperature"] == 0.7 and asked["n"] == 4
    assert asked["regex"], "a proposal pass is constrained unless asked otherwise"


def test_without_a_seed_nothing_claims_one(installed):
    """An unseeded run is a legitimate one; it must simply not look seeded."""
    installed("sampling_seed")
    engine = Engine()
    with pytest.raises(Exception):
        scorer_with(engine).propose("Answer the question.", "What is 2 + 2?",
                                    width=4, temperature=0.7)
    (asked,) = engine.asked
    assert "sampling_seed" not in asked


def test_a_beam_refuses_a_seed_rather_than_accepting_one():
    """There is no draw to fix, so a seed would describe something that did not
    happen -- the argument the temperature already makes on this backend."""
    from hde.backends.backend import TransformersScorer

    scorer = TransformersScorer.__new__(TransformersScorer)
    with pytest.raises(ValueError) as raised:
        scorer.propose("Answer the question.", "What is 2 + 2?", seed=3)
    assert "no draw" in str(raised.value)


def test_the_argument_parser_refuses_the_same_combination():
    import argparse

    from hde.candidates.proposing import add_proposal_arguments, check_proposal_arguments

    parser = add_proposal_arguments(argparse.ArgumentParser())
    arguments = parser.parse_args(["--backend", "transformers", "--seed", "3"])
    with pytest.raises(SystemExit):
        check_proposal_arguments(parser, arguments)

    arguments = parser.parse_args(["--backend", "sglang", "--seed", "3"])
    check_proposal_arguments(parser, arguments)
    assert arguments.seed == 3


# The risk the stubs above cannot reach: the seed is passed, but SGLang may expand
# one request's `n` samples into copies that share it. Then a seeded sampler
# returns `width` identical proposals, every question is degenerate, and the run
# records that artefact as a finding about the model rather than about the seed.
def watch(seen, distinct, **rest):
    from hde.candidates.proposing import watch_proposals

    return watch_proposals(seen, distinct, coded=0, **rest)


def test_a_seed_that_replaces_the_draw_stops_the_run():
    abort, _ = watch(20, [1] * 20, seeded=True, temperature=0.6)
    assert abort and "replacing it" in abort
    assert "seed+i" in abort, "it must say what to do instead"


def test_a_collapse_without_a_seed_is_still_only_reported():
    """It is what low temperature does, and the sweep exists to measure where."""
    abort, warning = watch(20, [1] * 20, seeded=False, temperature=0.6)
    assert abort is None and warning is not None


def test_a_collapse_at_a_temperature_that_explains_it_is_only_reported():
    """The sweep runs down to 0.1, where one answer per question is the finding."""
    abort, warning = watch(20, [1] * 20, seeded=True, temperature=0.1)
    assert abort is None and warning is not None


def test_one_question_with_a_single_answer_is_not_a_collapsed_sampler():
    """`2 + 2` can legitimately come back as ten fours; the claim is about all of them."""
    abort, _ = watch(20, [1] + [4] * 19, seeded=True, temperature=0.6)
    assert abort is None


def test_nothing_is_judged_before_there_is_enough_to_judge():
    assert watch(19, [1] * 19, seeded=True, temperature=0.6) == (None, None)


class Sampler:
    """A scorer whose only job is to answer the probe's two questions.

    `collapses` is whether a seed replaces the draw rather than fixing it, which
    is the engine behaviour the probe exists to settle. `varies` is whether the
    question itself has more than one plausible answer at this temperature, since
    one that does not says nothing about the seed either way.
    """

    def __init__(self, collapses, varies=True):
        self.collapses = collapses
        self.varies = varies
        self.asked = []

    def propose(self, state, instructions, width=10, max_new_tokens=32, temperature=0.7,
                seed=None, verified=True):
        self.asked.append((instructions, seed))
        distinct = 1 if not self.varies or (self.collapses and seed is not None) else width
        return [(f"{index % distinct}", -0.1) for index in range(width)]


def probe(scorer, questions=("q1", "q2", "q3")):
    from hde.candidates.proposing import probe_sampler

    return probe_sampler(scorer, "Answer.", questions, 4, 8, seed=7)


def test_an_engine_that_shares_one_seed_across_the_samples_is_caught_before_the_run():
    """Below temperature 0.5 the running watch is off by design, and there this
    artefact and the collapse the sweep measures are the same picture."""
    abort, facts = probe(Sampler(collapses=True))
    assert abort and "replacing the draw" in abort and "seed+i" in abort
    assert facts["probe_unseeded"] == [4] and facts["probe_seeded"] == [1]


def test_an_engine_whose_seed_fixes_the_draw_is_allowed_through():
    abort, facts = probe(Sampler(collapses=False))
    assert abort is None
    assert facts["probe"] == "the seed does not collapse the draw"
    assert facts["probe_seeded"] == [4]


def test_a_question_that_has_one_answer_anyway_decides_nothing():
    """It would otherwise look exactly like a collapsed sampler, so the probe moves
    on -- and having moved through all of them without an answer, it refuses.

    An undecided probe is not a pass: letting the run through would put it back in
    the blind spot below temperature 0.5 with a line in the record saying the
    sampler had been asked.
    """
    questions = [f"q{index}" for index in range(10)]
    abort, facts = probe(Sampler(collapses=True, varies=False), questions)
    assert abort and "still unknown" in abort
    assert facts["probe"] == "undecided"
    assert facts["probe_unseeded"] == [1] * 10, "it must try every question it was given"


def test_the_probe_asks_one_question_twice_and_stops_as_soon_as_it_knows():
    scorer = Sampler(collapses=False)
    probe(scorer)
    assert scorer.asked == [("q1", None), ("q1", 7)], scorer.asked


def test_the_probe_runs_at_a_temperature_of_its_own():
    """The run's temperature can be 0.1, where nothing varies and nothing is learnt."""
    from hde.candidates.proposing import PROBE_TEMPERATURE

    assert PROBE_TEMPERATURE >= 1.0
