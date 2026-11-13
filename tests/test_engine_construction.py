"""One entry point deciding which path a question takes, from the question itself.

Whether a question is read out from options it came with or from options built
for it used to be settled by which script was run. That is a fact about the
operator rather than about the question, and it is why the two paths could not
be reached through one call and why every caller had to remember to construct.

`criteria=None` now says it, and the engine acts on it.
"""
import numpy
import pytest

from hde._legacy.decision import SchemaDecisionEngine
from hde.candidates.construct import DEGENERATE
from hde.candidates.construction_settings import ConstructionSettings
from hde.data.schema import Choice, DecisionSchema


class Proposer:
    """A scorer that only proposes, recording what it was asked for."""

    def __init__(self, answers=("4", "5", "6")):
        self.answers = answers
        self.asked = []

    def propose(self, state, instructions, width=10, max_new_tokens=32, temperature=0.7,
                seed=None, verified=True):
        self.asked.append({"instructions": instructions, "width": width, "seed": seed,
                           "temperature": temperature, "verified": verified})
        return [(text, -0.1 * index) for index, text in enumerate(self.answers)]


def engine(proposer=None, **settings):
    return SchemaDecisionEngine(
        proposer or Proposer(),
        construction=ConstructionSettings(**settings) if settings or proposer else None, readout="coded")


def ranked(**settings):
    """Construction with the listing order left in rank order, on purpose."""
    return {"shuffle": None, **settings}


def test_a_question_that_came_with_options_is_left_alone():
    """In source order, untouched: the shuffle is for constructed candidates."""
    schema = DecisionSchema({"q": Choice(instructions="Pick", criteria={"A": "1", "B": "2"})})
    proposer = Proposer()
    built, notes = engine(proposer, shuffle="run").constructed("s", schema)
    assert built["q"].criteria == {"A": "1", "B": "2"}
    assert notes == {}
    assert proposer.asked == [], "nothing should have been proposed"


def test_a_question_without_options_has_them_built():
    schema = DecisionSchema({"q": Choice(instructions="What is 2+2?", criteria=None)})
    proposer = Proposer()
    built, notes = engine(proposer, **ranked(width=3)).constructed("s", schema)
    assert built["q"].enumerated
    assert list(built["q"].descriptions) == ["4", "5", "6"]
    assert notes["q"]["reason"] is None
    assert proposer.asked[0]["instructions"] == "What is 2+2?"
    assert proposer.asked[0]["verified"] is True, "the constrained shape, by default"


def test_both_kinds_in_one_schema_go_their_own_ways():
    schema = DecisionSchema({
        "open": Choice(instructions="What is 2+2?", criteria=None),
        "closed": Choice(instructions="Pick", criteria={"A": "1", "B": "2"})})
    proposer = Proposer()
    built, notes = engine(proposer, **ranked(width=3)).constructed("s", schema)
    assert built["closed"].criteria == {"A": "1", "B": "2"}
    assert built["open"].enumerated and list(built["open"].descriptions) == ["4", "5", "6"]
    assert set(notes) == {"open"}, "only the one that needed building is noted"
    assert len(proposer.asked) == 1


def test_the_shuffle_touches_only_the_constructed_candidates():
    """Native options keep source order; a constructed set is permuted, seeded by
    run and question, because in rank order the readout's slot preference is also
    a rank preference."""
    answers = [str(n) for n in range(8)]
    schema = DecisionSchema({name: Choice(instructions=f"q{name}", criteria=None)
                             for name in "abcd"})
    built, _ = engine(Proposer(answers), width=8).constructed("s", schema)
    orders = {tuple(built[name].descriptions) for name in "abcd"}
    assert len(orders) > 1, "a seed without the question would give one permutation"
    for order in orders:
        assert sorted(order) == sorted(answers), "a permutation, nothing gained or lost"

    # Rank order is available and has to be asked for: in it the slot a biased
    # readout favours is also one particular rank.
    in_rank, _ = engine(Proposer(answers), **ranked(width=8)).constructed("s", schema)
    assert list(in_rank["a"].descriptions) == answers


def test_a_construction_that_yields_no_choice_is_noted_not_hidden():
    """A benchmark whose proposals collapse is a fact about the benchmark, and a
    silent fallback would report it as an ordinary answer."""
    schema = DecisionSchema({"q": Choice(instructions="q", criteria=None)})
    built, notes = engine(Proposer(("4", "4", "4")), width=3).constructed("s", schema)
    assert notes["q"]["reason"] == DEGENERATE
    assert notes["q"]["candidates"] == ["4"]
    assert not built["q"].enumerated, "it is returned as it came, with nothing to read out"


def test_an_engine_that_was_not_given_settings_refuses_rather_than_proposing():
    """A proposal pass is the costly path -- generation before a readout -- and it
    should never start because a field was left out somewhere."""
    schema = DecisionSchema({"q": Choice(instructions="q", criteria=None)})
    with pytest.raises(ValueError, match="no candidate set"):
        SchemaDecisionEngine(Proposer(), readout="coded").constructed("s", schema)


def test_the_settings_a_run_used_are_recordable():
    settings = ConstructionSettings(width=6, temperature=0.3, seed=11, shuffle="run")
    recorded = settings.as_record()
    assert recorded["width"] == 6 and recorded["seed"] == 11
    assert recorded["shuffle"] == "run" and recorded["verified"] is True


@pytest.mark.parametrize("bad", [{"width": 1}, {"temperature": 0.0}, {"temperature": -1}])
def test_settings_that_cannot_produce_a_choice_are_refused(bad):
    with pytest.raises(ValueError):
        ConstructionSettings(**bad)


def test_the_prompt_an_enumerated_question_gets_is_unchanged_by_this_work():
    """A comparison run is in flight against the code before the conversion. For a
    question that arrives with its options, the rendered prompt must be the same
    string, or those numbers measure an interim state rather than this one."""
    from hde._legacy.prompting import unlabelled_suffix

    question = Choice(instructions="What is 2+2?", criteria={"A": "three", "B": "four"})
    assert unlabelled_suffix(question) == (
        "Question: What is 2+2?\nOptions:\nthree\nfour\nAnswer:")


class Readout(Proposer):
    """Proposes, then scores whatever it is asked to read out.

    Returns a hidden state, as the real coded readout does: it is read at the
    answer position and is what a decision head is fitted on. The content readout
    returns none, and that contrast is what `test_a_content_row_carries_no_hidden_state`
    is about -- a stub without one could not show it.
    """

    def score(self, state, schema):
        import numpy as np

        from hde._legacy.decision import RawFieldScores

        return tuple(
            RawFieldScores(name, np.arange(len(schema[name].choices), dtype=float),
                           7, {"system": "one"}, np.ones((1, 4), dtype=np.float32))
            for name in schema.names)


def test_evaluate_constructs_for_itself():
    """Leaving it to callers is what made the two paths reachable only through the
    script that remembered, and a question with no candidate set reached the
    readout and crashed there."""
    engine = SchemaDecisionEngine(
        Readout(), construction=ConstructionSettings(**ranked(width=3)), readout="coded")
    schema = DecisionSchema({"q": Choice(instructions="What is 2+2?", criteria=None)})
    found = engine.evaluate("Answer each question using its listed options.", schema)
    assert found.result.answers["q"].choice == "6", "the last slot, by this stub's logits"
    assert found.diagnostics["construction"]["q"]["reason"] is None


def test_one_candidate_is_the_answer_without_a_readout():
    """Scoring a set of one gives a point mass and statistics describing nothing."""
    engine = SchemaDecisionEngine(Readout(("4", "4", "4")),
                                  construction=ConstructionSettings(width=3), readout="coded")
    schema = DecisionSchema({"q": Choice(instructions="q", criteria=None)})
    found = engine.evaluate("s", schema)
    answer = found.result.answers["q"]
    assert answer.choice == "4"
    assert found.diagnostics["fields"]["q"]["readout"].startswith("skipped:")
    assert found.diagnostics["construction"]["q"]["reason"] == DEGENERATE


def test_no_candidates_is_no_answer_rather_than_a_guess():
    engine = SchemaDecisionEngine(Readout(()), construction=ConstructionSettings(width=3), readout="coded")
    schema = DecisionSchema({"q": Choice(instructions="q", criteria=None)})
    found = engine.evaluate("s", schema)
    assert found.result.answers["q"] is None
    assert found.diagnostics["construction"]["q"]["reason"]


def test_a_collapsed_question_does_not_stop_the_others():
    """The readout still runs for whatever survived construction."""
    engine = SchemaDecisionEngine(
        Readout(), construction=ConstructionSettings(**ranked(width=3)), readout="coded")
    schema = DecisionSchema({
        "built": Choice(instructions="q", criteria=None),
        "native": Choice(instructions="Pick", criteria={"A": "1", "B": "2"})})
    found = engine.evaluate("s", schema)
    assert found.result.answers["native"].choice == "B"
    assert found.result.answers["built"].choice == "6"


def test_the_proposal_does_not_inherit_the_readouts_state():
    """Once the readout's state says "use its listed options", handing it to the
    proposer tells the model to use options that do not exist -- which is the
    whole reason it is proposing."""
    proposer = Readout()
    engine = SchemaDecisionEngine(proposer, construction=ConstructionSettings(width=3), readout="coded")
    schema = DecisionSchema({"q": Choice(instructions="q", criteria=None)})
    engine.evaluate("Answer each question using its listed options.", schema)
    assert proposer.asked, "nothing was proposed"
    # `propose` is given the construction's own state, not the request's.
    assert ConstructionSettings().state == "Answer the question."


# ------------------------------------------------- the slow mode and the proposals


class Escalating(Readout):
    """Proposes, scores so that everything escalates, and records both slow prompts.

    `think` and `solve` both record what they were rendered with, so a test can
    read the prompt rather than trust the call site -- which is the only kind of
    test that would have caught this the first time.
    """

    def __init__(self, answers=("4", "5", "6")):
        super().__init__(answers)
        self.thought = []
        self.solved = []

    def think(self, state, schema, names, budget):
        import numpy as np

        from hde._legacy.decision import RawFieldScores
        from hde._legacy.prompting import question_body

        for name in names:
            question = schema[name]
            codes = [chr(65 + i) for i in range(len(question.choices))]
            self.thought.append(question_body(question, codes))
        return tuple(
            RawFieldScores(name, np.arange(len(schema[name].choices), dtype=float),
                           7, {"system": "two"}, None, generated_tokens=3)
            for name in names)

    def solve(self, state, question, budget, answer_tokens=48):
        from hde._legacy.prompting import build_open_thinking_messages

        rendered = " ".join(m["content"] for m in build_open_thinking_messages(state, question))
        self.solved.append(rendered)
        return "reasoning\nAnswer: 42", 5, False


def routed_engine(answers=("4", "5", "6")):
    from hde._legacy.heads import HeadMetadata
    from hde._legacy.routing import RoutingPolicy

    class AlwaysEscalate:
        metadata = HeadMetadata("m", "r", hidden_size=0, feature_layers=())

        def predict(self, features):
            import numpy as np
            # gain = p(slow only) - p(fast only) well above any lambda
            return np.array([[0.0, 0.0, 1.0, 0.0]])

    provider = Escalating(answers)
    return provider, SchemaDecisionEngine(
        provider, head=AlwaysEscalate(), routing=RoutingPolicy(0.0, budget=16),
        construction=ConstructionSettings(width=3), readout="coded")


def test_a_constructed_question_never_shows_the_slow_mode_its_own_proposals():
    """`think` renders every option into the prompt, so a constructed question sent
    there hands the slow mode the fast mode's guesses and confines it to them.

    Asserted by reading the rendered prompt. I twice claimed this was
    structurally impossible and cited a test that checked the wrong field; the
    leak was through `options` in `question_body`, which is what this reads.
    """
    provider, engine = routed_engine()
    schema = DecisionSchema({"q": Choice(instructions="What is 2+2?", criteria=None)})
    found = engine.evaluate("s", schema)

    assert provider.thought == [], "a constructed question must not reach think()"
    assert provider.solved, "it must reach solve() instead"
    (prompt,) = provider.solved
    for proposal in ("4", "5", "6"):
        assert proposal not in prompt, f"{proposal!r} leaked into the slow prompt"
    assert "What is 2+2?" in prompt

    # And the answer is what the slow mode wrote, not one of the proposals.
    assert found.result.answers["q"].choice == "42"
    assert found.diagnostics["fields"]["q"]["slow_free_form"] is True


def test_a_question_that_came_with_options_still_escalates_through_think():
    """The enumerated path is unchanged: there the options are the caller's, both
    modes answer the same question, and confining the slow mode is correct."""
    provider, engine = routed_engine()
    schema = DecisionSchema({"q": Choice(instructions="Pick", criteria={"A": "1", "B": "2"})})
    engine.evaluate("s", schema)
    assert provider.solved == [], "a native question must not go free-form"
    assert len(provider.thought) == 1
    assert '"description": "1"' in provider.thought[0], "its own options, as given"


def test_both_kinds_escalating_at_once_go_their_own_ways():
    provider, engine = routed_engine()
    schema = DecisionSchema({
        "built": Choice(instructions="What is 2+2?", criteria=None),
        "native": Choice(instructions="Pick", criteria={"A": "1", "B": "2"})})
    engine.evaluate("s", schema)
    assert len(provider.solved) == 1 and len(provider.thought) == 1
    assert "What is 2+2?" in provider.solved[0]
    assert "Pick" in provider.thought[0]


def test_the_free_form_slow_prompt_does_not_mention_listed_options():
    """The same fault the proposal step had, one call further on. The request's
    state tells the model to answer from listed options and this question has
    none -- and `open_three_policies`' all-slow baseline uses the option-free
    instruction, so a routed slow answer rendered differently would not be the
    same policy the routed column is compared against."""
    provider, engine = routed_engine()
    schema = DecisionSchema({"q": Choice(instructions="Compute 1/2+1/4.", criteria=None)})
    engine.evaluate("Answer each question using its listed options.", schema)
    (prompt,) = provider.solved
    assert "listed options" not in prompt, prompt
    assert ConstructionSettings().state in prompt
    assert "Compute 1/2+1/4." in prompt


def test_the_forward_calls_for_a_free_form_escalation_are_its_own():
    """`output_tokens` carries the think path's tokens too, so reusing it
    double-counts a request that escalated one of each; and solves run one after
    another rather than in a batch, so they add."""
    provider, engine = routed_engine()
    schema = DecisionSchema({
        "a": Choice(instructions="q one", criteria=None),
        "b": Choice(instructions="q two", criteria=None)})
    found = engine.evaluate("s", schema)
    # Two solves, five generated tokens each by the stub: two prefills and ten.
    assert found.diagnostics["forward_calls"] == 2 + 2 * (1 + 5)
    assert found.diagnostics["routing"]["generated_tokens"] == 10


def test_the_engine_can_construct_from_proposals_it_did_not_make():
    """The gate watches the raw proposals for a collapse and for option letters
    before they become a candidate set, and proposing again to construct would be
    a second draw rather than the same one."""
    proposer = Proposer(("ignored",))
    engine = SchemaDecisionEngine(proposer, construction=ConstructionSettings(width=3), readout="coded")
    schema = DecisionSchema({"q": Choice(instructions="q", criteria=None)})
    made = [("7", -0.1), ("8", -0.2)]
    built, notes = engine.constructed("s", schema, proposals={"q": made})
    assert proposer.asked == [], "it must not propose again"
    assert list(built["q"].descriptions) == ["7", "8"]
    assert notes["q"]["proposals"] == 2


def test_proposals_given_for_one_question_do_not_serve_another():
    proposer = Proposer(("4", "5"))
    engine = SchemaDecisionEngine(proposer, construction=ConstructionSettings(width=3), readout="coded")
    schema = DecisionSchema({"a": Choice(instructions="qa", criteria=None),
                             "b": Choice(instructions="qb", criteria=None)})
    built, _ = engine.constructed("s", schema, proposals={"a": [("7", -0.1), ("8", -0.2)]})
    assert list(built["a"].descriptions) == ["7", "8"]
    assert list(built["b"].descriptions) == ["4", "5"], "the other was proposed for"
    assert [call["instructions"] for call in proposer.asked] == ["qb"]


def test_constructed_candidates_are_shuffled_unless_rank_order_is_asked_for():
    """A scope, not an opt-in. In rank order the slot a biased readout favours is
    also one particular rank -- the runner-up on all but one of 478 questions --
    and rotation found the bias positional, so the confound outlives the labels
    the content readout removes."""
    from hde.candidates.construction_settings import DEFAULT_SHUFFLE

    assert ConstructionSettings().shuffle == DEFAULT_SHUFFLE
    assert ConstructionSettings(shuffle=None).shuffle is None
    # Fixed, so a resume lists a question the way the dead run did.
    assert ConstructionSettings().as_record()["shuffle"] == DEFAULT_SHUFFLE


def test_the_launcher_can_reach_both(capsys):
    """The flag existing with no path to it is how three faults here worked."""
    import argparse

    from hde.candidates.proposing import add_proposal_arguments

    parser = add_proposal_arguments(argparse.ArgumentParser())
    assert parser.parse_args([]).shuffle == ConstructionSettings().shuffle
    assert parser.parse_args(["--no-shuffle-candidates"]).shuffle is None
    assert parser.parse_args(["--shuffle-candidates", "run7"]).shuffle == "run7"


# ---------------------------------------------------------------- readout modes


def test_a_head_says_which_readout_its_features_came_from():
    from hde._legacy.heads import HeadMetadata

    assert HeadMetadata("m", "r", hidden_size=4, feature_layers=(-1,)).readout == "coded"
    assert HeadMetadata("m", "r", hidden_size=4, feature_layers=(-1,),
                        readout="content").readout == "content"


def test_a_coded_head_is_refused_under_the_content_readout():
    """The two produce different quantities on different scales -- a distribution
    over letter codes against a softmax over option scores -- so a head fitted to
    one would be reading numbers that do not mean what it learned they meant, and
    its output would look ordinary."""
    from hde._legacy.heads import HeadMetadata

    metadata = HeadMetadata("m", "r", hidden_size=4, feature_layers=(-1,))
    with pytest.raises(ValueError, match="different quantities"):
        metadata.check_serving("m", readout="content")
    metadata.check_serving("m", readout="coded")
    metadata.check_serving("m"), "no readout named means no claim to check"


def test_the_engine_refuses_the_mismatch_when_it_is_built():
    """At construction rather than at the first prediction, so a run does not load
    a model and then fail on question one."""
    from hde._legacy.heads import HeadMetadata

    class Head:
        metadata = HeadMetadata("m", "r", hidden_size=0, feature_layers=(),
                                readout="coded")

        def predict(self, features):
            raise AssertionError("it must refuse before predicting")

    # A coded-trained head under the content readout: refused when the engine is
    # built, before anything is loaded. The provider can score content, so the
    # head is what this is about rather than the provider's capability.
    with pytest.raises(ValueError, match="different quantities"):
        SchemaDecisionEngine(ContentScorer(), head=Head(), readout="content")
    SchemaDecisionEngine(ContentScorer(), head=Head(), readout="coded")

    # The mismatch check itself still bites, through the metadata.
    with pytest.raises(ValueError, match="different quantities"):
        Head.metadata.check_serving("m", readout="content")


def test_the_model_check_compares_the_engines_model_not_the_heads_own():
    """Passing the head's own id compared a value with itself and passed whatever
    it held, so the model check at that call site never checked anything."""
    from hde._legacy.heads import HeadMetadata

    class Head:
        metadata = HeadMetadata("trained-on-this", "r", hidden_size=0, feature_layers=())

        def predict(self, features):
            raise AssertionError("it must refuse before predicting")

    with pytest.raises(ValueError, match="different model"):
        SchemaDecisionEngine(Proposer(), head=Head(), model_id="serving-that", readout="coded")
    SchemaDecisionEngine(Proposer(), head=Head(), model_id="trained-on-this", readout="coded")


def test_an_unknown_readout_is_named():
    with pytest.raises(ValueError, match="unknown readout"):
        SchemaDecisionEngine(Proposer(), readout="vibes")


# ------------------------------------------------- the content readout, on the path


class ContentScorer(Proposer):
    """Scores each option's text. `sum` and `mean` disagree here on purpose.

    `hidden` is what a scorer built with feature layers returns: the state at the
    readout position, which is the last prefix token and so is one vector for the
    question rather than one per option. `ContentScorerNoLayers` is the other case.
    """

    hidden = numpy.ones((1, 4), dtype=numpy.float32)

    def score_options(self, state, question, content_free=False):
        from hde.readout.content_readout import OptionScore

        # "aa" is the better total; "bbbb" the better per token. So which option
        # comes back says which normalisation actually drove the readout.
        table = {"aa": (-2.0, 2, -1.0), "bbbb": (-3.0, 4, -1.5)}
        found = []
        for index, text in enumerate(question.descriptions):
            total, tokens, prior = table.get(text, (-9.0, 3, -1.0))
            found.append(OptionScore(index=index, text=text, tokens=tokens,
                                     total=total, prior=prior if content_free else None))
        return found, 0, self.hidden


class ContentScorerNoLayers(ContentScorer):
    """A scorer built with no feature layers, which returns no state."""

    hidden = None


def test_the_content_readout_is_on_the_serving_path():
    """It used to raise, because `evaluate` read out through provider.score
    whatever the engine said -- so selecting content recorded a readout the run
    did not use."""
    engine = SchemaDecisionEngine(ContentScorer(), readout="content")
    schema = DecisionSchema({"q": Choice(instructions="Pick",
                                         criteria={"A": "aa", "B": "bbbb"})})
    found = engine.evaluate("s", schema)
    assert found.result.answers["q"].choice == "A", "sum prefers the better total"
    assert found.diagnostics["readout"] == "content"
    assert found.diagnostics["normalisation"] == "sum"
    assert found.diagnostics["fields"]["q"]["provenance"]["readout"] == "content"


def test_the_normalisation_changes_which_option_is_read_out():
    """Otherwise the setting would be a label, which is the fault the raise existed
    to prevent one level up."""
    schema = DecisionSchema({"q": Choice(instructions="Pick",
                                         criteria={"A": "aa", "B": "bbbb"})})
    by_sum = SchemaDecisionEngine(ContentScorer(), readout="content",
                                  normalisation="sum").evaluate("s", schema)
    by_mean = SchemaDecisionEngine(ContentScorer(), readout="content",
                                   normalisation="mean").evaluate("s", schema)
    assert by_sum.result.answers["q"].choice == "A"
    assert by_mean.result.answers["q"].choice == "B", "mean prefers the better per token"
    assert by_mean.diagnostics["normalisation"] == "mean"


def test_sum_is_the_default_and_the_choice_is_recorded():
    """Chosen on confidence, length neutrality and short-answer accuracy, which
    agree across backbones, rather than on accuracy, which cannot separate it
    from mean on 1500 questions."""
    engine = SchemaDecisionEngine(ContentScorer(), readout="content")
    assert engine.normalisation == "sum"
    found = engine.evaluate("s", DecisionSchema(
        {"q": Choice(instructions="Pick", criteria={"A": "aa", "B": "bbbb"})}))
    assert found.diagnostics["normalisation"] == "sum"
    # The coded path records no normalisation rather than a default that did not apply.
    coded = SchemaDecisionEngine(Readout(), readout="coded").evaluate("s", DecisionSchema(
        {"q": Choice(instructions="Pick", criteria={"A": "1", "B": "2"})}))
    assert coded.diagnostics["normalisation"] is None
    assert coded.diagnostics["readout"] == "coded"


def test_only_pmi_pays_for_the_content_free_pass():
    """It is a second forward over every option, and sum and mean do not use it."""
    class Counting(ContentScorer):
        asked_content_free = []

        def score_options(self, state, question, content_free=False):
            Counting.asked_content_free.append(content_free)
            return super().score_options(state, question, content_free)

    schema = DecisionSchema({"q": Choice(instructions="Pick",
                                         criteria={"A": "aa", "B": "bbbb"})})
    for normalisation, expected in (("sum", False), ("mean", False), ("pmi", True)):
        Counting.asked_content_free.clear()
        SchemaDecisionEngine(Counting(), readout="content",
                             normalisation=normalisation).evaluate("s", schema)
        assert Counting.asked_content_free == [expected], normalisation


def test_a_provider_that_cannot_score_content_is_refused_when_the_engine_is_built():
    """Not at the first question. With content as the default, a provider without
    `score_options` -- SGLang, the project's standard backend -- would otherwise
    load a model and fail on question one."""
    with pytest.raises(NotImplementedError, match="cannot score option content"):
        SchemaDecisionEngine(Readout(), readout="content")
    SchemaDecisionEngine(Readout(), readout="coded")
    SchemaDecisionEngine(ContentScorer())


def test_an_unknown_normalisation_is_named():
    with pytest.raises(ValueError, match="unknown normalisation"):
        SchemaDecisionEngine(ContentScorer(), readout="content", normalisation="median")


def test_coded_is_what_an_engine_reads_out_with_unless_told_otherwise():
    """The user's ruling of 2026-10-01, which reverses the earlier one on cost.

    The number that moved it is from the serving backend: 0.0789 s against
    1.1945 s per question on the 27B (28284), 15.1x, where the transformers figure
    that the first ruling was taken under is 2.5x. What an optimised engine makes
    fast is the one forward pass coded needs, not the k sequences content needs.

    The part worth keeping from the first ruling is unchanged: whichever readout a
    run uses, it names it. The content readout's own advantages are accuracy, bias
    and confidence, none of which is cost, and `--readout content` buys them."""
    from hde._legacy.decision import DEFAULT_READOUT

    assert DEFAULT_READOUT == "coded"
    assert SchemaDecisionEngine(Readout()).readout == "coded"
    assert SchemaDecisionEngine(ContentScorer(), readout="content").readout == "content"


def test_a_head_cannot_be_served_under_the_readout_it_was_not_trained_on():
    """The guarantee survives the default moving, which is the point of testing the
    property rather than the direction. Under the content default a coded head was
    the one refused; under the coded default it is a content head, and every head
    trained so far is coded, so this is now the case that bites."""
    from hde._legacy.heads import HeadMetadata

    class ContentHead:
        metadata = HeadMetadata("m", "r", hidden_size=0, feature_layers=(),
                                readout="content")

        def predict(self, features):
            raise AssertionError("it must refuse before predicting")

    class CodedHead:
        metadata = HeadMetadata("m", "r", hidden_size=0, feature_layers=())

        def predict(self, features):
            raise AssertionError("it must refuse before predicting")

    with pytest.raises(ValueError, match="different quantities"):
        SchemaDecisionEngine(Readout(), head=ContentHead())
    SchemaDecisionEngine(ContentScorer(), head=ContentHead(), readout="content")

    # And the other way, so neither direction is correct merely by luck.
    with pytest.raises(ValueError, match="different quantities"):
        SchemaDecisionEngine(ContentScorer(), head=CodedHead(), readout="content")
    SchemaDecisionEngine(Readout(), head=CodedHead())


def test_the_constructed_path_follows_the_global_default_and_says_so():
    """It takes `DEFAULT_READOUT` rather than a readout of its own, so the user's
    ruling reaches the free-form path too -- which now means a free-form run reads
    letter codes unless it is told otherwise.

    That is a consequence worth naming rather than burying: constructed candidates
    are answer text, and labelling them A/B/C puts back the position prior the
    content readout was adopted to remove. `--readout content` is how a free-form
    run asks for it, and the MATH-500 line was designed around that readout."""
    import argparse

    from hde._legacy.decision import DEFAULT_READOUT
    from hde.candidates.proposing import add_proposal_arguments

    parser = add_proposal_arguments(argparse.ArgumentParser())
    parser.add_argument("--readout", choices=["coded", "content"], default=DEFAULT_READOUT)
    assert parser.parse_args([]).readout == DEFAULT_READOUT == "coded"
    assert parser.parse_args(["--readout", "content"]).readout == "content"


def test_the_fast_scores_come_from_the_readout_the_engine_is_set_to():
    """A caller that does not serve through `evaluate` still has to read out the
    way the engine says. `open_three_policies` called `provider.score` directly
    and so recorded a readout it was not using -- the same fault the engine's own
    refusal was written to prevent, one level up."""
    schema = DecisionSchema({"q": Choice(instructions="Pick",
                                         criteria={"A": "aa", "B": "bbbb"})})

    content = ContentScorer()
    (row,) = SchemaDecisionEngine(content, readout="content").readout_scores("s", schema)
    assert row.provenance["readout"] == "content"
    assert row.provenance["normalisation"] == "sum"
    # -2.0 for "aa" and -3.0 for "bbbb": the option scores, not code logits.
    assert list(row.logits) == [-2.0, -3.0]

    coded = Readout()
    (row,) = SchemaDecisionEngine(coded, readout="coded").readout_scores("s", schema)
    assert row.provenance.get("readout") != "content"
    assert list(row.logits) == [0.0, 1.0], "the coded stub's own logits"


def test_the_script_reads_out_through_the_engine():
    """Asserted on the source, because the fault was a call that bypassed it and
    a passing test suite said nothing."""

    from conftest import script_source
    source = script_source("open_three_policies")
    assert "engine.readout_scores(" in source
    assert "scorer.score(state, schema)" not in source, (
        "the fast column must not go round the engine's readout")
    assert '"readout": arguments.readout' in source, "and the run must record which"


def test_a_content_row_carries_the_state_its_scorer_produced():
    """It used to carry none whatever the scorer did, which made routing a
    coded-only capability: no state, no head trainable under this readout, and the
    engine rightly refusing to serve a coded head here. The state is the last prefix
    position -- one vector for the question, not one per option -- so it comes back
    beside the scores rather than inside them.

    A scorer built with no feature layers still returns none, and that is the case
    that must not silently become a zero vector."""
    schema = DecisionSchema({"q": Choice(instructions="Pick",
                                         criteria={"A": "aa", "B": "bbbb"})})
    (row,) = SchemaDecisionEngine(ContentScorer(),
                                  readout="content").readout_scores("s", schema)
    assert row.hidden is not None and row.hidden.shape == (1, 4)

    (row,) = SchemaDecisionEngine(ContentScorerNoLayers(),
                                  readout="content").readout_scores("s", schema)
    assert row.hidden is None, "no layers asked for, no state invented"

    coded = Readout()
    (row,) = SchemaDecisionEngine(coded, readout="coded").readout_scores("s", schema)
    assert row.hidden is not None, "the coded readout does carry one"
    # This half only shows the engine passes the stub's state through. That the
    # real scorer produces one is test_systemtwo's
    # `test_fast_path_returns_readout_hidden_states`, which checks the values
    # against the model's own hidden_states at both layers.


def test_a_content_head_can_now_be_served_under_the_content_readout():
    """The capability the state unblocks. The engine still refuses to mix the two
    readouts -- a softmax over option scores and a distribution over letter codes are
    different quantities -- but a head trained under content is now servable at all,
    which is what made routing a coded-only path."""
    from hde._legacy.heads import HeadMetadata

    class ContentHead:
        metadata = HeadMetadata("m", "r", hidden_size=4, feature_layers=(-1,),
                                readout="content")
        seen = []

        def predict(self, features):
            ContentHead.seen.append(features.shape)
            return numpy.array([[0.25, 0.25, 0.25, 0.25]])

    schema = DecisionSchema({"q": Choice(instructions="Pick",
                                         criteria={"A": "aa", "B": "bbbb"})})
    engine = SchemaDecisionEngine(ContentScorer(), head=ContentHead(), readout="content")
    engine.evaluate("s", schema)
    assert ContentHead.seen, "the head read the content row's state"

    # And the refusal that still has to hold.
    with pytest.raises(ValueError, match="different quantities"):
        SchemaDecisionEngine(Readout(), head=ContentHead(), readout="coded")
