"""The pipeline: one typed request in, a typed answer and the mode that produced it out.

The sequence is the paper's Figure 1. The fast-thinking mode encodes the prompt and
reads an answer at the boundary, producing on the same pass the hidden state the
self-routing block reads; the block returns the joint outcome; the gain it implies
is compared with the request's price; and only then, for the questions that clear
it, does the slow-thinking mode run.

The block and its policy are in :mod:`litjev.core.block`, the prompts in
:mod:`litjev.core.prompts`, and the typed objects this returns in
:mod:`litjev.core.pipeline_utils`.
"""

from dataclasses import dataclass, field  # noqa: F401
from time import perf_counter
from typing import Literal, Protocol  # noqa: F401

import numpy as np

from litjev.core.block import (
    OUTCOME_CLASSES,
    RoutingPolicy,
    escalation_gain,
    fast_confidence,
    should_escalate,
)
from litjev.core.pipeline_utils import (
    CONFIDENCE_METHOD,
    DEFAULT_READOUT,
    HEAD_CONFIDENCE_METHOD,
    NORMALISATIONS,
    RANKED_BY,
    READOUTS,
    ChoiceAnswer,
    DecisionResponse,
    Evaluation,
    NoulAnswer,
    RawFieldScores,
    ScoreAnswer,
    Usage,
    concentration,
)
from litjev.readout.scoring import calibrated_distribution


@dataclass
class Pass:
    """What one request accumulates as it crosses the pipeline's stages.

    One object rather than four dictionaries threaded through a two-hundred-line
    function: each stage takes it, fills its own part and hands it on, so a stage
    can be read and changed without the ones around it.

    `readable` is None when construction left nothing to choose between, which is
    a result rather than a failure and ends the pass early.
    """

    policy: object
    budget: int
    notes: dict = field(default_factory=dict)
    readable: object = None
    answers: dict = field(default_factory=dict)
    fields: dict = field(default_factory=dict)
    outcomes: dict = field(default_factory=dict)
    escalate: list = field(default_factory=list)
    scores: tuple = ()
    output_tokens: int = 0
    # The fast pass is a prefill and a readout on every request.
    forward_calls: int = 2


class SchemaDecisionEngine:
    def __init__(
        self,
        provider,
        temperature=1.0,
        model_id="unknown",
        calibration_fitted=False,
        head=None,
        routing=None,
        construction=None,
        readout=DEFAULT_READOUT,
        normalisation="sum",
    ):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        self.provider = provider
        # How to build a candidate set for a question that arrived without one.
        # None means this engine will not build them, and a question with
        # `criteria=None` is then an error rather than a silent proposal pass.
        self.construction = construction
        self.temperature = temperature
        self.model_id = model_id
        self.calibration_fitted = calibration_fitted
        if readout not in READOUTS:
            raise ValueError(f"unknown readout {readout!r}; use one of {sorted(READOUTS)}")
        if normalisation not in NORMALISATIONS:
            raise ValueError(
                f"unknown normalisation {normalisation!r}; use one of "
                f"{sorted(NORMALISATIONS)}")
        self.readout = readout
        # Which of the option scores is ranked on, under the content readout. The
        # default is `sum`, chosen on three axes that agree across backbones
        # rather than on accuracy, which cannot separate it from `mean` on 1500
        # questions: confidence AUROC (0.762 against 0.692 at 4B, 0.837 against
        # 0.749 at 27B), length neutrality within a question (+0.048 and +0.080
        # against +0.394 and +0.465), and accuracy where the answer is the
        # shortest option (0.701 against 0.591 at 27B). Recorded with every
        # result, because two runs under different normalisations are two
        # different measurements.
        self.normalisation = normalisation
        if readout == "content" and not hasattr(provider, "score_options"):
            raise NotImplementedError(
                f"{type(provider).__name__} cannot score option content, which is what "
                f"the default readout does; pass readout='coded' to serve letter codes. "
                f"Both shipped backends implement it, so this is a provider that does "
                f"not, and it is refused here rather than at the first question")
        self.head = head
        # Refused here rather than at the first prediction. A head fitted to one
        # readout's distribution and served under the other reads numbers that do
        # not mean what it learned they meant, and its output looks ordinary.
        if head is not None:
            # The engine's model, not the head's own, which would compare a value
            # against itself and pass whatever it held. "unknown" is the default
            # and is not a claim about which model this is, so it checks the
            # readout alone rather than failing every engine built without one.
            head.metadata.check_serving(
                self.model_id if model_id != "unknown" else head.metadata.model_id,
                readout=readout)
        self.routing = routing if routing is not None else RoutingPolicy()

    def decide(self, state, schema, routing=None):
        return self.evaluate(state, schema, routing).result

    def constructed(self, state, schema, proposals=None):
        """`(schema, notes)` with every question's candidate set present.

        A question arrives either with its answers or without them, and that is
        the one thing deciding which path it takes. It used to be decided by
        which script was run -- a fact about the operator rather than about the
        question -- which is why the two paths could not be reached through one
        entry point and why every caller had to remember to construct.

        A question that came with options is returned untouched, in source order.
        One that did not has candidates proposed for it and, if the run asks for
        it, listed in a shuffled order. `proposals` supplies them instead, for a
        caller that has to see the raw proposals before they become a candidate
        set -- the gate watches them for the collapse and for option letters, and
        proposing twice would be a second draw rather than the same one. `notes` carries per question what was
        built and why it may be unusable, because a question the construction
        could not turn into a choice is a fact about the construction and belongs
        in the record rather than in a silent fallback.
        """
        from litjev.candidates.construct import schema_from_beams
        from litjev.candidates.proposing import propose

        open_questions = [name for name in schema.names if not schema[name].enumerated]
        if not open_questions:
            return schema, {}
        if self.construction is None:
            raise ValueError(
                f"{len(open_questions)} question(s) arrived with no candidate set and this "
                f"engine was not given construction settings, so there is nothing to read "
                f"out. Pass ConstructionSettings, or send questions that carry their "
                f"options: {open_questions[:3]}")

        settings, questions, notes = self.construction, {}, {}
        for name in schema.names:
            question = schema[name]
            if question.enumerated:
                questions[name] = question
                continue
            # The construction's own state, not the request's: the request's tells
            # the model to answer from listed options, and there are none.
            made = (proposals or {}).get(name)
            if made is None:
                # The construction's state, not the request's: the request's tells
                # the model to answer from listed options, and there are none.
                made = propose(self.provider, settings.state, question.instructions,
                               settings.width, settings.max_new_tokens,
                               settings.temperature, seed=settings.seed,
                               verified=settings.verified)
            built, texts, reason = schema_from_beams(
                made, instructions=question.instructions, rank=settings.rank,
                shuffle=settings.shuffle, question_id=name)
            notes[name] = {"candidates": texts, "reason": reason,
                           "proposals": len(made)}
            if built is None:
                # Nothing to choose between. Marked rather than dropped: a
                # benchmark whose proposals collapse is a fact about the benchmark.
                questions[name] = question
            else:
                questions[name] = built
        return type(schema)(questions), notes

    def _distribution(self, row, question):
        if len(row.logits) != len(question.choices):
            raise RuntimeError("Scorer returned mismatched candidates")
        distribution = calibrated_distribution(
            row.logits, list(range(len(question.choices))), self.temperature
        )
        probabilities = dict(zip(question.choices, distribution.probabilities, strict=True))
        return distribution, probabilities

    @staticmethod
    def _answer(question, distribution, probabilities, confidence):
        if question.type == "noul":
            return NoulAnswer("noul", probabilities["true"])
        if question.type == "score":
            return ScoreAnswer(
                "score",
                sum(int(k) * p for k, p in probabilities.items()),
                dict(zip(question.choices, question.descriptions, strict=True)),
                probabilities,
                confidence,
            )
        return ChoiceAnswer(
            "choice", question.choices[distribution.winner_index], probabilities, confidence
        )

    def _outcome(self, row, question, distribution):
        """Decision head prediction for one fast readout, or None without a head."""
        if self.head is None:
            return None
        from litjev.core.block_utils import build_features

        hidden = row.hidden
        if not self.head.metadata.feature_layers:
            hidden = np.zeros(0, dtype=np.float32)  # stats-only head
        elif hidden is None:
            raise RuntimeError("Decision head requires readout hidden states from the scorer")
        features = build_features(hidden, distribution.probabilities, question.type)
        return self.head.predict(features)[0]

    def evaluate(self, state, schema, routing=None, router=None):
        """One request, in five stages, in the order of the paper's Figure 1.

        `router` is an optional `MarginRouter`, which routes *instead of* the
        priced head. It is passed in rather than held here because its cap is over
        recent decisions, so it has a lifetime -- one pass over a benchmark -- and
        that lifetime is the caller's knowledge, not the engine's. An engine that
        cached one per policy would silently carry a window across repeats.
        """
        clock = {}

        def timed(name, stage, *arguments):
            """One number per stage. A single `decision_seconds` says an
            escalation was expensive without saying which half spent it: the
            candidate construction in the schema stage is a model call too, and
            on a constructed benchmark it can cost more than the readout."""
            started = perf_counter()
            try:
                return stage(*arguments)
            finally:
                clock[name] = perf_counter() - started

        run = timed("policy_seconds", self._stage_policy, routing, router)
        run = timed("schema_seconds", self._stage_schema, run, state, schema)
        if run.readable is None:
            return self._evaluation(run.answers, run.fields, [], 0, run.policy, run.notes,
                                    clock=clock)
        run = timed("fast_seconds", self._stage_fast, run, state, router)
        run = timed("slow_seconds", self._stage_slow, run, state)
        return self._evaluation(
            run.answers, run.fields, run.escalate, run.output_tokens, run.policy, run.notes,
            budget=run.budget,
            forward_calls=run.forward_calls,
            input_tokens=run.scores[0].input_tokens,
            clock=clock,
        )

    def _stage_policy(self, routing, router):
        """Which policy decides, and with what budget. Reads no model."""
        policy = routing if routing is not None else self.routing
        if router is not None:
            if routing is not None and getattr(routing, "lambda_", None) is not None:
                raise ValueError(
                    "a margin router and a priced policy would both decide the same "
                    "question; pass one or the other")
            # The priced path is off: nothing is escalated by gain, and the head
            # is still read so that its outcome and confidence appear in the row.
            # **`budget=0` here was a RED, found by code-reviewer.** This same
            # field is what reaches `solve` and `think`, so every escalation the
            # margin rule made would have thought with **zero tokens** -- a
            # trained head against a threshold that cannot think, which would
            # have read as the threshold failing. The disabled policy is for the
            # *gain* path only; the budget is kept separately below.
            policy = RoutingPolicy(lambda_=0.0, budget=0)
        budget = router.policy.budget if router is not None else policy.budget
        # A price below any possible gain escalates every question, so it is a policy
        # the head cannot influence: always-think needs no head, and refusing it was
        # refusing to measure the baseline a benchmark compares the routing against.
        # `escalation_gain` is a difference of probabilities and so lies in [-1, 1],
        # which is why -1 is the boundary rather than a number chosen here.
        unconditional = policy.lambda_ <= -1.0
        if policy.enabled and self.head is None and not unconditional:
            raise ValueError(
                f"routing at lambda {policy.lambda_} needs a decision head to price the "
                f"gain against. Only a lambda at or below -1 escalates every question "
                f"regardless, which is always-think and needs no head")
        # The entry point constructs. Leaving it to callers is what made the two
        # paths reachable only through the script that remembered, and a question
        return Pass(policy=policy, budget=budget)

    def _stage_schema(self, run, state, schema):
        """Construct candidates where a question has none, and drop what cannot be read."""
        schema, run.notes = self.constructed(state, schema)

        # A construction that produced nothing to choose between never reaches a
        # readout: one candidate is the answer, since scoring a set of one gives a
        # point mass and statistics describing nothing; none is no answer at all.
        # Both are marked, because a benchmark whose proposals collapse is a fact
        # about the benchmark and a silent fallback would report it as an ordinary
        # answer.
        unreadable = {name: note for name, note in run.notes.items() if note["reason"]}
        for name, note in unreadable.items():
            found = note["candidates"]
            run.answers[name] = (ChoiceAnswer("choice", found[0], {found[0]: 1.0}, 0.0)
                             if found else None)
            run.fields[name] = {"construction": note, "system": "one",
                            "readout": "skipped: " + note["reason"]}
        kept = {name: schema[name] for name in schema.names if name not in unreadable}
        if not kept:
            # Every question collapsed during construction. That is a result --
            # which questions the construction could not turn into a choice -- so
            # it returns with the reasons rather than failing.
            return run
        run.readable = type(schema)(kept)
        return run

    def _stage_fast(self, run, state, router):
        """One prefill, one readout: the typed answer, and the escalate decision."""
        run.scores = self.readout_scores(state, run.readable)
        if tuple(row.name for row in run.scores) != run.readable.names:
            raise RuntimeError("Scorer returned mismatched run.fields")
        for row in run.scores:
            question = run.readable[row.name]
            distribution, probabilities = self._distribution(row, question)
            outcome = self._outcome(row, question, distribution)
            if outcome is None:
                confidence = concentration(distribution.probabilities)
                if run.policy.enabled and self.head is None:
                    # The unconditional case allowed above: there is no outcome to
                    # price and the run.policy escalates every question by construction,
                    # which is the always-think baseline.
                    run.escalate.append(row.name)
            else:
                confidence = fast_confidence(outcome)
                run.outcomes[row.name] = outcome
                if should_escalate(outcome, run.policy):
                    run.escalate.append(row.name)
            # **The margin rule reads the distribution, not the head**, so it is
            # decided here rather than inside `should_escalate` and it applies
            # whether or not a head is loaded. It is AnyJev's published rule and
            # the floor our head has to clear; the comparison exists on MMLU-Pro
            # (the fitted-nothing table) and not on JevBench, which is why it is
            # reachable from a benchmark run.
            if router is not None:
                if router.decide(distribution.probabilities):
                    run.escalate.append(row.name)
            run.answers[row.name] = self._answer(question, distribution, probabilities, confidence)
            run.fields[row.name] = {
                "logits": tuple(float(value) for value in row.logits),
                "probabilities": probabilities,
                "max_probability": distribution.winner_probability,
                "concentration": concentration(distribution.probabilities),
                "system": "one",
                "provenance": {**row.provenance, "temperature": self.temperature},
            }
            if outcome is not None:
                run.fields[row.name]["outcome"] = dict(
                    zip(OUTCOME_CLASSES, (float(p) for p in outcome), strict=True)
                )
                run.fields[row.name]["escalation_gain"] = escalation_gain(outcome)
        return run

    def _stage_slow(self, run, state):
        """Only the escalated questions, and only here."""
        run.output_tokens = 0
        run.forward_calls = 2
        # A constructed question never escalates to `think`. `think` renders the
        # question with every option in it and reads out their argmax, so a
        # question whose options are the fast mode's own proposals would hand them
        # to the slow mode and confine it to them -- the slow mode answering from
        # inside the fast mode's guesses, which is not the run.policy the
        # always-thinking column measures. It gets the question alone and run.answers
        # free-form. This is the ec288a1 fault, which I once asserted twice was
        # structurally impossible; it is prevented here by construction and by the
        # test that reads the rendered prompt.
        constructed_names = set(run.notes)
        open_escalations = [name for name in run.escalate if name in constructed_names]
        closed = [name for name in run.escalate if name not in constructed_names]
        for name in open_escalations:
            question = run.readable[name]
            # The construction's state, not the request's -- the same reason the
            # proposer gets it. The request's tells the model to answer from
            # listed options and this question has none; and the all-slow baseline
            # these runs compare against uses the option-free one, so a routed
            # slow answer rendered differently would not be the same run.policy.
            written, spent, truncated = self.provider.solve(
                self.construction.state, question.instructions, run.budget)
            run.output_tokens += spent
            # One prefill and one forward per generated token, per call. These run
            # one after another rather than in a batch, so they add rather than
            # taking the longest, and `run.output_tokens` already carries the think
            # path's tokens so it cannot be reused here.
            run.forward_calls += 1 + spent
            # Imported here: litjev.construct pulls in the optional maths
            # comparison, and the engine is served in deployments that have no
            # constructed questions and no reason to install it.
            from litjev.candidates.construct import extract_answer

            answer, closed_thinking = extract_answer(written)
            outcome = run.outcomes[name]
            run.answers[name] = ChoiceAnswer(
                "choice", answer, {answer: 1.0} if answer else {},
                float(outcome[0] + outcome[2]))
            run.fields[name].update({
                "system": "two", "generated_tokens": spent,
                "slow_free_form": True, "slow_text": written,
                "slow_truncated": bool(truncated),
                "slow_left_its_thinking": bool(closed_thinking),
            })
        if closed:
            slow = self.provider.think(state, run.readable, tuple(closed), run.budget)
            if tuple(row.name for row in slow) != tuple(closed):
                raise RuntimeError("Slow path returned mismatched run.fields")
            for row in slow:
                question = run.readable[row.name]
                distribution, probabilities = self._distribution(row, question)
                outcome = run.outcomes.get(row.name)
                # Without a head there is no joint outcome to read a confidence off, so
                # the slow answer reports the same concentration the fast one does. The
                # diagnostics already say which method produced it, so the two cases are
                # distinguishable in the record rather than only here.
                slow_confidence = (float(outcome[0] + outcome[2]) if outcome is not None
                                   else concentration(distribution.probabilities))
                run.answers[row.name] = self._answer(
                    question, distribution, probabilities, slow_confidence
                )
                run.output_tokens += row.generated_tokens
                run.fields[row.name].update(
                    {
                        "fast_logits": run.fields[row.name]["logits"],
                        "fast_probabilities": run.fields[row.name]["probabilities"],
                        "logits": tuple(float(value) for value in row.logits),
                        "probabilities": probabilities,
                        "max_probability": distribution.winner_probability,
                        "concentration": concentration(distribution.probabilities),
                        "system": "two",
                        "generated_tokens": row.generated_tokens,
                        # What the slow mode actually wrote. It was on the run.scores and
                        # never reached the diagnostics, so a benchmark could record
                        # that a question was escalated and not what came back --
                        # which is why 27866's re-scoring is impossible rather than
                        # pending. The typed answer is still assembled from the
                        # readout; this is the record, not the parse.
                        "reasoning": row.reasoning,
                        "slow_provenance": {**row.provenance, "temperature": self.temperature},
                    }
                )
            # One prefill, one forward per generated token, one final readout.
            run.forward_calls += 2 + max(row.generated_tokens for row in slow)
        return run


    def readout_scores(self, state, schema):
        """The fast mode's scores, through whichever readout this engine is set to.

        Public because a caller that does not serve through `evaluate` still has
        to read out the way the engine says. `open_three_policies` measures three
        policies rather than serving one, called `provider.score` directly, and so
        recorded a readout it was not using -- the same fault the engine's own
        refusal was written to prevent, one level up.
        """
        return (self._content_scores(state, schema) if self.readout == "content"
                else self.provider.score(state, schema))

    def _content_scores(self, state, schema):
        """Read out by scoring each option's own text, with no label between them.

        Returned in the shape the coded readout returns, so everything after this
        -- the distribution, the head's features, the answer -- is the same code.
        The scores are log-probabilities rather than logits over a shared
        vocabulary, which is why a head carries the readout it was trained under:
        a softmax over these is not the quantity a coded-trained head learned.

        The hidden state comes with the scores now, from the last prefix position --
        the readout position of a prompt that lists the options with no labels. It is
        not the coded readout's state and a head fitted to one does not transfer to
        the other, which `HeadMetadata.readout` records and `__init__` refuses on.
        Before it was returned, routing was a coded-only capability: a content row
        carried no state, so no head could be trained under this readout at all.
        """
        rows = []
        for name in schema.names:
            found, rewritten, hidden = self.provider.score_options(
                state, schema[name], content_free=self.normalisation == "pmi")
            ranked = {score.index: getattr(score, RANKED_BY[self.normalisation])
                      for score in found}
            rows.append(RawFieldScores(
                name, np.array([ranked[i] for i in sorted(ranked)], dtype=np.float64),
                0, {"system": "one", "readout": "content",
                    "normalisation": self.normalisation,
                    "rewritten_options": rewritten}, hidden))
        return tuple(rows)

    def _evaluation(self, answers, fields, escalate, output_tokens, policy, notes,
                    forward_calls=0, input_tokens=0, budget=None, clock=None):
        """The result, assembled the same way whether or not a readout happened.

        A request whose every question collapsed during construction still
        returns, with the reason per question, rather than failing: which
        questions the construction could not turn into a choice is the thing such
        a run is measuring.
        """
        return Evaluation(
            DecisionResponse(self.model_id, answers, Usage(input_tokens, output_tokens)),
            {
                "fields": fields,
                "forward_calls": forward_calls,
                "calibration_fitted": self.calibration_fitted,
                "confidence_method": HEAD_CONFIDENCE_METHOD if self.head else CONFIDENCE_METHOD,
                "construction": ({name: note for name, note in notes.items()} if notes
                                 else None),
                "readout": self.readout,
                "normalisation": self.normalisation if self.readout == "content" else None,
                # Per stage, so a caller can see where an escalation's time went.
                # The caller adds its own wall-clock around this, and what the
                # two do not account for is transport.
                "timing": {name: round(value, 6) for name, value in (clock or {}).items()},
                "routing": {
                    "enabled": policy.enabled,
                    "lambda": policy.lambda_,
                    # The budget actually spent against, which under a margin
                    # router is the router's and not the disabled policy's zero.
                    "budget": policy.budget if budget is None else budget,
                    "escalated": escalate,
                    "generated_tokens": output_tokens,
                },
            },
        )
